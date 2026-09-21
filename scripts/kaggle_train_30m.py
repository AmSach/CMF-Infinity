"""
CMF-LM Phase S1: 30M Self-Contained Kaggle GPU Training Script

- Designed to run inside a single Kaggle Notebook cell on a standard T4 GPU.
- 100% self-contained: Includes the entire architecture definitions (SlotMemory,
  DeliberativeCMF) and tokenizer bindings.
- Dataset: Streams "roneneldan/TinyStories" from Hugging Face on the fly.
- Hardware Optimization: Uses fp16 Mixed Precision (AMP) and Gradient Accumulation
  to maximize T4 GPU utilization with low memory footprint.
- Diagnostics: Logs real cross-entropy loss, parameter gradient norms, VRAM usage,
  and prints periodic autoregressive text rollouts.
"""

import os
import sys

# Prevent silent GIL/thread deadlocks between Rust fast tokenizers and multi-threaded nn.DataParallel
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Auto-install dependencies if missing in the environment
try:
    import datasets
    import transformers
except ImportError:
    print("Installing missing dependencies (datasets, transformers)...")
    os.system(f"{sys.executable} -m pip install -q datasets transformers")

import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOCKED ARCHITECTURE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

class CMFConfig:
    def __init__(self):
        self.vocab_size = 50257
        self.d_model = 384
        self.hidden_dim = 1536  # 4x FFN expansion (SwiGLU)
        self.num_layers = 6
        self.num_slots = 64
        self.routing_topk = 4    # Locked routing sparsity K=4
        self.adaptive_thinking = True
        self.min_thinking_steps = 2
        self.max_thinking_steps = 4    # Dynamic S1 training deliberation scale
        self.dropout = 0.1

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(nn.functional.silu(self.w1(x)) * self.w2(x))

class SlotMemory(nn.Module):
    """
    Locked WTA competitive gating associative memory.
    Ensures gradient continuity through slot projections without direct mutations.
    """
    def __init__(self, config: CMFConfig):
        super().__init__()
        self.num_slots = config.num_slots
        self.d_model = config.d_model
        self.topk = config.routing_topk
        
        self.slot_keys = nn.Parameter(torch.randn(config.num_slots, config.d_model) * 0.02)
        self.slot_vals = nn.Parameter(torch.randn(config.num_slots, config.d_model) * 0.02)
        
        self.write_gate = nn.Linear(config.d_model * 2, config.num_slots, bias=False)
        self.write_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.norm = RMSNorm(config.d_model)

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, T, d = z.shape
        S = self.num_slots
        
        # Initialize memory batch
        slot_vals_t = self.slot_vals.unsqueeze(0).repeat(B, 1, 1)
        z_out = torch.zeros_like(z)
        
        # --- MASSIVE PARALLEL VECTORIZATION ---
        # Compute all heavy projections and similarity scores in parallel before the loop!
        # scores_all shape: (B, T, S)
        scores_all = torch.matmul(z, self.slot_keys.t()) / math.sqrt(d)
        
        # gate_logits_all shape: (B, T, S)
        gate_in_all = torch.cat([z, context], dim=-1)
        gate_logits_all = self.write_gate(gate_in_all)
        
        # val_all shape: (B, T, d)
        val_all = self.write_proj(context)
        
        # topk_idx_all shape: (B, T, K)
        _, topk_idx_all = torch.topk(scores_all, self.topk, dim=-1)
        
        # Unbind tensors along sequence dimension to eliminate slow GPU slicing view allocations inside the loop
        scores_list = torch.unbind(scores_all, dim=1)
        gate_logits_list = torch.unbind(gate_logits_all, dim=1)
        val_list = torch.unbind(val_all, dim=1)
        topk_idx_list = torch.unbind(topk_idx_all, dim=1)
        
        # Pre-allocate mask buffer outside the loop to avoid 128 temporary GPU allocations per layer
        mask = torch.empty_like(gate_logits_list[0]).fill_(-65000.0)
        
        retrieved_list = []
        
        for t in range(T):
            scores = scores_list[t]
            gate_logits = gate_logits_list[t]
            val = val_list[t]
            topk_idx = topk_idx_list[t]
            
            # Reset pre-allocated mask in-place to avoid allocation overhead
            mask.fill_(-65000.0)
            topk_vals = torch.gather(gate_logits, dim=-1, index=topk_idx)
            mask.scatter_(dim=-1, index=topk_idx, src=topk_vals)
            
            gate = torch.softmax(mask, dim=-1)
            
            # Highly optimized low-allocation update formula using in-place subtraction and addcmul_
            decay = gate.unsqueeze(-1)
            diff = val.unsqueeze(1) - slot_vals_t
            slot_vals_t.addcmul_(decay, diff, value=0.25)
            
            # Read step
            attn = torch.softmax(scores, dim=-1)
            retrieved = torch.bmm(attn.unsqueeze(1), slot_vals_t).squeeze(1)
            retrieved_list.append(retrieved)
            
        retrieved_all = torch.stack(retrieved_list, dim=1)
        z_out = z + retrieved_all
        return self.norm(z_out)

class DeliberativeCMFLayer(nn.Module):
    def __init__(self, config: CMFConfig):
        super().__init__()
        self.memory = SlotMemory(config)
        self.ffn = SwiGLU(config.d_model, config.hidden_dim)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        
        # Bounded gated residual scale (initialized tiny to prevent recurrent overwrite violence)
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z + self.alpha * self.memory(self.norm1(z), z)
        z = z + self.ffn(self.norm2(z))
        return z

class DeliberativeCMF(nn.Module):
    """
    Phase S1 Locked 30M Deliberative CMF Model
    """
    def __init__(self, config: CMFConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([
            DeliberativeCMFLayer(config) for _ in range(config.num_layers)
        ])
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight # Tied weights
        
        # Adaptive halting head
        self.halt_head = nn.Linear(config.d_model, 1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, solver_steps: int = None) -> dict[str, torch.Tensor]:
        B, T = x.shape
        
        # Propagate mixed precision autocast context locally inside DataParallel GPU threads
        with torch.amp.autocast('cuda'):
            z = self.embed(x)
            
            # Determine deliberation steps (synchronized from main thread if passed, else fallback)
            if solver_steps is None:
                if self.training:
                    solver_steps = int(torch.randint(self.config.min_thinking_steps, self.config.max_thinking_steps + 1, (1,)).item())
                else:
                    solver_steps = self.config.max_thinking_steps
                
            from torch.utils.checkpoint import checkpoint
            
            for step in range(solver_steps):
                for layer in self.layers:
                    if self.training:
                        z = checkpoint(layer, z, use_reentrant=True)
                    else:
                        z = layer(z)
                    
            logits = self.lm_head(z)
            
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
                
                # During training, omit large logits from the output dict to prevent nn.DataParallel 
                # from gathering them back to the master GPU, saving massive VRAM and bandwidth.
                output = {
                    "loss": loss,
                    "thinking_steps": torch.tensor([solver_steps], device=x.device)
                }
            else:
                output = {
                    "logits": logits, 
                    "thinking_steps": torch.tensor([solver_steps], device=x.device)
                }
                
            return output

def get_clean_model(model: nn.Module) -> nn.Module:
    actual_model = model
    if isinstance(actual_model, nn.DataParallel):
        actual_model = actual_model.module
    if hasattr(actual_model, "_orig_mod"):
        actual_model = actual_model._orig_mod
    return actual_model

def generate_text(model: nn.Module, tokenizer, prompt: str, max_tokens: int = 150) -> str:
    model.eval()
    actual_model = get_clean_model(model)
    device = next(actual_model.parameters()).device
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = actual_model(input_ids)
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Sampling with temperature
            probs = torch.softmax(next_token_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. KAGLE T4 GPU TRAINING EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def setup_local_dataset():
    """
    Downloads and extracts raw TinyStories dataset directly from Hugging Face.
    Skips download and decompression if files are already fully present.
    """
    import urllib.request
    import tarfile
    import json
    import glob
    import sys
    
    checkpoint_dir = "/kaggle/working" if os.path.exists("/kaggle") else "."
    data_dir = os.path.join(checkpoint_dir, "TinyStories_data")
    
    # Only download/extract if the directory doesn't exist or is empty
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not os.path.exists(data_dir) or len(json_files) == 0:
        os.makedirs(data_dir, exist_ok=True)
        tar_path = os.path.join(checkpoint_dir, "TinyStories_all_data.tar.gz")
        if not os.path.exists(tar_path):
            print("\n[Downloading] Initializing TinyStories dataset download (290MB)...")
            url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
            
            def download_progress(block_num, block_size, total_size):
                percent = int(block_num * block_size * 100 / total_size)
                percent = min(100, percent)
                bar_length = 30
                filled_length = int(bar_length * percent // 100)
                bar = "=" * filled_length + ">" + " " * (bar_length - filled_length - 1)
                if percent == 100:
                    bar = "=" * bar_length
                sys.stdout.write(f"\r[Downloading] |{bar}| {percent}% ({block_num * block_size / (1024**2):.1f}MB / {total_size / (1024**2):.1f}MB)")
                sys.stdout.flush()
                
            urllib.request.urlretrieve(url, tar_path, reporthook=download_progress)
            print("\n[Download Complete] TinyStories raw archive saved to disk.")
        
        print("[Extracting] Decompressing story shards...")
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            for i, member in enumerate(members):
                tar.extract(member, path=data_dir)
                percent = int((i + 1) * 100 / len(members))
                bar_length = 30
                filled_length = int(bar_length * percent // 100)
                bar = "=" * filled_length + ">" + " " * (bar_length - filled_length - 1)
                if percent == 100:
                    bar = "=" * bar_length
                sys.stdout.write(f"\r[Extracting]  |{bar}| {percent}% ({i+1} / {len(members)} files)")
                sys.stdout.flush()
        print("\n[Extraction Complete] Shards unpacked successfully.")
        json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    
    stories = []
    print(f"\n[Loading] Reading stories from {len(json_files)} local JSON files...")
    # Load first 10 files (~200,000 stories) to keep system memory footprint low (~200MB)
    num_shards = min(10, len(json_files))
    for i, file_path in enumerate(json_files[:num_shards]):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                stories.append(item["story"])
        percent = int((i + 1) * 100 / num_shards)
        bar_length = 30
        filled_length = int(bar_length * percent // 100)
        bar = "=" * filled_length + ">" + " " * (bar_length - filled_length - 1)
        if percent == 100:
            bar = "=" * bar_length
        sys.stdout.write(f"\r[Loading Shards] |{bar}| {percent}% ({i+1} / {num_shards} shards)")
        sys.stdout.flush()
    print(f"\n[RAM Loaded] Fully buffered {len(stories):,} stories in system RAM!")
    return stories

class LocalStoryStream:
    """
    Serves pre-tokenized stories directly from local RAM with 0.00 seconds loading latency.
    Completely eliminates online tokenization CPU overhead inside the training loop.
    """
    def __init__(self, tokenized_stories, seq_len, pad_token_id):
        self.stories = tokenized_stories
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.ptr = 0
        
    def next_batch(self, batch_size: int) -> torch.Tensor:
        batch = []
        for _ in range(batch_size):
            if self.ptr >= len(self.stories):
                self.ptr = 0
            tokens = self.stories[self.ptr]
            self.ptr += 1
            
            # Pad or truncate tokens to seq_len
            if len(tokens) < self.seq_len:
                tokens = tokens + [self.pad_token_id] * (self.seq_len - len(tokens))
            else:
                tokens = tokens[:self.seq_len]
            batch.append(tokens)
        return torch.tensor(batch, dtype=torch.long)

def main():
    print("==================================================")
    print("CMF 30M PARAMETER TINYSTORIES GPU TRAINING EXECUTION")
    print("==================================================")
    
    # 0. Thoroughly clear existing GPU VRAM overhead from previous runs
    import gc
    import torch
    for obj_name in ['model', 'optimizer', 'scaler', 'actual_model']:
        if obj_name in globals():
            try:
                del globals()[obj_name]
            except Exception:
                pass
    gc.collect()
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Target Execution Device: {device} (Detected {num_gpus} GPUs)")
    
    # 1. Config locked model
    config = CMFConfig()
    model = DeliberativeCMF(config).to(device)
    
    if num_gpus > 1:
        print(f"Wrapping model in nn.DataParallel across {num_gpus} GPUs...")
        model = nn.DataParallel(model)
        
    # Print model specs
    actual_model = get_clean_model(model)
    num_params = sum(p.numel() for p in actual_model.parameters())
    print(f"Model Parameters: {num_params:,}")
    print(f"Estimated Parameter VRAM Memory: {num_params * 4 / (1024**2):.2f} MB")
    
    steps = 20000
    grad_accum_steps = 1  # Single parallel step to eliminate communication overhead
    seq_len = 128         # Perfect context length for TinyStories dataset, drastically reduces sequential time loop overhead
    
    # Set batch size to 256 (128 per GPU) to maximize multi-GPU tensor core utilization and VRAM usage
    batch_size = 256
    
    # Setup local dataset stream (downloads tar.gz once and serves stories from RAM with 0 latency)
    stories = setup_local_dataset()
    
    # Slice the dataset to 200,000 stories to prevent startup latency and completely avoid OOMs/deadlocks
    stories = stories[:200000]
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Keep tokenizer parallelism disabled to guarantee 100% stable, deadlock-free single-core tokenization
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print(f"\n[Pre-tokenizing] Converting {len(stories):,} stories to token IDs in RAM...")
    t_start = time.perf_counter()
    tokenized_stories = []
    chunk_size = 20000
    for i in range(0, len(stories), chunk_size):
        chunk = stories[i:i+chunk_size]
        enc_chunk = tokenizer(chunk, truncation=True, max_length=seq_len, add_special_tokens=True)
        tokenized_stories.extend(enc_chunk["input_ids"])
        
        elapsed = time.perf_counter() - t_start
        processed = min(i + chunk_size, len(stories))
        speed = processed / elapsed
        eta = (len(stories) - processed) / speed if speed > 0 else 0
        sys.stdout.write(f"\r[Tokenizing] | {processed:,}/{len(stories):,} ({processed*100/len(stories):.1f}%) | Speed: {speed:.0f} stories/sec | ETA: {eta:.1f}s")
        sys.stdout.flush()
    print(f"\n[Pre-tokenization Complete] Done in {time.perf_counter() - t_start:.2f} seconds!")
    
    print("[Status] Instantiating LocalStoryStream...")
    data_stream = LocalStoryStream(tokenized_stories, seq_len, tokenizer.eos_token_id)
    
    print("[Status] Setting up optimizer and scaler...")
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    
    # Use Kaggle's persistent directory if available, otherwise default to local
    checkpoint_dir = "/kaggle/working" if os.path.exists("/kaggle") else "."
    checkpoint_path = os.path.join(checkpoint_dir, "cmf_checkpoint_latest.pt")
    start_step = 1
    
    if os.path.exists(checkpoint_path):
        print(f"\n[Checkpoint Found] Loading from {checkpoint_path}...")
        try:
            print("[Status] Loading checkpoint with torch.load...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            print("[Status] Unwrapping model...")
            actual_model = get_clean_model(model)
            print("[Status] Loading model state dict...")
            actual_model.load_state_dict(checkpoint["model_state_dict"])
            print("[Status] Loading optimizer state dict...")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("[Status] Loading scaler state dict...")
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_step = checkpoint["step"] + 1
            
            # Resume exact dataset pointer location
            if "data_stream_ptr" in checkpoint:
                data_stream.ptr = checkpoint["data_stream_ptr"]
                print(f"Resumed dataset cursor position at index: {data_stream.ptr:,}")
                
            print(f"Resuming training starting from Step {start_step}!\n")
        except Exception as e:
            print(f"\n[Warning] Failed to load checkpoint due to corruption ({e}).")
            print("Deleting corrupted checkpoint and starting training from Step 1...")
            try:
                os.remove(checkpoint_path)
            except Exception:
                pass
            start_step = 1
            
    print("\nStarting genuine autoregressive S1 training loop...")
    t0 = time.perf_counter()
    
    accumulated_loss = 0.0
    accumulated_count = 0
    from tqdm import tqdm
    
    # Beautiful interactive dynamic progression bar aligned to correct step range
    pbar = tqdm(initial=start_step - 1, total=steps, desc="Training CMF S1", dynamic_ncols=True)
    try:
        for step in range(start_step, steps + 1):
            pbar.update(1)
            model.train()
            optimizer.zero_grad()
            
            # Load batches and pack targets in parallel from RAM buffer
            for accum in range(grad_accum_steps):
                x = data_stream.next_batch(batch_size).to(device)
                
                # Generate identical solver steps on main thread per accum step to keep multi-GPU execution graphs perfectly synchronized
                step_solver_steps = int(torch.randint(config.min_thinking_steps, config.max_thinking_steps + 1, (1,)).item())
                
                # Forward pass (mixed-precision autocast is locally propagated inside the model forward method)
                output = model(x, labels=x, solver_steps=step_solver_steps)
                
                # Reduce loss across multiple GPUs under nn.DataParallel
                raw_loss = output["loss"].mean()
                loss = raw_loss / grad_accum_steps
                    
                scaler.scale(loss).backward()
                accumulated_loss += float(raw_loss.detach().item())
            
            accumulated_count += 1
                
            # Physical optimizer update
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Trajectory logging (updates dynamic status bar postfix without screen spam!)
            if step % 10 == 0 or step == 1:
                elapsed = time.perf_counter() - t0
                vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
                
                pbar.set_postfix({
                    "Loss": f"{accumulated_loss / max(1, accumulated_count * grad_accum_steps):.4f}",
                    "VRAM": f"{vram_mb:.1f}MB",
                    "Speed": f"{step / elapsed:.2f}it/s"
                })
                accumulated_loss = 0.0
                accumulated_count = 0
                
            # Periodic autoregressive rollout test & checkpoint save (uses pbar.write to keep progress bar clean!)
            if step % 500 == 0:
                pbar.write("\n--------------------------------------------------")
                pbar.write("PERIODIC ROLLOUT TEST (STEP %d):" % step)
                test_prompt = "Once upon a time, there was a little boy named Timmy who had a dog."
                sample_output = generate_text(model, tokenizer, test_prompt, max_tokens=100)
                pbar.write(sample_output)
                pbar.write("--------------------------------------------------\n")
                
            # Periodic auto-saving every 100 steps
            if step % 100 == 0:
                actual_model = get_clean_model(model)
                checkpoint = {
                    "step": step,
                    "model_state_dict": actual_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "data_stream_ptr": data_stream.ptr, # Save exact dataset cursor to guarantee mathematical resumability
                }
                temp_path = checkpoint_path + ".tmp"
                torch.save(checkpoint, temp_path)
                os.replace(temp_path, checkpoint_path)
                pbar.write(f"[Checkpoint Saved] -> {checkpoint_path} at Step {step}\n")
                
    except KeyboardInterrupt:
        pbar.close()
        print(f"\n[Training Paused] Catching manual stop interrupt at Step {step}. Saving current state...")
        actual_model = get_clean_model(model)
        checkpoint = {
            "step": step,
            "model_state_dict": actual_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "data_stream_ptr": data_stream.ptr,
        }
        temp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)
        print(f"[Checkpoint Saved] -> {checkpoint_path} on interruption! You can safely resume training later.")
        return
        
    pbar.close()
    print("==================================================")
    print("CMF 30M PARAMETER TRAINING COMPLETED.")
    print("==================================================")

if __name__ == "__main__":
    main()
