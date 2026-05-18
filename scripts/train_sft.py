import os
import sys
import json
import time
import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField
from cmf.presets import get_preset
from cmf.generation import encode_to_tensor
from cmf.runtime import resolve_device

# Configure output to support UTF-8 on Windows consoles
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

class SFTDataset(Dataset):
    def __init__(self, tokenizer, data_items, max_seq_len=256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.encoded_pairs = []
        
        for item in data_items:
            instruction = item["instruction"].strip()
            response = item["response"].strip()
            
            # Format prompt with conversational markers
            prompt_str = f"User: {instruction}\nAssistant: "
            response_str = f"{response}\n"
            
            prompt_ids = encode_to_tensor(tokenizer, prompt_str)
            response_ids = encode_to_tensor(tokenizer, response_str)
            
            full_ids = torch.cat([prompt_ids, response_ids], dim=0)
            if full_ids.numel() > max_seq_len:
                # Truncate if it exceeds maximum context length
                full_ids = full_ids[:max_seq_len]
                prompt_len = min(prompt_ids.numel(), max_seq_len - 1)
            else:
                prompt_len = prompt_ids.numel()
                
            # Create a label tensor where prompt tokens are replaced by -100
            # (PyTorch's CrossEntropyLoss ignores -100 automatically!)
            labels = full_ids.clone()
            labels[:prompt_len] = -100
            
            self.encoded_pairs.append((full_ids, labels))
            
    def __len__(self):
        return len(self.encoded_pairs)
        
    def __getitem__(self, idx):
        full_ids, labels = self.encoded_pairs[idx]
        return full_ids, labels

def collate_sft(batch, pad_token_id=50256):
    # Dynamically pad sequences in the batch to the longest sequence
    max_len = max(x[0].numel() for x in batch)
    
    xs, ys = [], []
    for full_ids, labels in batch:
        pad_len = max_len - full_ids.numel()
        
        padded_x = torch.cat([full_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)], dim=0)
        padded_y = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)], dim=0)
        
        xs.append(padded_x)
        ys.append(padded_y)
        
    return torch.stack(xs), torch.stack(ys)

def get_synthetic_sft_dataset() -> list[dict]:
    # 200+ high-quality synthetic SFT instructions for facts, reasoning, and programming
    data = []
    
    # Factual QA
    facts = [
        ("what is cuda?", "CUDA is a parallel computing platform and application programming interface model created by NVIDIA for GPU acceleration."),
        ("what is cmf?", "CMF stands for Continuous Meaning Field, a deep learning architecture that represents symbols as paths in a continuous semantic space."),
        ("what is python?", "Python is a high-level, general-purpose, interpreted programming language known for readability and an extensive ecosystem."),
        ("explain deep learning.", "Deep learning is a subset of machine learning based on artificial neural networks with multiple layers that learn hierarchical representations of data."),
        ("what is transformer?", "A Transformer is a neural network architecture that relies on self-attention mechanisms to compute representations without sequential processing.")
    ]
    for q, a in facts:
        data.append({"instruction": q, "response": a})
        
    # Math & Coding Tasks
    for a in range(1, 20):
        for b in range(1, 20):
            data.append({
                "instruction": f"Calculate {a} + {b}.",
                "response": f"The sum of {a} and {b} is {a + b}."
            })
            data.append({
                "instruction": f"Solve: {a} + {b} = ?",
                "response": f"{a} + {b} = {a + b}."
            })
            
    # Text instruction tasks
    data.append({
        "instruction": "Write a short python function to compute factorial.",
        "response": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
    })
    data.append({
        "instruction": "What are the limitations of standard Transformers?",
        "response": "Standard Transformers are limited by discrete token states, quadratic context scaling, and mechanical step-by-step reasoning constraints."
    })
    
    return data

def main():
    print("=" * 70)
    print("🌌 CMF 120M SUPERVISED FINE-TUNING (SFT) ALIGNMENT PIPELINE")
    print("=" * 70)
    
    device = resolve_device("auto")
    print(f"Target Hardware device: {device}")
    
    # Load pretrained stable checkpoint
    checkpoint_path = ROOT / "checkpoint_stable.pt"
    if not checkpoint_path.exists():
        checkpoint_path = ROOT / "checkpoint_latest.pt"
        
    if not checkpoint_path.exists():
        print("Error: Pretrained base checkpoint not found in the root directory!")
        print("Please complete the pretraining step first.")
        sys.exit(1)
        
    print(f"Loading base model weights from: {checkpoint_path}...")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Build model using Deliberative reasoning preset
    preset_name = "infinity-reasoning-0.12b"
    preset = get_preset(preset_name)
    config = CMFConfig(**preset.config.__dict__)
    config.max_seq_len = 256
    
    print(f"Initializing deliberative reasoning model...")
    model = DeliberativeContinuousMeaningField(config)
    
    # Load base weights
    state_dict = payload["model"]
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        key = k
        while True:
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]
            elif key.startswith("module."):
                key = key[len("module."):]
            else:
                break
        cleaned_state_dict[key] = v
    model.load_state_dict(cleaned_state_dict)
    model = model.to(device)
    print("Pretrained base weights mapped successfully.")
    
    # Load Tokenizer
    print("Loading GPT-2 tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Prepare instruction data
    print("Loading alignment datasets...")
    data_items = get_synthetic_sft_dataset()
    
    # Check if a custom local JSON instruction set exists
    local_data_path = ROOT / "sft_dataset.json"
    if local_data_path.exists():
        try:
            with open(local_data_path, "r", encoding="utf-8") as f:
                local_items = json.load(f)
                data_items.extend(local_items)
                print(f"Successfully loaded {len(local_items)} custom prompts from {local_data_path}")
        except Exception as e:
            print(f"Warning loading local JSON: {e}")
            
    dataset = SFTDataset(tokenizer, data_items, max_seq_len=config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_sft)
    print(f"Total compiled alignment pairs: {len(dataset)}")
    
    # Setup optimizer and training params
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    epochs = 3
    print(f"Starting SFT Alignment for {epochs} epochs...")
    
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        step_count = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            
            # Compute cross entropy ONLY on response tokens (where labels are not -100)
            logits = output["logits"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = y[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            step_count += 1
            
        avg_loss = total_loss / step_count
        print(f"Epoch {epoch}/{epochs} | Average Alignment Loss: {avg_loss:.4f}")
        
    print("\nSFT Alignment complete! Saving final conversational model package...")
    out_package = ROOT / "cmf_120m_chat.package.pt"
    
    # Strip optimizer states to keep package lightweight
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "step": payload["training"]["step"],
        "tokenizer_name": "gpt2",
        "scope": "conversational_sft_aligned"
    }, out_package)
    
    print(f"Successfully saved conversational package to: {out_package}")
    print("=" * 70)

if __name__ == "__main__":
    main()
