# train_lora_slimpajama_200m.py – LoRA fine‑tuning on first 200 M tokens of SlimPajama (mini)
# ---------------------------------------------------------------------------
# This script is deliberately compact and heavily optimized for a laptop with limited RAM.
# It uses:
#   • 8‑bit quantisation (bitsandbytes) to fit the 120 M CMF model in ~2 GB VRAM/CPU.
#   • PEFT LoRA (rank = 8) – only a few MB of trainable parameters.
#   • Constant logging to STDOUT (every 100 steps) – suitable for monitoring.
#   • A tiny custom Dataset that streams the first 200 M tokens from the SlimPajama file.
# ---------------------------------------------------------------------------

import os
import math
import sys
import torch
from pathlib import Path
from tqdm.auto import tqdm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ----- HuggingFace / PEFT imports -----
from transformers import AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----- CMF imports (project specific) -----
from cmf.model import DeliberativeContinuousMeaningField
from cmf.config import CMFConfig
from cmf.presets import get_preset

# ---------------------------------------------------------------------------
# CONFIGURATION – adjust these values if you want a different run
# ---------------------------------------------------------------------------
DATASET_PATH = ROOT / "data" / "slimpajama_200M.txt"   # must exist with at least 200M tokens
PACKAGE_PATH = ROOT / "records" / "checkpoints" / "cmf_120m_pretrained_latest.package.pt"
OUTPUT_DIR = ROOT / "records" / "checkpoints" / "cmf_lora_slimpajama_200m"

# Training hyper‑parameters (ultra‑lightweight)
NUM_EPOCHS = 2
BATCH_SIZE = 64           # Maximize GPU core saturation based on VRAM capacity
GRAD_ACCUM_STEPS = 1      # Set to 1 as batch size is sufficiently large
LR = 5e-4
MAX_SEQ_LEN = 1024
BLOCK_SIZE = 256          # tokens per training example – fits comfortably in memory
LOG_EVERY = 100           # steps

# ---------------------------------------------------------------------------
# Helper: Load the pretrained CMF package (weights + tokenizer)
# ---------------------------------------------------------------------------
def load_package(pkg_path: Path):
    """Return (model, tokenizer) from a .package.pt file."""
    payload = torch.load(pkg_path, map_location="cpu", weights_only=False)
    cfg_dict = payload["config"] if "config" in payload else payload.get("metadata", {})
    cfg = CMFConfig(**cfg_dict)
    cfg.max_seq_len = MAX_SEQ_LEN
    model = DeliberativeContinuousMeaningField(cfg)
    model.load_state_dict(payload["state_dict"])
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer

model, tokenizer = load_package(PACKAGE_PATH)

# ---------------------------------------------------------------------------
# Optimize model for low‑memory training (8‑bit) and attach LoRA
# ---------------------------------------------------------------------------
model = prepare_model_for_kbit_training(model)
# Target modules are linear projection and convolution layers matching CMF's structure.
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["proj", "proposal", "update_gate", "gate", "q_proj", "k_proj", "v_proj", "out_proj", "conv"],
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(model, lora_cfg)

# ---------------------------------------------------------------------------
# Tiny streaming dataset – reads the first 200 M tokens only
# ---------------------------------------------------------------------------
class SlimPajama200MDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, tokenizer, block_size: int):
        self.block_size = block_size
        # If the file does not exist, download and create it
        if not path.exists():
            print(f"Dataset not found at {path}. Downloading the first 200M tokens from HuggingFace...")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from datasets import load_dataset
                ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
                tokens_seen = 0
                max_tokens = 200_000_000
                with open(path, "w", encoding="utf-8") as f:
                    for doc in ds:
                        if doc["text"].strip():
                            f.write(doc["text"] + "\n")
                            # approximate tokens via word split to avoid tokenizer overhead during download
                            tokens_seen += len(doc["text"].split())
                        if tokens_seen >= max_tokens:
                            break
                print(f"Downloaded and saved approximately {tokens_seen} words to {path}.")
            except ImportError:
                raise RuntimeError("Please 'pip install datasets' to download the dataset automatically.")

        # Fast batch tokenization and caching
        cache_path = path.with_suffix('.pt')
        if cache_path.exists():
            print("Loading pre-tokenized dataset from cache (instant)...")
            self.tokens = torch.load(cache_path)
        else:
            print(f"Tokenizing dataset using high-speed Rust batching...")
            import array
            import os
            tokens_array = array.array('i')
            file_size = os.path.getsize(path)
            
            with open(path, "r", encoding="utf-8") as f:
                lines = []
                with tqdm(total=file_size, unit="B", unit_scale=True, desc="Tokenizing dataset") as pbar:
                    for line in f:
                        lines.append(line.strip())
                        pbar.update(len(line.encode('utf-8', errors='ignore')))
                        if len(lines) >= 20000:
                            # Use HuggingFace's extremely fast Rust batch encoder
                            batch_encoded = tokenizer(lines, add_special_tokens=False)["input_ids"]
                            for ids in batch_encoded:
                                tokens_array.extend(ids)
                            lines.clear()
                    
                    if lines:
                        batch_encoded = tokenizer(lines, add_special_tokens=False)["input_ids"]
                        for ids in batch_encoded:
                            tokens_array.extend(ids)
                            
            self.tokens = torch.tensor(tokens_array, dtype=torch.long)
            self.tokens = self.tokens[:200_000_000]  # truncate to first 200 M tokens
            print(f"Saving tokenized cache to {cache_path}...")
            torch.save(self.tokens, cache_path)
        self.num_blocks = (len(self.tokens) - block_size) // block_size

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        block = self.tokens[start:end]
        # Return both input_ids and labels so HF Trainer can compute loss!
        return {"input_ids": block.clone(), "labels": block.clone()}

train_dataset = SlimPajama200MDataset(DATASET_PATH, tokenizer, BLOCK_SIZE)

import time
from transformers import TrainerCallback

class EtaCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % args.logging_steps == 0:
            elapsed = time.time() - getattr(self, 'start_time', time.time())
            steps_done = state.global_step
            steps_total = state.max_steps
            eta = (elapsed / steps_done) * (steps_total - steps_done) if steps_done else float('inf')
            print(f"[ETA] step {steps_done}/{steps_total} - elapsed {elapsed:.1f}s - ETA {eta/60:.1f}min")

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

# ---------------------------------------------------------------------------
# Trainer configuration – constant console logging
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LR,
    num_train_epochs=NUM_EPOCHS,
    fp16=True,
    logging_steps=10, # Log more frequently
    save_steps=500,
    save_total_limit=2,
    report_to=["none"],
    dataloader_num_workers=0,
    disable_tqdm=False,
)

class CMFTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # We pass input_ids and labels to the model.
        # It runs the forward pass (including adaptive thinking steps)
        # and returns a dictionary.
        labels = inputs.pop("labels", None)
        outputs = model(**inputs, labels=labels)
        
        # Base language modeling loss is now natively calculated correctly in model.py
        loss = outputs.get("loss")
        
        # Add ponder_loss if the model is using adaptive thinking
        ponder_loss = outputs.get("ponder_loss", 0.0)
        if isinstance(ponder_loss, torch.Tensor) and ponder_loss.requires_grad:
            # Weight the ponder penalty so it doesn't overpower the LM loss
            loss = loss + 0.01 * ponder_loss
            
        return (loss, outputs) if return_outputs else loss

trainer = CMFTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[EtaCallback()],
)

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n============================================================")
    print("Starting CMF LoRA Fine-Tuning...")
    print("============================================================")
    print(f"Package : {PACKAGE_PATH}")
    print(f"Dataset : {DATASET_PATH}")
    print(f"Output  : {OUTPUT_DIR}")

    # ── Adaptive checkpoint resumption ──────────────
    from transformers.trainer_utils import get_last_checkpoint
    
    resume_ckpt = None
    if OUTPUT_DIR.exists():
        resume_ckpt = get_last_checkpoint(str(OUTPUT_DIR))
        if resume_ckpt:
            print(f"\n>>> Resuming adaptively from latest checkpoint: {resume_ckpt}")
        else:
            print("\n>>> No checkpoint found – starting fresh.")
    else:
        print("\n>>> No checkpoint folder found – starting fresh.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
        trainer.save_model(str(OUTPUT_DIR))
        print("\n=== Training complete! Checkpoint saved to:", OUTPUT_DIR, "===")
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user – saving current state...")
        trainer.save_model(str(OUTPUT_DIR))
        print("[✔] Model saved. You can resume by re-running this script.")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        raise
    finally:
        print("Exiting...")
