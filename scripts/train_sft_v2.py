import os
import sys
import json
import time
import math
import argparse
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
                full_ids = full_ids[:max_seq_len]
                prompt_len = min(prompt_ids.numel(), max_seq_len - 1)
            else:
                prompt_len = prompt_ids.numel()
                
            # Create a label tensor where prompt tokens are replaced by -100
            labels = full_ids.clone()
            labels[:prompt_len] = -100
            
            self.encoded_pairs.append((full_ids, labels))
            
    def __len__(self):
        return len(self.encoded_pairs)
        
    def __getitem__(self, idx):
        full_ids, labels = self.encoded_pairs[idx]
        return full_ids, labels

def collate_sft(batch, pad_token_id=50256):
    max_len = max(x[0].numel() for x in batch)
    xs, ys = [], []
    for full_ids, labels in batch:
        pad_len = max_len - full_ids.numel()
        padded_x = torch.cat([full_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)], dim=0)
        padded_y = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)], dim=0)
        xs.append(padded_x)
        ys.append(padded_y)
    return torch.stack(xs), torch.stack(ys)

def get_active_prompts_dataset() -> list[dict]:
    # Custom high-quality SFT alignment data addressing the user's active failure cases
    return [
        {
            "instruction": "Q: 2+3=? A:",
            "response": "5"
        },
        {
            "instruction": "Q: 41+2=? A:",
            "response": "43"
        },
        {
            "instruction": "Fact: alice is bob. Fact: bob is developer. Q: what is alice? A:",
            "response": "alice is developer"
        },
        {
            "instruction": "Q: what is paris? A:",
            "response": "Paris is the capital and most populous city of France."
        },
        {
            "instruction": "System: You are CMF Infinity.\nUser: write a mathematical equation for meaning field.\nAssistant:",
            "response": "A continuous meaning field can be formulated as an ordinary differential equation over latent paths:\n\\frac{d\\mathbf{z}(t)}{dt} = \\mathbf{f}(\\mathbf{z}(t), \\mathbf{c}, t)\nwhere \\mathbf{z}(0) is the initial representation, \\mathbf{c} is the context field, and \\mathbf{f} is the neural vector field."
        }
    ]

def main():
    parser = argparse.ArgumentParser(description="CMF-v2 Supervised Fine-Tuning & Alignment script.")
    parser.add_argument("--base-checkpoint", type=str, default="checkpoint_latest.pt")
    parser.add_argument("--out-package", type=str, default="cmf_120m_aligned.package.pt")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--solver-method", type=str, default="symplectic", choices=["euler", "rk4", "symplectic"])
    parser.add_argument("--use-gmr", action="store_true", help="Enable Global Memory Router during SFT fine-tuning.")
    args = parser.parse_args()

    print("=" * 70)
    print("CMF-v2 SUPERVISED FINE-TUNING (SFT) ALIGNMENT PIPELINE")
    print("=" * 70)
    
    device = resolve_device("auto")
    print(f"Target Hardware device: {device}")
    
    checkpoint_path = ROOT / args.base_checkpoint
    if not checkpoint_path.exists():
        print(f"Error: Base checkpoint not found at {checkpoint_path}!")
        sys.exit(1)
        
    print(f"Loading base weights from: {checkpoint_path}...")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    preset_name = "infinity-reasoning-0.12b"
    preset = get_preset(preset_name)
    config = CMFConfig(**preset.config.__dict__)
    config.max_seq_len = 256
    config.solver_method = args.solver_method
    config.use_global_memory_router = args.use_gmr
    
    print(f"Initializing Deliberative CMF model (Method={args.solver_method}, GMR={args.use_gmr})...")
    model = DeliberativeContinuousMeaningField(config)
    
    state_dict = payload["model"] if "model" in payload else payload.get("model_state_dict", payload)
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
        
    # Load state_dict (transparently handled by our custom backward-compatibility mapping)
    model.load_state_dict(cleaned_state_dict)
    model = model.to(device)
    print("Base weights loaded and mapped successfully.")
    
    print("Loading GPT-2 tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load alignment datasets
    data_items = get_active_prompts_dataset()
    
    local_data_path = ROOT / "sft_dataset.json"
    if local_data_path.exists():
        try:
            with open(local_data_path, "r", encoding="utf-8") as f:
                local_items = json.load(f)
                data_items.extend(local_items)
                print(f"Loaded {len(local_items)} custom prompts from {local_data_path}")
        except Exception as e:
            print(f"Warning loading local JSON dataset: {e}")
            
    dataset = SFTDataset(tokenizer, data_items, max_seq_len=config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sft)
    print(f"Total compiled alignment pairs: {len(dataset)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    print(f"Starting SFT Alignment for {args.epochs} epochs...")
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        step_count = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            
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
            
        avg_loss = total_loss / max(1, step_count)
        print(f"Epoch {epoch}/{args.epochs} | Average Alignment Loss: {avg_loss:.4f}")
        
    print("\nSFT Alignment complete! Saving conversational model package...")
    out_path = ROOT / args.out_package
    torch.save({
        "model": model.state_dict(),
        "config": config.__dict__,
        "tokenizer": {"type": "hf_auto", "name": "gpt2", "vocab_size": tokenizer.vocab_size},
        "model_type": "deliberative_cmf",
        "training": {
            "epoch": args.epochs,
            "final_loss": avg_loss,
            "solver_method": args.solver_method,
            "use_global_memory_router": args.use_gmr
        }
    }, out_path)
    print(f"Successfully saved conversational package to: {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
