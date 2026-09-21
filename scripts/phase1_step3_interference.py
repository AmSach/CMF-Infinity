"""
CMF Checklist Phase 1 - Step 1.3: Memory Interference Test

This script tests whether the model's latent space supports compositional memory or collapses into semantic soup under high K bindings.
It evaluates retrieval accuracy across K active bindings [2, 4, 8, 16, 32].
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# Insert workspace root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmf.config import CMFConfig
from cmf.model import ParallelCMF
from cmf.memory_tasks import MultiBindingDataset, VOCAB_SIZE, PAD, measure_capacity_curve

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_memory_batches(dataset, batch_size: int, device: torch.device):
    batches = []
    for start in range(0, len(dataset), batch_size):
        chunk = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        ids = pad_sequence([s["input_ids"] for s in chunk],
                           batch_first=True, padding_value=PAD)
        lbl = pad_sequence([s["labels"] for s in chunk],
                           batch_first=True, padding_value=-100)
        batches.append((ids.to(device), lbl.to(device)))
    return batches

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print("CMF STEP 1.3 - COMPOSITIONAL MEMORY INTERFERENCE TEST")
    print("==================================================")
    print(f"Device: {device}\n")

    cfg = CMFConfig(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        hidden_dim=256,
        num_layers=3,
        num_slots=32,  # Give it enough slots to handle large K!
        solver_steps=4,
        dropout=0.0,
        tie_embeddings=False,
    )

    k_list = [2, 4, 8, 16, 32]
    results = {}

    configs = {
        "Full Attention (Baseline)": {"mode": "full", "mock_memory": False},
        "Slot Memory Only (No Attention)": {"mode": "none", "mock_memory": False}
    }

    print("Building compositional training dataset...")
    mb_train = MultiBindingDataset(2000, k_list=[2, 4, 8], gap=16, seed=42)
    train_batches = make_memory_batches(mb_train, batch_size=32, device=device)

    for name, opt in configs.items():
        print(f"\nTraining CMF under: {name}")
        set_seed(42)
        model = ParallelCMF(cfg).to(device)
        model.anchor.mode = opt["mode"]
        if opt["mock_memory"]:
            model.memory.forward = lambda z, context: torch.zeros_like(z)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        model.train()

        batch_iter = iter(train_batches * 10)
        for step in range(400):
            optimizer.zero_grad(set_to_none=True)
            ids, lbls = next(batch_iter)
            out = model(ids, labels=lbls)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        print(f"Evaluating capacity over K: {k_list}")
        capacity = measure_capacity_curve(
            model, k_list=k_list, n_per_k=100, gap=16, device=str(device)
        )
        results[name] = capacity

    # Save outputs
    out_dir = ROOT / "records"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "phase1_step3_interference.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[OK] Interference results saved to: {out_json}")

    # Plot
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.figure(figsize=(8, 5))
    
    colors = {
        "Full Attention (Baseline)": "#2b5c8f",
        "Slot Memory Only (No Attention)": "#dc2626"
    }
    
    markers = {
        "Full Attention (Baseline)": "o",
        "Slot Memory Only (No Attention)": "s"
    }

    for name, capacity in results.items():
        plt.plot(k_list, list(capacity.values()), label=name, color=colors[name], marker=markers[name], linewidth=2, markersize=8)

    plt.axhline(y=0.038, color="gray", linestyle="--", alpha=0.7, label="Random Baseline (1/26)")
    plt.title("Step 1.3: compositional Memory Interference Curve", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Number of simultaneous active bindings (K)", fontsize=11)
    plt.ylabel("Retrieval Accuracy", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.xticks(k_list)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    
    out_plot = out_dir / "phase1_step3_interference.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"[OK] Scientific plot saved to: {out_plot}")

if __name__ == "__main__":
    main()
