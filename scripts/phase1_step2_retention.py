"""
CMF Checklist Phase 1 - Step 1.2: Streaming Memory Tasks & Retention Curves

This script runs the retention degradation curve test for lengths [128, 512, 2048, 8192].
It compares two configurations:
  1. Full Attention (A)
  2. Slot Memory Only (D)

It plots and saves the retention curve to records/phase1_step2_retention.png.
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
from cmf.memory_tasks import KeyDoorDataset, VOCAB_SIZE, PAD, measure_retention_curve

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
    print("CMF STEP 1.2 - STREAMING RETENTION DEGRADATION CURVE")
    print("==================================================")
    print(f"Device: {device}\n")

    cfg = CMFConfig(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        hidden_dim=256,
        num_layers=3,
        num_slots=16,
        solver_steps=4,
        dropout=0.0,
        tie_embeddings=False,
    )

    gaps = [128, 512, 2048, 8192]
    results = {}

    configs = {
        "Full Attention (Baseline)": {"mode": "full", "mock_memory": False},
        "Slot Memory Only (No Attention)": {"mode": "none", "mock_memory": False}
    }

    print("Building training dataset on gaps [128, 512]...")
    kd_train = KeyDoorDataset(1600, gap_lengths=[128, 512], seed=42)
    train_batches = make_memory_batches(kd_train, batch_size=32, device=device)

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
        for step in range(300):
            optimizer.zero_grad(set_to_none=True)
            ids, lbls = next(batch_iter)
            out = model(ids, labels=lbls)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        print(f"Evaluating retention over gaps: {gaps}")
        retention = measure_retention_curve(
            model, gap_lengths=gaps, n_per_gap=100, device=str(device)
        )
        results[name] = retention

    # Save outputs
    out_dir = ROOT / "records"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "phase1_step2_retention.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[OK] Retention results saved to: {out_json}")

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

    for name, retention in results.items():
        plt.plot(gaps, list(retention.values()), label=name, color=colors[name], marker=markers[name], linewidth=2, markersize=8)

    plt.axhline(y=0.038, color="gray", linestyle="--", alpha=0.7, label="Random Baseline (1/26)")
    plt.title("Step 1.2: CMF Streaming Retention Degradation Curve", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Distractor Gap Length (tokens)", fontsize=11)
    plt.ylabel("Retrieval Accuracy", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.xscale("log", base=2)
    plt.xticks(gaps)
    plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    
    out_plot = out_dir / "phase1_step2_retention.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"[OK] Scientific plot saved to: {out_plot}")

if __name__ == "__main__":
    main()
