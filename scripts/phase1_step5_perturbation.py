"""
CMF Checklist Phase 1 - Step 1.5: Perturbation Recovery

This script tests whether CMF possesses true attractor basins that gracefully recover
from external noise injections. It injects varying levels of Gaussian noise into the
embeddings and measures loss degradation.
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
from cmf.memory_tasks import KeyDoorDataset, VOCAB_SIZE, PAD
from cmf.experiments import run_perturbation_test

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
    print("CMF STEP 1.5 - PERTURBATION RECOVERY & ATTRACTOR BASINS")
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

    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    results = {}

    configs = {
        "Full Attention (Baseline)": {"mode": "full", "mock_memory": False},
        "Slot Memory Only (No Attention)": {"mode": "none", "mock_memory": False}
    }

    print("Building training and evaluation datasets...")
    kd_train = KeyDoorDataset(1600, gap_lengths=[32, 64], seed=42)
    kd_eval  = KeyDoorDataset(200, gap_lengths=[32, 64], seed=42)
    
    train_batches = make_memory_batches(kd_train, batch_size=32, device=device)
    eval_batches  = make_memory_batches(kd_eval, batch_size=32, device=device)

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

        print(f"Evaluating noise perturbation levels: {noise_levels}")
        perturb_results = run_perturbation_test(
            model, eval_batches, device, noise_levels=noise_levels,
            output_dir=str(ROOT / "records" / "ablations" / "perturbation")
        )
        results[name] = perturb_results

    # Save outputs
    out_dir = ROOT / "records"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "phase1_step5_perturbation.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[OK] Perturbation results saved to: {out_json}")

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

    for name, perturb in results.items():
        # perturb keys are float or string depending on saving format
        sigmas = [float(k) for k in perturb.keys()]
        losses = list(perturb.values())
        plt.plot(sigmas, losses, label=name, color=colors[name], marker=markers[name], linewidth=2, markersize=8)

    plt.title("Step 1.5: CMF Perturbation Recovery Curve", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Gaussian Noise Std Dev (sigma)", fontsize=11)
    plt.ylabel("Evaluation Cross Entropy Loss", fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    
    out_plot = out_dir / "phase1_step5_perturbation.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"[OK] Scientific plot saved to: {out_plot}")

if __name__ == "__main__":
    main()
