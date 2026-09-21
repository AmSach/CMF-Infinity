"""
CMF Routing Showdown — Progressive Ablation Analysis

This script runs the scientific showdown test ladder:
  A. Full Attention
  B. Sparse Top-K Attention
  C. Local Window Attention
  D. Slot Memory Only (Fixed Slot Routing)
  E. Pure Latent Recurrent Dynamics (No Attention, No Memory)

It trains each configuration on:
  1. Key-Door Dataset (to measure retention curve vs gap length)
  2. Multi-Binding Dataset (to measure compositional binding capacity)

It saves metrics to records/routing_showdown.json and generates a beautiful,
publishable plot showing the comparative degradation curves.
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# Establish workspace root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmf.config import CMFConfig
from cmf.model import ParallelCMF
from cmf.memory_tasks import (
    KeyDoorDataset, MultiBindingDataset, VOCAB_SIZE, PAD,
    measure_retention_curve, measure_capacity_curve,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper to set seeds for deterministic comparisons
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────────────────────────
# Collate / Batching Helper
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# Main Showdown Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"CMF ROUTING SHOWDOWN - PROGRESSIVE ABLATION LADDER")
    print(f"==================================================")
    print(f"Running on device: {device}\n")

    # Define standard small model config
    cfg = CMFConfig(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        hidden_dim=256,
        num_layers=3,
        num_slots=16,
        solver_steps=4,
        dropout=0.0,
        tie_embeddings=False,
        routing_mode="full",
        routing_topk=4,
        routing_window=8,
    )

    # 5 Test Ladder Configurations
    configs = {
        "A. Full Attention": {
            "mode": "full",
            "mock_memory": False,
        },
        "B. Sparse Top-K Attention": {
            "mode": "sparse_topk",
            "mock_memory": False,
        },
        "C. Local Window Attention": {
            "mode": "local_window",
            "mock_memory": False,
        },
        "D. Slot Memory Only": {
            "mode": "none",
            "mock_memory": False,
        },
        "E. Pure Latent Recurrent": {
            "mode": "none",
            "mock_memory": True,
        }
    }

    results = {}

    # Define datasets
    gap_test_list = [16, 64, 128, 256]
    k_test_list = [1, 2, 4, 8]

    print("[1] Building training and evaluation datasets...")
    # Train datasets
    kd_train = KeyDoorDataset(1600, gap_lengths=[16, 64, 128])
    mb_train = MultiBindingDataset(1600, k_list=[1, 2, 4], gap=16)

    # Evaluation datasets are handled on-the-fly by cmf.memory_tasks helpers
    kd_train_batches = make_memory_batches(kd_train, batch_size=32, device=device)
    mb_train_batches = make_memory_batches(mb_train, batch_size=32, device=device)

    for name, opt in configs.items():
        print(f"\n--------------------------------------------------")
        print(f"Evaluating Config: {name}")
        print(f"--------------------------------------------------")

        # 1. Initialize fresh model
        set_seed(42)
        model = ParallelCMF(cfg).to(device)

        # Apply configuration options
        model.anchor.mode = opt["mode"]
        if opt["mock_memory"]:
            # Override SlotMemory to return zeros (Pure Latent Recurrent Dynamics)
            model.memory.forward = lambda z, context: torch.zeros_like(z)

        # 2. Train on KeyDoor task
        print("  Training on Key-Door (300 steps)...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        model.train()
        
        batch_iter = iter(kd_train_batches * 10)
        for step in range(300):
            optimizer.zero_grad(set_to_none=True)
            ids, lbls = next(batch_iter)
            out = model(ids, labels=lbls)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Measure Key-Door retention degradation curve
        print("  Measuring Key-Door retention curve...")
        retention = measure_retention_curve(
            model, gap_lengths=gap_test_list, n_per_gap=100, device=str(device)
        )

        # 3. Train on Multi-Binding task
        print("  Training on Multi-Binding (300 steps)...")
        set_seed(42)
        model = ParallelCMF(cfg).to(device)
        model.anchor.mode = opt["mode"]
        if opt["mock_memory"]:
            model.memory.forward = lambda z, context: torch.zeros_like(z)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        model.train()

        batch_iter = iter(mb_train_batches * 10)
        for step in range(300):
            optimizer.zero_grad(set_to_none=True)
            ids, lbls = next(batch_iter)
            out = model(ids, labels=lbls)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Measure Multi-Binding capacity curve
        print("  Measuring Multi-Binding capacity curve...")
        capacity = measure_capacity_curve(
            model, k_list=k_test_list, n_per_k=100, gap=16, device=str(device)
        )

        # Save metrics for this configuration
        results[name] = {
            "retention": {str(gap): acc for gap, acc in retention.items()},
            "capacity": {str(k): acc for k, acc in capacity.items()},
        }

    # Save to disk
    out_dir = ROOT / "records"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "routing_showdown.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[OK] Showdown metrics saved to: {out_json}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Plotting Comparative Scientific Figures
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[2] Generating comparative scientific plots...")

    # Professional scientific style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color palette
    colors = {
        "A. Full Attention": "#2b5c8f",          # Sleek Blue
        "B. Sparse Top-K Attention": "#4682b4",   # Muted Blue
        "C. Local Window Attention": "#5f9ea0",   # Teal
        "D. Slot Memory Only": "#d97706",         # Orange
        "E. Pure Latent Recurrent": "#dc2626",    # Crimson Red
    }

    markers = {
        "A. Full Attention": "o",
        "B. Sparse Top-K Attention": "s",
        "C. Local Window Attention": "^",
        "D. Slot Memory Only": "D",
        "E. Pure Latent Recurrent": "x",
    }

    # Plot 1: Key-Door Retention Curve
    for name, metrics in results.items():
        gaps = [int(g) for g in metrics["retention"].keys()]
        accs = list(metrics["retention"].values())
        ax1.plot(gaps, accs, label=name, color=colors[name], marker=markers[name], linewidth=2, markersize=8)

    ax1.set_title("Key-Door Retention Degradation Curve", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Distractor Gap Length (tokens)", fontsize=11)
    ax1.set_ylabel("Retrieval Accuracy", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(gap_test_list)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.axhline(y=0.038, color="gray", linestyle="--", alpha=0.7, label="Random Baseline (1/26)")
    ax1.legend(frameon=True, fontsize=10, loc="lower left")

    # Plot 2: Multi-Binding Capacity Curve
    for name, metrics in results.items():
        ks = [int(k) for k in metrics["capacity"].keys()]
        accs = list(metrics["capacity"].values())
        ax2.plot(ks, accs, label=name, color=colors[name], marker=markers[name], linewidth=2, markersize=8)

    ax2.set_title("Multi-Binding Capacity Curve", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Number of Simultaneous Bindings (K)", fontsize=11)
    ax2.set_ylabel("Retrieval Accuracy", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xticks(k_test_list)
    ax2.axhline(y=0.038, color="gray", linestyle="--", alpha=0.7, label="Random Baseline (1/26)")
    ax2.legend(frameon=True, fontsize=10, loc="lower left")

    plt.tight_layout()
    out_plot = out_dir / "routing_showdown.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"[OK] Scientific plot saved to: {out_plot}")
    print(f"==================================================")
    print(f"SHOWDOWN EXPERIMENT COMPLETE.")
    print(f"==================================================")

if __name__ == "__main__":
    main()
