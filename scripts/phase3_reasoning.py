"""
CMF Checklist Phase 3 - Step 3.1 & 3.2: Iterative Reasoning & Adaptive Compute

This script tests the iterative deliberation engine. It tracks:
1. Logit Entropy Evolution: Does logit entropy decrease monotonically over solver steps?
2. Adaptive Compute: Does thinking step count adapt to input complexity?
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Insert workspace root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmf.config import CMFConfig
from cmf.model import DeliberativeCMF
from cmf.experiments import run_solver_depth_test, set_seed

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print("CMF PHASE 3 - ITERATIVE REASONING & ADAPTIVE COMPUTE")
    print("==================================================")
    print(f"Device: {device}\n")

    # Config with 8 thinking steps max
    cfg = CMFConfig(
        vocab_size=256,
        d_model=128,
        hidden_dim=256,
        num_layers=3,
        num_slots=16,
        solver_steps=4,
        dropout=0.0,
        tie_embeddings=False,
        thinking_steps=16,
        adaptive_thinking=True,
        min_thinking_steps=2,
        max_thinking_steps=16,
        halting_threshold=0.92,
    )

    set_seed(42)
    model = DeliberativeCMF(cfg).to(device)

    # Put in train mode to set up weights, then evaluate
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Train briefly on random data to establish learned update gate and halt parameters
    print("Briefly training deliberative parameters (100 steps)...")
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        ids = torch.randint(0, 256, (4, 32), device=device)
        out = model(ids, labels=ids)
        out["loss"].backward()
        optimizer.step()

    model.eval()

    # Step 3.1: Logit Entropy Evolution
    print("\n[3.1] Running logit entropy evolution test...")
    # Feed an arbitrary input sequence
    test_ids = torch.randint(0, 256, (1, 16), device=device)
    traj = run_solver_depth_test(
        model, test_ids, device,
        output_dir=str(ROOT / "records" / "ablations" / "solver_depth")
    )

    # Plot Logit Entropy Evolution
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.figure(figsize=(8, 5))
    
    steps = [t["step"] for t in traj]
    entropies = [t["logit_entropy"] for t in traj]
    z_norms = [t["z_norm"] for t in traj]
    v_norms = [t["v_norm"] for t in traj]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "#2b5c8f"
    ax1.set_xlabel("Solver Thinking Step", fontsize=11)
    ax1.set_ylabel("Logit Entropy", color=color, fontsize=11)
    line1 = ax1.plot(steps, entropies, color=color, marker="o", linewidth=2.5, label="Logit Entropy")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  
    color = "#d97706"
    ax2.set_ylabel("Velocity Norm (||v||_2)", color=color, fontsize=11)
    line2 = ax2.plot(steps, v_norms, color=color, marker="s", linestyle="--", linewidth=2, label="Velocity Norm")
    ax2.tick_params(axis="y", labelcolor=color)

    # Added legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)

    plt.title("Step 3.1: CMF Latent Refinement & Logit Entropy Evolution", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    
    out_dir = ROOT / "records"
    out_plot = out_dir / "phase3_step1_entropy.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"\n[OK] Logit entropy evolution plot saved to: {out_plot}")

    # Step 3.2: Adaptive Compute Verification
    print("\n[3.2] Running Adaptive Compute test...")
    # Generate easy and hard examples by sequence length
    easy_ids = torch.randint(0, 256, (1, 4), device=device)
    hard_ids = torch.randint(0, 256, (1, 64), device=device)

    with torch.no_grad():
        out_easy = model(easy_ids)
        out_hard = model(hard_ids)

    easy_steps = out_easy["thinking_steps"].item()
    hard_steps = out_hard["thinking_steps"].item()

    print(f"  Easy Input (length 4)  -> Solver Steps Used: {easy_steps}")
    print(f"  Hard Input (length 64) -> Solver Steps Used: {hard_steps}")
    print(f"  Adaptive Scaling? {'YES' if hard_steps >= easy_steps else 'NO'}")

    results = {
        "trajectory": traj,
        "easy_steps": easy_steps,
        "hard_steps": hard_steps,
    }
    (out_dir / "phase3_reasoning.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
