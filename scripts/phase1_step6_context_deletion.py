"""
CMF Checklist Phase 1 - Step 1.6: Context Deletion Midway (Test A)

This script implements a brutal test of the continuous latent state memory:
Midway through the solver steps (e.g. steps >= 2 of 4), we completely zero out the
context landscape. This blocks all attention routing from querying history.
If the model can still retrieve the correct answer, it proves that the episodic
association was successfully compressed and stored in the evolving latent state z!
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
from cmf.memory_tasks import KeyDoorDataset, VOCAB_SIZE, PAD, val_tok

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
    print("CMF STEP 1.6 - TEST A: CONTEXT DELETION MIDWAY")
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

    print("Building datasets...")
    kd_train = KeyDoorDataset(1600, gap_lengths=[32, 64], seed=42)
    kd_eval  = KeyDoorDataset(200, gap_lengths=[32, 64], seed=42)
    
    train_batches = make_memory_batches(kd_train, batch_size=32, device=device)

    # 1. Train model with Full Attention
    print("\nTraining CMF with Full Attention...")
    set_seed(42)
    model = ParallelCMF(cfg).to(device)
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

    # 2. Evaluate with Context Deletion
    # We will define a custom forward pass that zeroes out context at step >= delete_after_step
    print("\nEvaluating with custom context deletion...")
    model.eval()

    def custom_forward_eval(input_ids: torch.Tensor, delete_midway: bool = False) -> torch.Tensor:
        B, T = input_ids.shape
        with torch.no_grad():
            emb     = model.embedding(input_ids)
            context = model.encoder(emb)
            z       = model.initial_state(context)

            steps = model.cfg.solver_steps
            dt    = 1.0 / steps

            for i in range(steps):
                tau = torch.full((B,), i * dt, dtype=z.dtype, device=z.device)
                
                # Brutal context deletion midway: zero out context landscape
                current_context = context
                if delete_midway and (i >= steps // 2):
                    current_context = torch.zeros_like(context)

                anchored = model.anchor(z, current_context)
                slot     = model.memory(z, current_context)
                velocity = model.field(anchored, current_context, slot, tau)
                z = z + dt * velocity

            logits = model.output(model.state_norm(z))
            return logits

    # Run evaluations
    correct_control = 0
    correct_deleted = 0
    total = len(kd_eval)

    for idx in range(total):
        s = kd_eval[idx]
        ids = s["input_ids"].unsqueeze(0).to(device)
        qpos = s["query_pos"]
        correct_tok = val_tok(s["value"])

        # Control Group (Normal)
        logits_control = custom_forward_eval(ids, delete_midway=False)
        pred_control = logits_control[0, qpos].argmax().item()
        correct_control += int(pred_control == correct_tok)

        # Experimental Group (Context Deleted Midway)
        logits_deleted = custom_forward_eval(ids, delete_midway=True)
        pred_deleted = logits_deleted[0, qpos].argmax().item()
        correct_deleted += int(pred_deleted == correct_tok)

    acc_control = correct_control / total
    acc_deleted = correct_deleted / total

    print(f"\nEvaluation Results:")
    print(f"  Control Group Accuracy (Normal): {acc_control*100:.1f}%")
    print(f"  Experimental Group Accuracy (Deleted Midway): {acc_deleted*100:.1f}%")
    print(f"  Random Baseline: 3.8%")

    # Plot results
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.figure(figsize=(6, 5))
    
    categories = ["Control (Normal)", "Context Deleted Midway", "Random Baseline"]
    accuracies = [acc_control, acc_deleted, 0.038]
    colors = ["#2b5c8f", "#d97706", "gray"]

    bars = plt.bar(categories, accuracies, color=colors, width=0.5)
    plt.title("Step 1.6: Context Deletion Midway (Test A)", fontsize=12, fontweight="bold", pad=10)
    plt.ylabel("Retrieval Accuracy", fontsize=11)
    plt.ylim(-0.05, 1.05)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f"{height*100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_dir = ROOT / "records"
    out_plot = out_dir / "phase1_step6_context_deletion.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"\n[OK] Scientific plot saved to: {out_plot}")

if __name__ == "__main__":
    main()
