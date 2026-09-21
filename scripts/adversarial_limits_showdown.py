"""
CMF Adversarial Limits Showdown

This script runs two brutal limit-mapping experiments:
1. Experiment A: Latent Bottleneck Sweep
   Sweeps latent_dim across [256, 128, 64, 32, 16, 8] under identical pretraining.
   Proves if retrieval survives below information-theoretic storage bounds.

2. Experiment B: Continuous ODE vs. Discrete Recurrent MLP
   Ablates the continuous time features and dt scaling. Compares:
     - Continuous ODE CMF
     - Discrete Recurrent CMF (dt = 1.0, no time embeddings)
   Measures loss convergence and retention accuracy at gap = 256.

Saves results and generates premium scientific degradation plots.
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
from cmf.memory_tasks import KeyDoorDataset, VOCAB_SIZE, PAD, val_tok

def set_seed(seed: int) -> None:
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

# ─────────────────────────────────────────────────────────────────────────────
# Custom Recurrent Discrete Model (Ablated Vector Field)
# ─────────────────────────────────────────────────────────────────────────────

class DiscreteRecurrentCMF(ParallelCMF):
    """
    Ablated CMF where:
    - dt is set to 1.0 (no scaling)
    - time features (tau) are disabled (zeroed out)
    - z = z + velocity (discrete MLP step, no continuous integration physics)
    """
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_states: bool = False,
        grad_ckpt: bool = False,
        routing_mode: str | None = None,
        log_trajectory: bool = False,
    ) -> dict[str, torch.Tensor]:
        if routing_mode is not None:
            _orig = self.anchor.mode
            self.anchor.mode = routing_mode

        B, T = input_ids.shape
        emb     = self.embedding(input_ids)
        context = self.encoder(emb, grad_ckpt=grad_ckpt)
        z       = self.initial_state(context)

        steps  = self.cfg.solver_steps
        traj   = [] if log_trajectory else None

        for i in range(steps):
            # ABLATION: Time features are completely zeroed out
            tau = torch.zeros((B,), dtype=z.dtype, device=z.device)
            anchored = self.anchor(z, context)
            slot     = self.memory(z, context)
            
            # ABLATION: No dt scaling (dt = 1.0)
            velocity = self.field(anchored, context, slot, tau)
            z = z + velocity  # Discrete update

            if traj is not None:
                traj.append({
                    "step": i,
                    "z_norm": z.norm(dim=-1).mean().item(),
                    "v_norm": velocity.norm(dim=-1).mean().item(),
                    "logit_entropy": float("nan"),
                })

        logits = self.output(self.state_norm(z))
        result = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:T].contiguous()
            result["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1), ignore_index=-100)
        return result

# ─────────────────────────────────────────────────────────────────────────────
# Main Adversarial limits runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print("CMF SYSTEMATIC FAILURE SURFACE MAPPING")
    print("==================================================")
    print(f"Device: {device}\n")

    # -------------------------------------------------------------------------
    # EXPERIMENT A: Latent Bottleneck Sweep
    # -------------------------------------------------------------------------
    print("--- [EXPERIMENT A] LATENT BOTTLENECK SWEEP ---")
    latent_dims = [256, 128, 64, 32, 16, 8]
    bottleneck_results = {}

    kd_train_64 = KeyDoorDataset(1200, gap_lengths=[64], seed=42)
    kd_eval_64  = KeyDoorDataset(100, gap_lengths=[64], seed=42)
    
    train_batches = make_memory_batches(kd_train_64, batch_size=32, device=device)

    for d in latent_dims:
        print(f"Training CMF with latent_dim = {d:3d}...")
        cfg = CMFConfig(
            vocab_size=VOCAB_SIZE,
            d_model=d,
            hidden_dim=d * 2,
            num_layers=3,
            num_slots=16,
            solver_steps=4,
            dropout=0.0,
            tie_embeddings=False,
        )

        set_seed(42)
        model = ParallelCMF(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        model.train()

        batch_iter = iter(train_batches)
        for step in range(300):
            optimizer.zero_grad(set_to_none=True)
            try:
                ids, lbls = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_batches)
                ids, lbls = next(batch_iter)
            out = model(ids, labels=lbls)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Evaluate retrieval accuracy
        model.eval()
        correct = 0
        for idx in range(len(kd_eval_64)):
            s = kd_eval_64[idx]
            ids = s["input_ids"].unsqueeze(0).to(device)
            qpos = s["query_pos"]
            correct_tok = val_tok(s["value"])
            out = model(ids)
            pred = out["logits"][0, qpos].argmax().item()
            correct += int(pred == correct_tok)

        acc = correct / len(kd_eval_64)
        bottleneck_results[d] = acc
        print(f"  d_model = {d:3d} -> Accuracy: {acc*100:.1f}%")

    # -------------------------------------------------------------------------
    # EXPERIMENT B: Continuous ODE vs. Discrete Recurrent MLP
    # -------------------------------------------------------------------------
    print("\n--- [EXPERIMENT B] CONTINUOUS ODE VS. DISCRETE RECURRENT MLP ---")
    
    cfg_b = CMFConfig(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        hidden_dim=256,
        num_layers=3,
        num_slots=16,
        solver_steps=4,
        dropout=0.0,
        tie_embeddings=False,
    )

    kd_train_256 = KeyDoorDataset(1200, gap_lengths=[256], seed=42)
    kd_eval_256  = KeyDoorDataset(100, gap_lengths=[256], seed=42)
    train_batches_b = make_memory_batches(kd_train_256, batch_size=32, device=device)

    architectures = {
        "Continuous CMF (ODE)": ParallelCMF(cfg_b),
        "Discrete Recurrent CMF": DiscreteRecurrentCMF(cfg_b)
    }

    ablation_results = {}

    for name, model in architectures.items():
        print(f"Training {name} on gap = 256...")
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        model.train()

        losses = []
        batch_iter = iter(train_batches_b)
        for step in range(300):
            optimizer.zero_grad(set_to_none=True)
            try:
                ids, lbls = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_batches_b)
                ids, lbls = next(batch_iter)
            out = model(ids, labels=lbls)
            loss = out["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        # Evaluate retrieval accuracy
        model.eval()
        correct = 0
        for idx in range(len(kd_eval_256)):
            s = kd_eval_256[idx]
            ids = s["input_ids"].unsqueeze(0).to(device)
            qpos = s["query_pos"]
            correct_tok = val_tok(s["value"])
            out = model(ids)
            pred = out["logits"][0, qpos].argmax().item()
            correct += int(pred == correct_tok)

        acc = correct / len(kd_eval_256)
        ablation_results[name] = {
            "accuracy": acc,
            "final_loss": losses[-1],
            "loss_history": losses[::10]  # Store downsampled loss
        }
        print(f"  {name} -> Accuracy: {acc*100:.1f}% | Final Loss: {losses[-1]:.4f}")

    # Save results to json
    out_dir = ROOT / "records"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "adversarial_limits.json"
    results = {
        "bottleneck": bottleneck_results,
        "ablation": ablation_results
    }
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[OK] Adversarial limits saved to: {out_json}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Plotting Comparative Figures
    # ─────────────────────────────────────────────────────────────────────────────
    print("\nGenerating multi-panel scientific scaling plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Latent Bottleneck Sweep
    ds_list = list(bottleneck_results.keys())
    accs_list = list(bottleneck_results.values())
    ax1.plot(ds_list, accs_list, color="#2b5c8f", marker="o", linewidth=2.5, markersize=8)
    ax1.axhline(y=0.038, color="gray", linestyle="--", alpha=0.7, label="Random Baseline (3.8%)")
    ax1.set_title("Experiment A: Latent Bottleneck Sweep", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Latent Space Dimension (d_model)", fontsize=11)
    ax1.set_ylabel("Key-Door Retrieval Accuracy", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ds_list)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.legend(frameon=True, fontsize=10)

    # Plot 2: Continuous vs Discrete Loss Convergence
    for name, data in ablation_results.items():
        history = data["loss_history"]
        steps_history = [i * 10 for i in range(len(history))]
        color = "#2b5c8f" if "Continuous" in name else "#dc2626"
        marker = "o" if "Continuous" in name else "s"
        ax2.plot(steps_history, history, label=f"{name} (Acc: {data['accuracy']*100:.1f}%)", color=color, marker=marker, linewidth=2)
    ax2.set_title("Experiment B: ODE vs Discrete Recurrent CMF", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Training Steps", fontsize=11)
    ax2.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax2.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    out_plot = out_dir / "adversarial_limits.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"[OK] Scientific plot saved to: {out_plot}")
    print("==================================================")
    print("ADVERSARIAL LIMITS EXPERIMENT COMPLETE.")
    print("==================================================")

if __name__ == "__main__":
    main()
