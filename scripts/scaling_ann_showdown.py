"""
CMF Scaling & ANN Showdown Experiment

Runs comparison of:
  - Full Attention (full)
  - Sparse Top-K (sparse_topk)
  - ANN Retrieval (ann)
  - Hierarchical Routing (hierarchical)

Across distractor lengths:
  - 128
  - 512
  - 2048
  - 8192

Measures:
  1. Retrieval accuracy
  2. Routing entropy (mean entropy of attention weights)
  3. Latency (milliseconds per forward pass)
  4. GPU Memory usage (MB allocated)
  5. Attention score matrix sparsity (%)
  6. Degradation curves

Saves results and generates a gorgeous comparative multi-panel scientific plot.
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

@torch.no_grad()
def measure_metrics(model, mode: str, gap: int, device: torch.device, n_samples: int = 50) -> dict:
    model.eval()
    model.anchor.mode = mode

    ds = KeyDoorDataset(n_samples, gap_lengths=[gap], seed=42)
    correct = 0
    latencies = []
    entropies = []
    sparsities = []

    # Warmup
    dummy = torch.randint(0, VOCAB_SIZE, (1, 64), device=device)
    for _ in range(5):
        _ = model(dummy)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_start = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

    for idx in range(len(ds)):
        s = ds[idx]
        ids = s["input_ids"].unsqueeze(0).to(device)
        qpos = s["query_pos"]
        correct_tok = val_tok(s["value"])

        # Measure latency
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        # We hook into model to grab attention scores
        out = model(ids)
        
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

        # Prediction accuracy
        pred = out["logits"][0, qpos].argmax().item()
        correct += int(pred == correct_tok)

    # Peak memory allocated during evaluation
    mem_peak = (torch.cuda.max_memory_allocated() / (1024 * 1024)) - mem_start  # MB
    if mem_peak < 0:
        mem_peak = 0.0

    accuracy = correct / n_samples
    avg_latency = sum(latencies) / len(latencies)

    # Let's estimate attention matrix properties by running one sample through
    # and tracking the proportion of values that are -inf (masked/sparse)
    sample_ids = ds[0]["input_ids"].unsqueeze(0).to(device)
    T = sample_ids.size(1)
    
    # We can retrieve attention matrix directly by running anchor query
    emb = model.embedding(sample_ids)
    context = model.encoder(emb)
    z = model.initial_state(context)
    
    # Run one step
    Q = model.anchor._split(model.anchor.q_proj(z))
    K = model.anchor._split(model.anchor.k_proj(context))
    scale = math.sqrt(model.anchor.head_dim)
    scores = torch.matmul(Q, K.transpose(-1, -2)) / scale
    
    # Apply causal mask
    causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal[None, None], float("-inf"))

    # Apply specific routing mask
    if mode == "local_window":
        row = torch.arange(T, device=device).unsqueeze(1)
        col = torch.arange(T, device=device).unsqueeze(0)
        outside = (row - col) > model.anchor.window
        scores = scores.masked_fill(outside[None, None], float("-inf"))
    elif mode == "sparse_topk":
        k = min(model.anchor.topk, T)
        topk_vals, _ = scores.topk(k, dim=-1)
        scores = scores.masked_fill(scores < topk_vals[..., -1:], float("-inf"))
    elif mode == "ann":
        C = 8
        pad_len = (C - T % C) % C
        scores_padded = F.pad(scores, (0, pad_len), value=float("-inf")) if pad_len > 0 else scores
        T_pad = scores_padded.size(-1)
        scores_blocked = scores_padded.view(1, model.anchor.n_heads, T, T_pad // C, C)
        scores_blocked_clean = scores_blocked.clamp(min=-1e9)
        block_max, _ = scores_blocked_clean.max(dim=-1)
        M = max(1, (T_pad // C) // 4)
        top_blocks_vals, _ = block_max.topk(M, dim=-1)
        threshold = top_blocks_vals[..., -1:]
        block_mask = block_max >= threshold
        token_mask = block_mask.unsqueeze(-1).expand(-1, -1, -1, -1, C).reshape(1, model.anchor.n_heads, T, T_pad)[..., :T]
        row_idx = torch.arange(T, device=device).unsqueeze(1)
        col_idx = torch.arange(T, device=device).unsqueeze(0)
        is_local = (row_idx - col_idx >= 0) & (row_idx - col_idx <= 16)
        final_mask = token_mask | is_local[None, None]
        scores = scores.masked_fill(~final_mask, float("-inf"))
    elif mode == "hierarchical":
        L = 32
        C = 8
        row_idx = torch.arange(T, device=device).unsqueeze(1)
        col_idx = torch.arange(T, device=device).unsqueeze(0)
        is_local = (row_idx - col_idx >= 0) & (row_idx - col_idx <= L)
        pad_len = (C - T % C) % C
        scores_padded = F.pad(scores, (0, pad_len), value=float("-inf")) if pad_len > 0 else scores
        T_pad = scores_padded.size(-1)
        scores_blocked = scores_padded.view(1, model.anchor.n_heads, T, T_pad // C, C)
        scores_blocked_clean = scores_blocked.clamp(min=-1e9)
        block_means = scores_blocked_clean.mean(dim=-1, keepdim=True)
        scores_hier = block_means.expand_as(scores_blocked).reshape(1, model.anchor.n_heads, T, T_pad)[..., :T]
        scores = torch.where(is_local[None, None], scores, scores_hier)
        scores = scores.masked_fill(causal[None, None], float("-inf"))

    # Compute sparsity (% of elements that are -inf)
    total_elements = scores.numel()
    inf_elements = torch.isinf(scores).sum().item()
    sparsity = (inf_elements / total_elements) * 100.0

    # Compute entropy of softmax weights
    # Ignore causal masked positions
    attn = F.softmax(scores, dim=-1)
    # entropy = -sum(p * log(p))
    entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=-1).mean().item()

    return {
        "accuracy": accuracy,
        "latency": avg_latency,
        "memory": mem_peak,
        "sparsity": sparsity,
        "entropy": entropy,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print("CMF SCALING & SPARSE ROUTING SHOWNOWN")
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
    modes = {
        "Full Attention (Dense)": "full",
        "Sparse Top-K (K=4)": "sparse_topk",
        "ANN Coarse Routing": "ann",
        "Hierarchical Memory": "hierarchical"
    }

    results = {mode_name: {} for mode_name in modes}

    # Standard training dataset
    print("Building training dataset on gaps...")
    kd_train = KeyDoorDataset(1600, gap_lengths=[32, 64, 128], seed=42)
    train_batches = make_memory_batches(kd_train, batch_size=32, device=device)

    # We will train one unified model with Full Attention
    # Since mode is dynamic, we can retarget the trained weights at evaluation!
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

    # Now evaluate all modes across all gaps!
    print("\n==================================================")
    print("RUNNING SCALING EVALUATIONS")
    print("==================================================")

    for mode_name, mode_id in modes.items():
        print(f"\nEvaluating: {mode_name}")
        for gap in gaps:
            metrics = measure_metrics(model, mode_id, gap, device, n_samples=50)
            results[mode_name][gap] = metrics
            print(f"  Gap={gap:5d} | Acc={metrics['accuracy']:.3f} | Latency={metrics['latency']:.2f}ms | Mem={metrics['memory']:.2f}MB | Sparsity={metrics['sparsity']:.1f}% | Entropy={metrics['entropy']:.4f}")

    # Save metrics
    out_dir = ROOT / "records"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "scaling_ann_showdown.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[OK] Showdown results saved to: {out_json}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Plotting Comparative Figures
    # ─────────────────────────────────────────────────────────────────────────────
    print("\nGenerating multi-panel scientific scaling plots...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "Full Attention (Dense)": "#2b5c8f",
        "Sparse Top-K (K=4)": "#4682b4",
        "ANN Coarse Routing": "#10b981", # Emerald
        "Hierarchical Memory": "#d97706"  # Amber
    }
    
    markers = {
        "Full Attention (Dense)": "o",
        "Sparse Top-K (K=4)": "s",
        "ANN Coarse Routing": "^",
        "Hierarchical Memory": "D"
    }

    # Plot 1: Retrieval Accuracy Degradation
    for name in modes:
        accs = [results[name][gap]["accuracy"] for gap in gaps]
        ax1.plot(gaps, accs, label=name, color=colors[name], marker=markers[name], linewidth=2, markersize=8)
    ax1.set_title("Retrieval Accuracy vs Distractor Gap", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Distractor Gap (tokens)", fontsize=11)
    ax1.set_ylabel("Retrieval Accuracy", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(gaps)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.legend(frameon=True, fontsize=10)

    # Plot 2: Latency Scaling
    for name in modes:
        latencies = [results[name][gap]["latency"] for gap in gaps]
        ax2.plot(gaps, latencies, label=name, color=colors[name], marker=markers[name], linewidth=2, markersize=8)
    ax2.set_title("Inference Latency vs Context Size", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Context Size (tokens)", fontsize=11)
    ax2.set_ylabel("Inference Latency (ms)", fontsize=11)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(gaps)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.legend(frameon=True, fontsize=10)

    # Plot 3: GPU Memory Usage
    for name in modes:
        memory = [results[name][gap]["memory"] for gap in gaps]
        ax3.plot(gaps, memory, label=name, color=colors[name], marker=markers[name], linewidth=2, markersize=8)
    ax3.set_title("Peak GPU Memory Overhead vs Context Size", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Context Size (tokens)", fontsize=11)
    ax3.set_ylabel("Memory Overhead (MB)", fontsize=11)
    ax3.set_xscale("log", base=2)
    ax3.set_xticks(gaps)
    ax3.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax3.legend(frameon=True, fontsize=10)

    # Plot 4: Routing Sparsity
    for name in modes:
        sparsity = [results[name][gap]["sparsity"] for gap in gaps]
        ax4.plot(gaps, sparsity, label=name, color=colors[name], marker=markers[name], linewidth=2, markersize=8)
    ax4.set_title("Routing Matrix Sparsity vs Context Size", fontsize=12, fontweight="bold", pad=10)
    ax4.set_xlabel("Context Size (tokens)", fontsize=11)
    ax4.set_ylabel("Sparsity (%)", fontsize=11)
    ax4.set_xscale("log", base=2)
    ax4.set_xticks(gaps)
    ax4.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax4.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    out_plot = out_dir / "scaling_ann_showdown.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"[OK] Scientific plot saved to: {out_plot}")
    print("==================================================")
    print("SHOWNOWN COMPLETE.")
    print("==================================================")

if __name__ == "__main__":
    main()
