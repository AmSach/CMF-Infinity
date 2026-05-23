"""
Phase 0 — Experiment infrastructure.

Every run must save:
    records/runs/<name>/run_config.json
    records/runs/<name>/metrics.jsonl
    records/runs/<name>/trajectory.csv
    records/runs/<name>/summary.json

This file also provides:
    run_routing_ablation()   — Phase 2.1: routing isolation curve
    run_solver_depth_test()  — Phase 3.1: logit evolution vs step count
    run_perturbation_test()  — Phase 1.5: recovery from noise injection
    environment_report()     — reproducibility metadata
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_report() -> dict:
    return {
        "python":   platform.python_version(),
        "torch":    torch.__version__,
        "cuda":     torch.version.cuda or "none",
        "device":   "cuda" if torch.cuda.is_available() else "cpu",
        "platform": platform.platform(),
        "pid":      os.getpid(),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Run config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    model_name: str
    param_count: int
    d_model: int
    num_slots: int
    solver_steps: int
    routing_mode: str
    preset: str
    dataset: str
    optimizer: str
    lr: float
    batch_size: int
    seq_len: int
    seed: int
    device: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment logger
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentLogger:
    """
    Context manager that writes a full experiment record to disk.

    Usage:
        cfg = RunConfig(...)
        with ExperimentLogger("records/runs/my_run", cfg) as log:
            for step, loss in training_loop():
                log.log(step, loss=loss)
                if trajectory:
                    log.log_trajectory(step, trajectory)
        # summary.json written on __exit__
    """

    def __init__(self, output_dir: str, cfg: Optional[RunConfig] = None):
        self.out = Path(output_dir)
        self.cfg = cfg
        self._metrics_fh  = None
        self._traj_fh     = None
        self._traj_writer = None
        self._steps: list[dict] = []
        self._t0 = time.time()

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        if self.cfg is not None:
            (self.out / "run_config.json").write_text(
                json.dumps({**asdict(self.cfg),
                            "environment": environment_report()}, indent=2))

        self._metrics_fh = open(self.out / "metrics.jsonl", "w")

        self._traj_fh = open(self.out / "trajectory.csv", "w", newline="")
        self._traj_writer = csv.DictWriter(
            self._traj_fh,
            fieldnames=["train_step", "solver_step", "z_norm", "v_norm",
                        "halt_prob", "logit_entropy"],
            extrasaction="ignore",
        )
        self._traj_writer.writeheader()
        return self

    def log(self, step: int, **kwargs: Any) -> None:
        row = {"step": step, **kwargs}
        self._steps.append(row)
        if self._metrics_fh:
            self._metrics_fh.write(json.dumps(row) + "\n")
            self._metrics_fh.flush()

    def log_trajectory(self, train_step: int, traj: list[dict]) -> None:
        if self._traj_writer is None:
            return
        for entry in traj:
            self._traj_writer.writerow({"train_step": train_step, **entry})
        if self._traj_fh:
            self._traj_fh.flush()

    def save(self) -> None:
        losses = [s["loss"] for s in self._steps if "loss" in s]
        summary = {
            "total_steps":   len(self._steps),
            "wall_seconds":  round(time.time() - self._t0, 2),
            "final_loss":    losses[-1] if losses else None,
            "best_loss":     min(losses) if losses else None,
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2))

    def __exit__(self, *_):
        self.save()
        for fh in [self._metrics_fh, self._traj_fh]:
            if fh:
                fh.close()


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainReport:
    model_name: str
    steps: int
    tokens: int
    initial_loss: float
    final_loss: float
    best_loss: float
    elapsed_sec: float
    parameters: int

    @property
    def loss_ratio(self) -> float:
        return self.final_loss / max(self.initial_loss, 1e-12)

    def to_dict(self) -> dict:
        return {**asdict(self), "loss_ratio": self.loss_ratio}


def evaluate_loss(
    model: nn.Module,
    batches: list[tuple[Tensor, Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for ids, lbls in batches:
            out = model(ids.to(device), labels=lbls.to(device))
            losses.append(float(out["loss"]))
    if not losses:
        raise ValueError("evaluate_loss: no batches")
    return sum(losses) / len(losses)


def train_fixed_steps(
    model_name: str,
    model: nn.Module,
    train_batches: Iterable[tuple[Tensor, Tensor]],
    eval_batches: list[tuple[Tensor, Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    steps: int,
    grad_accum: int = 1,
    clip_norm: float = 1.0,
    logger: Optional[ExperimentLogger] = None,
    log_traj_every: int = 0,
) -> TrainReport:
    model.to(device)
    initial_loss = evaluate_loss(model, eval_batches, device)
    model.train()
    t0 = time.perf_counter()
    losses: list[float] = []
    tokens = 0
    batch_iter = iter(train_batches)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for acc in range(grad_accum):
            try:
                ids, lbls = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_batches)
                ids, lbls = next(batch_iter)

            log_traj = log_traj_every > 0 and step % log_traj_every == 0 and acc == 0
            out  = model(ids.to(device), labels=lbls.to(device),
                         log_trajectory=log_traj)
            loss = out["loss"] / grad_accum
            loss.backward()
            step_loss += float(loss.detach() * grad_accum)
            tokens    += int(ids.numel())

            if log_traj and logger and "trajectory" in out:
                logger.log_trajectory(step, out["trajectory"])

        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        losses.append(step_loss)

        if logger:
            extra = {}
            if "thinking_steps" in out:
                extra["thinking_steps"] = int(out["thinking_steps"])
            if "halt_mean" in out:
                extra["halt_mean"] = float(out["halt_mean"])
            logger.log(step, loss=step_loss, **extra)

    final_loss = evaluate_loss(model, eval_batches, device)
    return TrainReport(
        model_name   = model_name,
        steps        = steps,
        tokens       = tokens,
        initial_loss = initial_loss,
        final_loss   = final_loss,
        best_loss    = min(losses),
        elapsed_sec  = round(time.perf_counter() - t0, 2),
        parameters   = count_parameters(model),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2.1 — Routing ablation curve
# ─────────────────────────────────────────────────────────────────────────────

def run_routing_ablation(
    model_factory,
    train_batches: list,
    eval_batches: list,
    device: torch.device,
    steps_per_mode: int = 100,
    output_dir: str = "records/ablations/routing",
    seed: int = 42,
) -> dict[str, dict]:
    """
    Phase 2.1: train identical models with each routing mode, compare loss.

    Returns {mode: {"initial_loss", "final_loss", "loss_ratio"}}.
    Saves routing_ablation.json to output_dir.

    One variable at a time (checklist Rule 2): only routing_mode changes.
    """
    modes = ["full", "sparse_topk", "local_window", "none"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for mode in modes:
        set_seed(seed)
        model = model_factory().to(device)
        model.anchor.mode = mode
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

        print(f"\n[routing_ablation] mode={mode}")
        report = train_fixed_steps(
            model_name=f"routing_{mode}",
            model=model, train_batches=train_batches,
            eval_batches=eval_batches, device=device,
            optimizer=opt, steps=steps_per_mode, grad_accum=1)

        results[mode] = {
            "initial_loss": report.initial_loss,
            "final_loss":   report.final_loss,
            "loss_ratio":   report.loss_ratio,
        }
        print(f"  initial={report.initial_loss:.4f}  final={report.final_loss:.4f}")

    (Path(output_dir) / "routing_ablation.json").write_text(
        json.dumps(results, indent=2))
    print(f"\nRouting ablation → {output_dir}/routing_ablation.json")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3.1 — Solver depth / logit evolution test
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_solver_depth_test(
    model,
    input_ids: Tensor,
    device: torch.device,
    output_dir: str = "records/ablations/solver_depth",
) -> list[dict]:
    """
    Phase 3.1: record logit entropy at every thinking step.

    If solver depth is real, entropy should decrease monotonically.
    If logits stabilise at step 1, solver depth adds no value.

    Returns list of {step, logit_entropy, z_norm, v_norm}.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.eval()
    model.to(device)

    out  = model(input_ids.to(device), log_trajectory=True)
    traj = out.get("trajectory", [])

    path = Path(output_dir) / "solver_depth.json"
    path.write_text(json.dumps(traj, indent=2))
    print(f"\nSolver depth test → {path}")
    for t in traj:
        print(f"  step={t['step']:2d}  entropy={t['logit_entropy']:.4f}"
              f"  z_norm={t['z_norm']:.4f}  v_norm={t['v_norm']:.4f}")
    return traj


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.5 — Perturbation recovery
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_perturbation_test(
    model,
    eval_batches: list[tuple[Tensor, Tensor]],
    device: torch.device,
    noise_levels: Optional[list[float]] = None,
    output_dir: str = "records/ablations/perturbation",
) -> dict[float, float]:
    """
    Phase 1.5: inject Gaussian noise into input embeddings, measure loss.

    Real attractor basins → graceful degradation.
    No memory → catastrophic collapse.
    Returns {noise_std: loss}.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.eval()
    model.to(device)
    results: dict[float, float] = {}

    original_embed = model.embedding

    for sigma in noise_levels:
        losses = []
        for ids, lbls in eval_batches:
            ids  = ids.to(device)
            lbls = lbls.to(device)
            # Patch: add noise to embeddings
            emb_clean = model.embedding(ids)
            noise     = torch.randn_like(emb_clean) * sigma

            # Temporarily replace embedding with noisy version
            class NoisyEmbed(nn.Module):
                def __init__(self, base, n): super().__init__(); self.base = base; self.n = n
                def forward(self, x): return self.base(x) + self.n

            model.embedding = NoisyEmbed(original_embed, noise)
            out = model(ids, labels=lbls)
            losses.append(float(out["loss"]))
            model.embedding = original_embed

        results[sigma] = sum(losses) / len(losses)
        print(f"  sigma={sigma:.3f}  loss={results[sigma]:.4f}")

    (Path(output_dir) / "perturbation.json").write_text(
        json.dumps({str(k): v for k, v in results.items()}, indent=2))
    return results
