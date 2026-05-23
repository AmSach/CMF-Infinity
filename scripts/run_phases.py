"""
CMF Phase Runner — executes all checklist phases sequentially.

Usage:
    python scripts/run_phases.py --phase all --preset 50m --device cpu
    python scripts/run_phases.py --phase 0    # infra + logging smoke test
    python scripts/run_phases.py --phase 1    # memory verification
    python scripts/run_phases.py --phase 2    # routing isolation
    python scripts/run_phases.py --phase 3    # iterative reasoning

Records written to:
    records/phases/phase_0_infra.json
    records/phases/phase_1_memory.json
    records/phases/phase_2_routing.json
    records/phases/phase_3_reasoning.json
    records/ablations/routing/routing_ablation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RECORDS = ROOT / "records"

import torch

from cmf.config import CMFConfig
from cmf.experiments import (
    ExperimentLogger, RunConfig, TrainReport,
    count_parameters, environment_report, evaluate_loss,
    run_routing_ablation, run_solver_depth_test,
    run_perturbation_test, set_seed, train_fixed_steps,
)
from cmf.memory_tasks import (
    KeyDoorDataset, MultiBindingDataset, ObjectPermanenceDataset,
    measure_retention_curve, measure_capacity_curve,
)
from cmf.presets import build_model, get_preset


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def save_phase(name: str, data: dict) -> None:
    path = RECORDS / "phases" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"phase": name, "timestamp": now(),
            "environment": environment_report(), **data}
    path.write_text(json.dumps(data, indent=2))
    print(f"\n[OK] {name} -> {path}")



def make_lm_batches(vocab_size: int, batch_size: int, seq_len: int,
                    n_batches: int, device: torch.device):
    """Synthetic random LM batches for structural tests."""
    batches = []
    for _ in range(n_batches):
        ids  = torch.randint(0, vocab_size, (batch_size, seq_len))
        lbls = ids.clone()
        batches.append((ids.to(device), lbls.to(device)))
    return batches


def make_memory_batches(dataset, batch_size: int, device: torch.device):
    """Pad + batch a memory task dataset."""
    from torch.nn.utils.rnn import pad_sequence
    from cmf.memory_tasks import PAD
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
# Phase 0 — Infrastructure
# ─────────────────────────────────────────────────────────────────────────────

def phase_0(preset_name: str, device: torch.device):
    print("\n=======================================")
    print("PHASE 0 - Infrastructure smoke test")
    print("=======================================")
    set_seed(42)

    model  = build_model(preset_name).to(device)
    preset = get_preset(preset_name)
    cfg    = preset.config
    params = count_parameters(model)
    print(f"  preset={preset_name}  params={params:,}  device={device}")

    # Smoke: forward + loss + backward
    ids  = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    out  = model(ids, labels=ids, log_trajectory=True)
    assert "logits" in out
    assert "loss"   in out
    out["loss"].backward()
    print(f"  loss={out['loss'].item():.4f}  OK")

    # Trajectory logging
    traj = out.get("trajectory", [])
    print(f"  trajectory steps logged: {len(traj)}")

    # Logger
    run_cfg = RunConfig(
        model_name=preset_name, param_count=params,
        d_model=cfg.d_model, num_slots=cfg.num_slots,
        solver_steps=cfg.solver_steps,
        routing_mode=cfg.routing_mode,
        preset=preset_name, dataset="synthetic",
        optimizer="AdamW", lr=3e-4,
        batch_size=2, seq_len=16, seed=42, device=str(device),
    )
    with ExperimentLogger(str(RECORDS / "runs" / "phase0_smoke"), run_cfg) as log:
        log.log(0, loss=out["loss"].item())
        if traj:
            log.log_trajectory(0, traj)

    save_phase("phase_0_infra", {
        "preset": preset_name, "params": params,
        "loss": out["loss"].item(),
        "trajectory_steps": len(traj),
        "status": "PASS",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Memory verification
# ─────────────────────────────────────────────────────────────────────────────

def phase_1(preset_name: str, device: torch.device):
    print("\n=======================================")
    print("PHASE 1 - Memory verification")
    print("=======================================")
    set_seed(42)

    # Build a model with the memory-task vocabulary
    from cmf.memory_tasks import VOCAB_SIZE
    cfg = CMFConfig(
        vocab_size=VOCAB_SIZE,
        d_model=128, hidden_dim=256,
        num_layers=3, num_slots=16,
        solver_steps=4, thinking_steps=4,
        dropout=0.0, tie_embeddings=False,
    )
    from cmf.model import ParallelCMF
    model = ParallelCMF(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)
    params = count_parameters(model)
    print(f"  memory-task model  params={params:,}")

    # ── 1.1: Retention curve ─────────────────────────────────────────────
    print("\n[1.1] Retention degradation curve (untrained baseline)")
    gaps = [16, 64, 128, 512]
    retention_baseline = measure_retention_curve(
        model, gap_lengths=gaps, n_per_gap=100, device=str(device))

    # Quick training on KeyDoor
    print("\n  Training on KeyDoor (500 steps) ...")
    ds = KeyDoorDataset(2000, gap_lengths=gaps)
    train_b = make_memory_batches(ds, batch_size=16, device=device)
    eval_b  = make_memory_batches(KeyDoorDataset(200, gap_lengths=gaps), 16, device)

    run_dir = str(RECORDS / "runs" / "phase1_keydoor")
    with ExperimentLogger(run_dir) as log:
        report = train_fixed_steps(
            "keydoor", model, train_b, eval_b, device, opt,
            steps=500, logger=log, log_traj_every=100)

    print(f"\n  training: {report.initial_loss:.4f} -> {report.final_loss:.4f}")

    print("\n[1.1] Retention curve (trained)")
    retention_trained = measure_retention_curve(
        model, gap_lengths=gaps, n_per_gap=100, device=str(device))

    # ── 1.4: Capacity curve ──────────────────────────────────────────────
    print("\n[1.4] Capacity curve (n_bindings vs accuracy)")
    capacity = measure_capacity_curve(
        model, k_list=[2, 4, 8], n_per_k=100, device=str(device))

    # ── 1.5: Perturbation recovery ───────────────────────────────────────
    print("\n[1.5] Perturbation recovery")
    perturb = run_perturbation_test(
        model, eval_b, device,
        output_dir=str(RECORDS / "ablations" / "perturbation"))

    # ── 1.6: Memory footprint constant ──────────────────────────────────
    from cmf.model import SlotMemory
    mem_params = sum(p.numel() for p in model.memory.parameters())
    print(f"\n[1.6] SlotMemory params = {mem_params}  (must not scale with seq_len)")

    save_phase("phase_1_memory", {
        "preset": preset_name,
        "memory_params": mem_params,
        "retention_baseline": {str(k): v for k, v in retention_baseline.items()},
        "retention_trained":  {str(k): v for k, v in retention_trained.items()},
        "capacity_curve":     {str(k): v for k, v in capacity.items()},
        "perturbation":       {str(k): v for k, v in perturb.items()},
        "train_initial_loss": report.initial_loss,
        "train_final_loss":   report.final_loss,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Routing isolation
# ─────────────────────────────────────────────────────────────────────────────

def phase_2(preset_name: str, device: torch.device):
    print("\n=======================================")
    print("PHASE 2 - Routing isolation")
    print("=======================================")
    set_seed(42)

    preset = get_preset(preset_name)
    cfg    = preset.config

    train_b = make_lm_batches(cfg.vocab_size, 4, 32, 200, device)
    eval_b  = make_lm_batches(cfg.vocab_size, 4, 32,  20, device)

    from cmf.model import ParallelCMF
    def factory(): return ParallelCMF(cfg)

    results = run_routing_ablation(
        factory, train_b, eval_b, device,
        steps_per_mode=100,
        output_dir=str(RECORDS / "ablations" / "routing"),
    )

    print("\nRouting ablation summary:")
    for mode, r in results.items():
        print(f"  {mode:15s}  Delta loss={r['initial_loss']:.4f}->{r['final_loss']:.4f}"
              f"  ratio={r['loss_ratio']:.3f}")

    save_phase("phase_2_routing", {"preset": preset_name, "results": results})


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Iterative reasoning
# ─────────────────────────────────────────────────────────────────────────────

def phase_3(preset_name: str, device: torch.device):
    print("\n=======================================")
    print("PHASE 3 - Iterative reasoning")
    print("=======================================")
    set_seed(42)

    from cmf.model import DeliberativeCMF
    preset = get_preset(preset_name)
    cfg_d  = CMFConfig(
        **{**preset.config.__dict__,
           "thinking_steps": 8,
           "adaptive_thinking": True,
           "min_thinking_steps": 2,
           "max_thinking_steps": 16},
    )
    model = DeliberativeCMF(cfg_d).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4)

    train_b = make_lm_batches(cfg_d.vocab_size, 4, 32, 200, device)
    eval_b  = make_lm_batches(cfg_d.vocab_size, 4, 32,  20, device)

    # 3.1: Solver depth — logit evolution
    print("\n[3.1] Logit evolution (untrained)")
    ids  = torch.randint(0, cfg_d.vocab_size, (1, 16), device=device)
    traj_before = run_solver_depth_test(
        model, ids, device,
        output_dir=str(RECORDS / "ablations" / "solver_depth_before"))

    # Train briefly
    print("\n  Training deliberative model (200 steps) ...")
    run_dir = str(RECORDS / "runs" / "phase3_deliberative")
    with ExperimentLogger(run_dir) as log:
        report = train_fixed_steps(
            "deliberative", model, train_b, eval_b, device, opt,
            steps=200, logger=log, log_traj_every=50)

    print(f"  {report.initial_loss:.4f} -> {report.final_loss:.4f}")

    print("\n[3.1] Logit evolution (trained)")
    traj_after = run_solver_depth_test(
        model, ids, device,
        output_dir=str(RECORDS / "ablations" / "solver_depth_after"))

    # 3.2: Adaptive compute
    print("\n[3.2] Adaptive compute (steps used per input)")
    model.eval()
    with torch.no_grad():
        out = model(ids)
    thinking_steps = int(out.get("thinking_steps", -1))
    print(f"  steps used: {thinking_steps}")

    save_phase("phase_3_reasoning", {
        "preset": preset_name,
        "traj_before": traj_before,
        "traj_after":  traj_after,
        "adaptive_steps_used": thinking_steps,
        "train_initial_loss": report.initial_loss,
        "train_final_loss":   report.final_loss,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",  default="all",
                        choices=["all", "0", "1", "2", "3"])
    parser.add_argument("--preset", default="tiny",
                        help="Preset name (tiny | 50m | 120m | ...)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device
                          if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    phases = {"0": phase_0, "1": phase_1, "2": phase_2, "3": phase_3}
    to_run = list(phases.keys()) if args.phase == "all" else [args.phase]

    for p in to_run:
        phases[p](args.preset, device)

    print("\n\nAll requested phases complete.")
    print(f"Records in: {RECORDS}/")


if __name__ == "__main__":
    main()
