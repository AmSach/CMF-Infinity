"""
CMF v3 Training Script.

Usage:
    python scripts/train.py --preset 120m --device cuda
    python scripts/train.py --preset 50m  --steps 5000 --batch 8 --seq_len 256
    python scripts/train.py --preset tiny --device cpu --steps 500  # smoke test

Writes checkpoints and logs to records/runs/<preset>_<timestamp>/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RECORDS = ROOT / "records"

import torch
import torch.nn as nn

from cmf.config import TrainingConfig
from cmf.experiments import (
    ExperimentLogger, RunConfig, count_parameters,
    evaluate_loss, set_seed, train_fixed_steps,
)
from cmf.presets import build_model, get_preset


def make_batches(vocab_size: int, batch_size: int, seq_len: int,
                 n: int, device: torch.device):
    for _ in range(n):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        yield ids, ids.clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset",   default="tiny")
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--steps",    type=int,   default=2000)
    parser.add_argument("--batch",    type=int,   default=4)
    parser.add_argument("--seq_len",  type=int,   default=64)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--log_traj", type=int,   default=200,
                        help="Log trajectory every N steps (0=off)")
    parser.add_argument("--ckpt_every", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device
                          if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    set_seed(args.seed)
    preset = get_preset(args.preset)
    model  = build_model(args.preset).to(device)
    params = count_parameters(model)

    print(f"[train] preset={args.preset}  params={params:,}  device={device}")
    print(f"        steps={args.steps}  batch={args.batch}  seq={args.seq_len}"
          f"  lr={args.lr}  grad_accum={args.grad_accum}")

    cfg = preset.config
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # LR warmup scheduler
    def warmup(step):
        warmup_steps = min(500, args.steps // 10)
        return min(1.0, step / max(warmup_steps, 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, warmup)

    train_b = list(make_batches(cfg.vocab_size, args.batch, args.seq_len, 500, device))
    eval_b  = list(make_batches(cfg.vocab_size, args.batch, args.seq_len,  20, device))

    run_name = f"{args.preset}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir  = str(RECORDS / "runs" / run_name)

    run_cfg = RunConfig(
        model_name=args.preset, param_count=params,
        d_model=cfg.d_model, num_slots=cfg.num_slots,
        solver_steps=cfg.solver_steps, routing_mode=cfg.routing_mode,
        preset=args.preset, dataset="synthetic_random",
        optimizer="AdamW", lr=args.lr,
        batch_size=args.batch, seq_len=args.seq_len,
        seed=args.seed, device=str(device),
    )

    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with ExperimentLogger(run_dir, run_cfg) as log:
        # Custom training loop with checkpointing
        model.train()
        initial_loss = evaluate_loss(model, eval_b, device)
        print(f"\n  initial_loss={initial_loss:.4f}")

        step       = 0
        best_loss  = float("inf")
        batch_iter = iter(train_b * 100)   # cycle

        import time as _time
        t0 = _time.perf_counter()

        while step < args.steps:
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0

            for acc in range(args.grad_accum):
                try:
                    ids, lbls = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(train_b * 100)
                    ids, lbls = next(batch_iter)

                log_traj = args.log_traj > 0 and step % args.log_traj == 0 and acc == 0
                out  = model(ids, labels=lbls, log_trajectory=log_traj)
                (out["loss"] / args.grad_accum).backward()
                step_loss += float(out["loss"].detach())

                if log_traj and "trajectory" in out:
                    log.log_trajectory(step, out["trajectory"])

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            step += 1

            extra = {}
            if "thinking_steps" in out:
                extra["thinking_steps"] = int(out["thinking_steps"])

            log.log(step, loss=step_loss / args.grad_accum,
                    lr=scheduler.get_last_lr()[0], **extra)

            if step_loss < best_loss:
                best_loss = step_loss

            if step % 100 == 0:
                elapsed = _time.perf_counter() - t0
                print(f"  step={step:5d}  loss={step_loss/args.grad_accum:.4f}"
                      f"  best={best_loss/args.grad_accum:.4f}"
                      f"  elapsed={elapsed:.0f}s")

            if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "opt_state":   opt.state_dict(),
                    "loss":        step_loss,
                }, str(ckpt_path))
                print(f"  [OK] checkpoint -> {ckpt_path}")

        final_loss = evaluate_loss(model, eval_b, device)
        print(f"\n  final_loss={final_loss:.4f}  (initial={initial_loss:.4f})")

        # Save final checkpoint
        torch.save({
            "step": step,
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "preset": args.preset,
        }, str(ckpt_dir / "final.pt"))
        print(f"  [OK] final checkpoint -> {ckpt_dir}/final.pt")
        print(f"  [OK] logs -> {run_dir}")


if __name__ == "__main__":
    main()
