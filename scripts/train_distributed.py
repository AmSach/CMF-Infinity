from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf import (
    CMFConfig,
    ParallelContinuousMeaningField,
    ContinuousMeaningField,
    DeliberativeContinuousMeaningField,
)
from cmf.checkpointing import save_model_package
from cmf.presets import estimate_cmf_parameters, get_preset
from cmf.runtime import resolve_device, synchronize_device
from cmf.scalable_data import (
    cached_lm_batches_from_shards,
    load_token_cache_manifest,
)

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = resolve_device("auto")
    return rank, local_rank, world_size, device

def build_model(model_type: str, config: CMFConfig) -> torch.nn.Module:
    if model_type == "parallel_cmf":
        return ParallelContinuousMeaningField(config)
    if model_type == "continuous_cmf":
        return ContinuousMeaningField(config)
    if model_type == "deliberative_cmf":
        return DeliberativeContinuousMeaningField(config)
    raise ValueError(f"Unsupported model_type: {model_type}")

def train(args: argparse.Namespace) -> None:
    rank, local_rank, world_size, device = setup_distributed()
    is_master = (rank == 0)

    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    preset = get_preset(args.preset)
    config = CMFConfig(**preset.config.__dict__)
    if args.seq_len:
        config.max_seq_len = args.seq_len

    if is_master:
        print(f"Distributed training: world_size={world_size}, device={device}")
        print(f"Model: {preset.display_name} ({estimate_cmf_parameters(config, model_type=preset.model_type):,} params)")

    model = build_model(preset.model_type, config)
    
    is_fsdp = False
    if world_size > 1:
        if args.fsdp and device.type == "cuda":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision
            
            # Cast CPU parameters to FP16 to avoid the initialization VRAM spike when FSDP copies unsharded params to GPU
            model = model.half()
            
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
            # Wrap on CPU first. FSDP will automatically shard and move parameters to local_rank GPU
            model = FSDP(model, device_id=local_rank, mixed_precision=mp_policy)
            is_fsdp = True
            if is_master:
                print("Using Fully Sharded Data Parallel (FSDP) with FP16 Mixed Precision.")
        else:
            model = model.to(device)
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    else:
        model = model.to(device)

    if args.compile:
        try:
            model = torch.compile(model)
            if is_master: print("torch.compile enabled.")
        except Exception as exc:
            if is_master: print(f"torch.compile failed: {exc}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=(device.type == "cuda"))
    if is_fsdp:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        scaler = ShardedGradScaler(enabled=args.amp and device.type == "cuda")
    else:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    
    # Load data
    # We use rank-specific seeds to ensure different ranks see different data shards/samples
    batches = cached_lm_batches_from_shards(
        args.token_cache_dir,
        seq_len=config.max_seq_len,
        batch_size=args.micro_batch_size,
        random_batches=True,
        seed=args.seed + rank,
        pin_memory=device.type == "cuda",
        batches_per_shard=args.cache_batches_per_shard,
    )

    model.train()
    start_time = time.perf_counter()
    tokens_seen = 0

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        
        for _ in range(args.grad_accum):
            x, y = next(batches)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                out = model(x, labels=y, gradient_checkpointing=args.gradient_checkpointing)
                loss = out["loss"] / args.grad_accum
            
            scaler.scale(loss).backward()
            step_loss += float(out["loss"].detach())
            tokens_seen += int(x.numel())

        if args.clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            if is_fsdp:
                model.clip_grad_norm_(args.clip_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()

        if is_master and (step + 1) % args.log_every == 0:
            elapsed = time.perf_counter() - start_time
            total_tokens = tokens_seen * world_size
            print(f"step={step+1}/{args.steps} loss={step_loss:.4f} tokens={total_tokens:,} tok/s={total_tokens/elapsed:.0f}")

        if is_master and (step + 1) % args.save_every == 0:
            checkpoint_path = args.checkpoint_dir / f"checkpoint_step_{step+1}.pt"
            if is_fsdp:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import StateDictType, FullStateDictConfig
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                    state_dict_to_save = model.state_dict()
                if is_master:
                    torch.save({
                        "model": state_dict_to_save,
                        "config": config,
                        "training": {"step": step + 1, "tokens": tokens_seen * world_size}
                    }, checkpoint_path)
                    print(f"Saved FSDP step checkpoint to {checkpoint_path}")
            else:
                save_model_package(
                    args.package_out,
                    model.module if world_size > 1 else model,
                    model_type=preset.model_type,
                    config=config,
                    training={"step": step + 1, "tokens": tokens_seen * world_size}
                )

    if is_master:
        print("Training complete.")
        if is_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict_to_save = model.state_dict()
            if is_master:
                torch.save({
                    "model": state_dict_to_save,
                    "config": config,
                    "model_type": preset.model_type
                }, args.package_out)
                print(f"Final gathered model package saved to {args.package_out}")
        else:
            save_model_package(
                args.package_out,
                model.module if world_size > 1 else model,
                model_type=preset.model_type,
                config=config,
                training={"step": args.steps, "tokens": tokens_seen * world_size}
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="infinity-0.5b")
    parser.add_argument("--token-cache-dir", type=Path, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cache-batches-per-shard", type=int, default=512)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--fsdp", action="store_true", help="Enable Fully Sharded Data Parallel (FSDP)")
    parser.add_argument("--package-out", type=Path, default=ROOT / "records" / "checkpoints" / "cmf_0.5b_final.package.pt")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "records" / "checkpoints" / "cmf_0.5b_steps")
    args = parser.parse_args()
    
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train(args)

if __name__ == "__main__":
    main()
