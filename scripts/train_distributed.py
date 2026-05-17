from __future__ import annotations

import argparse
import json
import os
import sys
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
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

def initialize_weights(model: torch.nn.Module, num_layers: int):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            
            # Scale down the output projection layers inside residual blocks and deliberative gates
            # to guarantee stable residual activation variance propagation in deep models
            if "proj" in name or "proposal" in name or "gate" in name or "update_gate" in name:
                with torch.no_grad():
                    module.weight.data.mul_(1.0 / math.sqrt(2.0 * num_layers))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
    
    # 1. Apply robust neural weight initialization
    if is_master:
        print("Initializing model weights using scaled residual initialization...")
    initialize_weights(model, config.num_layers)

    # 2. Check and load checkpoint BEFORE wrapping in FSDP/DDP to avoid CPU/GPU memory spikes
    checkpoint_exists = False
    start_step = 0
    tokens_seen = 0
    has_nan = False
    
    latest_ckpt = args.checkpoint_dir / "checkpoint_latest.pt"
    if not latest_ckpt.exists() and args.checkpoint_dir.exists():
        ckpt_files = sorted(args.checkpoint_dir.glob("checkpoint_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        if ckpt_files:
            latest_ckpt = ckpt_files[-1]

    if is_master:
        if latest_ckpt.exists():
            checkpoint_exists = True
            print(f"--- [RESUME] Found latest checkpoint at {latest_ckpt}. Restoring weights on Rank 0 CPU! ---")
            payload = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
            
            # Check for NaNs
            if "model" in payload:
                for k, v in payload["model"].items():
                    if torch.is_tensor(v) and torch.isnan(v).any():
                        has_nan = True
                        break
            
            if has_nan:
                print("\n--- [WARNING] Found NaN weights in checkpoint! Deleting corrupted checkpoint and resetting training from scratch to avoid infinite NaNs! ---\n")
                try:
                    latest_ckpt.unlink()
                except Exception as e:
                    print(f"Error unlinking corrupted checkpoint: {e}")
            else:
                # Load weights into the unwrapped model on Rank 0 CPU
                model.load_state_dict(payload["model"])
                start_step = payload["training"]["step"]
                tokens_seen = payload["training"].get("tokens", 0)
                print(f"Successfully loaded weights into unwrapped model on CPU. Ready for distributed wrapping!")

    if world_size > 1:
        # Broadcast metadata so all ranks agree on start_step, tokens_seen, and has_nan status
        metadata = [checkpoint_exists, start_step, tokens_seen, has_nan]
        dist.broadcast_object_list(metadata, src=0)
        checkpoint_exists, start_step, tokens_seen, has_nan = metadata

    is_fsdp = False
    if world_size > 1:
        if args.fsdp and device.type == "cuda":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
            
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
            import functools
            
            # Keep master weights in FP32 on CPU during sharding to preserve high-precision initialization
            mp_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float32
            )
            
            # Wrap any submodules with >= 1M parameters to enable layer-by-layer sharding
            my_auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=1_000_000
            )
            
            # Wrap on CPU first. FSDP will automatically shard and move parameters to local_rank GPU
            model = FSDP(
                model,
                device_id=local_rank,
                mixed_precision=mp_policy,
                auto_wrap_policy=my_auto_wrap_policy,
                sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                sync_module_states=True
            )
            is_fsdp = True
            if is_master:
                print("Using Fully Sharded Data Parallel (FSDP) with FP32 Master Weights, FP16 Gradient Reductions, and Size-Based Layer Auto-Wrapping.")




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

    use_fused = (device.type == "cuda") and not is_fsdp
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=use_fused)
    
    # 2. Add Cosine Warmup Learning Rate Scheduler for ultimate stability
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step: int):
        warmup_steps = 100
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, args.steps - warmup_steps))
        progress = min(1.0, progress)
        min_lr_ratio = 0.05
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Step the scheduler to match the restored step count
    if checkpoint_exists and not has_nan:
        for _ in range(start_step):
            scheduler.step()
        if is_master:
            print(f"Resumed scheduler state to step {start_step}")


    if is_fsdp:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        scaler = ShardedGradScaler(enabled=args.amp and device.type == "cuda")
    else:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    
    # 3. Load default gpt2 tokenizer safely on master for saving standalone packages
    tokenizer = None
    if is_master:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
            print(f"Warning: Could not load default gpt2 tokenizer: {e}")

    # 4. Checkpoint Resume System log status (FSDP handles parameters during wrap initialization)
    if checkpoint_exists and not has_nan:
        if is_master:
            print(f"Successfully synchronized loaded checkpoint across ranks! Resuming training from step {start_step}.")



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

    for step in range(start_step, args.steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        
        for _ in range(args.grad_accum):
            try:
                x, y = next(batches)
            except StopIteration:
                if is_master: print("--- Data exhausted. Ending training. ---")
                break
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
        scheduler.step()

        if is_master and (step + 1) % args.log_every == 0:
            elapsed = time.perf_counter() - start_time
            total_tokens = tokens_seen * world_size
            current_lr = scheduler.get_last_lr()[0]
            avg_loss = step_loss / args.grad_accum
            print(f"step={step+1}/{args.steps} loss={avg_loss:.4f} lr={current_lr:.3e} tokens={total_tokens:,} tok/s={total_tokens/max(1e-6, elapsed):.0f}")


        # Save latest checkpoint at configured step intervals atomically (overwrites previous to save disk space and maximize speed)
        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            if is_master or is_fsdp:
                checkpoint_path = args.checkpoint_dir / "checkpoint_latest.pt"
                temp_checkpoint_path = args.checkpoint_dir / "checkpoint_latest.pt.tmp"
                
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
                        }, temp_checkpoint_path)
                        if temp_checkpoint_path.exists():
                            temp_checkpoint_path.replace(checkpoint_path)
                        print(f"Saved FSDP latest checkpoint atomically to {checkpoint_path} (step {step+1})")
                else:
                    if is_master:
                        save_model_package(
                            temp_checkpoint_path,
                            model.module if world_size > 1 else model,
                            model_type=preset.model_type,
                            config=config,
                            tokenizer=tokenizer,
                            tokenizer_name="gpt2",
                            training={"step": step + 1, "tokens": tokens_seen * world_size}
                        )
                        if temp_checkpoint_path.exists():
                            temp_checkpoint_path.replace(checkpoint_path)
                        print(f"Saved latest model package atomically to {checkpoint_path} (step {step+1})")


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
                tokenizer=tokenizer,
                tokenizer_name="gpt2",
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
