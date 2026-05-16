from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import cycle
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf import (
    CMFConfig,
    ContinuousMeaningField,
    DeliberativeContinuousMeaningField,
    FastParallelContinuousMeaningField,
    ParallelContinuousMeaningField,
)
from cmf.checkpointing import load_model_package, save_model_package
from cmf.data import ByteTokenizer
from cmf.presets import estimate_cmf_parameters, get_preset
from cmf.runtime import resolve_device, synchronize_device
from cmf.scalable_data import (
    cached_lm_batches_from_shards,
    cached_lm_batches,
    iter_hf_text,
    iter_local_text,
    iter_token_batches_from_texts,
    load_token_cache,
    load_token_cache_manifest,
    synthetic_text_stream,
)


def build_tokenizer(args: argparse.Namespace, vocab_size: int | None = None):
    tokenizer_choice = args.tokenizer
    if tokenizer_choice == "auto":
        tokenizer_choice = "gpt2" if vocab_size and vocab_size > 256 else "byte"
    if tokenizer_choice == "byte":
        return ByteTokenizer(), None, 256
    if tokenizer_choice == "gpt2":
        if args.dry_run:
            return None, "gpt2", vocab_size or 50257
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install transformers to use --tokenizer gpt2.") from exc
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return tokenizer, "gpt2", tokenizer.vocab_size
    if tokenizer_choice == "hf":
        name = args.tokenizer_name
        if not name:
            raise ValueError("--tokenizer-name is required when --tokenizer hf is used.")
        if args.dry_run:
            return None, name, vocab_size or 50257
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install transformers to use --tokenizer hf.") from exc
        tokenizer = AutoTokenizer.from_pretrained(name)
        return tokenizer, name, tokenizer.vocab_size
    raise ValueError(tokenizer_choice)


def build_model(model_type: str, config: CMFConfig) -> torch.nn.Module:
    if model_type == "continuous_cmf":
        return ContinuousMeaningField(config)
    if model_type == "parallel_cmf":
        return ParallelContinuousMeaningField(config)
    if model_type == "deliberative_cmf":
        return DeliberativeContinuousMeaningField(config)
    if model_type == "fast_parallel_cmf":
        return FastParallelContinuousMeaningField(config)
    raise ValueError(f"Unsupported model_type: {model_type}")


def build_text_stream(args: argparse.Namespace):
    if args.text_file:
        lines = list(iter_local_text(args.text_file))
        if not lines:
            raise ValueError(f"{args.text_file} contained no non-empty lines.")
        return cycle(lines)
    if args.dataset:
        return iter_hf_text(
            args.dataset,
            name=args.dataset_name,
            split=args.split,
            text_column=args.text_column,
            streaming=True,
            limit=args.dataset_limit,
        )
    seed = (
        "continuous meaning field training stream. "
        "language data can be read in small batches without loading the full corpus. "
    )
    return synthetic_text_stream(seed, repeats=None)


def save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: CMFConfig,
    tokenizer_type: str,
    tokenizer_name: str | None,
    step: int,
    tokens: int,
    loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "cmf.training_checkpoint.v1",
            "step": step,
            "tokens": tokens,
            "loss": loss,
            "config": config.__dict__,
            "tokenizer": {"type": tokenizer_type, "name": tokenizer_name},
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def maybe_resume(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    payload = torch.load(path, map_location=device)
    if payload.get("format") != "cmf.training_checkpoint.v1":
        raise RuntimeError(f"{path} is not a CMF training checkpoint.")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    print(f"Resumed {path} at step {payload['step']} ({payload['tokens']} tokens).")
    return int(payload["step"]), int(payload["tokens"])


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    token_cache = None
    token_cache_meta = None
    cache_tokenizer_name = None
    cache_vocab_size = None
    if args.token_cache and args.token_cache_dir:
        raise ValueError("Use either --token-cache or --token-cache-dir, not both.")
    if args.token_cache:
        token_cache, token_cache_meta = load_token_cache(args.token_cache)
        cache_tokenizer = token_cache_meta.get("tokenizer", {}) if isinstance(token_cache_meta, dict) else {}
        cache_tokenizer_name = cache_tokenizer.get("name")
        cache_vocab_size = cache_tokenizer.get("vocab_size")
        if args.tokenizer == "auto" and cache_tokenizer_name:
            args.tokenizer = "hf" if cache_tokenizer_name != "gpt2" else "gpt2"
            args.tokenizer_name = cache_tokenizer_name
    elif args.token_cache_dir:
        token_cache_meta = {} if args.dry_run and not args.token_cache_dir.exists() else load_token_cache_manifest(args.token_cache_dir)
        cache_tokenizer = token_cache_meta.get("tokenizer", {}) if isinstance(token_cache_meta, dict) else {}
        cache_tokenizer_name = cache_tokenizer.get("name")
        cache_vocab_size = cache_tokenizer.get("vocab_size")
        if args.tokenizer == "auto" and cache_tokenizer_name:
            args.tokenizer = "hf" if cache_tokenizer_name != "gpt2" else "gpt2"
            args.tokenizer_name = cache_tokenizer_name

    init_model: torch.nn.Module | None = None
    init_payload = None
    if args.init_package:
        init_model, tokenizer, init_payload = load_model_package(args.init_package, device="cpu")
        token_spec = dict(init_payload.get("tokenizer", {}))
        tokenizer_name = token_spec.get("name")
        tokenizer_type = str(token_spec.get("type") or ("byte" if tokenizer_name is None else "hf_auto"))
        vocab_size = int(token_spec.get("vocab_size") or init_model.config.vocab_size)
        model_type = str(init_payload["model_type"])
        config = init_model.config
        config.vocab_size = vocab_size
        if args.seq_len is not None:
            config.max_seq_len = args.seq_len
        if cache_vocab_size and int(cache_vocab_size) != int(config.vocab_size):
            raise ValueError(
                f"Token cache vocab_size={cache_vocab_size} does not match package vocab_size={config.vocab_size}."
            )
        if cache_tokenizer_name and tokenizer_name and cache_tokenizer_name != tokenizer_name:
            raise ValueError(
                f"Token cache tokenizer={cache_tokenizer_name!r} does not match package tokenizer={tokenizer_name!r}."
            )
    else:
        preset = get_preset(args.preset) if args.preset else None
        model_type = preset.model_type if preset else args.model_type
        preset_config = preset.config if preset else None
        tokenizer, tokenizer_name, vocab_size = build_tokenizer(
            args,
            vocab_size=cache_vocab_size or (preset_config.vocab_size if preset_config else None),
        )
        tokenizer_type = "byte" if tokenizer_name is None else "hf_auto"

        if preset_config is not None:
            config = CMFConfig(**preset_config.__dict__)
            config.vocab_size = vocab_size
            if args.seq_len is not None:
                config.max_seq_len = args.seq_len
        else:
            config = CMFConfig(
                vocab_size=vocab_size,
                d_model=args.d_model,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                max_seq_len=args.seq_len or 1024,
                solver_steps_per_token=args.solver_steps,
                dropout=args.dropout,
                tie_embeddings=args.tie_embeddings,
            )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "preset": args.preset,
                    "init_package": str(args.init_package) if args.init_package else None,
                    "model_type": model_type,
                    "config": config.__dict__,
                    "parameter_estimate": estimate_cmf_parameters(config, model_type=model_type),
                    "device": str(device),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    train_seq_len = args.seq_len or config.max_seq_len
    model = init_model.to(device) if init_model is not None else build_model(model_type, config).to(device)
    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("torch.compile enabled.")
        except Exception as exc:
            print(f"torch.compile unavailable, continuing eager: {exc}")

    adamw_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.fused_adamw and device.type == "cuda":
        adamw_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    except TypeError:
        adamw_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    start_step, tokens_seen = maybe_resume(args.resume, model, optimizer, device) if args.resume else (0, 0)

    if token_cache is not None:
        batches = cached_lm_batches(
            token_cache,
            seq_len=train_seq_len,
            batch_size=args.micro_batch_size,
            num_batches=None,
            random_batches=not args.sequential_cache_batches,
            seed=args.seed,
            pin_memory=device.type == "cuda",
        )
        data_source = {"token_cache": str(args.token_cache), **(token_cache_meta or {})}
    elif args.token_cache_dir:
        batches = cached_lm_batches_from_shards(
            args.token_cache_dir,
            seq_len=train_seq_len,
            batch_size=args.micro_batch_size,
            num_batches=None,
            random_batches=not args.sequential_cache_batches,
            seed=args.seed,
            pin_memory=device.type == "cuda",
            batches_per_shard=args.cache_batches_per_shard,
        )
        data_source = {"token_cache_dir": str(args.token_cache_dir), **(token_cache_meta or {})}
    else:
        texts = build_text_stream(args)
        batches = iter_token_batches_from_texts(
            texts,
            tokenizer,
            seq_len=train_seq_len,
            batch_size=args.micro_batch_size,
            max_batches=None,
        )
        data_source = {"dataset": args.dataset, "dataset_name": args.dataset_name, "text_file": str(args.text_file) if args.text_file else None}
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    model.train()
    start = time.perf_counter()
    last_loss = float("nan")

    for step in range(start_step, args.steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = next(batches)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                out = model(x, labels=y)
                loss = out["loss"] / args.grad_accum
            scaler.scale(loss).backward()
            tokens_seen += int(x.numel())
            last_loss = float(out["loss"].detach().cpu())

        if args.clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_every == 0:
            synchronize_device(device)
            elapsed = time.perf_counter() - start
            print(
                f"step={step + 1} loss={last_loss:.4f} "
                f"tokens={tokens_seen:,} tok/s={tokens_seen / max(elapsed, 1e-6):.0f}",
                flush=True,
            )
        if args.checkpoint and (step + 1) % args.save_every == 0:
            model_for_save = getattr(model, "_orig_mod", model)
            save_training_checkpoint(
                args.checkpoint,
                model=model_for_save,
                optimizer=optimizer,
                config=config,
                tokenizer_type=tokenizer_type,
                tokenizer_name=tokenizer_name,
                step=step + 1,
                tokens=tokens_seen,
                loss=last_loss,
            )
            print(f"saved_checkpoint={args.checkpoint} step={step + 1} tokens={tokens_seen:,}", flush=True)

    if args.package_out:
        model_for_save = getattr(model, "_orig_mod", model)
        save_model_package(
            args.package_out,
            model_for_save,
            model_type=model_type,
            config=config,
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            training={
                "steps": args.steps,
                "tokens": tokens_seen,
                "seq_len": train_seq_len,
                "micro_batch_size": args.micro_batch_size,
                "grad_accum": args.grad_accum,
                "final_loss": last_loss,
                "data_source": data_source,
            },
        )
        print(f"Saved inference package: {args.package_out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scalable CMF training loop for local files or streaming HF datasets.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", help="Hugging Face dataset id, for example a FineWeb-Edu style corpus.")
    parser.add_argument("--dataset-name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--token-cache", type=Path, help="Local token cache produced by scripts/prepare_hf_token_cache.py.")
    parser.add_argument("--token-cache-dir", type=Path, help="Directory of sharded token caches produced by scripts/prepare_hf_token_shards.py.")
    parser.add_argument("--preset", choices=["infinity-0.00037b", "infinity-0.12b", "infinity-reasoning-0.12b", "infinity-0.203b", "infinity-8b"])
    parser.add_argument("--model-type", choices=["continuous_cmf", "parallel_cmf", "deliberative_cmf", "fast_parallel_cmf"], default="parallel_cmf")
    parser.add_argument("--tokenizer", choices=["auto", "byte", "gpt2", "hf"], default="auto")
    parser.add_argument("--tokenizer-name", help="Hugging Face tokenizer name when --tokenizer hf is used.")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--solver-steps", type=int, default=1)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tie-embeddings", action="store_true")
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tf32", action="store_true", help="Allow TF32 matmul/cudnn on CUDA devices.")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile for the model.")
    parser.add_argument("--fused-adamw", action="store_true", help="Use fused AdamW when available on CUDA.")
    parser.add_argument("--sequential-cache-batches", action="store_true", help="Read cached-token batches sequentially instead of randomly.")
    parser.add_argument("--cache-batches-per-shard", type=int, default=1024, help="How many batches to draw before rotating sharded token-cache files.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "records" / "checkpoints" / "large_scale_train.pt")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-package", type=Path, help="Initialize training from a CMF inference package.")
    parser.add_argument("--package-out", type=Path, default=ROOT / "records" / "checkpoints" / "large_scale_cmf.package.pt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
