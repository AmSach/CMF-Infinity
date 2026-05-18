from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers to prepare an HF token cache.") from exc
    return AutoTokenizer.from_pretrained(name)


def iter_dataset_rows(args: argparse.Namespace):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to prepare an HF token cache.") from exc

    kwargs = {"split": args.split, "streaming": not args.no_streaming}
    if args.dataset_name:
        dataset = load_dataset(args.dataset, args.dataset_name, **kwargs)
    else:
        dataset = load_dataset(args.dataset, **kwargs)

    for idx, row in enumerate(dataset):
        text = row.get(args.text_column)
        if isinstance(text, str) and text.strip():
            yield idx, text


def tensor_dtype(name: str) -> torch.dtype:
    if name == "int32":
        return torch.int32
    if name == "int64":
        return torch.int64
    raise ValueError(f"Unsupported dtype: {name}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def existing_resume_state(output_dir: Path) -> tuple[int, int, int, list[dict[str, Any]]]:
    sidecars = sorted(output_dir.glob("tokens_*.pt.json"))
    if not sidecars:
        return 0, 0, 0, []
    shards: list[dict[str, Any]] = []
    tokens_seen = 0
    rows_seen = 0
    last_source_row = -1
    for sidecar in sidecars:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        shard_path = output_dir / str(payload["path"])
        if not shard_path.exists():
            continue
        shards.append(payload)
        tokens_seen += int(payload["tokens_count"])
        rows_seen += int(payload["rows"])
        last_source_row = max(last_source_row, int(payload.get("source_row_end", -1)))
    return tokens_seen, rows_seen, last_source_row + 1, shards


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local sharded token caches from an HF text dataset.")
    parser.add_argument("--dataset", default="HuggingFaceTB/smollm-corpus")
    parser.add_argument("--dataset-name", default="fineweb-edu-dedup")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--shard-tokens", type=int, default=25_000_000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "records" / "data" / "chinchilla_gpt2_120m")
    parser.add_argument("--dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--append-eos", action="store_true")
    parser.add_argument("--log-every-rows", type=int, default=1000)
    parser.add_argument("--tokenize-batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-ahead", type=int, default=0, help="Limit tokenization to at most N shards ahead of currently training shard to save disk space (0 or negative to disable)")
    args = parser.parse_args()

    if args.target_tokens < 1:
        raise ValueError("--target-tokens must be positive.")
    if args.shard_tokens < 1024:
        raise ValueError("--shard-tokens must be at least 1024.")
    if args.output_dir.exists() and args.overwrite:
        for path in args.output_dir.glob("tokens_*.pt*"):
            path.unlink()
        manifest = args.output_dir / "manifest.json"
        if manifest.exists():
            manifest.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_name)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    dtype = tensor_dtype(args.dtype)
    start = time.perf_counter()

    tokens_seen = 0
    rows_seen = 0
    skip_until_source_row = 0
    shards: list[dict[str, Any]] = []
    if args.resume:
        tokens_seen, rows_seen, skip_until_source_row, shards = existing_resume_state(args.output_dir)
        print(
            f"resume tokens={tokens_seen:,} rows={rows_seen:,} next_source_row={skip_until_source_row:,}",
            flush=True,
        )
    if tokens_seen >= args.target_tokens:
        print("Target already satisfied; writing complete manifest.", flush=True)

    shard_idx = len(shards)
    shard_chunks: list[torch.Tensor] = []
    shard_tokens = 0
    shard_rows = 0
    shard_source_start: int | None = None
    shard_source_end: int | None = None
    pending_texts: list[str] = []
    pending_rows: list[int] = []

    def write_manifest(complete: bool) -> None:
        payload = {
            "format": "cmf.token_cache_dir.v1",
            "complete": complete,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "dataset": args.dataset,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "text_column": args.text_column,
            "target_tokens": args.target_tokens,
            "tokens_count": tokens_seen,
            "rows": rows_seen,
            "shard_tokens": args.shard_tokens,
            "dtype": args.dtype,
            "tokenizer": {
                "type": "hf_auto",
                "name": args.tokenizer_name,
                "vocab_size": getattr(tokenizer, "vocab_size", None),
                "eos_token_id": eos_token_id,
            },
            "shards": shards,
        }
        write_json(args.output_dir / "manifest.json", payload)

    def flush_shard(force: bool = False) -> None:
        nonlocal shard_idx, shard_chunks, shard_tokens, shard_rows, shard_source_start, shard_source_end
        if not shard_chunks:
            return
        if not force and shard_tokens < args.shard_tokens:
            return
        tokens = torch.cat(shard_chunks).contiguous().to(dtype=dtype)
        if tokens.numel() > args.shard_tokens and not force:
            keep = tokens[: args.shard_tokens].contiguous()
            remainder = tokens[args.shard_tokens :].contiguous()
        else:
            keep = tokens
            remainder = torch.empty(0, dtype=dtype)

        filename = f"tokens_{shard_idx:06d}.pt"
        shard_meta = {
            "format": "cmf.token_cache_shard.v1",
            "path": filename,
            "shard_index": shard_idx,
            "tokens_count": int(keep.numel()),
            "rows": shard_rows,
            "source_row_start": shard_source_start,
            "source_row_end": shard_source_end,
            "dtype": args.dtype,
            "tokenizer": {
                "type": "hf_auto",
                "name": args.tokenizer_name,
                "vocab_size": getattr(tokenizer, "vocab_size", None),
                "eos_token_id": eos_token_id,
            },
        }
        torch.save({**shard_meta, "tokens": keep}, args.output_dir / filename)
        write_json(args.output_dir / f"{filename}.json", shard_meta)
        shards.append(shard_meta)
        write_manifest(complete=False)
        elapsed = time.perf_counter() - start
        print(
            f"wrote {filename} shard_tokens={keep.numel():,} total_tokens={tokens_seen:,} "
            f"rows={rows_seen:,} tok/s={tokens_seen / max(elapsed, 1e-9):.0f}",
            flush=True,
        )
        shard_idx += 1
        shard_chunks = [remainder] if remainder.numel() else []
        shard_tokens = int(remainder.numel())
        shard_rows = 0 if remainder.numel() == 0 else shard_rows
        shard_source_start = None if remainder.numel() == 0 else shard_source_start
        shard_source_end = None if remainder.numel() == 0 else shard_source_end

    def add_encoded(encoded: list[int], row_idx: int) -> None:
        nonlocal tokens_seen, rows_seen, shard_tokens, shard_rows, shard_source_start, shard_source_end
        if args.append_eos and eos_token_id is not None:
            encoded.append(int(eos_token_id))
        if not encoded:
            return
        remaining = args.target_tokens - tokens_seen
        if remaining <= 0:
            return
        if len(encoded) > remaining:
            encoded = encoded[:remaining]
        shard_chunks.append(torch.tensor(encoded, dtype=torch.long))
        token_count = len(encoded)
        tokens_seen += token_count
        rows_seen += 1
        shard_tokens += token_count
        shard_rows += 1
        shard_source_start = row_idx if shard_source_start is None else min(shard_source_start, row_idx)
        shard_source_end = row_idx if shard_source_end is None else max(shard_source_end, row_idx)
        flush_shard(force=shard_tokens >= args.shard_tokens)

    def flush_pending() -> None:
        if not pending_texts:
            return
        encoded_batch = tokenizer(
            pending_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        rows = list(pending_rows)
        pending_texts.clear()
        pending_rows.clear()
        for encoded, row_idx in zip(encoded_batch, rows):
            add_encoded(encoded, row_idx)
            if tokens_seen >= args.target_tokens:
                break

    if tokens_seen < args.target_tokens:
        for row_idx, text in iter_dataset_rows(args):
            if row_idx < skip_until_source_row:
                continue
            pending_texts.append(text)
            pending_rows.append(row_idx)
            if len(pending_texts) >= args.tokenize_batch_size:
                # Adaptive Disk Throttle:
                # Check if the tokenizer is too far ahead of the trainer.
                if args.max_ahead > 0:
                    while True:
                        existing_shards = sorted(args.output_dir.glob("tokens_*.pt"))
                        if existing_shards:
                            indices = []
                            for p in existing_shards:
                                parts = p.stem.split("_")
                                if len(parts) >= 2 and parts[-1].isdigit():
                                    indices.append(int(parts[-1]))
                            if indices:
                                min_active = min(indices)
                                if shard_idx - min_active >= args.max_ahead:
                                    print(f"[Adaptive Throttle] Shard {shard_idx} is {shard_idx - min_active} shards ahead of training (min active: {min_active}). Pausing downloader to save disk space...", end="\r", flush=True)
                                    time.sleep(2)
                                    continue
                        break

                flush_pending()
            if rows_seen and rows_seen % args.log_every_rows == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"rows={rows_seen:,} source_row={row_idx:,} tokens={tokens_seen:,} "
                    f"tok/s={tokens_seen / max(elapsed, 1e-9):.0f}",
                    flush=True,
                )
            if tokens_seen >= args.target_tokens:
                break

    flush_pending()
    flush_shard(force=True)
    write_manifest(complete=tokens_seen >= args.target_tokens)
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "tokens": tokens_seen,
                "target_tokens": args.target_tokens,
                "rows": rows_seen,
                "shards": len(shards),
                "elapsed_sec": elapsed,
                "tokens_per_sec": tokens_seen / max(elapsed, 1e-9),
                "complete": tokens_seen >= args.target_tokens,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
