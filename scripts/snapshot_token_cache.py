from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def completed_shards(source_dir: Path) -> list[dict[str, Any]]:
    sidecars = sorted(source_dir.glob("tokens_*.pt.json"))
    shards = []
    for sidecar in sidecars:
        meta = read_json(sidecar)
        shard_path = source_dir / str(meta.get("path", sidecar.name.removesuffix(".json")))
        if shard_path.exists():
            shards.append({**meta, "_source_path": str(shard_path)})
    shards.sort(key=lambda item: int(item.get("shard_index", 0)))
    return shards


def clear_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["tokens_*.pt", "tokens_*.pt.json", "manifest.json"]:
        for path in output_dir.glob(pattern):
            path.unlink()


def existing_snapshot_ok(output_dir: Path, target_tokens: int) -> bool:
    manifest = output_dir / "manifest.json"
    if not manifest.exists():
        return False
    payload = read_json(manifest)
    return (
        payload.get("format") == "cmf.token_cache_dir.v1"
        and bool(payload.get("complete"))
        and int(payload.get("tokens_count", 0)) >= target_tokens
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot completed token-cache shards into a fixed-size offline cache.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.target_tokens < 1:
        raise ValueError("--target-tokens must be positive.")
    if not args.source_dir.exists():
        raise FileNotFoundError(args.source_dir)
    if existing_snapshot_ok(args.output_dir, args.target_tokens) and not args.overwrite:
        print(f"Existing snapshot is ready: {args.output_dir}")
        return
    if args.output_dir.exists() and not args.overwrite and any(args.output_dir.iterdir()):
        raise RuntimeError(f"{args.output_dir} is not empty. Use --overwrite or choose another output dir.")

    shards = completed_shards(args.source_dir)
    available = sum(int(shard.get("tokens_count", 0)) for shard in shards)
    if available < args.target_tokens:
        raise RuntimeError(
            f"Only {available:,} completed tokens are available in {args.source_dir}; "
            f"need {args.target_tokens:,}."
        )

    clear_output(args.output_dir)
    source_manifest = read_json(args.source_dir / "manifest.json") if (args.source_dir / "manifest.json").exists() else {}
    copied: list[dict[str, Any]] = []
    tokens_written = 0
    rows_written = 0

    for out_idx, shard in enumerate(shards):
        if tokens_written >= args.target_tokens:
            break
        source_path = Path(str(shard["_source_path"]))
        source_count = int(shard["tokens_count"])
        remaining = args.target_tokens - tokens_written
        out_name = f"tokens_{out_idx:06d}.pt"
        out_path = args.output_dir / out_name

        out_meta = {key: value for key, value in shard.items() if not key.startswith("_")}
        out_meta["path"] = out_name
        out_meta["shard_index"] = out_idx
        out_meta["snapshot_source"] = str(source_path)

        if source_count <= remaining:
            shutil.copy2(source_path, out_path)
            out_meta["tokens_count"] = source_count
        else:
            payload = torch.load(source_path, map_location="cpu")
            tokens = payload["tokens"][:remaining].contiguous()
            out_meta["tokens_count"] = int(tokens.numel())
            out_meta["rows"] = None
            torch.save({**out_meta, "tokens": tokens}, out_path)

        write_json(args.output_dir / f"{out_name}.json", out_meta)
        copied.append(out_meta)
        tokens_written += int(out_meta["tokens_count"])
        rows = out_meta.get("rows")
        if isinstance(rows, int):
            rows_written += rows
        print(f"snapshot {out_name}: total_tokens={tokens_written:,}", flush=True)

    tokenizer = source_manifest.get("tokenizer") or (copied[0].get("tokenizer") if copied else None)
    manifest = {
        "format": "cmf.token_cache_dir.v1",
        "complete": tokens_written >= args.target_tokens,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "dataset": source_manifest.get("dataset"),
        "dataset_name": source_manifest.get("dataset_name"),
        "split": source_manifest.get("split"),
        "text_column": source_manifest.get("text_column"),
        "target_tokens": args.target_tokens,
        "tokens_count": tokens_written,
        "rows": rows_written if rows_written else None,
        "dtype": source_manifest.get("dtype"),
        "tokenizer": tokenizer,
        "snapshot_source_dir": str(args.source_dir),
        "shards": copied,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "source_dir": str(args.source_dir),
                "tokens": tokens_written,
                "target_tokens": args.target_tokens,
                "shards": len(copied),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
