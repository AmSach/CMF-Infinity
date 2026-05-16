from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
        if args.max_rows is not None and idx >= args.max_rows:
            break
        text = row.get(args.text_column)
        if isinstance(text, str) and text.strip():
            yield idx, text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local token cache from an HF text dataset.")
    parser.add_argument("--dataset", default="HuggingFaceTB/smollm-corpus")
    parser.add_argument("--dataset-name", default="fineweb-edu-dedup")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "records" / "data" / "smollm_fineweb_edu_gpt2_2m.pt")
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--append-eos", action="store_true")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--tokenize-batch-size", type=int, default=64)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_name)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    chunks: list[torch.Tensor] = []
    rows = 0
    tokens_seen = 0
    start = time.perf_counter()

    pending_texts: list[str] = []
    last_row_idx = 0

    def flush_pending() -> None:
        nonlocal rows, tokens_seen
        if not pending_texts:
            return
        encoded_batch = tokenizer(
            pending_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        pending_texts.clear()
        for encoded in encoded_batch:
            if args.append_eos and eos_token_id is not None:
                encoded.append(int(eos_token_id))
            if not encoded:
                continue
            if tokens_seen + len(encoded) > args.max_tokens:
                encoded = encoded[: max(0, args.max_tokens - tokens_seen)]
            if encoded:
                chunks.append(torch.tensor(encoded, dtype=torch.long))
                tokens_seen += len(encoded)
                rows += 1
            if tokens_seen >= args.max_tokens:
                break

    for row_idx, text in iter_dataset_rows(args):
        last_row_idx = row_idx
        pending_texts.append(text)
        if len(pending_texts) >= args.tokenize_batch_size:
            flush_pending()
            if rows % args.log_every == 0 or tokens_seen >= args.max_tokens:
                elapsed = time.perf_counter() - start
                print(
                    f"rows={rows:,} source_row={row_idx:,} tokens={tokens_seen:,} "
                    f"tok/s={tokens_seen / max(elapsed, 1e-9):.0f}",
                    flush=True,
                )
        if tokens_seen >= args.max_tokens:
            break

    if tokens_seen < args.max_tokens:
        flush_pending()
        elapsed = time.perf_counter() - start
        print(
            f"rows={rows:,} source_row={last_row_idx:,} tokens={tokens_seen:,} "
            f"tok/s={tokens_seen / max(elapsed, 1e-9):.0f}",
            flush=True,
        )

    if not chunks:
        raise RuntimeError("No tokens were collected from the dataset.")

    tokens = torch.cat(chunks).contiguous()
    payload = {
        "format": "cmf.token_cache.v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "text_column": args.text_column,
        "tokenizer": {
            "type": "hf_auto",
            "name": args.tokenizer_name,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "eos_token_id": eos_token_id,
        },
        "rows": rows,
        "tokens_count": int(tokens.numel()),
        "tokens": tokens,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    sidecar.write_text(
        json.dumps({key: value for key, value in payload.items() if key != "tokens"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sidecar": str(sidecar),
                "rows": rows,
                "tokens": int(tokens.numel()),
                "elapsed_sec": elapsed,
                "tokens_per_sec": int(tokens.numel()) / max(elapsed, 1e-9),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
