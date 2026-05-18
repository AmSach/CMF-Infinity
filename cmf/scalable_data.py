from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
from pathlib import Path
from typing import Any

import torch


def iter_local_text(path: str | Path) -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield line


def iter_hf_text(
    dataset: str,
    *,
    name: str | None = None,
    split: str = "train",
    text_column: str = "text",
    streaming: bool = True,
    limit: int | None = None,
) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the optional 'datasets' dependency to stream HF corpora.") from exc

    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if name:
        data = load_dataset(dataset, name, **kwargs)
    else:
        data = load_dataset(dataset, **kwargs)

    for idx, row in enumerate(data):
        if limit is not None and idx >= limit:
            break
        text = row.get(text_column)
        if isinstance(text, str) and text.strip():
            yield text


def encode_text(tokenizer: Any, text: str) -> torch.Tensor:
    encoded = tokenizer.encode(text)
    if isinstance(encoded, torch.Tensor):
        return encoded.to(dtype=torch.long).flatten()
    if isinstance(encoded, list):
        return torch.tensor(encoded, dtype=torch.long)
    if hasattr(encoded, "ids"):
        return torch.tensor(encoded.ids, dtype=torch.long)
    raise TypeError(f"Unsupported tokenizer output: {type(encoded).__name__}")


def iter_token_batches_from_texts(
    texts: Iterable[str],
    tokenizer: Any,
    *,
    seq_len: int,
    batch_size: int,
    max_batches: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    buffer = torch.empty(0, dtype=torch.long)
    pending: list[torch.Tensor] = []
    pending_tokens = 0
    produced = 0
    samples_needed = batch_size * (seq_len + 1)

    for text in texts:
        tokens = encode_text(tokenizer, text)
        if tokens.numel() == 0:
            continue
        pending.append(tokens)
        pending_tokens += int(tokens.numel())
        if pending_tokens:
            buffer = torch.cat([buffer, *pending]) if buffer.numel() else torch.cat(pending)
            pending.clear()
            pending_tokens = 0
        while buffer.numel() >= samples_needed:
            window = buffer[:samples_needed].view(batch_size, seq_len + 1)
            yield window[:, :-1].contiguous(), window[:, 1:].contiguous()
            produced += 1
            if max_batches is not None and produced >= max_batches:
                return
            buffer = buffer[samples_needed:]


def load_token_cache(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)

    if torch.is_tensor(payload):
        return payload.to(dtype=torch.long).flatten(), {"format": "raw_tensor"}
    if not isinstance(payload, dict) or "tokens" not in payload:
        raise ValueError(f"{path} is not a CMF token cache.")
    tokens = payload["tokens"]
    if not torch.is_tensor(tokens):
        raise ValueError(f"{path} token cache field 'tokens' is not a tensor.")
    metadata = {key: value for key, value in payload.items() if key != "tokens"}
    return tokens.to(dtype=torch.long).flatten(), metadata


def list_token_cache_shards(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    manifest = root / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        shards = payload.get("shards", [])
        paths = [root / str(item["path"]) for item in shards if "path" in item]
    else:
        paths = sorted(root.glob("*.pt"))
    import time
    paths = [path for path in paths if path.exists() and not path.name.endswith(".package.pt")]
    if not paths:
        print(f"--- [JIT Loader] Waiting for first token cache shard to appear in {root}... ---")
        while not paths:
            time.sleep(1.0)
            if manifest.exists():
                try:
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    shards = payload.get("shards", [])
                    paths = [root / str(item["path"]) for item in shards if "path" in item]
                except Exception:
                    pass
            else:
                paths = sorted(root.glob("*.pt"))
            paths = [path for path in paths if path.exists() and not path.name.endswith(".package.pt")]
    return paths


def load_token_cache_manifest(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    if root.is_file():
        _tokens, metadata = load_token_cache(root)
        return metadata
    manifest = root / "manifest.json"
    if not manifest.exists():
        return {"format": "cmf.token_cache_dir.v1", "shards": [str(path) for path in list_token_cache_shards(root)]}
    return json.loads(manifest.read_text(encoding="utf-8"))


def cached_lm_batches(
    tokens: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    num_batches: int | None = None,
    stride: int | None = None,
    random_batches: bool = True,
    seed: int = 0,
    pin_memory: bool = False,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if tokens.ndim != 1:
        raise ValueError(f"tokens must be 1D, got {tuple(tokens.shape)}")
    if tokens.numel() <= seq_len + 1:
        raise ValueError("token cache must contain more tokens than seq_len + 1")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    tokens = tokens.to(dtype=torch.long).contiguous()
    max_start = tokens.numel() - seq_len - 1
    produced = 0
    cursor = 0
    stride = stride or seq_len
    generator = torch.Generator()
    generator.manual_seed(seed)

    while num_batches is None or produced < num_batches:
        if random_batches:
            starts = torch.randint(0, max_start, (batch_size,), generator=generator)
        else:
            starts = (torch.arange(batch_size) * stride + cursor) % max_start
            cursor = int((cursor + batch_size * stride) % max_start)

        windows = torch.stack([tokens[int(start) : int(start) + seq_len + 1] for start in starts])
        x = windows[:, :-1].contiguous()
        y = windows[:, 1:].contiguous()
        if pin_memory:
            x = x.pin_memory()
            y = y.pin_memory()
        produced += 1
        yield x, y


def cached_lm_batches_from_shards(
    path: str | Path,
    *,
    seq_len: int,
    batch_size: int,
    num_batches: int | None = None,
    random_batches: bool = True,
    seed: int = 0,
    pin_memory: bool = False,
    batches_per_shard: int = 1024,
    delete_consumed: bool = False,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    import queue
    import threading

    shards = list_token_cache_shards(path)
    if batches_per_shard < 1:
        raise ValueError("batches_per_shard must be positive")

    produced = 0
    epoch = 0
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Worker thread to load shards asynchronously in the background
    def shard_loader_worker(shard_paths, q):
        for p in shard_paths:
            try:
                tokens, metadata = load_token_cache(p)
                q.put((tokens, metadata, p), block=True)
            except Exception as e:
                print(f"Error asynchronously loading shard {p}: {e}")
                q.put(None, block=True)
        q.put(None, block=True) # Sentinel to signal end of stream

    while num_batches is None or produced < num_batches:
        shards = list_token_cache_shards(path)
        # If delete_consumed is active, do not shuffle shards to maintain lockstep order across ranks
        if random_batches and len(shards) > 1 and not delete_consumed:
            order = torch.randperm(len(shards), generator=generator).tolist()
            shard_order = [shards[idx] for idx in order]
        else:
            shard_order = shards

        # Spawn the background pre-loader thread to buffer up to 2 shards in CPU RAM
        preload_queue = queue.Queue(maxsize=2)
        loader_thread = threading.Thread(
            target=shard_loader_worker,
            args=(shard_order, preload_queue),
            daemon=True
        )
        loader_thread.start()

        for shard_idx in range(len(shard_order)):
            remaining = None if num_batches is None else num_batches - produced
            if remaining is not None and remaining <= 0:
                return

            # Retrieve pre-loaded token tensor from background queue instantly (Zero I/O block!)
            payload = preload_queue.get()
            if payload is None:
                break
            tokens, _metadata, shard_path = payload

            if random_batches:
                shard_batches = batches_per_shard if remaining is None else min(batches_per_shard, remaining)
                iterator = cached_lm_batches(
                    tokens,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    num_batches=shard_batches,
                    random_batches=True,
                    seed=seed + epoch * max(1, len(shards)) + shard_idx,
                    pin_memory=pin_memory,
                )
                for batch in iterator:
                    produced += 1
                    yield batch
                
                # Delete consumed shard files to preserve strict disk limits
                if delete_consumed:
                    try:
                        p = Path(shard_path)
                        p.unlink(missing_ok=True)
                        p.with_suffix(".pt.json").unlink(missing_ok=True)
                        p.with_name(p.name + ".json").unlink(missing_ok=True)
                        print(f"\n--- [Disk Cleanup] Successfully deleted consumed shard {p.name} ---", flush=True)
                    except Exception as e:
                        print(f"Warning: Failed to delete consumed shard {shard_path}: {e}")
                continue

            max_start = tokens.numel() - seq_len - 1
            cursor = 0
            while cursor < max_start:
                if num_batches is not None and produced >= num_batches:
                    return
                starts = (torch.arange(batch_size) * seq_len + cursor) % max_start
                windows = torch.stack([tokens[int(start) : int(start) + seq_len + 1] for start in starts])
                x = windows[:, :-1].contiguous()
                y = windows[:, 1:].contiguous()
                if pin_memory:
                    x = x.pin_memory()
                    y = y.pin_memory()
                produced += 1
                cursor += batch_size * seq_len
                yield x, y
            
            # Delete consumed shard files to preserve strict disk limits (sequential path)
            if delete_consumed:
                try:
                    p = Path(shard_path)
                    p.unlink(missing_ok=True)
                    p.with_suffix(".pt.json").unlink(missing_ok=True)
                    p.with_name(p.name + ".json").unlink(missing_ok=True)
                    print(f"\n--- [Disk Cleanup] Successfully deleted consumed shard {p.name} ---", flush=True)
                except Exception as e:
                    print(f"Warning: Failed to delete consumed shard {shard_path}: {e}")
        epoch += 1


def synthetic_text_stream(seed_text: str, *, repeats: int | None = None) -> Iterator[str]:
    idx = 0
    while repeats is None or idx < repeats:
        yield seed_text
        idx += 1
