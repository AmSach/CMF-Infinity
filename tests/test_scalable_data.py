import pytest

torch = pytest.importorskip("torch")

from cmf.data import ByteTokenizer
from cmf.scalable_data import (
    cached_lm_batches,
    cached_lm_batches_from_shards,
    iter_token_batches_from_texts,
    load_token_cache,
    load_token_cache_manifest,
    synthetic_text_stream,
)


def test_streaming_batches_from_text_iterable():
    tokenizer = ByteTokenizer()
    texts = synthetic_text_stream("abcdefg " * 8, repeats=4)
    batches = iter_token_batches_from_texts(
        texts,
        tokenizer,
        seq_len=8,
        batch_size=2,
        max_batches=2,
    )

    first = next(batches)
    assert first[0].shape == (2, 8)
    assert first[1].shape == (2, 8)
    assert first[0].dtype == torch.long


def test_cached_lm_batches_shapes_and_cache_load(tmp_path):
    cache_path = tmp_path / "tokens.pt"
    torch.save({"format": "cmf.token_cache.v1", "tokens": torch.arange(64)}, cache_path)
    tokens, meta = load_token_cache(cache_path)

    batch = next(cached_lm_batches(tokens, seq_len=8, batch_size=3, random_batches=False))

    assert meta["format"] == "cmf.token_cache.v1"
    assert batch[0].shape == (3, 8)
    assert batch[1].shape == (3, 8)
    assert torch.equal(batch[1], batch[0] + 1)


def test_sharded_cached_lm_batches(tmp_path):
    shard_dir = tmp_path / "cache"
    shard_dir.mkdir()
    torch.save({"format": "cmf.token_cache_shard.v1", "tokens": torch.arange(40)}, shard_dir / "tokens_000000.pt")
    torch.save({"format": "cmf.token_cache_shard.v1", "tokens": torch.arange(40, 80)}, shard_dir / "tokens_000001.pt")
    (shard_dir / "manifest.json").write_text(
        """
        {
          "format": "cmf.token_cache_dir.v1",
          "tokenizer": {"type": "hf_auto", "name": "gpt2", "vocab_size": 50257},
          "shards": [
            {"path": "tokens_000000.pt"},
            {"path": "tokens_000001.pt"}
          ]
        }
        """,
        encoding="utf-8",
    )

    meta = load_token_cache_manifest(shard_dir)
    batches = cached_lm_batches_from_shards(
        shard_dir,
        seq_len=8,
        batch_size=2,
        random_batches=False,
        batches_per_shard=1,
    )
    batch = next(batches)
    second = next(batches)

    assert meta["format"] == "cmf.token_cache_dir.v1"
    assert batch[0].shape == (2, 8)
    assert torch.equal(batch[1], batch[0] + 1)
    assert second[0][0, 0].item() == 16
