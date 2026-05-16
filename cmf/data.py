from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Union, Optional
from pathlib import Path

import torch


TOY_TEXT = (
    "continuous meaning field flows through language. "
    "latent states move as smooth semantic trajectories. "
    "dilated convolutions build the prompt landscape. "
)

SMALL_LM_TEXT = (
    TOY_TEXT
    + "the solver traces a path and the decoder maps points back to tokens. "
    + "phase zero proves learning, phase one compares tiny baselines. "
    + "phase two measures context length, phase three times integration. "
    + "phase four records the geometry of the trajectory. "
)


@dataclass(frozen=True)
class ByteTokenizer:
    vocab_size: int = 256

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        values = [int(v) % self.vocab_size for v in token_ids.detach().cpu().flatten()]
        return bytes(values).decode("utf-8", errors="replace")


def repeated_corpus(text: str, min_bytes: int, tokenizer: Optional[Union[ByteTokenizer, 'SimpleBPETokenizer']] = None) -> torch.Tensor:
    if tokenizer is None:
        tokenizer = ByteTokenizer()
    encoded = tokenizer.encode(text)
    repeats = max(1, min_bytes // max(1, encoded.numel() * (1 if isinstance(tokenizer, ByteTokenizer) else 2)) + 1)
    return encoded.repeat(repeats)


def load_text_corpus(path: Union[str, Path]) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def cyclic_lm_batches(
    data: torch.Tensor,
    seq_len: int,
    batch_size: int,
    num_batches: int | None = None,
    stride: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if data.ndim != 1:
        raise ValueError(f"data must be a 1D tensor, got {tuple(data.shape)}")
    if data.numel() <= seq_len + 1:
        raise ValueError("data must contain more tokens than seq_len + 1")

    stride = stride or seq_len
    cursor = 0
    produced = 0
    max_start = data.numel() - seq_len - 1

    while num_batches is None or produced < num_batches:
        inputs = []
        labels = []
        for _ in range(batch_size):
            start = cursor % max_start
            chunk = data[start : start + seq_len + 1]
            inputs.append(chunk[:-1])
            labels.append(chunk[1:])
            cursor += stride
        produced += 1
        yield torch.stack(inputs), torch.stack(labels)


def fixed_eval_batches(
    text: str,
    seq_len: int,
    batch_size: int,
    num_batches: int,
    min_bytes: int = 8192,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    data = repeated_corpus(text, min_bytes=min_bytes)
    return list(cyclic_lm_batches(data, seq_len, batch_size, num_batches))

