from __future__ import annotations

import itertools
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf import CMFConfig, ContinuousMeaningField, TrainingConfig
from cmf.data import ByteTokenizer
from cmf.runtime import resolve_device
from cmf.training import train_with_gradient_accumulation


def make_batches(text: str, seq_len: int, batch_size: int):
    tokenizer = ByteTokenizer()
    data = tokenizer.encode(text)
    max_start = len(data) - seq_len - 1
    cursor = 0
    while True:
        xs = []
        ys = []
        for _ in range(batch_size):
            start = cursor % max_start
            chunk = data[start : start + seq_len + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            cursor += seq_len
        yield torch.stack(xs), torch.stack(ys)


def main() -> None:
    device = resolve_device("auto")
    seed_text = (
        "continuous meaning field flows through language. "
        "the model learns a smooth direction through latent space. "
    )
    text = seed_text * 256

    model_config = CMFConfig(vocab_size=256, d_model=96, hidden_dim=192, num_layers=5)
    train_config = TrainingConfig(max_steps=25, log_every=5)
    model = ContinuousMeaningField(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    batches = make_batches(
        text,
        seq_len=64,
        batch_size=train_config.micro_batch_size,
    )
    state = train_with_gradient_accumulation(
        model,
        itertools.islice(batches, train_config.max_steps * train_config.gradient_accumulation_steps),
        optimizer,
        train_config,
        device,
    )

    tokenizer = ByteTokenizer()
    prompt = tokenizer.encode("continuous meaning ").unsqueeze(0).to(device)
    generated = model.generate(prompt, max_new_tokens=80, temperature=0.8, top_k=40)
    decoded = tokenizer.decode(generated[0])

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "toy_generation.txt").write_text(decoded, encoding="utf-8")
    print(f"optimizer_steps={state.step} tokens={state.tokens}")
    print(decoded)


if __name__ == "__main__":
    main()
