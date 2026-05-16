from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.checkpointing import load_model_package
from cmf.runtime import resolve_device


@torch.no_grad()
def greedy_generate(model: torch.nn.Module, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    generated = input_ids
    model.eval()
    for _ in range(max_new_tokens):
        output = model(generated)
        next_token = torch.argmax(output["logits"][:, -1], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict CMF package inference.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, tokenizer, payload = load_model_package(args.checkpoint, device=device)
    encoded = tokenizer.encode(args.prompt)
    if not isinstance(encoded, torch.Tensor):
        encoded = torch.tensor(encoded, dtype=torch.long)
    input_ids = encoded.unsqueeze(0).to(device)
    generated = greedy_generate(model, input_ids, args.max_new_tokens)
    print(f"format={payload['format']} model_type={payload['model_type']} device={device}")
    print(tokenizer.decode(generated[0]))


if __name__ == "__main__":
    main()
