from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf import CMFConfig
from cmf.checkpointing import load_legacy_state_dict, load_model_package
from cmf.runtime import resolve_device


def load_gpt2_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers to use the GPT-2 tokenizer.") from exc
    return AutoTokenizer.from_pretrained("gpt2")


def build_model(args: argparse.Namespace, device: torch.device):
    if args.package:
        model, tokenizer, payload = load_model_package(
            args.package,
            device=device,
            expected_model_type="continuous_cmf",
        )
        return model, tokenizer, payload

    tokenizer = load_gpt2_tokenizer()
    config = CMFConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        hidden_dim=3072,
        num_layers=6,
    )
    model = load_legacy_state_dict(
        args.weights,
        model_type="continuous_cmf",
        config=config,
        device=device,
        strict=True,
    )
    payload = {
        "format": "legacy_state_dict",
        "model_type": "continuous_cmf",
        "config": config.__dict__,
        "tokenizer": {"type": "hf_auto", "name": "gpt2", "vocab_size": tokenizer.vocab_size},
    }
    return model, tokenizer, payload


def chat(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, tokenizer, payload = build_model(args, device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print("--- CMF GPT-2-tokenizer interactive engine ---")
    print(f"Device: {device}")
    print(f"Checkpoint format: {payload.get('format')}")
    print(f"Parameters: {params:,}")
    print("Type 'exit' to quit.")

    while True:
        prompt = input("Prompt > ")
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt.strip():
            continue

        ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        print(f"CMF > {tokenizer.decode(output[0])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive CMF chat with strict checkpoint loading.")
    parser.add_argument("--package", type=Path, help="CMF model package with config/tokenizer metadata.")
    parser.add_argument("--weights", type=Path, default=ROOT / "cmf_120m_weights.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    chat(args)


if __name__ == "__main__":
    main()
