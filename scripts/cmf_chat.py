from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf import CMFConfig, ContinuousMeaningField
from cmf.checkpointing import load_model_package
from cmf.runtime import resolve_device


def load_grammar_artifacts(args: argparse.Namespace, device: torch.device):
    if args.package:
        return load_model_package(args.package, device=device, expected_model_type="continuous_cmf")[:2]

    if not args.weights.exists():
        raise FileNotFoundError(f"{args.weights} not found. Run scripts/cmf_local_grammar_train.py first.")
    if not args.tokenizer.exists():
        raise FileNotFoundError(f"{args.tokenizer} not found. Run scripts/cmf_local_grammar_train.py first.")

    config = CMFConfig(
        vocab_size=1000,
        d_model=128,
        hidden_dim=256,
        num_layers=8,
        adaptive_steps=True,
    )
    model = ContinuousMeaningField(config).to(device)
    state = torch.load(args.weights, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Grammar checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    with args.tokenizer.open("rb") as handle:
        tokenizer = pickle.load(handle)
    if getattr(tokenizer, "vocab_size", None) != config.vocab_size:
        raise RuntimeError(
            f"Tokenizer vocab_size {getattr(tokenizer, 'vocab_size', None)} "
            f"does not match model vocab_size {config.vocab_size}."
        )
    return model, tokenizer


def start_chat(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, tokenizer = load_grammar_artifacts(args, device)
    model.eval()

    print("CMF grammar chat")
    print(f"Device: {device}")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("CMF > ")
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        tokens = tokenizer.encode(user_input).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tokens)
            next_token = torch.argmax(outputs["logits"][:, -1], dim=-1)
            response = tokenizer.decode(next_token)
        steps = outputs.get("solver_steps")
        step_text = f" in {int(steps.item())} solver steps" if steps is not None else ""
        print(f"Next token{step_text}: {response}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive grammar-checkpoint CMF chat.")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "infinity_weights_grammar.pt")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer.pkl")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    start_chat(args)


if __name__ == "__main__":
    main()
