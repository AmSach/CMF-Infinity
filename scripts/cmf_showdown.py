from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf import CMFConfig, ContinuousMeaningField
from cmf.baselines import TinyGPTLM
from cmf.runtime import resolve_device, synchronize_device


def load_gpt2_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers to use the GPT-2 tokenizer.") from exc
    return AutoTokenizer.from_pretrained("gpt2")


def strict_load(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    missing, unexpected = model.load_state_dict(torch.load(path, map_location=device), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"{path} mismatch: missing={missing}, unexpected={unexpected}")


@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    generated = input_ids
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(generated)["logits"][:, -1] / max(temperature, 1e-6)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return generated


def showdown(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    tokenizer = load_gpt2_tokenizer()
    vocab_size = tokenizer.vocab_size

    config = CMFConfig(vocab_size=vocab_size, d_model=768, hidden_dim=3072, num_layers=6)
    cmf = ContinuousMeaningField(config).to(device)
    gpt = TinyGPTLM(
        vocab_size=vocab_size,
        d_model=768,
        nhead=12,
        num_layers=12,
        hidden_dim=3072,
        max_seq_len=args.max_seq_len,
    ).to(device)

    strict_load(cmf, args.cmf_weights, device)
    strict_load(gpt, args.gpt_weights, device)
    cmf.eval()
    gpt.eval()

    print("--- Side-by-side CMF vs GPT checkpoint showdown ---")
    print(f"Device: {device}")

    for prompt in args.prompt:
        print(f"\nPrompt: {prompt}")
        ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

        synchronize_device(device)
        start = time.perf_counter()
        c_ids = greedy_generate(cmf, ids, max_new_tokens=args.max_new_tokens)
        synchronize_device(device)
        print(f"CMF: {tokenizer.decode(c_ids[0])} | {time.perf_counter() - start:.3f}s")

        synchronize_device(device)
        start = time.perf_counter()
        g_ids = greedy_generate(gpt, ids, max_new_tokens=args.max_new_tokens)
        synchronize_device(device)
        print(f"GPT: {tokenizer.decode(g_ids[0])} | {time.perf_counter() - start:.3f}s")

    test_ids = torch.randint(0, vocab_size, (1, args.max_seq_len), device=device)
    print(f"\nLatency benchmark ({args.max_seq_len} tokens)")
    for name, model in [("CMF", cmf), ("GPT", gpt)]:
        synchronize_device(device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(test_ids)
        synchronize_device(device)
        print(f"{name}: {(time.perf_counter() - start) * 1000.0:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict checkpoint showdown. Fails if either model is missing.")
    parser.add_argument("--cmf-weights", type=Path, default=ROOT / "cmf_120m_weights.pt")
    parser.add_argument("--gpt-weights", type=Path, default=ROOT / "gpt_120m_weights.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument(
        "--prompt",
        action="append",
        default=[
            "The capital of France is",
            "If A is B and B is C, then A is",
            "Wikipedia is a free online",
        ],
    )
    args = parser.parse_args()
    showdown(args)


if __name__ == "__main__":
    main()
