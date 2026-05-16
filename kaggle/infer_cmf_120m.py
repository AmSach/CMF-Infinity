from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.checkpointing import load_model_package
from cmf.generation import decode_tokens, encode_to_tensor, generate_ids, trim_assistant_response
from cmf.runtime import resolve_device


def default_package() -> Path:
    candidates = [
        Path("/kaggle/working/cmf_120m/checkpoints/cmf_120m_kaggle.package.pt"),
        ROOT / "pretrained" / "cmf_120m_gate.package.pt",
        ROOT / "records" / "checkpoints" / "cmf_120m_gate.package.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt a CMF 120M package.")
    parser.add_argument("--package", type=Path, default=default_package())
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, tokenizer, payload = load_model_package(args.package, device=device)
    model.eval()

    input_ids = encode_to_tensor(tokenizer, args.prompt).unsqueeze(0).to(device)
    with torch.no_grad():
        output_ids = generate_ids(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            max_context_tokens=payload.get("config", {}).get("max_seq_len"),
        )
    print(trim_assistant_response(decode_tokens(tokenizer, output_ids[0])))


if __name__ == "__main__":
    main()
