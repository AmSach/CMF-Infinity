from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.checkpointing import load_model_package
from cmf.generation import decode_tokens, encode_to_tensor, generate_ids
from cmf.infinity_runtime import DeliberationConfig, EvidenceMemory, deliberative_generate_text
from cmf.runtime import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot CMF Infinity text generation.")
    parser.add_argument("package", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--knowledge-file", type=Path, help="Optional text file used as retrieval-grounded evidence.")
    parser.add_argument("--deliberation-steps", type=int, default=1)
    parser.add_argument("--deliberation-candidates", type=int, default=1)
    parser.add_argument("--open-ended-deliberation", action="store_true")
    parser.add_argument("--max-deliberation-seconds", type=float)
    parser.add_argument("--evidence-top-k", type=int, default=4)
    parser.add_argument("--print-trace", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, tokenizer, payload = load_model_package(args.package, device=device)
    max_context = int(payload.get("config", {}).get("max_seq_len", 2048))
    memory = EvidenceMemory.from_path(args.knowledge_file) if args.knowledge_file else None
    use_deliberation = (
        args.open_ended_deliberation
        or args.deliberation_steps > 1
        or args.deliberation_candidates > 1
        or memory is not None
    )
    if use_deliberation:
        wall_time = args.max_deliberation_seconds
        if args.open_ended_deliberation and wall_time is None:
            wall_time = 30.0
        result = deliberative_generate_text(
            model,
            tokenizer,
            args.prompt,
            device=device,
            config=DeliberationConfig(
                max_thinking_steps=None if args.open_ended_deliberation else args.deliberation_steps,
                candidates_per_step=args.deliberation_candidates,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_wall_time_s=wall_time,
                evidence_top_k=args.evidence_top_k,
            ),
            memory=memory,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            max_context_tokens=max_context,
        )
        print(result.text)
        if args.print_trace:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        input_ids = encode_to_tensor(tokenizer, args.prompt).unsqueeze(0).to(device)
        output_ids = generate_ids(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            max_context_tokens=max_context,
        )
        print(decode_tokens(tokenizer, output_ids[0]))


if __name__ == "__main__":
    main()
