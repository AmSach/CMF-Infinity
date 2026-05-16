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
from cmf.generation import ChatState, decode_tokens, encode_to_tensor, generate_ids, trim_assistant_response
from cmf.infinity_runtime import DeliberationConfig, EvidenceMemory, deliberative_generate_text
from cmf.runtime import resolve_device


def load_memory(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_memory(path: Path | None, row: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="CMF Infinity chatbot for trained model packages.")
    parser.add_argument("package", type=Path, help="CMF model package emitted by training.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--system", default=ChatState().system_prompt)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--memory", type=Path, default=ROOT / "records" / "chat_memory.jsonl")
    parser.add_argument("--no-memory", action="store_true")
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
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    state = ChatState(system_prompt=args.system)
    memory_path = None if args.no_memory else args.memory
    evidence_memory = EvidenceMemory.from_path(args.knowledge_file) if args.knowledge_file else None
    use_deliberation = (
        args.open_ended_deliberation
        or args.deliberation_steps > 1
        or args.deliberation_candidates > 1
        or evidence_memory is not None
    )

    for row in load_memory(memory_path)[-8:]:
        if row.get("user"):
            state.add_user(str(row["user"]))
        if row.get("assistant"):
            state.add_assistant(str(row["assistant"]))

    print(f"CMF Infinity chat loaded: {payload.get('model_type')} on {device}")
    print(f"Max context tokens: {max_context}")
    print("Type 'exit' to quit.")

    while True:
        user_text = input("You > ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue
        state.add_user(user_text)
        prompt = state.render()
        if use_deliberation:
            wall_time = args.max_deliberation_seconds
            if args.open_ended_deliberation and wall_time is None:
                wall_time = 30.0
            result = deliberative_generate_text(
                model,
                tokenizer,
                prompt,
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
                memory=evidence_memory,
                retrieval_query=user_text,
                eos_token_id=eos_token_id,
                max_context_tokens=max_context,
            )
            response = trim_assistant_response(result.text)
            if args.print_trace:
                print(
                    f"trace: stopped={result.stopped_reason} "
                    f"steps={len(result.steps)} score={result.score:.4f}"
                )
        else:
            input_ids = encode_to_tensor(tokenizer, prompt).unsqueeze(0).to(device)
            with torch.no_grad():
                output_ids = generate_ids(
                    model,
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    eos_token_id=eos_token_id,
                    max_context_tokens=max_context,
                )
            new_tokens = output_ids[0, input_ids.size(1):]
            response = trim_assistant_response(decode_tokens(tokenizer, new_tokens))
            if not response:
                response = trim_assistant_response(decode_tokens(tokenizer, output_ids[0]))
        print(f"CMF > {response}\n")
        state.add_assistant(response)
        append_memory(memory_path, {"user": user_text, "assistant": response})


if __name__ == "__main__":
    main()
