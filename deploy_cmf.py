"""
CMF-Infinity Model Deployment Template
=======================================
Use this template to load any trained CMF checkpoint and run it as a chatbot,
compression engine, or autonomous agent after training completes on Kaggle.

Usage:
    python deploy_cmf.py --package /path/to/cmf_agi_120m_final.package.pt
"""

import argparse
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.checkpointing import load_model_package
from cmf.generation import (
    ChatState,
    encode_to_tensor,
    decode_tokens,
    generate_ids,
    trim_assistant_response,
)


def load_model(package_path: str, device: str = "auto"):
    """Load a trained CMF package. Works for any scale: 120M to 500B."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Loading model from: {package_path}")
    model, tokenizer, payload = load_model_package(package_path, device=dev)
    model.eval()
    model_type = payload.get("model_type", "unknown")
    config = payload.get("config", {})
    step = payload.get("training", {}).get("step", "?")
    tokens = payload.get("training", {}).get("tokens", "?")
    print(f"  Model type : {model_type}")
    print(f"  d_model    : {config.get('d_model', '?')}")
    print(f"  num_layers : {config.get('num_layers', '?')}")
    print(f"  Trained for: {step:,} steps / {tokens:,} tokens" if isinstance(step, int) else f"  Trained for: {step} steps")
    print(f"  Device     : {device}\n")
    return model, tokenizer, dev


def chat(model, tokenizer, device, args):
    """Interactive chat loop."""
    state = ChatState(system_prompt=args.system_prompt)
    print("=" * 60)
    print("CMF-Infinity Chat Interface")
    print("Type 'quit' to exit, 'reset' to clear history.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "reset":
            state = ChatState(system_prompt=args.system_prompt)
            print("[History cleared]")
            continue
        if not user_input:
            continue

        state.add_user(user_input)
        prompt = state.render()

        input_ids = encode_to_tensor(tokenizer, prompt).unsqueeze(0).to(device)

        with torch.no_grad():
            generated = generate_ids(
                model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
                max_context_tokens=args.max_context_tokens,
            )

        new_tokens = generated[0, input_ids.size(1):]
        response = decode_tokens(tokenizer, new_tokens)
        response = trim_assistant_response("Assistant:" + response)

        state.add_assistant(response)
        print(f"\nCMF: {response}")


def compress(model, tokenizer, device, args):
    """Compute bits-per-byte on a file — uses the model as a compression oracle."""
    import math
    path = Path(args.compress_file)
    data = path.read_bytes()
    print(f"Computing compression score for: {path} ({len(data):,} bytes)")

    # Encode as byte tokens
    ids = torch.tensor(list(data), dtype=torch.long).unsqueeze(0).to(device)

    total_nll = 0.0
    chunk = args.max_context_tokens
    num_chunks = 0

    with torch.no_grad():
        for start in range(0, ids.size(1) - 1, chunk):
            x = ids[:, start:start + chunk]
            y = ids[:, start + 1:start + chunk + 1]
            if x.size(1) < 2:
                continue
            # Pad y to same length as x
            if y.size(1) < x.size(1):
                y = torch.cat([y, x[:, -1:]], dim=1)
            out = model(x, labels=y)
            total_nll += float(out["loss"]) * x.size(1)
            num_chunks += x.size(1)

    bpb = total_nll / (num_chunks * math.log(2))
    print(f"  Bits-per-byte (BPB): {bpb:.4f}")
    print(f"  (Lower = better compression. gzip ~= 3.5 BPB, ideal = 1.0 BPB)")
    return bpb


def main():
    parser = argparse.ArgumentParser(description="CMF-Infinity Deployment")
    parser.add_argument("--package", type=str, required=True,
                        help="Path to .package.pt file from training")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to run on")
    parser.add_argument("--mode", type=str, default="chat",
                        choices=["chat", "compress"],
                        help="Run as chatbot or compression benchmark")

    # Chat options
    parser.add_argument("--system-prompt", type=str,
                        default="You are CMF Infinity, a precise and intelligent assistant. Answer directly and accurately.",
                        help="System prompt for chat mode")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum tokens to generate per response")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (higher = more creative)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K sampling filter")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-P (nucleus) sampling filter")
    parser.add_argument("--repetition-penalty", type=float, default=1.15,
                        help="Repetition penalty (1.0 = disabled)")
    parser.add_argument("--max-context-tokens", type=int, default=1024,
                        help="Max context window to feed to model")

    # Compression options
    parser.add_argument("--compress-file", type=str, default=None,
                        help="File to compute compression score for (compress mode)")

    args = parser.parse_args()

    model, tokenizer, device = load_model(args.package, args.device)

    if args.mode == "chat":
        chat(model, tokenizer, device, args)
    elif args.mode == "compress":
        if not args.compress_file:
            print("Error: --compress-file required for compress mode")
            sys.exit(1)
        compress(model, tokenizer, device, args)


if __name__ == "__main__":
    main()
