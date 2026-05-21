import argparse
import sys
from pathlib import Path

import torch

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.checkpointing import load_model_package
from cmf.generation import decode_tokens, encode_to_tensor
from cmf.infinity_runtime import DeliberationConfig, deliberative_generate_text
from cmf.runtime import resolve_device

def stream_generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    thinking_steps: int,
    max_context_tokens: int,
) -> None:
    """Generate tokens one‑by‑one and print them as they appear.

    This function mirrors the logic inside ``generate_ids`` but prints each token as soon
    as it is sampled, giving a "real‑time" feel similar to chat UI streaming.
    """
    # Encode prompt once
    input_ids = encode_to_tensor(tokenizer, prompt).unsqueeze(0).to(device)
    generated = input_ids
    # Configure deliberation (the model's internal thinking loops)
    model.config.adaptive_thinking = False
    model.config.thinking_steps = thinking_steps

    for _ in range(max_new_tokens):
        # Respect max_context to avoid OOM on long prompts
        model_input = generated
        if max_context_tokens is not None and generated.size(1) > max_context_tokens:
            model_input = generated[:, -max_context_tokens:]
        # Forward pass – model returns a dict with "logits"
        output = model(model_input)
        logits = output["logits"][:, -1]  # (batch, vocab)
        # Apply repetition penalty (same as in generate_ids)
        logits = apply_repetition_penalty(logits, generated, repetition_penalty)
        # Sample next token
        next_token = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        # Append token to sequence
        generated = torch.cat([generated, next_token], dim=1)
        # Decode and print the newly generated token (no newline, flush immediately)
        token_str = decode_tokens(tokenizer, next_token.squeeze(0)).strip()
        print(token_str, end="", flush=True)
        # Stop early if eos token encountered
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None and int(next_token.squeeze()) == eos_id:
            break
    print()  # final newline for prompt

# Helper functions copied from cmf.generation (slightly simplified)
def apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    for batch_idx in range(generated.size(0)):
        token_ids = torch.unique(generated[batch_idx])
        token_logits = logits[batch_idx, token_ids]
        logits[batch_idx, token_ids] = torch.where(
            token_logits < 0,
            token_logits * penalty,
            token_logits / penalty,
        )
    return logits

def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = logits.clone()
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits.masked_fill(remove, float("-inf")))
    return filtered

def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float = 0.95,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-6)
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
    logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def main() -> None:
    parser = argparse.ArgumentParser(description="Stream tokens from a CMF package with adjustable temperature, steps, etc.")
    parser.add_argument("package", type=Path, help="Path to .package.pt file produced by package_checkpoint.py")
    parser.add_argument("--prompt", required=True, help="Prompt string to feed the model")
    parser.add_argument("--device", default="auto", help="Device to run on (auto, cpu, cuda)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top‑k sampling size")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top‑p (nucleus) probability mass")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Penalty to discourage repeats")
    parser.add_argument("--deliberation-steps", type=int, default=8, help="Number of internal thinking steps (model.config.thinking_steps)")
    parser.add_argument("--max-context-tokens", type=int, default=None, help="If set, truncate context to this many tokens")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Loading package from {args.package} on {device}…")
    model, tokenizer, payload = load_model_package(args.package, device=device)

    # Stream generation
    stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        thinking_steps=args.deliberation_steps,
        max_context_tokens=args.max_context_tokens,
    )

if __name__ == "__main__":
    main()
