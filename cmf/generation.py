from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ChatTurn:
    role: str
    content: str


@dataclass
class ChatState:
    system_prompt: str = (
        "You are CMF Infinity, a careful research assistant. "
        "Answer directly, admit uncertainty, and keep claims grounded."
    )
    turns: list[ChatTurn] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.turns.append(ChatTurn("User", content))

    def add_assistant(self, content: str) -> None:
        self.turns.append(ChatTurn("Assistant", content))

    def render(self) -> str:
        lines = [f"System: {self.system_prompt.strip()}", ""]
        for turn in self.turns:
            lines.append(f"{turn.role}: {turn.content.strip()}")
        lines.append("Assistant:")
        return "\n".join(lines)


def encode_to_tensor(tokenizer: Any, text: str) -> torch.Tensor:
    encoded = tokenizer.encode(text)
    if isinstance(encoded, torch.Tensor):
        return encoded.to(dtype=torch.long).flatten()
    if isinstance(encoded, list):
        return torch.tensor(encoded, dtype=torch.long)
    if hasattr(encoded, "ids"):
        return torch.tensor(encoded.ids, dtype=torch.long)
    raise TypeError(f"Unsupported tokenizer output type: {type(encoded).__name__}")


def decode_tokens(tokenizer: Any, token_ids: torch.Tensor) -> str:
    try:
        return tokenizer.decode(token_ids)
    except TypeError:
        return tokenizer.decode(token_ids.detach().cpu().flatten().tolist())


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
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


@torch.no_grad()
def generate_ids(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    eos_token_id: int | None = None,
    max_context_tokens: int | None = None,
) -> torch.Tensor:
    model.eval()
    generated = input_ids
    for _ in range(max_new_tokens):
        model_input = generated
        if max_context_tokens is not None and generated.size(1) > max_context_tokens:
            model_input = generated[:, -max_context_tokens:]
        output = model(model_input)
        logits = output["logits"][:, -1]
        logits = apply_repetition_penalty(logits, generated, repetition_penalty)
        next_token = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated = torch.cat([generated, next_token], dim=1)
        if eos_token_id is not None and int(next_token[0, 0]) == eos_token_id:
            break
    return generated


def trim_assistant_response(text: str) -> str:
    if "Assistant:" in text:
        text = text.rsplit("Assistant:", 1)[-1]
    for marker in ("\nUser:", "\nSystem:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()
