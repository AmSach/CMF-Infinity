from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .data import ByteTokenizer, cyclic_lm_batches, repeated_corpus
from .experiments import count_parameters


FACTS = {
    "mercury": "planet",
    "venus": "planet",
    "mars": "planet",
    "python": "language",
    "paris": "france",
    "tokyo": "japan",
    "cuda": "gpu",
    "transformer": "attention",
    "cmf": "field",
    "mamba": "state",
}

CHAIN_FACTS = [
    ("alice", "bob", "developer"),
    ("charlie", "david", "scientist"),
    ("eve", "frank", "engineer"),
    ("grace", "heidi", "artist"),
    ("ivan", "judy", "lawyer"),
]


@dataclass(frozen=True)
class TaskPrompt:
    prompt: str
    answer: str
    kind: str


def make_fact_corpus(repeats: int = 128) -> str:
    rows = []
    for _ in range(repeats):
        for key, value in FACTS.items():
            rows.append(f"Q: what is {key}? A: {value}\n")
    return "".join(rows)


def make_addition_corpus(max_value: int = 49, repeats: int = 4, seed: int = 0) -> str:
    rng = random.Random(seed)
    pairs = [(a, b) for a in range(max_value + 1) for b in range(max_value + 1)]
    rows = []
    for _ in range(repeats):
        rng.shuffle(pairs)
        for a, b in pairs:
            rows.append(f"Q: {a}+{b}=? A: {a + b}\n")
    return "".join(rows)


def make_chain_reasoning_corpus(repeats: int = 40) -> str:
    rows = []
    for _ in range(repeats):
        for sub, obj, role in CHAIN_FACTS:
            rows.append(f"Fact: {sub} is {obj}. Fact: {obj} is {role}. Q: what is {sub}? A: {role}\n")
    return "".join(rows)


def make_mixed_reasoning_corpus(seed: int = 0) -> str:
    return (
        make_fact_corpus(repeats=64) + 
        make_addition_corpus(max_value=39, repeats=2, seed=seed) +
        make_chain_reasoning_corpus(repeats=32)
    )


def make_task_prompts() -> list[TaskPrompt]:
    prompts: list[TaskPrompt] = []
    for key, value in FACTS.items():
        prompts.append(TaskPrompt(prompt=f"Q: what is {key}? A: ", answer=value, kind="fact"))
    for a, b in [(2, 3), (7, 8), (12, 5), (19, 20), (31, 8), (39, 0)]:
        prompts.append(TaskPrompt(prompt=f"Q: {a}+{b}=? A: ", answer=str(a + b), kind="addition_seen_range"))
    for a, b in [(41, 2), (50, 7), (63, 8), (91, 5)]:
        prompts.append(TaskPrompt(prompt=f"Q: {a}+{b}=? A: ", answer=str(a + b), kind="addition_extrapolate"))
    for sub, obj, role in CHAIN_FACTS:
        prompts.append(TaskPrompt(prompt=f"Fact: {sub} is {obj}. Fact: {obj} is {role}. Q: what is {sub}? A: ", answer=role, kind="chain_reasoning"))
    return prompts


def make_lm_batches(
    text: str,
    seq_len: int,
    batch_size: int,
    num_batches: int,
    min_bytes: int = 65536,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    data = repeated_corpus(text, min_bytes=min_bytes)
    return cyclic_lm_batches(
        data,
        seq_len=seq_len,
        batch_size=batch_size,
        num_batches=num_batches,
        stride=max(1, seq_len // 3),
    )


def parameter_match_report(
    candidate: nn.Module,
    baseline: nn.Module,
    *,
    tolerance: float = 0.02,
) -> dict[str, float | int | bool]:
    candidate_params = count_parameters(candidate)
    baseline_params = count_parameters(baseline)
    ratio = candidate_params / max(baseline_params, 1)
    relative_delta = abs(candidate_params - baseline_params) / max(baseline_params, 1)
    return {
        "candidate_parameters": candidate_params,
        "baseline_parameters": baseline_params,
        "candidate_div_baseline": ratio,
        "relative_delta": relative_delta,
        "tolerance": tolerance,
        "matched": relative_delta <= tolerance,
    }


def extract_answer(text: str) -> str:
    match = re.search(r"A:[ \t]*([^\n\r]*)", text)
    if not match:
        return ""
    return match.group(1).strip().split(" ")[0]


def evaluate_prompt_accuracy(
    model,
    tokenizer: ByteTokenizer,
    prompts: list[TaskPrompt],
    device: torch.device,
    max_new_tokens: int = 16,
) -> dict:
    model.eval()
    rows = []
    correct = 0
    by_kind: dict[str, list[bool]] = {}
    with torch.no_grad():
        for item in prompts:
            input_ids = tokenizer.encode(item.prompt).unsqueeze(0).to(device)
            generated = input_ids
            for _ in range(max_new_tokens):
                output = model(generated)
                next_token = torch.argmax(output["logits"][:, -1], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                if int(next_token[0, 0]) in (10, 13):
                    break
            decoded = tokenizer.decode(generated[0])
            answer = extract_answer(decoded)
            is_correct = answer == item.answer
            correct += int(is_correct)
            by_kind.setdefault(item.kind, []).append(is_correct)
            rows.append(
                {
                    "kind": item.kind,
                    "prompt": item.prompt,
                    "expected": item.answer,
                    "generated": decoded,
                    "answer": answer,
                    "correct": is_correct,
                }
            )

    by_kind_accuracy = {
        kind: sum(values) / max(1, len(values)) for kind, values in by_kind.items()
    }
    return {
        "accuracy": correct / max(1, len(prompts)),
        "correct": correct,
        "total": len(prompts),
        "by_kind_accuracy": by_kind_accuracy,
        "rows": rows,
    }


def evaluate_candidate_accuracy(
    model,
    tokenizer: ByteTokenizer,
    prompts: list[TaskPrompt],
    device: torch.device,
    numeric_max: int = 120,
) -> dict:
    fact_candidates = sorted(set(FACTS.values()))
    numeric_candidates = [str(value) for value in range(numeric_max + 1)]
    rows = []
    correct = 0
    by_kind: dict[str, list[bool]] = {}
    model.eval()

    with torch.no_grad():
        for item in prompts:
            candidates = (
                numeric_candidates
                if item.kind.startswith("addition")
                else (fact_candidates if item.kind == "fact" else sorted(set(r for _, _, r in CHAIN_FACTS)))
            )
            scored = [
                (candidate, score_candidate(model, tokenizer, item.prompt, candidate, device))
                for candidate in candidates
            ]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            best_answer = scored[0][0]
            is_correct = best_answer == item.answer
            correct += int(is_correct)
            by_kind.setdefault(item.kind, []).append(is_correct)
            rows.append(
                {
                    "kind": item.kind,
                    "prompt": item.prompt,
                    "expected": item.answer,
                    "best_answer": best_answer,
                    "best_score": scored[0][1],
                    "expected_rank": next(
                        idx + 1
                        for idx, (candidate, _score) in enumerate(scored)
                        if candidate == item.answer
                    ),
                    "correct": is_correct,
                }
            )

    by_kind_accuracy = {
        kind: sum(values) / max(1, len(values)) for kind, values in by_kind.items()
    }
    return {
        "accuracy": correct / max(1, len(prompts)),
        "correct": correct,
        "total": len(prompts),
        "by_kind_accuracy": by_kind_accuracy,
        "rows": rows,
    }


def score_candidate(
    model,
    tokenizer: ByteTokenizer,
    prompt: str,
    candidate: str,
    device: torch.device,
) -> float:
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + candidate + "\n")
    input_ids = full_ids[:-1].unsqueeze(0).to(device)
    labels = full_ids[1:].unsqueeze(0).to(device)
    output = model(input_ids)
    log_probs = torch.log_softmax(output["logits"], dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    start = max(0, prompt_ids.numel() - 1)
    return float(token_log_probs[0, start:].sum().detach().cpu())
