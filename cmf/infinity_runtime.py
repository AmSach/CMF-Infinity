from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from .generation import (
    decode_tokens,
    encode_to_tensor,
    generate_ids,
    trim_assistant_response,
)


_TERM_RE = re.compile(r"[A-Za-z0-9_]+")


def _terms(text: str) -> list[str]:
    return [term.lower() for term in _TERM_RE.findall(text) if len(term) > 1]


def _term_set(text: str) -> set[str]:
    return set(_terms(text))


@dataclass(frozen=True)
class EvidenceItem:
    """A retrievable text chunk used to ground CMF generation."""

    text: str
    source: str = "memory"
    score: float = 0.0

    def compact(self, max_chars: int = 600) -> str:
        text = " ".join(self.text.split())
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 3)].rstrip() + "..."


@dataclass
class EvidenceMemory:
    """Small dependency-free lexical memory for factual grounding.

    This is intentionally simple: it is a local retrieval layer, not a claim of
    learned factual knowledge. Stronger deployments can replace it with vector
    search while preserving the same retrieve/render interface.
    """

    items: list[EvidenceItem] = field(default_factory=list)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        source: str = "memory",
        chunk_chars: int = 900,
        overlap_chars: int = 120,
    ) -> "EvidenceMemory":
        chunks: list[EvidenceItem] = []
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        for paragraph_idx, paragraph in enumerate(paragraphs or [text]):
            if len(paragraph) <= chunk_chars:
                chunks.append(EvidenceItem(paragraph, source=f"{source}#{paragraph_idx + 1}"))
                continue

            start = 0
            chunk_idx = 1
            step = max(1, chunk_chars - overlap_chars)
            while start < len(paragraph):
                chunk = paragraph[start : start + chunk_chars].strip()
                if chunk:
                    chunks.append(
                        EvidenceItem(
                            chunk,
                            source=f"{source}#{paragraph_idx + 1}.{chunk_idx}",
                        )
                    )
                start += step
                chunk_idx += 1
        return cls(chunks)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        chunk_chars: int = 900,
        overlap_chars: int = 120,
    ) -> "EvidenceMemory":
        source_path = Path(path)
        text = source_path.read_text(encoding="utf-8")
        return cls.from_text(
            text,
            source=str(source_path),
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
        )

    def retrieve(self, query: str, *, k: int = 4) -> list[EvidenceItem]:
        query_terms = _term_set(query)
        if not query_terms or not self.items or k <= 0:
            return []

        scored: list[EvidenceItem] = []
        for item in self.items:
            item_terms = _term_set(item.text)
            if not item_terms:
                continue
            overlap = query_terms & item_terms
            if not overlap:
                continue
            precision = len(overlap) / max(1, len(item_terms))
            recall = len(overlap) / max(1, len(query_terms))
            phrase_bonus = 0.25 if query.lower().strip() in item.text.lower() else 0.0
            score = (2.0 * precision * recall) / max(precision + recall, 1e-9)
            scored.append(EvidenceItem(item.text, item.source, score + phrase_bonus))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]

    def render(self, query: str, *, k: int = 4, max_chars_per_item: int = 600) -> str:
        evidence = self.retrieve(query, k=k)
        if not evidence:
            return ""
        lines = ["Evidence:"]
        for idx, item in enumerate(evidence, start=1):
            lines.append(f"[{idx}] ({item.source}) {item.compact(max_chars_per_item)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class DeliberationConfig:
    """Generation-time controller for open-ended CMF thinking loops."""

    max_thinking_steps: int | None = 4
    min_thinking_steps: int = 1
    candidates_per_step: int = 2
    max_new_tokens: int = 256
    temperature: float = 0.8
    min_temperature: float = 0.25
    temperature_decay: float = 0.9
    top_k: int | None = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    improvement_epsilon: float = 1e-3
    patience: int = 2
    max_wall_time_s: float | None = None
    evidence_top_k: int = 4
    evidence_weight: float = 0.15
    self_consistency_weight: float = 0.05
    length_penalty: float = 0.002

    def validate(self) -> None:
        if self.max_thinking_steps is not None and self.max_thinking_steps < 1:
            raise ValueError("max_thinking_steps must be positive or None")
        if self.min_thinking_steps < 1:
            raise ValueError("min_thinking_steps must be positive")
        if self.max_thinking_steps is not None and self.min_thinking_steps > self.max_thinking_steps:
            raise ValueError("min_thinking_steps cannot exceed max_thinking_steps")
        if self.candidates_per_step < 1:
            raise ValueError("candidates_per_step must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.max_thinking_steps is None and self.max_wall_time_s is None:
            raise ValueError(
                "open-ended deliberation requires max_wall_time_s or another external stop condition"
            )


@dataclass(frozen=True)
class DeliberationCandidate:
    text: str
    score: float
    mean_logprob: float
    evidence_score: float
    consensus_score: float
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "mean_logprob": self.mean_logprob,
            "evidence_score": self.evidence_score,
            "consensus_score": self.consensus_score,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class DeliberationStep:
    step: int
    elapsed_s: float
    best_score: float
    converged: bool
    candidates: list[DeliberationCandidate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "elapsed_s": self.elapsed_s,
            "best_score": self.best_score,
            "converged": self.converged,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class DeliberationResult:
    text: str
    score: float
    grounded_prompt: str
    evidence: list[EvidenceItem]
    steps: list[DeliberationStep]
    stopped_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "grounded_prompt": self.grounded_prompt,
            "evidence": [
                {"source": item.source, "score": item.score, "text": item.text}
                for item in self.evidence
            ],
            "steps": [step.to_dict() for step in self.steps],
            "stopped_reason": self.stopped_reason,
        }


def build_grounded_prompt(prompt: str, evidence: Iterable[EvidenceItem]) -> str:
    evidence = list(evidence)
    if not evidence:
        return prompt
    lines = [
        "Use the evidence below when it is relevant. If the evidence is insufficient, say so.",
        "",
        "Evidence:",
    ]
    for idx, item in enumerate(evidence, start=1):
        lines.append(f"[{idx}] ({item.source}) {item.compact()}")
    lines.extend(["", "Task:", prompt.strip()])
    return "\n".join(lines)


def _score_evidence_alignment(text: str, evidence: list[EvidenceItem]) -> float:
    if not evidence:
        return 0.0
    answer_terms = _term_set(text)
    if not answer_terms:
        return 0.0
    evidence_terms = set()
    for item in evidence:
        evidence_terms.update(_term_set(item.text))
    if not evidence_terms:
        return 0.0
    return len(answer_terms & evidence_terms) / max(1, len(answer_terms))


def _consensus_scores(texts: list[str]) -> list[float]:
    if not texts:
        return []
    term_sets = [_term_set(text) for text in texts]
    scores: list[float] = []
    for idx, terms in enumerate(term_sets):
        if not terms:
            scores.append(0.0)
            continue
        others = [other for other_idx, other in enumerate(term_sets) if other_idx != idx and other]
        if not others:
            scores.append(0.0)
            continue
        similarities = [
            len(terms & other) / max(1, len(terms | other))
            for other in others
        ]
        scores.append(sum(similarities) / len(similarities))
    return scores


@torch.no_grad()
def score_token_continuation(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    *,
    device: torch.device | str,
    max_context_tokens: int | None = None,
) -> float:
    """Return mean log probability for a continuation under the current model."""
    prompt_ids = prompt_ids.detach().to(device=device, dtype=torch.long).flatten()
    continuation_ids = continuation_ids.detach().to(device=device, dtype=torch.long).flatten()
    if continuation_ids.numel() == 0:
        return float("-inf")

    if max_context_tokens is not None:
        keep_prompt = max(1, max_context_tokens - int(continuation_ids.numel()))
        prompt_ids = prompt_ids[-keep_prompt:]

    full_ids = torch.cat([prompt_ids, continuation_ids])
    if full_ids.numel() < 2:
        return float("-inf")
    input_ids = full_ids[:-1].unsqueeze(0)
    labels = full_ids[1:].unsqueeze(0)
    output = model(input_ids)
    log_probs = torch.log_softmax(output["logits"], dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    start = max(0, prompt_ids.numel() - 1)
    continuation_log_probs = token_log_probs[0, start:]
    if continuation_log_probs.numel() == 0:
        return float("-inf")
    return float(continuation_log_probs.mean().detach().cpu())


def _score_generated_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    text: str,
    *,
    device: torch.device | str,
    evidence: list[EvidenceItem],
    consensus_score: float,
    config: DeliberationConfig,
    max_context_tokens: int | None,
) -> DeliberationCandidate:
    continuation_ids = encode_to_tensor(tokenizer, text).to(device)
    mean_logprob = score_token_continuation(
        model,
        prompt_ids,
        continuation_ids,
        device=device,
        max_context_tokens=max_context_tokens,
    )
    evidence_score = _score_evidence_alignment(text, evidence)
    length_cost = math.log1p(max(0, int(continuation_ids.numel())))
    score = (
        mean_logprob
        + config.evidence_weight * evidence_score
        + config.self_consistency_weight * consensus_score
        - config.length_penalty * length_cost
    )
    return DeliberationCandidate(
        text=text,
        score=score,
        mean_logprob=mean_logprob,
        evidence_score=evidence_score,
        consensus_score=consensus_score,
        token_count=int(continuation_ids.numel()),
    )


def deliberative_generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    device: torch.device | str,
    config: DeliberationConfig | None = None,
    memory: EvidenceMemory | None = None,
    retrieval_query: str | None = None,
    eos_token_id: int | None = None,
    max_context_tokens: int | None = None,
) -> DeliberationResult:
    """Generate text using repeated candidate creation and verifier scoring."""
    config = config or DeliberationConfig()
    config.validate()

    query = retrieval_query or prompt
    evidence = memory.retrieve(query, k=config.evidence_top_k) if memory is not None else []
    grounded_prompt = build_grounded_prompt(prompt, evidence)
    prompt_ids = encode_to_tensor(tokenizer, grounded_prompt).unsqueeze(0).to(device)
    flat_prompt_ids = prompt_ids[0]

    start_time = time.perf_counter()
    best_candidate: DeliberationCandidate | None = None
    stagnant_steps = 0
    steps: list[DeliberationStep] = []
    stopped_reason = "max_thinking_steps"

    step_idx = 0
    while True:
        elapsed = time.perf_counter() - start_time
        if config.max_wall_time_s is not None and elapsed >= config.max_wall_time_s:
            stopped_reason = "max_wall_time_s"
            break
        if config.max_thinking_steps is not None and step_idx >= config.max_thinking_steps:
            stopped_reason = "max_thinking_steps"
            break

        temperature = max(
            config.min_temperature,
            config.temperature * (config.temperature_decay ** step_idx),
        )
        generated_texts: list[str] = []
        for candidate_idx in range(config.candidates_per_step):
            candidate_temperature = max(
                config.min_temperature,
                temperature * (1.0 + 0.05 * candidate_idx),
            )
            output_ids = generate_ids(
                model,
                prompt_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=candidate_temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                eos_token_id=eos_token_id,
                max_context_tokens=max_context_tokens,
            )
            new_ids = output_ids[0, prompt_ids.size(1) :]
            text = trim_assistant_response(decode_tokens(tokenizer, new_ids))
            if text:
                generated_texts.append(text)

        consensus = _consensus_scores(generated_texts)
        candidates = [
            _score_generated_text(
                model,
                tokenizer,
                flat_prompt_ids,
                text,
                device=device,
                evidence=evidence,
                consensus_score=consensus[idx],
                config=config,
                max_context_tokens=max_context_tokens,
            )
            for idx, text in enumerate(generated_texts)
        ]
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)

        improved = False
        if candidates:
            round_best = candidates[0]
            if (
                best_candidate is None
                or round_best.score > best_candidate.score + config.improvement_epsilon
            ):
                best_candidate = round_best
                stagnant_steps = 0
                improved = True
            else:
                stagnant_steps += 1
        else:
            stagnant_steps += 1

        converged = (
            step_idx + 1 >= config.min_thinking_steps
            and stagnant_steps >= config.patience
        )
        steps.append(
            DeliberationStep(
                step=step_idx + 1,
                elapsed_s=time.perf_counter() - start_time,
                best_score=best_candidate.score if best_candidate is not None else float("-inf"),
                converged=converged,
                candidates=candidates,
            )
        )
        if converged:
            stopped_reason = "converged"
            break
        if not improved and config.max_thinking_steps is None and config.max_wall_time_s is None:
            stopped_reason = "no_external_stop"
            break
        step_idx += 1

    if best_candidate is None:
        best_candidate = DeliberationCandidate(
            text="",
            score=float("-inf"),
            mean_logprob=float("-inf"),
            evidence_score=0.0,
            consensus_score=0.0,
            token_count=0,
        )

    return DeliberationResult(
        text=best_candidate.text,
        score=best_candidate.score,
        grounded_prompt=grounded_prompt,
        evidence=evidence,
        steps=steps,
        stopped_reason=stopped_reason,
    )
