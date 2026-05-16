import math

import pytest

torch = pytest.importorskip("torch")

from cmf.data import ByteTokenizer
from cmf.infinity_runtime import (
    DeliberationConfig,
    EvidenceMemory,
    build_grounded_prompt,
    score_token_continuation,
)
from cmf.solver import integrate_adaptive


def test_evidence_memory_retrieves_relevant_chunk():
    memory = EvidenceMemory.from_text(
        "CUDA accelerates tensor kernels on NVIDIA GPUs.\n\n"
        "Paris is the capital city of France.",
        source="facts",
    )

    results = memory.retrieve("What accelerates tensor kernels?", k=1)

    assert results
    assert "CUDA" in results[0].text
    assert results[0].score > 0


def test_grounded_prompt_includes_evidence_and_task():
    evidence = EvidenceMemory.from_text("CMF uses a continuous latent field.").retrieve("CMF field")

    prompt = build_grounded_prompt("Explain CMF.", evidence)

    assert "Evidence:" in prompt
    assert "Task:" in prompt
    assert "continuous latent field" in prompt


def test_open_ended_deliberation_requires_stop_condition():
    config = DeliberationConfig(max_thinking_steps=None, max_wall_time_s=None)

    with pytest.raises(ValueError):
        config.validate()


def test_score_token_continuation_uniform_model():
    class UniformModel(torch.nn.Module):
        def forward(self, input_ids):
            batch, seq_len = input_ids.shape
            return {"logits": torch.zeros(batch, seq_len, 256, device=input_ids.device)}

    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode("A: ")
    continuation_ids = tokenizer.encode("field")

    score = score_token_continuation(
        UniformModel(),
        prompt_ids,
        continuation_ids,
        device="cpu",
    )

    assert math.isclose(score, -math.log(256), rel_tol=1e-5)


def test_adaptive_solver_uses_unit_time_horizon_for_curved_fields():
    z0 = torch.zeros(1, 1)
    context = torch.zeros(1, 1)

    def field(_z, _context, tau):
        return tau.unsqueeze(-1) * 10.0

    z, steps = integrate_adaptive(
        z0,
        context,
        field,
        min_steps=1,
        max_steps=8,
        curvature_threshold=0.05,
    )

    assert steps > 1
    assert 0.0 < float(z[0, 0]) < 10.0


def test_adaptive_solver_keeps_constant_fields_cheap():
    z0 = torch.zeros(2, 3)
    context = torch.zeros(2, 3)

    def field(_z, _context, tau):
        return torch.ones(tau.size(0), 3)

    z, steps = integrate_adaptive(
        z0,
        context,
        field,
        min_steps=1,
        max_steps=8,
        curvature_threshold=0.05,
    )

    assert steps == 1
    assert torch.allclose(z, torch.ones_like(z))
