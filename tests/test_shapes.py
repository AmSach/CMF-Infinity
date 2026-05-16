import pytest

torch = pytest.importorskip("torch")

from cmf import (
    CMFConfig,
    ContinuousMeaningField,
    DeliberativeContinuousMeaningField,
    FastParallelContinuousMeaningField,
    ParallelContinuousMeaningField,
)
from cmf.fast_integrator import euler_integrate_precomputed


def test_cmf_forward_shapes():
    config = CMFConfig(vocab_size=32, d_model=16, hidden_dim=32, num_layers=3)
    model = ContinuousMeaningField(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))

    output = model(input_ids, labels=labels, return_states=True)

    assert output["logits"].shape == (2, 8, config.vocab_size)
    assert output["states"].shape == (2, 8, config.d_model)
    assert output["loss"].ndim == 0


def test_parallel_cmf_forward_shapes():
    config = CMFConfig(vocab_size=32, d_model=16, hidden_dim=32, num_layers=3)
    model = ParallelContinuousMeaningField(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))

    output = model(input_ids, labels=labels, return_states=True)

    assert output["logits"].shape == (2, 8, config.vocab_size)
    assert output["states"].shape == (2, 8, config.d_model)
    assert output["loss"].ndim == 0


def test_fast_parallel_cmf_forward_shapes():
    config = CMFConfig(vocab_size=32, d_model=16, hidden_dim=32, num_layers=3)
    model = FastParallelContinuousMeaningField(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))

    output = model(input_ids, labels=labels, return_states=True)

    assert output["logits"].shape == (2, 8, config.vocab_size)
    assert output["states"].shape == (2, 8, config.d_model)
    assert output["loss"].ndim == 0


def test_deliberative_cmf_forward_shapes():
    config = CMFConfig(
        vocab_size=32,
        d_model=16,
        hidden_dim=32,
        num_layers=2,
        thinking_steps=3,
        adaptive_thinking=True,
        min_thinking_steps=1,
        max_thinking_steps=3,
    )
    model = DeliberativeContinuousMeaningField(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))

    output = model(input_ids, labels=labels, return_states=True)

    assert output["logits"].shape == (2, 8, config.vocab_size)
    assert output["states"].shape == (2, 3, 8, config.d_model)
    assert output["loss"].ndim == 0
    assert output["thinking_steps"].item() == 3
    assert 0.0 <= output["halt_mean"].item() <= 1.0


def test_precomputed_euler_matches_cumsum():
    z0 = torch.zeros(2, 3)
    velocity = torch.ones(2, 4, 3)
    states = euler_integrate_precomputed(z0, velocity, dt=0.25)

    expected = torch.tensor(
        [
            [[0.25, 0.25, 0.25], [0.50, 0.50, 0.50], [0.75, 0.75, 0.75], [1.00, 1.00, 1.00]],
            [[0.25, 0.25, 0.25], [0.50, 0.50, 0.50], [0.75, 0.75, 0.75], [1.00, 1.00, 1.00]],
        ]
    )
    assert torch.allclose(states, expected)
