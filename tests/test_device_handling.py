import pytest

torch = pytest.importorskip("torch")

from cmf import CMFConfig, ParallelContinuousMeaningField
from cmf.data import ByteTokenizer
from cmf.experiments import benchmark_forward
from cmf.runtime import resolve_device


def test_resolve_device_cpu_is_explicit():
    assert resolve_device("cpu").type == "cpu"


def test_cpu_benchmark_does_not_require_cuda():
    model = ParallelContinuousMeaningField(
        CMFConfig(vocab_size=32, d_model=8, hidden_dim=16, num_layers=1)
    )
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("cpu benchmark smoke")[:8].unsqueeze(0) % 32
    labels = ids.clone()

    result = benchmark_forward(
        model,
        ids,
        labels,
        torch.device("cpu"),
        iterations=1,
        warmup=0,
    )

    assert result["tokens_per_sec"] > 0
    assert "cuda_peak_memory_mb" not in result
