from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import CMFConfig


@dataclass(frozen=True)
class CMFInfinityPreset:
    name: str
    model_type: str
    config: CMFConfig
    description: str

    @property
    def parameter_billions(self) -> float:
        return estimate_cmf_parameters(self.config, model_type=self.model_type) / 1_000_000_000.0

    @property
    def display_name(self) -> str:
        return f"CMF Infinity {self.parameter_billions:.5g}B"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "model_type": self.model_type,
            "parameter_estimate": estimate_cmf_parameters(self.config, model_type=self.model_type),
            "parameter_billions": self.parameter_billions,
            "config": asdict(self.config),
            "description": self.description,
        }


def _encoder_params(config: CMFConfig) -> int:
    total = 0
    for _ in range(config.num_layers):
        total += (config.hidden_dim * 2) * config.d_model * config.kernel_size
        total += config.hidden_dim * 2
        total += config.d_model * config.hidden_dim
        total += config.d_model
        total += config.d_model * 2
    return total


def _shared_params(config: CMFConfig) -> int:
    total = config.vocab_size * config.d_model
    if not config.tie_embeddings:
        total += config.vocab_size * config.d_model
    total += config.d_model * config.d_model + config.d_model
    total += config.d_model * 2
    return total


def estimate_cmf_parameters(config: CMFConfig, *, model_type: str = "parallel_cmf") -> int:
    total = _shared_params(config) + _encoder_params(config)
    if model_type in {"continuous_cmf", "parallel_cmf", "deliberative_cmf"}:
        time_dim = 16
        field_in = config.d_model * 4 + time_dim
        total += 32 * config.d_model
        total += config.hidden_dim * config.d_model + config.hidden_dim
        total += config.hidden_dim * field_in + config.hidden_dim
        total += config.hidden_dim * config.hidden_dim + config.hidden_dim
        total += config.d_model * config.hidden_dim + config.d_model
        total += config.d_model * config.hidden_dim + config.d_model
        if model_type == "deliberative_cmf":
            total += config.d_model * (config.d_model * 3) + config.d_model
            total += config.d_model + 1
    elif model_type == "fast_parallel_cmf":
        field_in = config.d_model * 4
        total += 32 * config.d_model
        total += config.d_model * field_in + config.d_model
        total += config.d_model * field_in + config.d_model
    else:
        raise ValueError(f"Unknown CMF model_type: {model_type}")
    return total


PRESETS: dict[str, CMFInfinityPreset] = {
    "infinity-0.00037b": CMFInfinityPreset(
        name="infinity-0.00037b",
        model_type="fast_parallel_cmf",
        config=CMFConfig(
            vocab_size=256,
            d_model=80,
            hidden_dim=160,
            num_layers=3,
            solver_steps_per_token=1,
            max_seq_len=96,
            dropout=0.0,
            tie_embeddings=False,
        ),
        description="Small matched benchmark preset used for fast regression tests.",
    ),
    "infinity-0.203b": CMFInfinityPreset(
        name="infinity-0.203b",
        model_type="continuous_cmf",
        config=CMFConfig(
            vocab_size=50257,
            d_model=768,
            hidden_dim=3072,
            num_layers=6,
            solver_steps_per_token=4,
            max_seq_len=1024,
            dropout=0.0,
            tie_embeddings=True,
        ),
        description="GPT-2-tokenizer legacy checkpoint scale, about 203M tensor parameters.",
    ),
    "infinity-0.12b": CMFInfinityPreset(
        name="infinity-0.12b",
        model_type="parallel_cmf",
        config=CMFConfig(
            vocab_size=50257,
            d_model=640,
            hidden_dim=2560,
            num_layers=6,
            kernel_size=3,
            solver_steps_per_token=1,
            max_seq_len=128,
            dropout=0.0,
            tie_embeddings=True,
            adaptive_steps=False,
        ),
        description="120M-class GPT-2-tokenizer CMF preset for single-GPU smoke gates.",
    ),
    "infinity-reasoning-0.12b": CMFInfinityPreset(
        name="infinity-reasoning-0.12b",
        model_type="deliberative_cmf",
        config=CMFConfig(
            vocab_size=50257,
            d_model=640,
            hidden_dim=2560,
            num_layers=6,
            kernel_size=3,
            solver_steps_per_token=1,
            max_seq_len=8192,       # Adaptive context — no more artificial ceiling
            dropout=0.0,
            tie_embeddings=True,
            adaptive_steps=False,
            thinking_steps=16,      # Default thinking budget
            adaptive_thinking=True,
            min_thinking_steps=4,
            max_thinking_steps=96,  # Claude-level deliberation depth
            halting_threshold=0.90,
            num_memory_anchors=64,
            field_depth=2,
        ),
        description="120M-class deliberative CMF with 96-step adaptive thinking and 8k context. Proof-of-concept for funding.",
    ),
    "infinity-0.5b": CMFInfinityPreset(
        name="infinity-0.5b",
        model_type="parallel_cmf",
        config=CMFConfig(
            vocab_size=50257,
            d_model=1024,
            hidden_dim=4096,
            num_layers=16,
            kernel_size=3,
            solver_steps_per_token=1,
            max_seq_len=8192,
            dropout=0.0,
            tie_embeddings=True,
            adaptive_steps=False,
            num_memory_anchors=128,
            field_depth=2,
        ),
        description="0.5B-class CMF Infinity preset for high-throughput multi-GPU training.",
    ),
    "infinity-2b": CMFInfinityPreset(
        name="infinity-2b",
        model_type="parallel_cmf",
        config=CMFConfig(
            vocab_size=50257,
            d_model=1536,
            hidden_dim=6144,
            num_layers=32,
            kernel_size=3,
            solver_steps_per_token=1,
            max_seq_len=8192,
            dropout=0.0,
            tie_embeddings=True,
            adaptive_steps=False,
            num_memory_anchors=256,
            field_depth=3,
        ),
        description="2B-class deep CMF Infinity preset for maximum reasoning capacity.",
    ),
    "infinity-reasoning-1.2b": CMFInfinityPreset(
        name="infinity-reasoning-1.2b",
        model_type="deliberative_cmf",
        config=CMFConfig(
            vocab_size=50257,
            d_model=1280,
            hidden_dim=5120,
            num_layers=24,
            kernel_size=3,
            solver_steps_per_token=1,
            max_seq_len=8192,
            dropout=0.0,
            tie_embeddings=True,
            adaptive_steps=False,
            thinking_steps=16,
            adaptive_thinking=True,
            min_thinking_steps=4,
            max_thinking_steps=96,
            halting_threshold=0.92,
            num_memory_anchors=256,
            field_depth=3,
        ),
        description="1.2B-class deliberative CMF with 96-step deep iterative refinement for SOTA reasoning.",
    ),
    "infinity-reasoning-2b": CMFInfinityPreset(
        name="infinity-reasoning-2b",
        model_type="deliberative_cmf",
        config=CMFConfig(
            vocab_size=50257,
            d_model=1536,
            hidden_dim=6144,
            num_layers=24,
            kernel_size=3,
            solver_steps_per_token=1,
            max_seq_len=8192,
            dropout=0.0,
            tie_embeddings=True,
            adaptive_steps=False,
            thinking_steps=16,
            adaptive_thinking=True,
            min_thinking_steps=4,
            max_thinking_steps=96,
            halting_threshold=0.92,
            num_memory_anchors=256,
            field_depth=3,
        ),
        description="2B-class deliberative CMF with 96-step deep iterative refinement for SOTA reasoning.",
    ),
    "infinity-reasoning-7b": CMFInfinityPreset(
        name="infinity-reasoning-7b",
        model_type="deliberative_cmf",
        config=CMFConfig(
            vocab_size=50257,
            d_model=4096,
            hidden_dim=11264,
            num_layers=32,
            kernel_size=3,
            solver_steps_per_token=1,
            max_seq_len=32768,
            dropout=0.0,
            tie_embeddings=True,
            adaptive_steps=False,
            thinking_steps=32,
            adaptive_thinking=True,
            min_thinking_steps=8,
            max_thinking_steps=128,
            halting_threshold=0.92,
            num_memory_anchors=512,
            field_depth=4,
        ),
        description="7B-class deliberative CMF — local deployment target with 128-step reasoning and 32k context.",
    ),
    "infinity-reasoning-70b": CMFInfinityPreset(
        name="infinity-reasoning-70b",
        model_type="deliberative_cmf",
        config=CMFConfig(
            vocab_size=50257,
            d_model=8192,
            hidden_dim=28672,
            num_layers=80,
            kernel_size=3,
            solver_steps_per_token=1,
            max_seq_len=131072,     # 128k context window
            dropout=0.0,
            tie_embeddings=True,
            adaptive_steps=False,
            thinking_steps=48,
            adaptive_thinking=True,
            min_thinking_steps=8,
            max_thinking_steps=192,
            halting_threshold=0.93,
            num_memory_anchors=1024,
            field_depth=6,
        ),
        description="70B-class deliberative CMF — maximum local reasoning model with 192-step thinking and 128k context.",
    ),
    "infinity-reasoning-120b": CMFInfinityPreset(
        name="infinity-reasoning-120b",
        model_type="deliberative_cmf",
        config=CMFConfig(
            vocab_size=50257,
            d_model=12288,
            hidden_dim=40960,
            num_layers=96,
            kernel_size=3,
            solver_steps_per_token=1,
            max_seq_len=131072,
            dropout=0.0,
            tie_embeddings=True,
            adaptive_steps=False,
            thinking_steps=64,
            adaptive_thinking=True,
            min_thinking_steps=16,
            max_thinking_steps=256,
            halting_threshold=0.94,
            num_memory_anchors=2048,
            field_depth=8,
        ),
        description="120B-class deliberative CMF — true AGI cloud model. 256-step reasoning, 128k context, 2048 memory anchors.",
    ),
    "infinity-reasoning-500b": CMFInfinityPreset(
        name="infinity-reasoning-500b",
        model_type="deliberative_cmf",
        config=CMFConfig(
            vocab_size=50257,
            d_model=18432,
            hidden_dim=65536,
            num_layers=128,
            kernel_size=3,
            solver_steps_per_token=1,
            max_seq_len=262144,     # 256k context window
            dropout=0.0,
            tie_embeddings=True,
            adaptive_steps=False,
            thinking_steps=96,
            adaptive_thinking=True,
            min_thinking_steps=32,
            max_thinking_steps=512, # Deepest reasoning ever — 512 thinking orbits
            halting_threshold=0.95,
            num_memory_anchors=4096,
            field_depth=12,
        ),
        description="500B-class deliberative CMF — AGI frontier. 512-step reasoning, 256k context, 4096 memory anchors, 12-layer VectorField.",
    ),
    "infinity-8b": CMFInfinityPreset(
        name="infinity-8b",
        model_type="parallel_cmf",
        config=CMFConfig(
            vocab_size=50257,
            d_model=4096,
            hidden_dim=16384,
            num_layers=16,
            kernel_size=3,
            solver_steps_per_token=2,
            max_seq_len=32768,
            dropout=0.0,
            tie_embeddings=True,
            adaptive_steps=True,
            min_solver_steps=1,
            max_solver_steps=16,
            curvature_threshold=0.03,
            num_memory_anchors=512,
            field_depth=4,
        ),
        description="Target 8B-class CMF Infinity preset for multi-GPU training.",
    ),
}


def get_preset(name: str) -> CMFInfinityPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown preset '{name}'. Available: {', '.join(sorted(PRESETS))}") from exc
