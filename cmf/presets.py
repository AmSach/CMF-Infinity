"""
CMF v3 presets — only sizes that can actually be trained.

Removed from v1:
  - "infinity-reasoning-500b", "infinity-reasoning-120b", "infinity-reasoning-70b"
    These exist in the codebase only as config dicts. Nobody has trained them.
    Listing them creates false impressions and wastes time when someone tries
    to instantiate a 500B model on a laptop.

Kept:
  - tiny    (sanity / CI)
  - 50m     (ablation experiments — fast iteration)
  - 120m    (primary proof-of-concept, matches existing kaggle scripts)
  - 500m    (next scale step after 120m validates)

Each preset has model_type so the factory knows which class to build.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from .config import CMFConfig


@dataclass(frozen=True)
class CMFPreset:
    name: str
    model_type: str   # "parallel" | "deliberative"
    config: CMFConfig
    description: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "config": asdict(self.config),
            "description": self.description,
        }


PRESETS: dict[str, CMFPreset] = {
    # ── CI / unit tests ──────────────────────────────────────────────────
    "tiny": CMFPreset(
        name="tiny",
        model_type="parallel",
        config=CMFConfig(
            vocab_size=256,
            d_model=64,
            hidden_dim=128,
            num_layers=2,
            num_slots=8,
            solver_steps=4,
            thinking_steps=4,
            dropout=0.0,
            tie_embeddings=False,
        ),
        description="Minimal model for unit tests and CI.",
    ),

    # ── Ablation scale ───────────────────────────────────────────────────
    "50m": CMFPreset(
        name="50m",
        model_type="parallel",
        config=CMFConfig(
            vocab_size=50257,
            d_model=512,
            hidden_dim=2048,
            num_layers=6,
            num_slots=32,
            solver_steps=6,
            dropout=0.1,
            routing_mode="sparse_topk",
            routing_topk=16,
        ),
        description="50M ablation model. Fast iteration, fits on 6GB GPU.",
    ),

    # ── Primary proof-of-concept ─────────────────────────────────────────
    "120m": CMFPreset(
        name="120m",
        model_type="parallel",
        config=CMFConfig(
            vocab_size=50257,
            d_model=768,
            hidden_dim=3072,
            num_layers=8,
            num_slots=64,
            solver_steps=8,
            dropout=0.1,
            routing_mode="sparse_topk",
            routing_topk=16,
        ),
        description="120M parallel CMF. Primary training target.",
    ),

    # ── Deliberative variant of 120m ─────────────────────────────────────
    "120m-deliberative": CMFPreset(
        name="120m-deliberative",
        model_type="deliberative",
        config=CMFConfig(
            vocab_size=50257,
            d_model=768,
            hidden_dim=3072,
            num_layers=8,
            num_slots=64,
            solver_steps=6,
            thinking_steps=8,
            adaptive_thinking=True,
            min_thinking_steps=2,
            max_thinking_steps=16,
            halting_threshold=0.5,
            dropout=0.1,
            routing_mode="sparse_topk",
            routing_topk=16,
        ),
        description="120M deliberative CMF. Tests iterative reasoning hypothesis.",
    ),

    # ── Next scale step ──────────────────────────────────────────────────
    "500m": CMFPreset(
        name="500m",
        model_type="parallel",
        config=CMFConfig(
            vocab_size=50257,
            d_model=1024,
            hidden_dim=4096,
            num_layers=16,
            num_slots=128,
            solver_steps=8,
            dropout=0.1,
            routing_mode="sparse_topk",
            routing_topk=32,
        ),
        description="500M parallel CMF. Only attempt after 120m experiments pass.",
    ),
}


def get_preset(name: str) -> CMFPreset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"Unknown preset '{name}'. Available: {', '.join(sorted(PRESETS))}")


def build_model(name: str):
    """Return an instantiated model from a preset name."""
    from .model import ParallelCMF, DeliberativeCMF
    preset = get_preset(name)
    if preset.model_type == "deliberative":
        return DeliberativeCMF(preset.config)
    return ParallelCMF(preset.config)


def _encoder_params(config: CMFConfig) -> int:
    total = 0
    kernel_size = getattr(config, "kernel_size", 3)
    for _ in range(config.num_layers):
        total += (config.hidden_dim * 2) * config.d_model * kernel_size
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
    if model_type in {"continuous_cmf", "parallel_cmf", "deliberative_cmf", "parallel", "deliberative"}:
        time_dim = 16
        field_in = config.d_model * 4 + time_dim
        total += 32 * config.d_model
        total += config.hidden_dim * config.d_model + config.hidden_dim
        total += config.hidden_dim * field_in + config.hidden_dim
        total += config.hidden_dim * config.hidden_dim + config.hidden_dim
        total += config.d_model * config.hidden_dim + config.d_model
        total += config.d_model * config.hidden_dim + config.d_model
        if model_type in {"deliberative_cmf", "deliberative"}:
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

