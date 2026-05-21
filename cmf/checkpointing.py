from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .baselines import TinyGPTLM
from .config import CMFConfig
from .data import ByteTokenizer
from .model import (
    ContinuousMeaningField,
    DeliberativeContinuousMeaningField,
    ParallelContinuousMeaningField,
)
from .tokenizer import SimpleBPETokenizer


CHECKPOINT_FORMAT = "cmf.model_package.v1"

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "continuous_cmf": ContinuousMeaningField,
    "parallel_cmf": ParallelContinuousMeaningField,
    "deliberative_cmf": DeliberativeContinuousMeaningField,
    "tiny_gpt": TinyGPTLM,
}


def config_to_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    raise TypeError(f"Unsupported config type: {type(config).__name__}")


def tokenizer_to_spec(tokenizer: Any, *, name: str | None = None) -> dict[str, Any]:
    if isinstance(tokenizer, ByteTokenizer):
        return {"type": "byte", "vocab_size": tokenizer.vocab_size}
    if isinstance(tokenizer, SimpleBPETokenizer):
        return {
            "type": "simple_bpe",
            "vocab_size": tokenizer.vocab_size,
            "vocab": tokenizer.vocab,
            "merges": tokenizer.merges,
        }
    if name is not None:
        vocab_size = getattr(tokenizer, "vocab_size", None)
        return {"type": "hf_auto", "name": name, "vocab_size": vocab_size}
    raise TypeError(
        "Tokenizer metadata is required. Pass a ByteTokenizer, SimpleBPETokenizer, "
        "or an explicit Hugging Face tokenizer name."
    )


def tokenizer_from_spec(spec: dict[str, Any]) -> Any:
    kind = spec.get("type")
    if kind == "byte":
        return ByteTokenizer(vocab_size=int(spec.get("vocab_size", 256)))
    if kind == "simple_bpe":
        tokenizer = SimpleBPETokenizer(vocab_size=int(spec["vocab_size"]))
        tokenizer.vocab = dict(spec["vocab"])
        tokenizer.merges = dict(spec["merges"])
        tokenizer.token_to_id = {value: key for key, value in tokenizer.vocab.items()}
        return tokenizer
    if kind == "hf_auto":
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "This checkpoint uses a Hugging Face tokenizer. Install transformers "
                "or load with a locally reconstructed tokenizer."
            ) from exc
        return AutoTokenizer.from_pretrained(str(spec["name"]))
    raise ValueError(f"Unsupported tokenizer spec: {kind!r}")


def _make_model(model_type: str, config: dict[str, Any]) -> nn.Module:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Known: {sorted(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[model_type]
    if model_type == "tiny_gpt":
        return cls(**config)
    return cls(CMFConfig(**config))


def save_model_package(
    path: str | Path,
    model: nn.Module,
    *,
    model_type: str,
    config: Any,
    tokenizer: Any,
    tokenizer_name: str | None = None,
    training: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    config_dict = config_to_dict(config)
    tokenizer_spec = tokenizer_to_spec(tokenizer, name=tokenizer_name)
    package = {
        "format": CHECKPOINT_FORMAT,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "model_type": model_type,
        "config": config_dict,
        "tokenizer": tokenizer_spec,
        "state_dict": model.state_dict(),
        "training": training or {},
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, path)


def load_model_package(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    expected_model_type: str | None = None,
    strict: bool = True,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a CMF model package. Expected format {CHECKPOINT_FORMAT}. "
            "Legacy raw state_dict files must be loaded with an explicit config and tokenizer."
        )
    model_type = str(payload["model_type"])
    if expected_model_type is not None and model_type != expected_model_type:
        raise ValueError(f"Expected model_type {expected_model_type}, found {model_type}.")
    model = _make_model(model_type, dict(payload["config"]))
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device)
    tokenizer = tokenizer_from_spec(dict(payload["tokenizer"]))
    return model, tokenizer, payload


def load_legacy_state_dict(
    path: str | Path,
    *,
    model_type: str,
    config: Any,
    device: torch.device | str,
    strict: bool = True,
) -> nn.Module:
    model = _make_model(model_type, config_to_dict(config))
    state = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"Legacy checkpoint {path} did not contain a state_dict.")
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Legacy checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device)


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and payload.get("format") == CHECKPOINT_FORMAT:
        state = payload["state_dict"]
        return {
            "format": payload["format"],
            "model_type": payload["model_type"],
            "config": payload["config"],
            "tokenizer": {
                key: value
                for key, value in payload["tokenizer"].items()
                if key not in {"vocab", "merges"}
            },
            "parameters": sum(t.numel() for t in state.values() if torch.is_tensor(t)),
        }
    if isinstance(payload, dict):
        return {
            "format": "legacy_state_dict_or_training_checkpoint",
            "keys": list(payload.keys())[:20],
            "parameters": sum(t.numel() for t in payload.values() if torch.is_tensor(t)),
        }
    return {"format": type(payload).__name__}
