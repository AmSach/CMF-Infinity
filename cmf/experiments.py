from __future__ import annotations

import json
import math
import os
import random
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from .runtime import peak_memory_mb, reset_peak_memory, synchronize_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class TrainReport:
    model_name: str
    steps: int
    micro_batches: int
    tokens: int
    initial_loss: float
    final_loss: float
    best_loss: float
    elapsed_sec: float
    parameters: int

    @property
    def loss_ratio(self) -> float:
        return self.final_loss / max(self.initial_loss, 1e-12)

    def to_dict(self) -> dict[str, float | int | str]:
        data = asdict(self)
        data["loss_ratio"] = self.loss_ratio
        return data


def _loss_to_float(output: dict[str, torch.Tensor]) -> float:
    loss = output.get("loss")
    if loss is None:
        raise ValueError("model output did not include loss")
    return float(loss.detach().cpu())


def evaluate_loss(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for input_ids, labels in batches:
            output = model(input_ids.to(device), labels=labels.to(device))
            losses.append(_loss_to_float(output))
    if not losses:
        raise ValueError("evaluate_loss requires at least one batch")
    return float(sum(losses) / len(losses))


def train_fixed_steps(
    model_name: str,
    model: nn.Module,
    train_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    eval_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    steps: int,
    grad_accum_steps: int = 1,
    clip_grad_norm: float = 1.0,
    use_amp: bool = False,
) -> TrainReport:
    model.to(device)
    initial_loss = evaluate_loss(model, eval_batches, device)
    model.train()
    start = time.perf_counter()
    losses = []
    tokens = 0
    micro_batches = 0
    batch_iter = iter(train_batches)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        for _accum_idx in range(grad_accum_steps):
            input_ids, labels = next(batch_iter)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp and device.type == "cuda",
            ):
                output = model(input_ids, labels=labels)
                raw_loss = output["loss"]
                loss = raw_loss / grad_accum_steps
            scaler.scale(loss).backward()
            step_losses.append(float(raw_loss.detach().cpu()))
            tokens += int(input_ids.numel())
            micro_batches += 1
        if clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(sum(step_losses) / len(step_losses)))

    final_loss = evaluate_loss(model, eval_batches, device)
    elapsed = time.perf_counter() - start
    return TrainReport(
        model_name=model_name,
        steps=steps,
        micro_batches=micro_batches,
        tokens=tokens,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=min(losses + [final_loss]),
        elapsed_sec=elapsed,
        parameters=count_parameters(model),
    )


def benchmark_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None,
    device: torch.device,
    iterations: int = 5,
    warmup: int = 1,
) -> dict[str, float]:
    model.to(device)
    model.eval()
    input_ids = input_ids.to(device)
    labels = labels.to(device) if labels is not None else None

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids, labels=labels)

    reset_peak_memory(device)
    synchronize_device(device)

    tracemalloc.start()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            output = model(input_ids, labels=labels)
            logits = output["logits"]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite logits in benchmark")
    synchronize_device(device)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tokens = int(input_ids.numel() * iterations)
    result = {
        "elapsed_sec": elapsed,
        "iterations": float(iterations),
        "tokens": float(tokens),
        "tokens_per_sec": tokens / max(elapsed, 1e-12),
        "ms_per_iteration": elapsed * 1000.0 / max(iterations, 1),
        "python_peak_memory_mb": peak / (1024.0 * 1024.0),
    }
    cuda_peak = peak_memory_mb(device)
    if cuda_peak is not None:
        result["cuda_peak_memory_mb"] = cuda_peak
    return result


def finite_dict_values(data: dict) -> bool:
    for value in data.values():
        if isinstance(value, dict):
            if not finite_dict_values(value):
                return False
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and not finite_dict_values(item):
                    return False
                if isinstance(item, float) and not math.isfinite(item):
                    return False
        elif isinstance(value, float) and not math.isfinite(value):
            return False
    return True


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown_report(path: Path, title: str, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append("```json")
    lines.append(json.dumps(data, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def environment_report() -> dict[str, str | int | bool | None]:
    return {
        "cwd": os.getcwd(),
        "python": os.sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
