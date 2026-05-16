from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .config import TrainingConfig


@dataclass
class TrainState:
    step: int = 0
    tokens: int = 0


def train_with_gradient_accumulation(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
) -> TrainState:
    model.train()
    state = TrainState()
    optimizer.zero_grad(set_to_none=True)

    for micro_step, (input_ids, labels) in enumerate(batches, start=1):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        output = model(input_ids, labels=labels)
        loss = output["loss"] / config.gradient_accumulation_steps
        loss.backward()
        state.tokens += int(input_ids.numel())

        if micro_step % config.gradient_accumulation_steps == 0:
            if config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state.step += 1

            if state.step >= config.max_steps:
                break

    return state

