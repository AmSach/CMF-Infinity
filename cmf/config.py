from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CMFConfig:
    # ── Vocabulary / embedding ──────────────────────────────────────────
    vocab_size: int = 50257
    d_model: int = 512
    tie_embeddings: bool = True

    # ── Dilated CNN encoder ─────────────────────────────────────────────
    num_layers: int = 6
    hidden_dim: int = 2048        # conv channel width (gated, so *2 inside)
    kernel_size: int = 3
    causal: bool = True
    dropout: float = 0.1

    # ── Slot memory ─────────────────────────────────────────────────────
    # Fixed capacity — O(num_slots), NOT O(seq_len)
    num_slots: int = 64

    # ── Solver ─────────────────────────────────────────────────────────
    solver_steps: int = 8          # default fixed steps
    solver_method: str = "euler"   # "euler" | "rk4"
    adaptive_solver: bool = False
    min_solver_steps: int = 2
    max_solver_steps: int = 16
    curvature_threshold: float = 0.05

    # ── Deliberation (thinking steps) ──────────────────────────────────
    thinking_steps: int = 8
    adaptive_thinking: bool = False
    min_thinking_steps: int = 2
    max_thinking_steps: int = 16
    halting_threshold: float = 0.5

    # ── Routing mode (ablation axis) ────────────────────────────────────
    # "full" | "sparse_topk" | "local_window" | "none"
    routing_mode: str = "sparse_topk"
    routing_topk: int = 16
    routing_window: int = 64

    # ── Misc ─────────────────────────────────────────────────────────────
    max_seq_len: int = 2048


@dataclass
class TrainingConfig:
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 10_000
    warmup_steps: int = 500
    clip_grad_norm: float = 1.0
    log_every: int = 50
    eval_every: int = 500
    use_amp: bool = False

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps
