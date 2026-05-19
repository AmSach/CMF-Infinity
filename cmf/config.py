from dataclasses import dataclass


@dataclass
class CMFConfig:
    vocab_size: int = 256
    d_model: int = 128
    hidden_dim: int = 256
    num_layers: int = 6
    kernel_size: int = 3
    dropout: float = 0.1
    solver_steps_per_token: int = 4
    solver_method: str = "euler"
    max_seq_len: int = 256
    causal: bool = True
    tie_embeddings: bool = True
    # Adaptive Flow settings
    adaptive_steps: bool = False
    min_solver_steps: int = 1
    max_solver_steps: int = 4
    curvature_threshold: float = 0.05
    # Deliberative latent refinement settings
    thinking_steps: int = 2
    adaptive_thinking: bool = False
    min_thinking_steps: int = 1
    max_thinking_steps: int = 4
    halting_threshold: float = 0.85
    # CMF-v2 upgrades
    use_global_memory_router: bool = False


@dataclass
class TrainingConfig:
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 1000
    clip_grad_norm: float = 1.0
    log_every: int = 25

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps
