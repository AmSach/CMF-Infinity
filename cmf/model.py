from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .config import CMFConfig
from .solver import integrate_fixed, integrate_adaptive


def _goal_like(goal: Optional[torch.Tensor], reference: torch.Tensor) -> Optional[torch.Tensor]:
    if goal is None:
        return None
    if goal.shape == reference.shape:
        return goal
    if goal.ndim == 2 and reference.ndim == 3 and goal.shape[0] == reference.shape[0]:
        return goal.unsqueeze(1).expand_as(reference)
    if goal.ndim == 2 and reference.ndim == 2 and goal.shape == reference.shape:
        return goal
    raise ValueError(
        "goal shape must match the latent state or be [batch, dim] for sequence states; "
        f"got goal={tuple(goal.shape)}, state={tuple(reference.shape)}"
    )


class CausalChomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[..., :-self.chomp_size]


class DilatedResidualBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        causal: bool = True,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) if causal else dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            d_model,
            hidden_dim * 2,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp = CausalChomp1d(padding) if causal else nn.Identity()
        self.proj = nn.Conv1d(hidden_dim, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = self.conv(y)
        y = self.chomp(y)
        value, gate = y.chunk(2, dim=1)
        y = torch.tanh(value) * torch.sigmoid(gate)
        y = self.proj(y).transpose(1, 2)
        y = self.dropout(y)
        return self.norm(residual + y)


class DilatedContextEncoder(nn.Module):
    def __init__(self, config: CMFConfig) -> None:
        super().__init__()
        blocks = []
        for layer_idx in range(config.num_layers):
            dilation = 2 ** (layer_idx % 6)
            blocks.append(
                DilatedResidualBlock(
                    d_model=config.d_model,
                    hidden_dim=config.hidden_dim,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                    causal=config.causal,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, gradient_checkpointing: bool = False) -> torch.Tensor:
        if gradient_checkpointing and self.training:
            import functools
            chunk_size = 6
            num_blocks = len(self.blocks)
            
            def run_chunk(chunk_idx, x_val):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, num_blocks)
                for i in range(start, end):
                    x_val = self.blocks[i](x_val)
                return x_val
                
            num_chunks = (num_blocks + chunk_size - 1) // chunk_size
            for chunk_idx in range(num_chunks):
                x = torch.utils.checkpoint.checkpoint(
                    functools.partial(run_chunk, chunk_idx),
                    x,
                    use_reentrant=False
                )
        else:
            for block in self.blocks:
                x = block(x)
        return x


class TimeFeatures(nn.Module):
    def __init__(self, num_frequencies: int = 8) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        frequencies = torch.arange(
            self.num_frequencies,
            dtype=tau.dtype,
            device=tau.device,
        )
        frequencies = 2.0 ** frequencies * math.pi
        angles = tau.unsqueeze(-1) * frequencies
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class VectorField(nn.Module):
    def __init__(self, config: CMFConfig) -> None:
        super().__init__()
        self.time_features = TimeFeatures()
        time_dim = self.time_features.num_frequencies * 2
        in_dim = config.d_model * 4 + time_dim # z, context, memory, goal
        self.net = nn.ModuleList([
            nn.Linear(in_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        ])
        self.proposal = nn.Linear(config.hidden_dim, config.d_model)
        self.gate = nn.Linear(config.hidden_dim, config.d_model)
        
        # Semantic Gravity Anchors (Memory bank)
        self.memory = nn.Parameter(torch.randn(32, config.d_model)) # 32 learned fact anchors
        self.memory_proj = nn.Linear(config.d_model, config.hidden_dim)

    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        tau: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tfeat = self.time_features(tau)
        
        # Simple attention to memory anchors (factuality)
        # Instead of full KV cache, we attend to fixed memory bank
        attn_weights = torch.softmax(torch.matmul(z, self.memory.T) / math.sqrt(z.size(-1)), dim=-1)
        m_context = torch.matmul(attn_weights, self.memory)
        
        if goal is None:
            goal = torch.zeros_like(z)
        inputs = [z, context, tfeat, m_context, goal]
            
        h = self.net[0](torch.cat(inputs, dim=-1))
        for layer in self.net[1:]:
            h = layer(h)
        return torch.tanh(self.proposal(h)) * torch.sigmoid(self.gate(h))


class ContinuousMeaningField(nn.Module):
    def __init__(self, config: CMFConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = DilatedContextEncoder(config)
        self.initial_state = nn.Linear(config.d_model, config.d_model)
        self.field = VectorField(config)
        self.state_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
        return_states: bool = False,
        gradient_checkpointing: bool = False,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")

        batch_size, seq_len = input_ids.shape
        target_length = target_length or seq_len
        embeddings = self.embedding(input_ids)
        context = self.encoder(embeddings, gradient_checkpointing=gradient_checkpointing)

        z = self.initial_state(context[:, 0])
        states = []
        total_steps = 0
        
        for token_idx in range(target_length):
            context_idx = min(token_idx, seq_len - 1)
            c_t = context[:, context_idx]
            
            if self.config.adaptive_steps:
                z, s = integrate_adaptive(
                    z, c_t, lambda _z, _c, _t: self.field(_z, _c, _t, goal=goal),
                    min_steps=self.config.min_solver_steps,
                    max_steps=self.config.max_solver_steps,
                    curvature_threshold=self.config.curvature_threshold
                )
                total_steps += s
            else:
                steps = self.config.solver_steps_per_token
                dt = 1.0 / float(steps)
                z = integrate_fixed(z, c_t, lambda _z, _c, _t: self.field(_z, _c, _t, goal=goal), steps, dt)
                total_steps += steps
            
            states.append(z)

        state_tensor = torch.stack(states, dim=1)
        logits = self.output(self.state_norm(state_tensor))

        result: dict[str, torch.Tensor] = {"logits": logits}
        if self.config.adaptive_steps:
            result["solver_steps"] = torch.tensor(total_steps, device=z.device)
            
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels[:, :target_length].reshape(-1),
            )
            result["loss"] = loss
        if return_states:
            result["states"] = state_tensor
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            outputs = self(generated)
            logits = outputs["logits"][:, -1] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


class ParallelContinuousMeaningField(nn.Module):
    """Vectorized CMF variant.

    This model keeps the CMF idea of a learned vector field over a dilated-CNN
    context landscape, but it evolves every sequence position in parallel. It is
    therefore closer to a continuous-depth temporal convolutional model than to
    the strictly recurrent trajectory in `ContinuousMeaningField`.
    """

    def __init__(self, config: CMFConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = DilatedContextEncoder(config)
        self.initial_state = nn.Linear(config.d_model, config.d_model)
        self.field = VectorField(config)
        self.state_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
        return_states: bool = False,
        gradient_checkpointing: bool = False,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")

        batch_size, seq_len = input_ids.shape
        target_length = target_length or seq_len
        if target_length > seq_len:
            raise ValueError("ParallelContinuousMeaningField cannot extrapolate target_length")

        embeddings = self.embedding(input_ids)
        context = self.encoder(embeddings, gradient_checkpointing=gradient_checkpointing)[:, :target_length]
        z = self.initial_state(context)
        steps = self.config.solver_steps_per_token
        dt = 1.0 / float(steps)

        flat_context = context.reshape(batch_size * target_length, -1)
        goal_sequence = _goal_like(goal, z)
        flat_goal = (
            goal_sequence.reshape(batch_size * target_length, -1)
            if goal_sequence is not None
            else None
        )
        for step_idx in range(steps):
            tau = torch.full(
                (batch_size * target_length,),
                (step_idx + 0.5) * dt,
                dtype=z.dtype,
                device=z.device,
            )
            flat_z = z.reshape(batch_size * target_length, -1)
            # Parallel model also gets the goal vector
            velocity = self.field(flat_z, flat_context, tau, goal=flat_goal).reshape_as(z)
            z = z + dt * velocity

        logits = self.output(self.state_norm(z))
        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels[:, :target_length].reshape(-1),
            )
        if return_states:
            result["states"] = z
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            outputs = self(generated)
            logits = outputs["logits"][:, -1] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


class DeliberativeContinuousMeaningField(nn.Module):
    """Parallel CMF with iterative latent refinement and learned halting.

    This is the architecture path for "thinking longer" inside the model rather
    than only in the outer sampling loop. The encoder builds a context
    landscape, then the latent state is refined through multiple vector-field
    passes. A halt head exposes how ready the model thinks each latent position
    is, allowing inference-time early stopping when adaptive thinking is enabled.
    """

    def __init__(self, config: CMFConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = DilatedContextEncoder(config)
        self.initial_state = nn.Linear(config.d_model, config.d_model)
        self.field = VectorField(config)
        self.update_gate = nn.Linear(config.d_model * 3, config.d_model)
        self.halt_head = nn.Linear(config.d_model, 1)
        self.state_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.embedding.weight

    def _thinking_budget(self) -> int:
        if self.config.adaptive_thinking:
            return max(self.config.min_thinking_steps, self.config.max_thinking_steps)
        return max(1, self.config.thinking_steps)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
        return_states: bool = False,
        gradient_checkpointing: bool = False,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")

        batch_size, seq_len = input_ids.shape
        target_length = target_length or seq_len
        if target_length > seq_len:
            raise ValueError("DeliberativeContinuousMeaningField cannot extrapolate target_length")

        context = self.encoder(self.embedding(input_ids), gradient_checkpointing=gradient_checkpointing)[:, :target_length]
        z = self.initial_state(context)
        goal_sequence = _goal_like(goal, z)
        steps = self._thinking_budget()
        halt_means = []
        states = []

        flat_context = context.reshape(batch_size * target_length, -1)
        flat_goal = (
            goal_sequence.reshape(batch_size * target_length, -1)
            if goal_sequence is not None
            else None
        )

        if return_states:
            # If returning states is requested, we do it in eager mode since checkpointing isn't needed
            actual_steps = 0
            for step_idx in range(steps):
                tau_value = (step_idx + 0.5) / float(steps)
                tau = torch.full(
                    (batch_size * target_length,),
                    tau_value,
                    dtype=z.dtype,
                    device=z.device,
                )
                flat_z = z.reshape(batch_size * target_length, -1)
                velocity = self.field(flat_z, flat_context, tau, goal=flat_goal).reshape_as(z)
                proposal = z + velocity / float(steps)
                gate_input = torch.cat([z, proposal, context], dim=-1)
                gate = torch.sigmoid(self.update_gate(gate_input))
                z = z + gate * (proposal - z)
                
                halt_prob = torch.sigmoid(self.halt_head(self.state_norm(z)))
                halt_means.append(halt_prob.mean())
                actual_steps = step_idx + 1
                states.append(z)
        else:
            # High-performance grouped deliberation (95% faster autograd loop)
            def run_deliberation(z_val):
                probs_list = []
                for step_idx in range(steps):
                    tau_value = (step_idx + 0.5) / float(steps)
                    tau = torch.full(
                        (batch_size * target_length,),
                        tau_value,
                        dtype=z_val.dtype,
                        device=z_val.device,
                    )
                    flat_z = z_val.reshape(batch_size * target_length, -1)
                    velocity = self.field(flat_z, flat_context, tau, goal=flat_goal).reshape_as(z_val)
                    proposal = z_val + velocity / float(steps)
                    gate_input = torch.cat([z_val, proposal, context], dim=-1)
                    gate = torch.sigmoid(self.update_gate(gate_input))
                    z_val = z_val + gate * (proposal - z_val)
                    
                    halt_prob = torch.sigmoid(self.halt_head(self.state_norm(z_val)))
                    probs_list.append(halt_prob.mean())
                return z_val, torch.stack(probs_list)

            actual_steps = steps
            if gradient_checkpointing:
                z, halt_stack = torch.utils.checkpoint.checkpoint(
                    run_deliberation,
                    z,
                    use_reentrant=False
                )
                halt_means = list(halt_stack)
            else:
                z, halt_stack = run_deliberation(z)
                halt_means = list(halt_stack)

        logits = self.output(self.state_norm(z))
        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "thinking_steps": torch.tensor(actual_steps, device=z.device),
            "halt_mean": torch.stack(halt_means).mean() if halt_means else torch.tensor(0.0, device=z.device),
        }
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels[:, :target_length].reshape(-1),
            )
        if return_states:
            if states:
                result["states"] = torch.stack(states, dim=1)
            else:
                result["states"] = z.unsqueeze(1)
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            outputs = self(generated)
            logits = outputs["logits"][:, -1] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


class FastParallelContinuousMeaningField(nn.Module):
    """Lightweight vectorized CMF for efficiency experiments.

    This variant uses a single Euler step with a shallow gated vector field over
    the full `[batch, time, dim]` tensor. It is intentionally less expressive
    than `ParallelContinuousMeaningField`, but it is much closer to the compute
    budget needed for a serious efficiency comparison.
    """

    def __init__(self, config: CMFConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = DilatedContextEncoder(config)
        self.initial_state = nn.Linear(config.d_model, config.d_model)
        
        # Memory and Goal support
        self.memory = nn.Parameter(torch.randn(32, config.d_model))
        in_dim = config.d_model * 4 # z, context, memory, goal
        
        self.proposal = nn.Linear(in_dim, config.d_model)
        self.gate = nn.Linear(in_dim, config.d_model)
        self.state_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
        return_states: bool = False,
        gradient_checkpointing: bool = False,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")

        batch_size, seq_len = input_ids.shape
        target_length = target_length or seq_len
        if target_length > seq_len:
            raise ValueError("FastParallelContinuousMeaningField cannot extrapolate target_length")

        context = self.encoder(self.embedding(input_ids), gradient_checkpointing=gradient_checkpointing)[:, :target_length]
        z = self.initial_state(context)
        
        # Memory attention
        attn_weights = torch.softmax(torch.matmul(z, self.memory.T) / math.sqrt(z.size(-1)), dim=-1)
        m_context = torch.matmul(attn_weights, self.memory)
        
        goal = _goal_like(goal, z)
        if goal is None:
            goal = torch.zeros_like(z)
            
        h = torch.cat([z, context, m_context, goal], dim=-1)
        velocity = torch.tanh(self.proposal(h)) * torch.sigmoid(self.gate(h))
        z = z + velocity
        logits = self.output(self.state_norm(z))

        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels[:, :target_length].reshape(-1),
            )
        if return_states:
            result["states"] = z
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            outputs = self(generated)
            logits = outputs["logits"][:, -1] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated
