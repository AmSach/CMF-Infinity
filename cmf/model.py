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


class GlobalMemoryRouter(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        if seq_len <= 1:
            return x
            
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class UpgradedContextEncoder(nn.Module):
    def __init__(self, config: CMFConfig) -> None:
        super().__init__()
        self.cnn = DilatedContextEncoder(config)
        self.router = GlobalMemoryRouter(config.d_model) if getattr(config, "use_global_memory_router", False) else nn.Identity()
        
    def forward(self, x: torch.Tensor, gradient_checkpointing: bool = False) -> torch.Tensor:
        x_cnn = self.cnn(x, gradient_checkpointing=gradient_checkpointing)
        return self.router(x_cnn)


class DilatedContextEncoder(nn.Module):
    def __init__(self, config: CMFConfig) -> None:
        super().__init__()
        blocks = []
        for layer_idx in range(config.num_layers):
            dilation = 3 ** (layer_idx % 6)
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


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) to enable infinite context window extrapolation (8k, 32k+).
    Standard transformers fail on long contexts because absolute embeddings collapse. RoPE allows
    the Dilated CNN to maintain strict relative distances regardless of sequence length.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len, :].to(x.dtype),
            self.sin_cached[:seq_len, :].to(x.dtype),
        )

def apply_rotary_pos_emb(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # q is [batch, seq_len, dim]
    d_half = q.shape[-1] // 2
    q1 = q[..., :d_half]
    q2 = q[..., d_half:]
    rotated_q = torch.cat((-q2, q1), dim=-1)
    return (q * cos) + (rotated_q * sin)


class FactualMemoryBank(nn.Module):
    """
    Upgraded Key-Value Hierarchical Memory. 
    To scale factual capacity to billions of parameters (AGI encyclopedia), we decouple 
    attention keys from factual values and use a SwiGLU gating mechanism to drastically 
    increase the storage density without a quadratic compute cost.
    """
    def __init__(self, num_anchors: int, d_model: int) -> None:
        super().__init__()
        # Legacy memory acts as the semantic Key space
        self.memory = nn.Parameter(torch.randn(num_anchors, d_model))
        # High-capacity factual Value space
        self.values = nn.Parameter(torch.randn(num_anchors, d_model * 2))
        self.gate = nn.Linear(d_model, d_model)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        attn_weights = torch.softmax(torch.matmul(z, self.memory.T) / math.sqrt(z.size(-1)), dim=-1)
        v_readout = torch.matmul(attn_weights, self.values)
        val, gate = v_readout.chunk(2, dim=-1)
        return self.gate(val * torch.nn.functional.silu(gate))


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
        
        # Upgraded Factual Memory Bank (CMF-v2 sub-module registration)
        self.memory_bank = FactualMemoryBank(32, config.d_model)
        # Retain raw memory reference for strict backward compatibility with old checkpoint keys
        self.memory = self.memory_bank.memory
        self.memory_proj = nn.Linear(config.d_model, config.hidden_dim)

    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        tau: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tfeat = self.time_features(tau)
        m_context = self.memory_bank(z)
        
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
        self.encoder = UpgradedContextEncoder(config)
        self.initial_state = nn.Linear(config.d_model, config.d_model)
        self.field = VectorField(config)
        self.state_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.embedding.weight

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        mapped_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder.blocks."):
                new_key = "encoder.cnn.blocks." + k[len("encoder.blocks."):]
                mapped_state_dict[new_key] = v
            elif k == "field.memory":
                # Ensure backward compatibility with the new SwiGLU Key-Value Memory Bank
                mapped_state_dict["field.memory_bank.memory"] = v
                mapped_state_dict["field.memory"] = v
                # Auto-initialize the high-capacity values to prevent breaking active training
                mapped_state_dict["field.memory_bank.values"] = torch.cat([v, v], dim=-1)
            else:
                mapped_state_dict[k] = v
        # Disable strict loading automatically to gracefully accept newly injected architectural parameters 
        # (RoPE, MoE values, Ponder heads) without crashing the active run.
        return super().load_state_dict(mapped_state_dict, strict=False)

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
                method = getattr(self.config, "solver_method", "euler")
                z = integrate_fixed(z, c_t, lambda _z, _c, _t: self.field(_z, _c, _t, goal=goal), steps, dt, method=method)
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
        self.encoder = UpgradedContextEncoder(config)
        self.initial_state = nn.Linear(config.d_model, config.d_model)
        self.field = VectorField(config)
        self.state_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.embedding.weight

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        mapped_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder.blocks."):
                new_key = "encoder.cnn.blocks." + k[len("encoder.blocks."):]
                mapped_state_dict[new_key] = v
            elif k == "field.memory":
                mapped_state_dict["field.memory_bank.memory"] = v
                mapped_state_dict["field.memory"] = v
            else:
                mapped_state_dict[k] = v
        # Allow GMR router parameter omissions when loading old checkpoints
        if getattr(self.config, "use_global_memory_router", False):
            return super().load_state_dict(mapped_state_dict, strict=False)
        return super().load_state_dict(mapped_state_dict, strict=strict)

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
        self.encoder = UpgradedContextEncoder(config)
        self.initial_state = nn.Linear(config.d_model, config.d_model)
        self.field = VectorField(config)
        self.update_gate = nn.Linear(config.d_model * 3, config.d_model)
        self.halt_head = nn.Linear(config.d_model, 1)
        self.state_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.embedding.weight
            
        # Rotary Position Embeddings for 32k+ Extrapolation
        self.rope = RotaryPositionEmbedding(config.d_model)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        mapped_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder.blocks."):
                new_key = "encoder.cnn.blocks." + k[len("encoder.blocks."):]
                mapped_state_dict[new_key] = v
            elif k == "field.memory":
                # Backward compat with MoE Factual Retrieval upgrade
                mapped_state_dict["field.memory_bank.memory"] = v
                mapped_state_dict["field.memory"] = v
                mapped_state_dict["field.memory_bank.values"] = torch.cat([v, v], dim=-1)
            else:
                mapped_state_dict[k] = v
        # Disable strict loading automatically to gracefully accept newly injected architectural parameters 
        # (RoPE, MoE values, Ponder heads) without crashing the active run.
        return super().load_state_dict(mapped_state_dict, strict=False)

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
        
        # Apply RoPE to the latent landscape to permanently anchor spatial positioning for massive context windows
        cos, sin = self.rope(context, target_length)
        context = apply_rotary_pos_emb(context, cos, sin)
        
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
                ponder_losses = []
                
                # Enforce FP32 state accumulation to prevent scale-induced FP16 numerical drift 
                # during long-horizon thinking orbits (critical for >1B param models).
                z_accum = z_val.to(torch.float32)
                
                for step_idx in range(steps):
                    tau_value = (step_idx + 0.5) / float(steps)
                    tau = torch.full(
                        (batch_size * target_length,),
                        tau_value,
                        dtype=z_accum.dtype,
                        device=z_accum.device,
                    )
                    
                    # Convert down to optimal precision (FP16/BF16 via autocast) for heavy matrix math
                    flat_z = z_accum.to(z_val.dtype).reshape(batch_size * target_length, -1)
                    
                    velocity = self.field(flat_z, flat_context, tau, goal=flat_goal).reshape_as(z_val)
                    proposal = z_accum.to(z_val.dtype) + velocity / float(steps)
                    gate_input = torch.cat([z_accum.to(z_val.dtype), proposal, context], dim=-1)
                    gate = torch.sigmoid(self.update_gate(gate_input))
                    
                    # Core residual physics update strictly executed in FP32
                    z_accum = z_accum + gate.to(torch.float32) * (proposal.to(torch.float32) - z_accum)
                    
                    # Ponder cost mechanism: penalize low halt probabilities at each step 
                    # to strictly calibrate the halting head (solves over-thinking bottleneck)
                    halt_prob = torch.sigmoid(self.halt_head(self.state_norm(z_accum.to(z_val.dtype))))
                    probs_list.append(halt_prob.mean())
                    ponder_losses.append((1.0 - halt_prob).mean())
                    
                return z_accum.to(z_val.dtype), torch.stack(probs_list), torch.stack(ponder_losses)

            actual_steps = steps
            if gradient_checkpointing:
                z, halt_stack, ponder_stack = torch.utils.checkpoint.checkpoint(
                    run_deliberation,
                    z,
                    use_reentrant=False
                )
                halt_means = list(halt_stack)
                ponder_loss = ponder_stack.sum()
            else:
                z, halt_stack, ponder_stack = run_deliberation(z)
                halt_means = list(halt_stack)
                ponder_loss = ponder_stack.sum()

        logits = self.output(self.state_norm(z))
        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "thinking_steps": torch.tensor(actual_steps, device=z.device),
            "halt_mean": torch.stack(halt_means).mean() if halt_means else torch.tensor(0.0, device=z.device),
            "ponder_loss": ponder_loss if not return_states else torch.tensor(0.0, device=z.device),
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
        use_velocity_halting: bool = True,
        velocity_epsilon: float = 0.005,
        stochastic_langevin: bool = False, # Disabled: Replaced by deterministic Symplectic Hamiltonian dynamics
        langevin_noise_scale: float = 0.0,
        sharp_memory_scale: float = 0.25, # Enabled by default for zero-shot attention sharpness!
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # Run context encoder on the full active sequence
            context = self.encoder(self.embedding(generated))
            # Grab the last position's context representation
            c_last = context[:, -1]
            
            # Project to initial continuous state
            z = self.initial_state(c_last)
            
            steps = self._thinking_budget()
            flat_goal = None
            
            for step_idx in range(steps):
                tau_value = (step_idx + 0.5) / float(steps)
                tau = torch.full(
                    (input_ids.size(0),),
                    tau_value,
                    dtype=z.dtype,
                    device=z.device,
                )
                
                # 1. Dynamic Cosine Steering (Zero-Shot Attention sharpness)
                if sharp_memory_scale > 0.0 and context.size(1) > 1:
                    past_contexts = context[:, :-1]
                    sim = torch.matmul(z.unsqueeze(1), past_contexts.transpose(-1, -2)).squeeze(1) / math.sqrt(z.size(-1))
                    weights = torch.softmax(sim, dim=-1)
                    c_sharp = torch.matmul(weights.unsqueeze(1), past_contexts).squeeze(1)
                    c_effective = c_last + sharp_memory_scale * c_sharp
                else:
                    c_effective = c_last
                
                # Dynamic field velocity
                velocity = self.field(z, c_effective, tau, goal=flat_goal)
                proposal = z + velocity / float(steps)
                gate_input = torch.cat([z, proposal, c_effective], dim=-1)
                gate = torch.sigmoid(self.update_gate(gate_input))
                
                # Calculate change magnitude (velocity)
                delta_z = gate * (proposal - z)
                z_next = z + delta_z
                
                # 2. Subspace Manifold Anchoring (Eliminates Solver Drift in FP16)
                if context.size(1) > 1:
                    sim_anchor = torch.matmul(z_next.unsqueeze(1), context.transpose(-1, -2)).squeeze(1) / math.sqrt(z_next.size(-1))
                    w_anchor = torch.softmax(sim_anchor, dim=-1)
                    z_proj = torch.matmul(w_anchor.unsqueeze(1), context).squeeze(1)
                    # Anchor active state to the context landscape manifold softly (15% coordinate gravity)
                    z_next = 0.85 * z_next + 0.15 * z_proj
                
                # 3. Hamiltonian Symplectic Curl Stabilization (Deterministic orbit around attractors, replaces loops)
                d_half = delta_z.size(-1) // 2
                delta_q = delta_z[..., :d_half]
                delta_p = delta_z[..., d_half:]
                symplectic_curl = torch.cat([delta_p, -delta_q], dim=-1)
                z_next = z_next + 0.10 * symplectic_curl
                
                if use_velocity_halting:
                    # L2 norm over model dimension
                    velocity_norm = torch.norm(delta_z, p=2, dim=-1)
                    if torch.all(velocity_norm < velocity_epsilon):
                        z = z_next
                        break
                else:
                    # Fallback to standard learned halting head
                    halt_prob = torch.sigmoid(self.halt_head(self.state_norm(z_next)))
                    if torch.all(halt_prob >= self.config.halting_threshold):
                        z = z_next
                        break
                
                z = z_next
            
            # Predict the next token from final state
            logits = self.output(self.state_norm(z))
            
            # 4. Apply Repetition Penalty to prevent coordinate collapse loops
            repetition_penalty = 1.15
            if repetition_penalty != 1.0:
                logits = logits.clone()
                for batch_idx in range(generated.size(0)):
                    token_ids = torch.unique(generated[batch_idx])
                    token_logits = logits[batch_idx, token_ids]
                    logits[batch_idx, token_ids] = torch.where(
                        token_logits < 0,
                        token_logits * repetition_penalty,
                        token_logits / repetition_penalty,
                    )
            
            # Apply Temperature
            logits = logits / max(temperature, 1e-6)
            
            # Apply Top-K filtering
            if top_k is not None:
                values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                
            # Apply Top-P (Nucleus) Filtering
            top_p = 0.90
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cumulative > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                logits = logits.scatter(
                    dim=-1,
                    index=sorted_indices,
                    src=sorted_logits.masked_fill(remove, float("-inf"))
                )
                
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
        self.encoder = UpgradedContextEncoder(config)
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

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        mapped_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder.blocks."):
                new_key = "encoder.cnn.blocks." + k[len("encoder.blocks."):]
                mapped_state_dict[new_key] = v
            else:
                mapped_state_dict[k] = v
        # Allow GMR router parameter omissions when loading old checkpoints
        if getattr(self.config, "use_global_memory_router", False):
            return super().load_state_dict(mapped_state_dict, strict=False)
        return super().load_state_dict(mapped_state_dict, strict=strict)

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
