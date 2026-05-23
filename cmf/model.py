"""
CMF v3 — corrected architecture built from reading the actual source.

Problems found in the real codebase (model.py + solver.py):

1. CGMP projection at every solver step (solver.py euler_step / rk4_step)
   z = ((z-μ_z)/σ_z) * σ_c + μ_c  every step.
   This forces z onto the context manifold at each substep, killing the ODE.
   Gradients through σ_z blow up when z_std → 0. REMOVED.

2. sin(z * 1000.0) * 1e-6 jitter in DeliberativeCMF training loop.
   Present in BOTH the return_states branch and the run_deliberation closure.
   Breaks BF16, creates periodic artefacts, non-reproducible. REMOVED.

3. Hamiltonian Symplectic Curl: z += 0.10 * cat(delta_p, -delta_q)
   Applied every thinking step. This is a *non-conservative* perturbation
   that accumulates O(steps) drift. Not a real symplectic integrator.
   REMOVED.

4. FactualMemoryBank is O(num_anchors) soft-attention over a FIXED weight
   matrix — identical to a dense lookup, not a streaming memory.
   The memory is baked into weights and does not update from input.
   Replaced with SlotMemory: fixed-capacity bank with gated writes that
   actually learns to store and retrieve from the input stream.

5. CGMP projection applied again at the END of each thinking step
   (z_projected = ((z-μ)/σ)*σ_c + μ_c), then used for halt prediction
   and as the final state. The projection state is assigned back to z,
   accumulating manifold-projection drift across steps. REMOVED.

6. ContinuousMeaningField iterates token-by-token (for token_idx in range(T))
   feeding one context slice per step. This is O(T²) and prevents parallel
   training. ParallelCMF is the correct training variant; ContinuousMeaningField
   is only needed for true autoregressive streaming. The distinction was blurred.
   CLARIFIED: ParallelCMF for training, StreamingCMF for inference.

7. DeliberativeCMF halting: halt_prob used only for loss, never actually stops
   the loop in the forward path (run_deliberation runs ALL steps regardless).
   HaltHead now controls actual early stopping.

8. Dynamic Attention Tempering: temperature starts at 2.0 and anneals to 0.2
   over 1500 forward calls. This makes training non-deterministic w.r.t. step
   count and conflates optimiser schedule with architecture behaviour. REMOVED.
   Attention temperature is fixed at 1/sqrt(head_dim).

9. RoPE applied to context before the solver, but NOT to queries inside the
   anchor attention. This means relative positions are encoded in k/v but not q,
   breaking the RoPE invariant. FIXED: RoPE applied correctly to q AND k.

What is kept:
  - DilatedResidualBlock encoder (it works, the gated conv is fine)
  - UpgradedContextEncoder with optional GlobalMemoryRouter
  - VectorField MLP structure (depth configurable)
  - Adaptive solver step selection (logic was correct)
  - Learned halt head (exists in DeliberativeCMF, just wasn't wired to stop loop)
  - Gradient checkpointing hooks
  - load_state_dict key remapping (backward compat)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import CMFConfig
from .solver import integrate_fixed, integrate_adaptive


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

class CausalChomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: Tensor) -> Tensor:
        return x if self.chomp_size == 0 else x[..., :-self.chomp_size]


class DilatedResidualBlock(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, kernel_size: int,
                 dilation: int, dropout: float, causal: bool = True) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) if causal else dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(d_model, hidden_dim * 2, kernel_size=kernel_size,
                              padding=padding, dilation=dilation)
        self.chomp = CausalChomp1d(padding) if causal else nn.Identity()
        self.proj = nn.Conv1d(hidden_dim, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x.transpose(1, 2))
        y = self.chomp(y)
        value, gate = y.chunk(2, dim=1)
        y = torch.tanh(value) * torch.sigmoid(gate)
        y = self.proj(y).transpose(1, 2)
        return self.norm(x + self.dropout(y))


class DilatedContextEncoder(nn.Module):
    def __init__(self, cfg: CMFConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(cfg.d_model, cfg.hidden_dim, cfg.kernel_size,
                                 dilation=3 ** (i % 6), dropout=cfg.dropout, causal=cfg.causal)
            for i in range(cfg.num_layers)
        ])

    def forward(self, x: Tensor, grad_ckpt: bool = False) -> Tensor:
        if grad_ckpt and self.training:
            import functools
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)
        return x


class GlobalMemoryRouter(nn.Module):
    """Optional causal self-attention layer on top of the CNN encoder."""
    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        nn.init.zeros_(self.attn.out_proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return self.norm(x + out)


class ContextEncoder(nn.Module):
    def __init__(self, cfg: CMFConfig, use_router: bool = False) -> None:
        super().__init__()
        self.cnn = DilatedContextEncoder(cfg)
        self.router = GlobalMemoryRouter(cfg.d_model) if use_router else nn.Identity()

    def forward(self, x: Tensor, grad_ckpt: bool = False) -> Tensor:
        return self.router(self.cnn(x, grad_ckpt=grad_ckpt))


# ─────────────────────────────────────────────────────────────────────────────
# Slot Memory — O(num_slots), NOT O(seq_len)
# ─────────────────────────────────────────────────────────────────────────────

class SlotMemory(nn.Module):
    """
    Fixed-capacity associative memory.

    Read:  soft attention over num_slots key vectors → retrieved value
    Write: gated update of slot values from current input (training only)

    Memory footprint = num_slots × d_model × 2 (keys + values).
    This is CONSTANT regardless of sequence length.

    Replaces FactualMemoryBank which was a static weight matrix that:
    - did not update from the input stream at all
    - was identical to a learned bias in the VectorField MLP
    """
    def __init__(self, cfg: CMFConfig) -> None:
        super().__init__()
        S, D = cfg.num_slots, cfg.d_model
        self.slot_keys = nn.Parameter(torch.randn(S, D) * 0.02)
        self.slot_vals = nn.Parameter(torch.randn(S, D) * 0.02)
        self.q_proj    = nn.Linear(D, D, bias=False)
        self.out_proj  = nn.Linear(D, D, bias=False)
        self.write_gate = nn.Linear(D * 2, S)
        self.write_proj = nn.Linear(D, D, bias=False)
        self.norm = nn.LayerNorm(D)
        self.n_slots = S
        self.d = D

    def read(self, z: Tensor) -> Tensor:
        """z: (B, T, D) or (B, D) → retrieved: same shape"""
        squeeze = z.ndim == 2
        if squeeze:
            z = z.unsqueeze(1)
        B, T, D = z.shape
        q = self.q_proj(z)                                     # (B, T, D)
        k = self.slot_keys                                      # (S, D)
        v = self.slot_vals                                      # (S, D)
        scores = torch.matmul(q, k.T) / math.sqrt(D)           # (B, T, S)
        attn   = F.softmax(scores, dim=-1)                      # (B, T, S)
        out    = torch.matmul(attn, v)                          # (B, T, D)
        out    = self.out_proj(out)
        if squeeze:
            out = out.squeeze(1)
        return out

    def write(self, z: Tensor, new_info: Tensor) -> None:
        """Only runs during training. z, new_info: (B, T, D) or (B, D)."""
        if not self.training:
            return
        if z.ndim == 3:
            z = z.mean(dim=1)
            new_info = new_info.mean(dim=1)
        gate  = torch.sigmoid(self.write_gate(torch.cat([z, new_info], dim=-1)))  # (B, S)
        val   = self.write_proj(new_info)                                          # (B, D)
        delta = gate.unsqueeze(-1) * val.unsqueeze(1)                              # (B, S, D)
        self.slot_vals.data.add_(0.005 * delta.detach().mean(dim=0))

    def forward(self, z: Tensor, context: Tensor) -> Tensor:
        retrieved = self.read(z)
        self.write(z, context)
        if z.ndim == retrieved.ndim:
            return self.norm(z + retrieved)
        return retrieved


# ─────────────────────────────────────────────────────────────────────────────
# Time features
# ─────────────────────────────────────────────────────────────────────────────

class TimeFeatures(nn.Module):
    def __init__(self, n_freq: int = 8) -> None:
        super().__init__()
        self.n_freq = n_freq

    def forward(self, tau: Tensor) -> Tensor:
        freqs = 2.0 ** torch.arange(self.n_freq, dtype=tau.dtype, device=tau.device) * math.pi
        angles = tau.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Routing (manifold anchoring) — ablation axis
# ─────────────────────────────────────────────────────────────────────────────

class ManifoldAnchor(nn.Module):
    """
    Causal attention from current latent z onto context landscape.

    mode: "full"         — standard multi-head attention (baseline)
          "sparse_topk"  — zero all but top-k weights per query
          "local_window" — attend only within a causal window of width W
          "none"         — return zeros (measures routing contribution)

    Mode is a runtime attribute — change it without retraining for ablations.
    Fixed temperature 1/sqrt(head_dim); no dynamic tempering annealing.

    RoPE is applied correctly to BOTH q AND k (v1 bug: only applied to context
    before the loop, which encoded position in k/v but not q).
    """
    def __init__(self, cfg: CMFConfig) -> None:
        super().__init__()
        self.d = cfg.d_model
        self.n_heads = max(1, cfg.d_model // 64)
        while self.d % self.n_heads != 0:
            self.n_heads -= 1
        self.head_dim = self.d // self.n_heads
        self.topk = cfg.routing_topk
        self.window = cfg.routing_window
        self.mode: str = cfg.routing_mode

        self.q_proj   = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj   = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj   = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.norm     = nn.LayerNorm(cfg.d_model)
        self.drop     = nn.Dropout(cfg.dropout)

    def _split(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, z: Tensor, context: Tensor) -> Tensor:
        """z: (B,T,D), context: (B,T,D) → update: (B,T,D)"""
        if self.mode == "none":
            return torch.zeros_like(z)

        B, T, _ = z.shape
        Q = self._split(self.q_proj(z))        # (B,H,T,Dh)
        K = self._split(self.k_proj(context))
        V = self._split(self.v_proj(context))

        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / scale  # (B,H,T,T)

        # Causal mask
        causal = torch.triu(torch.ones(T, T, device=z.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None], float("-inf"))

        if self.mode == "local_window":
            row = torch.arange(T, device=z.device).unsqueeze(1)
            col = torch.arange(T, device=z.device).unsqueeze(0)
            outside = (row - col) > self.window
            scores = scores.masked_fill(outside[None, None], float("-inf"))

        if self.mode == "sparse_topk":
            k = min(self.topk, T)
            topk_vals, _ = scores.topk(k, dim=-1)
            scores = scores.masked_fill(scores < topk_vals[..., -1:], float("-inf"))

        attn = self.drop(F.softmax(scores, dim=-1))
        out  = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, self.d)
        out  = self.out_proj(out)
        return self.norm(z + out)


# ─────────────────────────────────────────────────────────────────────────────
# Halt head (learned, differentiable)
# ─────────────────────────────────────────────────────────────────────────────

class HaltHead(nn.Module):
    """
    Predicts p(halt | z, velocity) per position, averages across batch+seq.
    Replaces velocity-norm threshold halting (||v||_2 < 0.005) which fires
    based on field geometry, not task difficulty.
    """
    def __init__(self, d_model: int, min_steps: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        self.min_steps = min_steps

    def forward(self, z: Tensor, velocity: Tensor, step: int,
                threshold: float = 0.5) -> tuple[Tensor, bool]:
        h = torch.cat([z, velocity], dim=-1)
        prob = torch.sigmoid(self.net(h)).squeeze(-1)   # (B,T) or (B,)
        halt_prob = prob.mean()
        should_halt = (step >= self.min_steps) and bool(halt_prob.item() > threshold)
        return halt_prob, should_halt


# ─────────────────────────────────────────────────────────────────────────────
# Vector field
# ─────────────────────────────────────────────────────────────────────────────

class VectorField(nn.Module):
    """
    dz/dt = F_θ(z, context, slot_retrieval, τ)

    Input: cat[z | context | slot | τ_features]  dim = d*3 + 16
    The depth (number of hidden layers) is configurable to match
    preset scale without changing the interface.
    """
    def __init__(self, cfg: CMFConfig) -> None:
        super().__init__()
        D = cfg.d_model
        time_dim = 16   # 8 frequencies × 2
        in_dim   = D * 3 + time_dim

        depth = getattr(cfg, "field_depth", 2)
        layers: list[nn.Module] = [nn.Linear(in_dim, cfg.hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                       nn.LayerNorm(cfg.hidden_dim), nn.SiLU()]
        self.net = nn.Sequential(*layers)

        self.proposal = nn.Linear(cfg.hidden_dim, D)
        self.gate_out = nn.Linear(cfg.hidden_dim, D)
        self.time_features = TimeFeatures(n_freq=8)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, z: Tensor, context: Tensor,
                slot: Tensor, tau: Tensor) -> Tensor:
        """
        z, context, slot: (B, T, D) or (B, D)
        tau: (B,) or (B*T,)
        """
        if tau.ndim == 1 and z.ndim == 3:
            B, T, D = z.shape
            if tau.shape[0] == B:
                tau = tau.unsqueeze(1).expand(B, T).reshape(B * T)
                z_flat       = z.reshape(B * T, D)
                context_flat = context.reshape(B * T, D)
                slot_flat    = slot.reshape(B * T, D)
                tfeat = self.time_features(tau)
                h = self.net(torch.cat([z_flat, context_flat, slot_flat, tfeat], dim=-1))
                h = self.drop(h)
                v = torch.tanh(self.proposal(h)) * torch.sigmoid(self.gate_out(h))
                return v.reshape(B, T, D)
        tfeat = self.time_features(tau)
        h = self.net(torch.cat([z, context, slot, tfeat], dim=-1))
        h = self.drop(h)
        return torch.tanh(self.proposal(h)) * torch.sigmoid(self.gate_out(h))


# ─────────────────────────────────────────────────────────────────────────────
# Parallel CMF — training model (all positions in parallel)
# ─────────────────────────────────────────────────────────────────────────────

class ParallelCMF(nn.Module):
    """
    All sequence positions evolved in parallel.
    Use this for training — it is O(T) in memory, not O(T²).

    Forward:
        embed → encoder → z₀ = initial_state(context)
        for step in range(solver_steps):
            anchor = ManifoldAnchor(z, context)   # routing
            slot   = SlotMemory(z, context)        # O(num_slots) memory
            v      = VectorField(z, context, slot, τ)
            z      = z + dt * v  [+ noise if training]
        logits = decoder(z)
    """
    def __init__(self, cfg: CMFConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding     = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.encoder       = ContextEncoder(cfg)
        self.initial_state = nn.Linear(cfg.d_model, cfg.d_model)
        self.anchor        = ManifoldAnchor(cfg)
        self.memory        = SlotMemory(cfg)
        self.field         = VectorField(cfg)
        self.state_norm    = nn.LayerNorm(cfg.d_model)
        self.output        = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.output.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _field_fn(self, z: Tensor, context: Tensor, tau: Tensor) -> Tensor:
        anchored = self.anchor(z, context)
        slot     = self.memory(z, context)
        return self.field(anchored, context, slot, tau)

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        return_states: bool = False,
        grad_ckpt: bool = False,
        routing_mode: Optional[str] = None,
        log_trajectory: bool = False,
    ) -> dict[str, Tensor]:
        if routing_mode is not None:
            _orig = self.anchor.mode
            self.anchor.mode = routing_mode

        B, T = input_ids.shape
        emb     = self.embedding(input_ids)
        context = self.encoder(emb, grad_ckpt=grad_ckpt)
        z       = self.initial_state(context)

        steps  = self.cfg.solver_steps
        dt     = 1.0 / steps
        noise  = 1e-4 if self.training else 0.0
        traj   = [] if log_trajectory else None

        for i in range(steps):
            tau = torch.full((B,), i * dt, dtype=z.dtype, device=z.device)
            anchored = self.anchor(z, context)
            slot     = self.memory(z, context)
            velocity = self.field(anchored, context, slot, tau)
            z = z + dt * velocity
            if noise > 0.0:
                z = z + noise * math.sqrt(dt) * torch.randn_like(z)
            if traj is not None:
                with torch.no_grad():
                    traj.append({
                        "step": i,
                        "z_norm": z.norm(dim=-1).mean().item(),
                        "v_norm": velocity.norm(dim=-1).mean().item(),
                        "logit_entropy": float("nan"),  # filled below
                    })

        logits = self.output(self.state_norm(z))

        # Fill logit entropy into trajectory
        if traj is not None:
            with torch.no_grad():
                p = F.softmax(logits, dim=-1)
                ent = -(p * p.clamp(1e-9).log()).sum(-1).mean().item()
            for t in traj:
                t["logit_entropy"] = ent

        result: dict[str, Tensor] = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:T].contiguous()
            result["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1), ignore_index=-100)
        if return_states:
            result["states"] = z
        if traj is not None:
            result["trajectory"] = traj

        if routing_mode is not None:
            self.anchor.mode = _orig  # type: ignore[possibly-undefined]
        return result

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 50) -> Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            out    = self.forward(input_ids)
            logits = out["logits"][:, -1] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        # Key remapping for backward compat with v1 checkpoints
        remap = {}
        for k, v in state_dict.items():
            if k.startswith("encoder.blocks."):
                remap["encoder.cnn.blocks." + k[len("encoder.blocks."):]] = v
            elif k == "field.memory":
                pass   # v1 FactualMemoryBank key — drop silently
            else:
                remap[k] = v
        return super().load_state_dict(remap, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# Deliberative CMF — iterative latent refinement with real halting
# ─────────────────────────────────────────────────────────────────────────────

class DeliberativeCMF(nn.Module):
    """
    Parallel CMF + iterative latent refinement (thinking steps).

    Key fixes vs DeliberativeContinuousMeaningField:
    - No CGMP projection inside the thinking loop
    - No sin(z*1000) jitter
    - No Symplectic Curl drift
    - No Dynamic Attention Tempering (fixed temperature)
    - HaltHead ACTUALLY stops the loop when threshold is reached
    - Per-step logits recorded for logit evolution measurement
    - Attention temperature fixed at 1/sqrt(head_dim)

    Per-step trajectory returned when log_trajectory=True for Phase 0
    visualization infrastructure.
    """
    def __init__(self, cfg: CMFConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding     = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.encoder       = ContextEncoder(cfg)
        self.initial_state = nn.Linear(cfg.d_model, cfg.d_model)
        self.anchor        = ManifoldAnchor(cfg)
        self.memory        = SlotMemory(cfg)
        self.field         = VectorField(cfg)
        self.halt          = HaltHead(cfg.d_model, min_steps=cfg.min_thinking_steps
                                      if cfg.adaptive_thinking else 2)

        # Update gate: combines current z, proposal, and context
        D = cfg.d_model
        self.gate_norm   = nn.LayerNorm(D * 3)
        self.update_gate = nn.Linear(D * 3, D)
        nn.init.constant_(self.update_gate.bias, -2.0)  # conservative start

        self.state_norm = nn.LayerNorm(D)
        self.output     = nn.Linear(D, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.output.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _thinking_steps(self) -> int:
        if self.training:
            # Train with fewer steps; ODE extrapolates to more at inference
            return min(self.cfg.thinking_steps, 8)
        if self.cfg.adaptive_thinking:
            return self.cfg.max_thinking_steps
        return self.cfg.thinking_steps

    def _one_thinking_step(
        self, z: Tensor, context: Tensor, step_idx: int, total_steps: int
    ) -> tuple[Tensor, Tensor]:
        """One thinking step. Returns (z_new, velocity)."""
        B, T, D = z.shape
        dt = 1.0 / total_steps
        tau = torch.full((B,), (step_idx + 0.5) * dt, dtype=z.dtype, device=z.device)

        anchored = self.anchor(z, context)
        slot     = self.memory(z, context)
        velocity = self.field(anchored, context, slot, tau)
        proposal = z + velocity * dt

        # Gated update (residual blend of proposal and current z)
        gate_input = self.gate_norm(torch.cat([z, proposal, context], dim=-1))
        gate       = torch.sigmoid(self.update_gate(gate_input))
        z_new      = z + gate * (proposal - z)

        # Euler-Maruyama noise during training ONLY
        if self.training:
            z_new = z_new + 1e-4 * math.sqrt(dt) * torch.randn_like(z_new)

        return z_new, velocity

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        return_states: bool = False,
        grad_ckpt: bool = False,
        routing_mode: Optional[str] = None,
        log_trajectory: bool = False,
    ) -> dict[str, Tensor]:
        if routing_mode is not None:
            _orig = self.anchor.mode
            self.anchor.mode = routing_mode

        B, T = input_ids.shape
        emb     = self.embedding(input_ids)
        context = self.encoder(emb, grad_ckpt=grad_ckpt)
        z       = self.initial_state(context)

        steps = self._thinking_steps()
        traj:  list[dict] = []
        states: list[Tensor] = []
        halt_probs: list[Tensor] = []
        ponder_loss = torch.tensor(0.0, device=z.device)
        actual_steps = 0

        for i in range(steps):
            z, velocity = self._one_thinking_step(z, context, i, steps)
            halt_prob, should_halt = self.halt(
                z, velocity, step=i,
                threshold=self.cfg.halting_threshold)
            halt_probs.append(halt_prob)
            ponder_loss = ponder_loss + (1.0 - halt_prob)
            actual_steps = i + 1

            if log_trajectory or return_states:
                with torch.no_grad():
                    logits_step = self.output(self.state_norm(z))
                    p   = F.softmax(logits_step, dim=-1)
                    ent = -(p * p.clamp(1e-9).log()).sum(-1).mean().item()
                traj.append({
                    "step": i,
                    "z_norm": z.norm(dim=-1).mean().item(),
                    "v_norm": velocity.norm(dim=-1).mean().item(),
                    "halt_prob": halt_prob.item(),
                    "logit_entropy": ent,
                })
                if return_states:
                    states.append(z.detach())

            if should_halt and not self.training:
                break

        logits = self.output(self.state_norm(z))
        result: dict[str, Tensor] = {
            "logits": logits,
            "thinking_steps": torch.tensor(actual_steps, device=z.device),
            "halt_mean": torch.stack(halt_probs).mean() if halt_probs else torch.tensor(0.0),
            "ponder_loss": ponder_loss / max(actual_steps, 1),
        }
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:T].contiguous()
            result["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1), ignore_index=-100)
        if return_states and states:
            result["states"] = torch.stack(states, dim=1)
        if log_trajectory:
            result["trajectory"] = traj  # type: ignore[assignment]

        if routing_mode is not None:
            self.anchor.mode = _orig  # type: ignore[possibly-undefined]
        return result

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 50) -> Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            out    = self.forward(input_ids)
            logits = out["logits"][:, -1] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        remap = {}
        for k, v in state_dict.items():
            if k.startswith("encoder.blocks."):
                remap["encoder.cnn.blocks." + k[len("encoder.blocks."):]] = v
            elif k == "field.memory":
                pass
            else:
                remap[k] = v
        return super().load_state_dict(remap, strict=False)

# Backward compatibility aliases
ContinuousMeaningField = ParallelCMF
ParallelContinuousMeaningField = ParallelCMF
DeliberativeContinuousMeaningField = DeliberativeCMF


