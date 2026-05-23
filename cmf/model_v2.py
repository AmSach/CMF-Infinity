import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class CMFv2Config:
    def __init__(
        self,
        vocab_size: int = 200,
        d_model: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_slots: int = 8,
        thinking_steps: int = 4,
        dropout: float = 0.0,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.thinking_steps = thinking_steps
        self.dropout = dropout


class MemorySlotReadHead(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, z: torch.Tensor, M: torch.Tensor, temp: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: [batch, d_model]
        # M: [batch, num_slots, d_model]
        batch_size = z.size(0)
        
        q = self.q_proj(z).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, 1, D_h]
        k = self.k_proj(M).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, S, D_h]
        v = self.v_proj(M).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, S, D_h]
        
        sim = torch.matmul(q, k.transpose(-1, -2)) / (temp * math.sqrt(self.head_dim)) # [B, H, 1, S]
        attn = torch.softmax(sim, dim=-1)
        
        out = torch.matmul(attn, v) # [B, H, 1, D_h]
        out = out.transpose(1, 2).contiguous().view(batch_size, self.d_model)
        
        # Calculate entropy of attention over slots
        p = torch.clamp(attn, min=1e-9)
        entropy = -torch.sum(p * torch.log2(p), dim=-1).mean(dim=1) # [batch, 1]
        
        return self.out_proj(out), entropy


class MemorySlotWriteHead(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.write_gate = nn.Linear(d_model * 2, 1)
        nn.init.constant_(self.write_gate.bias, -2.0)

    def forward(self, z: torch.Tensor, e_t: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        # z: [batch, d_model]
        # M: [batch, num_slots, d_model]
        # Returns updated Memory slots M_new
        batch_size = z.size(0)
        num_slots = M.size(1)
        
        q = self.q_proj(z).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, 1, D_h]
        k = self.k_proj(M).view(batch_size, num_slots, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, S, D_h]
        v = self.v_proj(z).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, 1, D_h]
        
        # Calculate write address with sharp softmax (temp 0.1) to solve gradient diffusion
        sim = torch.matmul(q, k.transpose(-1, -2)) / (0.1 * math.sqrt(self.head_dim)) # [B, H, 1, S]
        attn = torch.softmax(sim, dim=-1) # [B, H, 1, S]
        
        # Expand write value to slots
        # attn: [B, H, 1, S], v: [B, H, 1, D_h]
        # write_delta = attn^T * v -> [B, H, S, D_h]
        write_delta = torch.matmul(attn.transpose(-1, -2), v) # [B, H, S, D_h]
        write_delta = write_delta.transpose(1, 2).contiguous().view(batch_size, num_slots, self.d_model)
        
        gate = torch.sigmoid(self.write_gate(torch.cat([z, e_t], dim=-1))).unsqueeze(-1) # [B, 1, 1]
        
        return M + gate * write_delta


class VectorFieldV2(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 3 + 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.time_frequencies = 8

    def forward(self, z: torch.Tensor, x: torch.Tensor, m: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # z: [batch, d_model]
        # x: [batch, d_model] (current observation)
        # m: [batch, d_model] (memory read)
        # tau: [batch] (time step index)
        frequencies = torch.arange(
            self.time_frequencies,
            dtype=tau.dtype,
            device=tau.device,
        )
        frequencies = 2.0 ** frequencies * math.pi
        angles = tau.unsqueeze(-1) * frequencies
        t_feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        inp = torch.cat([z, x, m, t_feats], dim=-1)
        return self.net(inp)


class PersistentStateCMF(nn.Module):
    def __init__(self, config: CMFv2Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        self.read_head = MemorySlotReadHead(config.d_model)
        self.write_head = MemorySlotWriteHead(config.d_model)
        
        self.field = VectorFieldV2(config.d_model, config.hidden_dim)
        
        self.boundary_gate = nn.Linear(config.d_model * 2, config.d_model)
        self.boundary_norm = nn.LayerNorm(config.d_model * 2)
        nn.init.constant_(self.boundary_gate.bias, -1.0) # start conservative (keep memory)
        
        self.update_gate = nn.Linear(config.d_model * 4, config.d_model)
        self.gate_norm = nn.LayerNorm(config.d_model * 4)
        nn.init.constant_(self.update_gate.bias, -2.0) # default to conservative ODE updates
        
        self.state_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size)
        
        # Symplectic stabilizer states
        self.register_buffer("initial_slots", torch.zeros(config.num_slots, config.d_model))
        nn.init.normal_(self.initial_slots, mean=0.0, std=0.5)
        self.register_buffer("tau_steps", torch.tensor([(i + 0.5) / config.thinking_steps for i in range(config.thinking_steps)]))
        self._forward_calls = 0

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: [batch, d_model] initialized to zero or small noise
        # M: [batch, num_slots, d_model] initialized to default slots
        z = torch.zeros(batch_size, self.config.d_model, device=device)
        M = self.initial_slots.unsqueeze(0).repeat(batch_size, 1, 1).clone()
        return z, M

    def step_forward(
        self,
        x_token: torch.Tensor,
        z_prev: torch.Tensor,
        M_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_token: [batch_size] (current token input)
        # z_prev: [batch_size, d_model]
        # M_prev: [batch_size, num_slots, d_model]
        
        batch_size = x_token.size(0)
        e_t = self.input_proj(self.embedding(x_token)) # [B, d_model]
        
        steps = self.config.thinking_steps
        
        # Apply discrete transition gate at token sequence boundary
        gate_boundary = torch.sigmoid(self.boundary_gate(self.boundary_norm(torch.cat([z_prev, e_t], dim=-1))))
        z_accum = (1.0 - gate_boundary).to(torch.float32) * z_prev.to(torch.float32) + gate_boundary.to(torch.float32) * e_t.to(torch.float32)
        M = M_prev.clone()
        
        # Pre-compute memory slot statistics once for scale projection
        m_mean = M.mean(dim=-1).mean(dim=1, keepdim=True)
        m_std = M.std(dim=-1).mean(dim=1, keepdim=True)
        
        # Deliberation loop
        for step_idx in range(steps):
            tau = self.tau_steps[step_idx].expand(batch_size).to(device=e_t.device, dtype=e_t.dtype)
            
            # Read from memory slots using projected current state to match query-key scales
            z_curr = z_accum.to(e_t.dtype)
            z_curr_mean = z_curr.mean(dim=-1, keepdim=True)
            z_curr_std = z_curr.std(dim=-1, keepdim=True)
            z_curr_proj = ((z_curr - z_curr_mean) / torch.clamp(z_curr_std, min=1e-6)) * m_std + m_mean
            
            m_t, entropy = self.read_head(z_curr_proj, M, temp=0.1) # [B, d_model] using sharp temp 0.1
            
            # Compute vector field velocity
            velocity = self.field(z_curr, e_t, m_t, tau)
            proposal = z_curr + velocity / float(steps)
            
            # Gated continuous update
            gate_input = torch.cat([z_curr, proposal, e_t, m_t], dim=-1)
            gate = torch.sigmoid(self.update_gate(self.gate_norm(gate_input)))
            
            delta_z = gate.to(torch.float32) * (proposal.to(torch.float32) - z_accum)
            z_accum = z_accum + delta_z
            
            # Hamiltonian Symplectic Curl Stabilizer
            d_half = delta_z.size(-1) // 2
            delta_q = delta_z[..., :d_half]
            delta_p = delta_z[..., d_half:]
            symplectic_curl = torch.cat([delta_p, -delta_q], dim=-1)
            z_accum = z_accum + 0.10 * symplectic_curl
            
            # Langevin Noise (if training)
            if self.training:
                noise = torch.randn_like(z_accum) * 1e-4
                z_accum = z_accum + noise
                
        z_final = z_accum.to(e_t.dtype)
        # Context-Guided Manifold Projection (CGMP) ONCE at the end
        z_mean = z_final.mean(dim=-1, keepdim=True)
        z_std = z_final.std(dim=-1, keepdim=True)
        m_mean = M.mean(dim=-1).mean(dim=1, keepdim=True)
        m_std = M.std(dim=-1).mean(dim=1, keepdim=True)
        z_final_proj = ((z_final - z_mean) / torch.clamp(z_std, min=1e-6)) * m_std + m_mean
        
        # Update memory slots based on finalized state after deliberation finishes
        M = self.write_head(z_final_proj, e_t, M)
        
        logits = self.output(self.state_norm(z_final_proj))
        return logits, z_final, M

    def forward(
        self,
        input_ids: torch.Tensor,
        z_init: Optional[torch.Tensor] = None,
        M_init: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Batch forward pass for sequence training:
        # Processes sequence step-by-step in streaming mode (no attention to history).
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if self.training:
            self._forward_calls += 1
            
        if z_init is None or M_init is None:
            z, M = self.init_state(batch_size, device)
        else:
            z, M = z_init, M_init
            
        logits_list = []
        for t in range(seq_len):
            x_t = input_ids[:, t]
            logits, z, M = self.step_forward(x_t, z, M)
            logits_list.append(logits)
            
        # Stack logits: [batch, seq_len, vocab_size]
        all_logits = torch.stack(logits_list, dim=1)
        
        return {
            "logits": all_logits,
            "z_final": z,
            "M_final": M
        }
