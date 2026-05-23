import sys, torch, math
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.model_v2 import PersistentStateCMF, CMFv2Config

device = "cpu"
torch.manual_seed(42)

VOCAB_SIZE = 50
YES = 30
NO = 31

def generate_maze_sequence(batch_size, length):
    x = torch.zeros(batch_size, length, dtype=torch.long, device=device)
    y = torch.full((batch_size, length), -100, dtype=torch.long, device=device)
    
    pos = torch.randint(0, 10, (batch_size,), device=device)
    x[:, 0] = 10 + pos
    
    key_colors = torch.randint(0, 2, (batch_size,), device=device)
    door_colors = torch.randint(0, 2, (batch_size,), device=device)
    
    decisions = torch.randint(0, 2, (batch_size, length), device=device)
    actions = torch.where(decisions == 0, 22, 23)
    x[:, 1:-2] = actions[:, 1:-2]
    
    key_step = length // 3
    x[:, key_step] = 20 + key_colors
    
    x[:, length - 2] = 24 + door_colors
    x[:, length - 1] = 26
    
    targets = torch.where(key_colors == door_colors, YES, NO)
    y[:, -1] = targets
    return x, y

cmf_config = CMFv2Config(
    vocab_size=50,
    d_model=64,
    hidden_dim=128,
    num_layers=4,
    num_slots=8,
    thinking_steps=4,
)

model = PersistentStateCMF(cmf_config).to(device)

x, y = generate_maze_sequence(2, 25)

# Forward pass
model.train()
out = model(x)
loss = F.cross_entropy(out["logits"][:, -1, :], y[:, -1])
loss.backward()

print("=" * 80)
print("GRADIENT DIAGNOSTIC REPORT FOR CMF V2")
print("=" * 80)
print(f"Loss: {loss.item():.4f}\n")

print("Gradient Norms by Parameter:")
print("-" * 50)
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        param_norm = param.norm().item()
        print(f"{name:<50} | Param Norm: {param_norm:.4e} | Grad Norm: {grad_norm:.4e}")
    else:
        print(f"{name:<50} | NO GRADIENT")

# Let's inspect the intermediate activations of a step
z, M = model.init_state(2, device)

# Let's inspect step_forward manually step-by-step
print("\n" + "=" * 80)
print("STATE & ACTIVATION INSPECTION STEP-BY-STEP")
print("=" * 80)

for t in range(25):
    x_t = x[:, t]
    
    # We trace step_forward manually
    batch_size = x_t.size(0)
    e_t = model.input_proj(model.embedding(x_t))
    
    steps = model.config.thinking_steps
    z_accum = z.clone()
    M_prev = M.clone()
    
    print(f"\nToken Step {t} | Input Token: {x_t.tolist()}")
    print(f"  Slots M std: {M_prev.std(dim=-1).mean().item():.4f} | M mean: {M_prev.mean().item():.4f}")
    
    for step_idx in range(steps):
        tau_value = (step_idx + 0.5) / float(steps)
        tau = torch.full((batch_size,), tau_value, dtype=z_accum.dtype, device=z_accum.device)
        
        z_curr = z_accum.to(e_t.dtype)
        m_t, entropy = model.read_head(z_curr, M, temp=1.0)
        
        velocity = model.field(z_curr, e_t, m_t, tau)
        proposal = z_curr + velocity / float(steps)
        
        gate_input = torch.cat([z_curr, proposal, e_t, m_t], dim=-1)
        gate_val = torch.sigmoid(model.update_gate(model.gate_norm(gate_input)))
        
        delta_z = gate_val * (proposal - z_accum)
        z_accum = z_accum + delta_z
        
        print(f"    Delib Step {step_idx} | Velocity Norm: {velocity.norm(dim=-1).mean().item():.4f} | Update Gate Mean: {gate_val.mean().item():.4f}")
        
    z_final = z_accum.to(e_t.dtype)
    # CGMP
    z_mean = z_final.mean(dim=-1, keepdim=True)
    z_std = z_final.std(dim=-1, keepdim=True)
    m_mean = M.mean(dim=-1).mean(dim=1, keepdim=True)
    m_std = M.std(dim=-1).mean(dim=1, keepdim=True)
    z_final_proj = ((z_final - z_mean) / torch.clamp(z_std, min=1e-6)) * m_std + m_mean
    
    # Write Head
    q = model.write_head.q_proj(z_final_proj)
    k = model.write_head.k_proj(M)
    # Sim
    num_slots = M.size(1)
    q_view = q.view(batch_size, 1, model.write_head.num_heads, model.write_head.head_dim).transpose(1, 2)
    k_view = k.view(batch_size, num_slots, model.write_head.num_heads, model.write_head.head_dim).transpose(1, 2)
    sim = torch.matmul(q_view, k_view.transpose(-1, -2)) / math.sqrt(model.write_head.head_dim)
    attn = torch.softmax(sim, dim=-1)
    
    w_gate = torch.sigmoid(model.write_head.write_gate(torch.cat([z_final_proj, e_t], dim=-1)))
    print(f"  Write Gate: {w_gate.mean().item():.4f} | Attn Entropy: {-torch.sum(attn * torch.log(attn + 1e-9), dim=-1).mean().item():.4f}")
    
    M = model.write_head(z_final_proj, e_t, M)
    z = z_final_proj
