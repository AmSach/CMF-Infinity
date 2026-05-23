import sys, time, torch, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load config and model
print("Loading model and weights...")
ckpt = torch.load("checkpoint_latest_new.pt", map_location="cpu", weights_only=False)
cfg_src = ckpt["config"]
cfg_dict = cfg_src if isinstance(cfg_src, dict) else cfg_src.__dict__.copy()
sd = ckpt["model"]
mem_keys = [k for k in sd if "field.memory" in k and "bank" not in k]
if mem_keys:
    cfg_dict["num_memory_anchors"] = sd[mem_keys[0]].shape[0]
config = CMFConfig(**cfg_dict)

model = DeliberativeContinuousMeaningField(config).to(device)
clean = {}
for k, v in sd.items():
    k2 = k.replace("_orig_mod.module.", "").replace("module.", "")
    clean[k2] = v
model.load_state_dict(clean, strict=False)

# =====================================================================
# TEST 1: Activation Magnitudes & Hidden State Variance over Solver Steps
# =====================================================================
def analyze_solver_steps():
    print("\n" + "="*80)
    print("ANALYSIS 1: ACTIVATION MAGNITUDES & VARIANCE OVER SOLVER STEPS")
    print("="*80)
    
    batch_size = 32
    target_length = 64
    x = torch.randint(0, 1000, (batch_size, target_length), device=device)
    y = torch.randint(0, 1000, (batch_size, target_length), device=device)
    
    model.train()
    
    # Run encoder, initial state
    with torch.no_grad():
        emb = model.embedding(x)
        context = model.encoder(emb)
        cos, sin = model.rope(context, target_length)
        from cmf.model import apply_rotary_pos_emb
        context = apply_rotary_pos_emb(context, cos, sin)
        z = model.initial_state(context)
        
    flat_context = context.reshape(batch_size * target_length, -1)
    z_accum = z.to(torch.float32).clone()
    steps = 8
    step_anchor_scale = min(0.9, 0.15 * (8.0 / steps))
    
    print(f"{'Step':<6} | {'z L2 Norm':<12} | {'z Variance':<12} | {'Vel L2 Norm':<12} | {'Gate Mean':<12}")
    print("-" * 60)
    
    for step_idx in range(steps):
        tau_value = (step_idx + 0.5) / float(steps)
        tau = torch.full(
            (batch_size * target_length,),
            tau_value,
            dtype=z_accum.dtype,
            device=z_accum.device,
        )
        flat_z = z_accum.to(z.dtype).reshape(batch_size * target_length, -1)
        
        with torch.no_grad():
            velocity = model.field(flat_z, flat_context, tau).reshape_as(z)
            proposal = z_accum.to(z.dtype) + velocity / float(steps)
            
            if target_length > 1:
                sim_anchor = torch.matmul(proposal, context.transpose(-1, -2)) / math.sqrt(proposal.size(-1))
                mask = torch.triu(torch.full((target_length, target_length), float("-inf"), device=proposal.device), diagonal=1)
                sim_anchor = sim_anchor + mask
                w_anchor = torch.softmax(sim_anchor, dim=-1)
                z_proj = torch.matmul(w_anchor, context)
                proposal = (1.0 - step_anchor_scale) * proposal + step_anchor_scale * z_proj
                
            gate_input = torch.cat([z_accum.to(z.dtype), proposal, context], dim=-1)
            gate = torch.sigmoid(model.update_gate(model.gate_norm(gate_input)))
            delta_z = gate.to(torch.float32) * (proposal.to(torch.float32) - z_accum)
            z_accum = z_accum + delta_z
            
            # Symplectic curl
            d_half = delta_z.size(-1) // 2
            delta_q = delta_z[..., :d_half]
            delta_p = delta_z[..., d_half:]
            symplectic_curl = torch.cat([delta_p, -delta_q], dim=-1)
            z_accum = z_accum + 0.10 * symplectic_curl
            
        z_l2 = torch.norm(z_accum, dim=-1).mean().item()
        z_var = z_accum.var(dim=-1).mean().item()
        vel_l2 = torch.norm(velocity, dim=-1).mean().item()
        gate_mean = gate.mean().item()
        
        print(f"{step_idx:<6d} | {z_l2:<12.4f} | {z_var:<12.6f} | {vel_l2:<12.4f} | {gate_mean:<12.4f}")
        
# =====================================================================
# TEST 2: Gradient Norms Per Layer
# =====================================================================
def analyze_gradient_norms():
    print("\n" + "="*80)
    print("ANALYSIS 2: GRADIENT NORMS PER LAYER")
    print("="*80)
    
    batch_size = 16
    target_length = 64
    x = torch.randint(0, 1000, (batch_size, target_length), device=device)
    y = torch.randint(0, 1000, (batch_size, target_length), device=device)
    
    model.train()
    model.zero_grad()
    
    out = model(x, labels=y)
    loss = out["loss"]
    loss.backward()
    
    # Group gradients by layer category
    layer_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            # find category
            cat = "other"
            if "embedding" in name:
                cat = "Embedding"
            elif "encoder" in name:
                cat = "Context Encoder"
            elif "initial_state" in name:
                cat = "Initial State Projector"
            elif "field.memory" in name:
                cat = "Factual Memory Bank"
            elif "field.net" in name:
                cat = "VectorField MLP"
            elif "field.proposal" in name or "field.gate" in name:
                cat = "VectorField Heads"
            elif "update_gate" in name or "gate_norm" in name:
                cat = "Update Gate"
            elif "halt_head" in name:
                cat = "Halting Head"
            elif "output" in name:
                cat = "Language Modeling Head"
                
            norm = torch.norm(p.grad).item()
            if cat not in layer_grads:
                layer_grads[cat] = []
            layer_grads[cat].append(norm)
            
    print(f"{'Layer Category':<30} | {'Mean Grad Norm':<18} | {'Max Grad Norm':<18}")
    print("-" * 72)
    for cat, norms in layer_grads.items():
        mean_norm = sum(norms) / len(norms)
        max_norm = max(norms)
        print(f"{cat:<30} | {mean_norm:<18.6f} | {max_norm:<18.6f}")

# =====================================================================
# TEST 3: Hidden State L2 Norm across Sequence Lengths (Boundary Stability)
# =====================================================================
def analyze_seq_lengths():
    print("\n" + "="*80)
    print("ANALYSIS 3: HIDDEN STATE L2 NORM ACROSS SEQUENCE LENGTHS")
    print("="*80)
    
    seq_lengths = [16, 32, 64, 128, 256, 512, 1024]
    
    print(f"{'Seq Length':<12} | {'Final z L2 Norm':<18} | {'Final z Var':<18} | {'Status':<15}")
    print("-" * 70)
    
    model.eval()
    for l in seq_lengths:
        x = torch.randint(0, 1000, (8, l), device=device)
        with torch.no_grad():
            out = model(x, return_states=True)
            # Find the final state z
            z_final = out["states"][-1]  # shape: [B, L, d_model]
            
        z_l2 = torch.norm(z_final, dim=-1).mean().item()
        z_var = z_final.var(dim=-1).mean().item()
        
        # Determine status
        status = "HEALTHY"
        if z_l2 > 50.0 or math.isnan(z_l2):
            status = "EXPLODING"
        elif z_l2 < 0.1:
            status = "COLLAPSED"
            
        print(f"{l:<12d} | {z_l2:<18.4f} | {z_var:<18.6f} | {status:<15}")

if __name__ == "__main__":
    analyze_solver_steps()
    analyze_gradient_norms()
    analyze_seq_lengths()
