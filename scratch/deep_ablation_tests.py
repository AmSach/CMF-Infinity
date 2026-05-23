import sys, time, torch, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Task configuration (Interference/Retrieval)
VOCAB_SIZE = 100
KEY_A = 0
EQUALS = 1
VAL_COLORS = [2, 3, 4, 5]  # blue, red, green, yellow
GARBAGE_MIN = 6
GARBAGE_MAX = 59
QUERY_A = 60

def generate_batch(batch_size, distance):
    seq_len = distance + 4
    x = torch.randint(GARBAGE_MIN, GARBAGE_MAX + 1, (batch_size, seq_len), device=device)
    y = torch.full((batch_size, seq_len), -100, device=device)
    for b in range(batch_size):
        val = VAL_COLORS[torch.randint(0, len(VAL_COLORS), (1,)).item()]
        x[b, 0] = KEY_A
        x[b, 1] = EQUALS
        x[b, 2] = val
        x[b, -1] = QUERY_A
        y[b, -1] = val
    return x, y

# Initialize model (No RoPE for pure semantic field)
import cmf.model
cmf.model.apply_rotary_pos_emb = lambda x, cos, sin: x

config = CMFConfig(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    hidden_dim=128,
    num_layers=4,
    max_seq_len=5000,
    adaptive_thinking=False,
    thinking_steps=4,
)
model = DeliberativeContinuousMeaningField(config).to(device)

# Train a small model to perform the retrieval task first
print("Training CMF baseline on retrieval...")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()
for step in range(200):
    dist = torch.randint(50, 150, (1,)).item()
    x, y = generate_batch(16, dist)
    opt.zero_grad()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out["logits"][:, -1, :], y[:, -1])
    loss.backward()
    opt.step()
print("Training complete.")
model.eval()

# =====================================================================
# PART 1 & 2: PERTURBATION, ABLATIONS, AND INFORMATION RECOVERY
# =====================================================================
print("\n" + "="*80)
print("PART 1 & 2: PERTURBATION, ABLATIONS, AND STATE INFORMATION RECOVERY")
print("="*80)

# We will implement custom forward passes to inject noise under different ablations:
# 1. Standard (with CGMP)
# 2. No CGMP (Disable projection during solver steps)
# 3. Frozen Context (Context attention is disabled or context is set to a constant after step 1)

def run_ablation_perturbation(ablation_type="standard", noise_type="gaussian", noise_std=0.3):
    """
    ablation_type: "standard", "no_cgmp", "frozen_context"
    noise_type: "gaussian", "adversarial"
    """
    correct = 0
    total = 30
    
    # We will track target token embedding similarity over the 4 solver steps
    # to see if the state recovers information (denoises) dynamically
    similarities = [[] for _ in range(4)]
    
    for _ in range(total):
        x, y = generate_batch(1, 100)
        target_token = y[0, -1].item()
        target_embed = model.embedding.weight[target_token].detach()
        
        # We need to run the forward pass step-by-step to inject noise
        with torch.no_grad():
            context = model.encoder(model.embedding(x))
            z = model.initial_state(context)
            
            # Initial state z
            batch_size, seq_len, d_model = z.shape
            flat_context = context.reshape(batch_size * seq_len, -1)
            
            steps = 4
            step_anchor_scale = min(0.9, 0.15 * (8.0 / steps))
            
            z_accum = z.clone()
            
            # For "frozen_context", we save context at step 1 and freeze it (no attention updates)
            orig_context = context.clone()
            
            for step_idx in range(steps):
                tau_value = (step_idx + 0.5) / float(steps)
                tau = torch.full((batch_size * seq_len,), tau_value, dtype=z_accum.dtype, device=z_accum.device)
                
                # Fetch velocity
                flat_z = z_accum.reshape(batch_size * seq_len, -1)
                velocity = model.field(flat_z, flat_context, tau).reshape_as(z_accum)
                proposal = z_accum + velocity / float(steps)
                
                # Subspace Manifold Anchoring
                if seq_len > 1 and ablation_type != "frozen_context":
                    z_proj = torch.nn.functional.scaled_dot_product_attention(
                        proposal.unsqueeze(1), context.unsqueeze(1), context.unsqueeze(1), is_causal=True
                    ).squeeze(1)
                    proposal = (1.0 - step_anchor_scale) * proposal + step_anchor_scale * z_proj
                elif ablation_type == "frozen_context":
                    # For frozen context, anchoring uses the original context but with no dynamic updates
                    z_proj = torch.nn.functional.scaled_dot_product_attention(
                        proposal.unsqueeze(1), orig_context.unsqueeze(1), orig_context.unsqueeze(1), is_causal=True
                    ).squeeze(1)
                    proposal = (1.0 - step_anchor_scale) * proposal + step_anchor_scale * z_proj
                
                gate_input = torch.cat([z_accum, proposal, context], dim=-1)
                gate = torch.sigmoid(model.update_gate(model.gate_norm(gate_input)))
                
                delta_z = gate * (proposal - z_accum)
                z_next = z_accum + delta_z
                
                # Hamiltonian Symplectic Curl
                d_half = delta_z.size(-1) // 2
                delta_q = delta_z[..., :d_half]
                delta_p = delta_z[..., d_half:]
                symplectic_curl = torch.cat([delta_p, -delta_q], dim=-1)
                z_next = z_next + 0.10 * symplectic_curl
                
                z_accum = z_next
                
                # NOISE INJECTION AT STEP 1 (after step_idx == 0)
                if step_idx == 0 and noise_std > 0:
                    if noise_type == "gaussian":
                        perturbation = torch.randn_like(z_accum) * noise_std
                        z_accum = z_accum + perturbation
                    elif noise_type == "adversarial":
                        # Push state towards the WRONG target basin
                        # We find a wrong color and use its embedding direction
                        wrong_color = VAL_COLORS[0] if target_token != VAL_COLORS[0] else VAL_COLORS[1]
                        wrong_embed = model.embedding.weight[wrong_color].detach()
                        # Add a vector in the direction of the wrong embedding
                        perturbation = wrong_embed.view(1, 1, -1) * noise_std
                        z_accum = z_accum + perturbation
                
                # Context-Guided Manifold Projection (CGMP)
                if ablation_type != "no_cgmp":
                    z_mean = z_accum.mean(dim=-1, keepdim=True)
                    z_std = z_accum.std(dim=-1, keepdim=True)
                    c_mean = context.mean(dim=-1, keepdim=True)
                    c_std = context.std(dim=-1, keepdim=True)
                    z_accum = ((z_accum - z_mean) / torch.clamp(z_std, min=1e-6)) * c_std + c_mean
                
                # Track target similarity of z at query token (position -1)
                sim = torch.nn.functional.cosine_similarity(z_accum[0, -1], target_embed, dim=-1).item()
                similarities[step_idx].append(sim)
                
            # Decode final prediction
            logits = model.output(model.state_norm(z_accum))
            pred = torch.argmax(logits[0, -1, :]).item()
            if pred == target_token:
                correct += 1
                
    avg_sims = [sum(sims)/len(sims) for sims in similarities]
    acc = correct / total
    return acc, avg_sims

# Run tests
test_cases = [
    ("Standard + Gaussian Noise (0.3)", "standard", "gaussian", 0.3),
    ("Standard + Adversarial Noise (0.3)", "standard", "adversarial", 0.3),
    ("No CGMP + Gaussian Noise (0.3)", "no_cgmp", "gaussian", 0.3),
    ("No CGMP + Adversarial Noise (0.3)", "no_cgmp", "adversarial", 0.3),
    ("Frozen Context + Gaussian Noise (0.3)", "frozen_context", "gaussian", 0.3),
]

print(f"{'Condition':<40} | {'Accuracy':<10} | {'Cosine Sim w/ Target across Steps 1 -> 4':<40}")
print("-" * 100)
for name, ab_t, n_t, std in test_cases:
    acc, sims = run_ablation_perturbation(ab_t, n_t, std)
    sims_str = " -> ".join(f"{s:.3f}" for s in sims)
    print(f"{name:<40} | {acc:<10.1%} | {sims_str:<40}")

# =====================================================================
# PART 3: COMPONENT ATTRIBUTION (REMOVING HELPERS)
# =====================================================================
print("\n" + "="*80)
print("PART 3: COMPONENT ATTRIBUTION (REMOVING HELPERS ON RETRIEVAL)")
print("="*80)

def run_eval_ablation(no_curl=False, no_anchoring=False, no_gating=False):
    correct = 0
    total = 30
    
    for _ in range(total):
        x, y = generate_batch(1, 100)
        target_token = y[0, -1].item()
        
        with torch.no_grad():
            context = model.encoder(model.embedding(x))
            z = model.initial_state(context)
            
            batch_size, seq_len, d_model = z.shape
            flat_context = context.reshape(batch_size * seq_len, -1)
            
            steps = 4
            step_anchor_scale = min(0.9, 0.15 * (8.0 / steps))
            z_accum = z.clone()
            
            for step_idx in range(steps):
                tau_value = (step_idx + 0.5) / float(steps)
                tau = torch.full((batch_size * seq_len,), tau_value, dtype=z_accum.dtype, device=z_accum.device)
                
                flat_z = z_accum.reshape(batch_size * seq_len, -1)
                velocity = model.field(flat_z, flat_context, tau).reshape_as(z_accum)
                
                if no_gating:
                    # Euler integration directly without gate scaling
                    proposal = z_accum + velocity / float(steps)
                else:
                    proposal = z_accum + velocity / float(steps)
                    
                # Anchoring
                if seq_len > 1 and not no_anchoring:
                    z_proj = torch.nn.functional.scaled_dot_product_attention(
                        proposal.unsqueeze(1), context.unsqueeze(1), context.unsqueeze(1), is_causal=True
                    ).squeeze(1)
                    proposal = (1.0 - step_anchor_scale) * proposal + step_anchor_scale * z_proj
                
                if no_gating:
                    z_next = proposal
                else:
                    gate_input = torch.cat([z_accum, proposal, context], dim=-1)
                    gate = torch.sigmoid(model.update_gate(model.gate_norm(gate_input)))
                    delta_z = gate * (proposal - z_accum)
                    z_next = z_accum + delta_z
                
                # Curl
                if not no_curl and not no_gating:
                    d_half = delta_z.size(-1) // 2
                    delta_q = delta_z[..., :d_half]
                    delta_p = delta_z[..., d_half:]
                    symplectic_curl = torch.cat([delta_p, -delta_q], dim=-1)
                    z_next = z_next + 0.10 * symplectic_curl
                
                z_accum = z_next
                
                # Projection
                z_mean = z_accum.mean(dim=-1, keepdim=True)
                z_std = z_accum.std(dim=-1, keepdim=True)
                c_mean = context.mean(dim=-1, keepdim=True)
                c_std = context.std(dim=-1, keepdim=True)
                z_accum = ((z_accum - z_mean) / torch.clamp(z_std, min=1e-6)) * c_std + c_mean
                
            logits = model.output(model.state_norm(z_accum))
            pred = torch.argmax(logits[0, -1, :]).item()
            if pred == target_token:
                correct += 1
                
    return correct / total

print(f"  CMF Baseline (All active)         | Accuracy: {run_eval_ablation():.1%}")
print(f"  Ablation: No Symplectic Curl      | Accuracy: {run_eval_ablation(no_curl=True):.1%}")
print(f"  Ablation: No Manifold Anchoring   | Accuracy: {run_eval_ablation(no_anchoring=True):.1%}")
print(f"  Ablation: No Update Gating        | Accuracy: {run_eval_ablation(no_gating=True):.1%}")
print(f"  Ablation: No Curl + No Anchoring  | Accuracy: {run_eval_ablation(no_curl=True, no_anchoring=True):.1%}")
print("="*80)
