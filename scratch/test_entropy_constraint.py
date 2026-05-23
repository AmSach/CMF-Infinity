import sys, time, torch, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

VOCAB_SIZE = 180
KEY_A = 0
EQUALS = 1
QUERY_A = 2
QUESTION_MARK = 3

VAL_COLORS = [4, 5, 6, 7]
GARBAGE_TOKENS = list(range(10, 120))

def get_distractor_seq(distance):
    val = VAL_COLORS[torch.randint(0, len(VAL_COLORS), (1,)).item()]
    distractors = torch.randint(GARBAGE_TOKENS[0], GARBAGE_TOKENS[-1] + 1, (distance,))
    x = torch.cat([
        torch.tensor([KEY_A, EQUALS, val]),
        distractors,
        torch.tensor([QUERY_A, QUESTION_MARK])
    ]).unsqueeze(0).to(device)
    y = torch.full_like(x, -100)
    y[0, -1] = val
    return x, y

print("Initializing CMF model...")
config = CMFConfig(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    hidden_dim=128,
    num_layers=4,
    max_seq_len=25000,
    adaptive_thinking=False,
    thinking_steps=4,
    use_global_memory_router=False,
)
import cmf.model
cmf.model.apply_rotary_pos_emb = lambda x, cos, sin: x

model = DeliberativeContinuousMeaningField(config).to(device)

# We will patch the model's forward pass to return the attention entropy as well.
# Let's inspect DeliberativeContinuousMeaningField's forward pass in model.py
# and construct a patched forward pass that collects the entropy of each anchoring attention step.

def forward_with_entropy(self, input_ids, labels=None, goal=None, target_length=None, return_states=False, gradient_checkpointing=False):
    batch_size, seq_len = input_ids.shape
    target_length = target_length or seq_len
    
    context = self.encoder(self.embedding(input_ids), gradient_checkpointing=gradient_checkpointing)[:, :target_length]
    cos, sin = self.rope(context, target_length)
    context = cmf.model.apply_rotary_pos_emb(context, cos, sin)
    
    z = self.initial_state(context)
    
    flat_context = context.reshape(batch_size * target_length, -1)
    flat_goal = None
    
    steps = self.config.thinking_steps
    step_anchor_scale = min(0.9, 0.15 * (8.0 / steps))
    
    z_accum = z.to(torch.float32)
    
    entropies = []
    
    for step_idx in range(steps):
        tau_value = (step_idx + 0.5) / float(steps)
        tau = torch.full((batch_size * target_length,), tau_value, dtype=z_accum.dtype, device=z_accum.device)
        
        flat_z = z_accum.to(z.dtype).reshape(batch_size * target_length, -1)
        velocity = self.field(flat_z, flat_context, tau, goal=flat_goal).reshape_as(z)
        proposal = z_accum.to(z.dtype) + velocity / float(steps)
        
        # Anchoring Attention
        if target_length > 1 and step_anchor_scale > 0:
            q = self.anchor_q(proposal).view(batch_size, target_length, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.anchor_k(context).view(batch_size, target_length, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.anchor_v(context).view(batch_size, target_length, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute raw attention to get entropy
            sim = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.full((target_length, target_length), float("-inf"), device=sim.device), diagonal=1)
            sim = sim + mask
            w_anchor = torch.softmax(sim, dim=-1)
            
            # Calculate entropy for the query token (last position)
            # w_anchor shape: [batch, heads, target_length, target_length]
            # We look at the query position [:, :, -1, :] (attention weights for the last token)
            query_weights = w_anchor[:, :, -1, :]
            query_weights = torch.clamp(query_weights, min=1e-9)
            step_entropy = -torch.sum(query_weights * torch.log2(query_weights), dim=-1).mean()
            entropies.append(step_entropy)
            
            z_proj = torch.matmul(w_anchor, v)
            z_proj = z_proj.transpose(1, 2).reshape(batch_size, target_length, self.config.d_model)
            z_proj = self.anchor_out(z_proj)
            proposal = (1.0 - step_anchor_scale) * proposal + step_anchor_scale * z_proj
            
        gate_input = torch.cat([z_accum.to(z.dtype), proposal, context], dim=-1)
        gate = torch.sigmoid(self.update_gate(self.gate_norm(gate_input)))
        
        delta_z = gate.to(torch.float32) * (proposal.to(torch.float32) - z_accum)
        z_accum = z_accum + delta_z
        
        # Curl
        d_half = delta_z.size(-1) // 2
        delta_q = delta_z[..., :d_half]
        delta_p = delta_z[..., d_half:]
        symplectic_curl = torch.cat([delta_p, -delta_q], dim=-1)
        z_accum = z_accum + 0.10 * symplectic_curl
        
        # CGMP
        z_mean = z_accum.mean(dim=-1, keepdim=True)
        z_std = z_accum.std(dim=-1, keepdim=True)
        c_mean = context.mean(dim=-1, keepdim=True)
        c_std = context.std(dim=-1, keepdim=True)
        z_accum = ((z_accum - z_mean) / torch.clamp(z_std, min=1e-6)) * c_std + c_mean
        
    z_final = z_accum.to(z.dtype)
    logits = self.output(self.state_norm(z_final))
    
    result = {"logits": logits, "entropies": torch.stack(entropies)}
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:target_length].contiguous()
        result["loss"] = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
    return result

# Bind patched forward pass
model.forward = lambda *args, **kwargs: forward_with_entropy(model, *args, **kwargs)

print("\n" + "="*80)
print("TRAINING CMF WITH AN ATTENTION ENTROPY PENALTY")
print("="*80)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# We set beta=0.05 (scale of entropy penalty)
BETA = 0.05
steps = 1500

model.train()
for step in range(steps + 1):
    x, y = get_distractor_seq(torch.randint(50, 150, (1,)).item())
    
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
        out = model(x, labels=y)
        ce_loss = out["loss"]
        # Entropy penalty: we want to MINIMIZE attention entropy, so we add it to the loss
        entropy_loss = out["entropies"].mean()
        loss = ce_loss + BETA * entropy_loss
        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if step % 250 == 0:
        print(f"  Step {step:4d} | CE Loss: {ce_loss.item():.4f} | Avg Attention Entropy: {entropy_loss.item():.4f} bits | Total Loss: {loss.item():.4f}")

# =====================================================================
# EVALUATION: DISTRACTOR SATURATION (UP TO 10,000 DISTRACTORS)
# =====================================================================
print("\n" + "="*80)
print("EVALUATING SCALING AND ATTENTION ENTROPY COLLAPSE AFTER PENALIZATION")
print("="*80)

distances = [100, 1000, 5000, 10000]
num_trials = 50

print(f"{'Distance':<10} | {'CMF Accuracy':<15}")
print("-" * 30)

model.eval()
for d in distances:
    correct = 0
    for _ in range(num_trials):
        x, y = get_distractor_seq(d)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
                out = model(x)
                pred = torch.argmax(out["logits"][0, -1, :]).item()
                label = y[0, -1].item()
                if pred == label:
                    correct += 1
        if device == "cuda": torch.cuda.empty_cache()
    print(f"{d:<10d} | {correct/num_trials:<15.1%}")

print("\n" + "="*80)
print("SHANNON ENTROPY PATHWAY AFTER PENALIZATION (500 distractors)")
print("="*80)

cmf_entropies = [[] for _ in range(4)]
for _ in range(50):
    x, y = get_distractor_seq(500)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
            out = model(x)
            entropies = out["entropies"].cpu().tolist()
            for step_idx, e in enumerate(entropies):
                cmf_entropies[step_idx].append(e)

for step_idx in range(4):
    avg_e = sum(cmf_entropies[step_idx]) / len(cmf_entropies[step_idx])
    print(f"CMF Deliberation Step {step_idx+1:<2d}      | {avg_e:.4f} bits")
print("="*80)
