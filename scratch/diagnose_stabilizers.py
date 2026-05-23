import sys, io, math, torch, tiktoken
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField

CKPT_NEW = Path("e:/CMF/checkpoint_latest_new.pt")
enc = tiktoken.get_encoding("gpt2")
text = "The capital of France is Paris."
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_clean_model():
    ckpt = torch.load(str(CKPT_NEW), map_location="cpu", weights_only=False)
    cfg_src = ckpt["config"]
    cfg_dict = cfg_src if isinstance(cfg_src, dict) else cfg_src.__dict__.copy()
    sd = ckpt["model"]
    mem_keys = [k for k in sd if "field.memory" in k and "bank" not in k]
    if mem_keys:
        cfg_dict["num_memory_anchors"] = sd[mem_keys[0]].shape[0]
    config = CMFConfig(**cfg_dict)
    model = DeliberativeContinuousMeaningField(config)
    clean = {}
    for k, v in sd.items():
        k2 = k.replace("_orig_mod.module.", "").replace("module.", "")
        clean[k2] = v
    model.load_state_dict(clean, strict=False)
    model.eval()
    return model.to(device)

def run_eval(model, steps, curl_scale=0.10, anchor_scale=0.15, project_every_step=True):
    ids = torch.tensor([enc.encode(text)], dtype=torch.long, device=device)
    x = ids[:, :-1]
    y = ids[:, 1:]
    
    # Temporarily patch the forward pass of model to respect our custom parameters
    original_forward = model.forward
    
    def patched_forward(input_ids, labels=None, goal=None, target_length=None, return_states=False, gradient_checkpointing=False):
        batch_size, seq_len = input_ids.shape
        target_length = target_length or seq_len
        context = model.encoder(model.embedding(input_ids))[:, :target_length]
        cos, sin = model.rope(context, target_length)
        context = apply_rotary_pos_emb(context, cos, sin)
        z = model.initial_state(context)
        
        flat_context = context.reshape(batch_size * target_length, -1)
        flat_goal = None
        
        z_accum = z.to(torch.float32)
        
        for step_idx in range(steps):
            tau_value = (step_idx + 0.5) / float(steps)
            tau = torch.full((batch_size * target_length,), tau_value, dtype=z_accum.dtype, device=z_accum.device)
            flat_z = z_accum.to(z.dtype).reshape(batch_size * target_length, -1)
            
            velocity = model.field(flat_z, flat_context, tau, goal=flat_goal).reshape_as(z)
            proposal = z_accum.to(z.dtype) + velocity / float(steps)
            
            # Subspace Manifold Anchoring
            if anchor_scale == "scaled":
                step_anchor_scale = 0.15 / steps
            elif anchor_scale == "preserved":
                step_anchor_scale = 0.15 * (8.0 / steps)
            else:
                step_anchor_scale = anchor_scale
                
            if target_length > 1 and step_anchor_scale > 0:
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
            
            # Hamiltonian Symplectic Curl
            if curl_scale > 0:
                d_half = delta_z.size(-1) // 2
                delta_q = delta_z[..., :d_half]
                delta_p = delta_z[..., d_half:]
                symplectic_curl = torch.cat([delta_p, -delta_q], dim=-1)
                z_accum = z_accum + curl_scale * symplectic_curl
                
            # Context-Guided Manifold Projection (CGMP)
            if project_every_step:
                z_mean = z_accum.mean(dim=-1, keepdim=True)
                z_std = z_accum.std(dim=-1, keepdim=True)
                c_mean = context.mean(dim=-1, keepdim=True)
                c_std = context.std(dim=-1, keepdim=True)
                z_accum = ((z_accum - z_mean) / torch.clamp(z_std, min=1e-6)) * c_std + c_mean
                
        if not project_every_step:
            # Project only at the very end
            z_mean = z_accum.mean(dim=-1, keepdim=True)
            z_std = z_accum.std(dim=-1, keepdim=True)
            c_mean = context.mean(dim=-1, keepdim=True)
            c_std = context.std(dim=-1, keepdim=True)
            z_accum = ((z_accum - z_mean) / torch.clamp(z_std, min=1e-6)) * c_std + c_mean
            
        z_final = z_accum.to(z.dtype)
        logits = model.output(model.state_norm(z_final))
        
        result = {"logits": logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:target_length].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss
        return result

    # We need apply_rotary_pos_emb in local scope
    from cmf.model import apply_rotary_pos_emb
    
    with torch.no_grad():
        res = patched_forward(x, labels=y)
    return res["loss"].item()

# Test various configurations
model = load_clean_model()

configs = {
    "A: Default (curl=0.10, anchor=0.15, project=every_step)": dict(curl_scale=0.10, anchor_scale=0.15, project_every_step=True),
    "B: No Curl (curl=0.00, anchor=0.15, project=every_step)": dict(curl_scale=0.00, anchor_scale=0.15, project_every_step=True),
    "C: No Anchor (curl=0.10, anchor=0.00, project=every_step)": dict(curl_scale=0.10, anchor_scale=0.00, project_every_step=True),
    "D: Project Only at End (curl=0.10, anchor=0.15, project=end)": dict(curl_scale=0.10, anchor_scale=0.15, project_every_step=False),
    "E: Clean ODE (curl=0.00, anchor=0.00, project=end)": dict(curl_scale=0.00, anchor_scale=0.00, project_every_step=False),
    "F: Scaled Anchor (curl=0.10, anchor=0.15/S, project=every)": dict(curl_scale=0.10, anchor_scale="scaled", project_every_step=True),
    "G: Scaled Anchor + End Proj (curl=0.10, anchor=0.15/S, project=end)": dict(curl_scale=0.10, anchor_scale="scaled", project_every_step=False),
    "H: Preserved Anchor at S=8 (curl=0.10, anchor=1.2/S, project=every)": dict(curl_scale=0.10, anchor_scale="preserved", project_every_step=True),
    "I: Preserved Anchor + End Proj (curl=0.10, anchor=1.2/S, project=end)": dict(curl_scale=0.10, anchor_scale="preserved", project_every_step=False),
}

steps_to_test = [1, 2, 4, 8, 16, 32]

print("="*80)
print(f"{'Configuration':<55} | " + " | ".join(f"S={s:<2d}" for s in steps_to_test))
print("="*80)

for name, kwargs in configs.items():
    row_vals = []
    for s in steps_to_test:
        val = run_eval(model, s, **kwargs)
        row_vals.append(f"{val:.4f}")
    print(f"{name:<55} | " + " | ".join(row_vals))

print("="*80)
