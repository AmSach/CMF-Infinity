import sys, time, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def trace_step():
    print("Loading config...")
    ckpt = torch.load("checkpoint_latest_new.pt", map_location="cpu", weights_only=False)
    cfg_src = ckpt["config"]
    cfg_dict = cfg_src if isinstance(cfg_src, dict) else cfg_src.__dict__.copy()
    sd = ckpt["model"]
    mem_keys = [k for k in sd if "field.memory" in k and "bank" not in k]
    if mem_keys:
        cfg_dict["num_memory_anchors"] = sd[mem_keys[0]].shape[0]
        
    config = CMFConfig(**cfg_dict)
    print("Initializing fresh model...")
    model = DeliberativeContinuousMeaningField(config).to(device)
    model.train()
    
    print("Generating batch...")
    x = torch.randint(0, 1000, (64, 64), device=device)
    y = torch.randint(0, 1000, (64, 64), device=device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    t0 = time.time()
    
    # 1. Forward Pass
    print("[TRACE] Starting embedding...")
    emb = model.embedding(x)
    print(f"[TRACE] Embedding shape: {emb.shape} | Time: {time.time()-t0:.4f}s")
    
    print("[TRACE] Starting encoder...")
    context = model.encoder(emb)[:, :64]
    print(f"[TRACE] Context shape: {context.shape} | Time: {time.time()-t0:.4f}s")
    
    print("[TRACE] Starting rope & initial state...")
    cos, sin = model.rope(context, 64)
    from cmf.model import apply_rotary_pos_emb
    context = apply_rotary_pos_emb(context, cos, sin)
    z = model.initial_state(context)
    print(f"[TRACE] Initial state shape: {z.shape} | Time: {time.time()-t0:.4f}s")
    
    print("[TRACE] Starting ODE solver integration...")
    # Trace inside ODE solver
    batch_size = 64
    target_length = 64
    flat_context = context.reshape(batch_size * target_length, -1)
    flat_goal = None
    z_accum = z.to(torch.float32)
    steps = 8
    step_anchor_scale = min(0.9, 0.15 * (8.0 / steps))
    
    for step_idx in range(steps):
        t_sub = time.time()
        tau_value = (step_idx + 0.5) / float(steps)
        tau = torch.full(
            (batch_size * target_length,),
            tau_value,
            dtype=z_accum.dtype,
            device=z_accum.device,
        )
        flat_z = z_accum.to(z.dtype).reshape(batch_size * target_length, -1)
        
        velocity = model.field(flat_z, flat_context, tau, goal=flat_goal).reshape_as(z)
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
        
        d_half = delta_z.size(-1) // 2
        delta_q = delta_z[..., :d_half]
        delta_p = delta_z[..., d_half:]
        symplectic_curl = torch.cat([delta_p, -delta_q], dim=-1)
        z_accum = z_accum + 0.10 * symplectic_curl
        
        # Langevin Stochastic Noise Injection (Regularization during training)
        if model.training:
            noise = torch.randn_like(z_accum) * 1e-4
            jitter = torch.sin(z_accum * 1000.0) * 1e-6
            z_accum = z_accum + noise + jitter
            
        z_temp = z_accum.to(z.dtype)
        z_mean = z_temp.mean(dim=-1, keepdim=True)
        z_std = z_temp.std(dim=-1, keepdim=True)
        c_mean = context.mean(dim=-1, keepdim=True)
        c_std = context.std(dim=-1, keepdim=True)
        z_temp_proj = ((z_temp - z_mean) / torch.clamp(z_std, min=1e-6)) * c_std + c_mean
        
        halt_prob = torch.sigmoid(model.halt_head(model.state_norm(z_temp_proj)))
        
        print(f"  [ODE] Step {step_idx} | Time: {time.time()-t_sub:.4f}s")
        
    z_accum_temp = z_accum.to(z.dtype)
    z_mean = z_accum_temp.mean(dim=-1, keepdim=True)
    z_std = z_accum_temp.std(dim=-1, keepdim=True)
    c_mean = context.mean(dim=-1, keepdim=True)
    c_std = context.std(dim=-1, keepdim=True)
    z_final_proj = ((z_accum_temp - z_mean) / torch.clamp(z_std, min=1e-6)) * c_std + c_mean
    
    print(f"[TRACE] Finished solver | Time: {time.time()-t0:.4f}s")
    
    # 2. Output logits
    print("[TRACE] Starting logits...")
    logits = model.output(model.state_norm(z_final_proj))
    print(f"[TRACE] Logits shape: {logits.shape} | Time: {time.time()-t0:.4f}s")
    
    # 3. Loss computation
    print("[TRACE] Starting loss...")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = y[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    print(f"[TRACE] Loss: {loss.item()} | Time: {time.time()-t0:.4f}s")
    
    # 4. Backward Pass
    print("[TRACE] Starting backward pass...")
    loss.backward()
    print(f"[TRACE] Finished backward pass | Time: {time.time()-t0:.4f}s")
    
    # 5. Optimizer step
    print("[TRACE] Starting optimizer step...")
    opt.step()
    print(f"[TRACE] Finished step | Time: {time.time()-t0:.4f}s")

if __name__ == "__main__":
    import math
    trace_step()
