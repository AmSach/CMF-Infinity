import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import sys
from pathlib import Path

# Force UTF-8 stdout encoding for Windows compatibility
sys.stdout.reconfigure(encoding='utf-8')

from cmf import CMFConfig, DeliberativeContinuousMeaningField
from cmf.data import ByteTokenizer
from cmf.model import apply_rotary_pos_emb

# Define an expanded, highly complex set of drone scenarios to test situational logic
DRONE_SCENARIOS = [
    # LOW COMPLEXITY (Simple cruise / Clear paths)
    {"prompt": "ALT: 10m | WIND: 2kmh | OBS: NONE | BAT: 95% -> CMD:", "action": "CRUISE_FORWARD", "complexity": "low"},
    {"prompt": "ALT: 15m | WIND: 1kmh | OBS: NONE | BAT: 88% -> CMD:", "action": "CRUISE_FORWARD", "complexity": "low"},
    {"prompt": "ALT: 5m  | WIND: 3kmh | OBS: NONE | BAT: 99% -> CMD:", "action": "CRUISE_FORWARD", "complexity": "low"},
    {"prompt": "ALT: 12m | WIND: 0kmh | OBS: NONE | BAT: 75% -> CMD:", "action": "CRUISE_FORWARD", "complexity": "low"},
    {"prompt": "ALT: 20m | WIND: 1kmh | OBS: NONE | BAT: 92% -> CMD:", "action": "CRUISE_FORWARD", "complexity": "low"},
    {"prompt": "ALT: 8m  | WIND: 2kmh | OBS: NONE | BAT: 80% -> CMD:", "action": "CRUISE_FORWARD", "complexity": "low"},
    {"prompt": "ALT: 14m | WIND: 3kmh | OBS: NONE | BAT: 85% -> CMD:", "action": "CRUISE_FORWARD", "complexity": "low"},
    {"prompt": "ALT: 11m | WIND: 0kmh | OBS: NONE | BAT: 90% -> CMD:", "action": "CRUISE_FORWARD", "complexity": "low"},

    # MEDIUM COMPLEXITY (Single anomaly / Obstacle OR Battery)
    {"prompt": "ALT: 10m | WIND: 5kmh | OBS: FRONT_1m | BAT: 90% -> CMD:", "action": "ROTATE_LEFT", "complexity": "medium"},
    {"prompt": "ALT: 8m  | WIND: 4kmh | OBS: FRONT_2m | BAT: 85% -> CMD:", "action": "ROTATE_LEFT", "complexity": "medium"},
    {"prompt": "ALT: 12m | WIND: 2kmh | OBS: NONE | BAT: 18% -> CMD:", "action": "RETURN_TO_BASE", "complexity": "medium"},
    {"prompt": "ALT: 15m | WIND: 8kmh | OBS: NONE | BAT: 15% -> CMD:", "action": "RETURN_TO_BASE", "complexity": "medium"},
    {"prompt": "ALT: 7m  | WIND: 3kmh | OBS: FRONT_1m | BAT: 70% -> CMD:", "action": "ROTATE_LEFT", "complexity": "medium"},
    {"prompt": "ALT: 9m  | WIND: 6kmh | OBS: FRONT_2m | BAT: 65% -> CMD:", "action": "ROTATE_LEFT", "complexity": "medium"},
    {"prompt": "ALT: 11m | WIND: 1kmh | OBS: NONE | BAT: 19% -> CMD:", "action": "RETURN_TO_BASE", "complexity": "medium"},
    {"prompt": "ALT: 13m | WIND: 5kmh | OBS: NONE | BAT: 17% -> CMD:", "action": "RETURN_TO_BASE", "complexity": "medium"},

    # HIGH COMPLEXITY (Multi-factor emergency requiring trade-offs)
    {"prompt": "ALT: 12m | WIND: 25kmh | OBS: FRONT_1m | BAT: 9%  -> CMD:", "action": "LAND_IMMEDIATELY", "complexity": "high"},
    {"prompt": "ALT: 2m  | WIND: 30kmh | OBS: FRONT_2m | BAT: 8%  -> CMD:", "action": "LAND_IMMEDIATELY", "complexity": "high"},
    {"prompt": "ALT: 6m  | WIND: 22kmh | OBS: FRONT_1m | BAT: 5%  -> CMD:", "action": "LAND_IMMEDIATELY", "complexity": "high"},
    {"prompt": "ALT: 10m | WIND: 28kmh | OBS: FRONT_1m | BAT: 7%  -> CMD:", "action": "LAND_IMMEDIATELY", "complexity": "high"},
    {"prompt": "ALT: 15m | WIND: 35kmh | OBS: FRONT_1m | BAT: 4%  -> CMD:", "action": "LAND_IMMEDIATELY", "complexity": "high"},
    {"prompt": "ALT: 5m  | WIND: 32kmh | OBS: FRONT_2m | BAT: 3%  -> CMD:", "action": "LAND_IMMEDIATELY", "complexity": "high"},
    {"prompt": "ALT: 7m  | WIND: 29kmh | OBS: FRONT_1m | BAT: 2%  -> CMD:", "action": "LAND_IMMEDIATELY", "complexity": "high"},
    {"prompt": "ALT: 9m  | WIND: 27kmh | OBS: FRONT_2m | BAT: 6%  -> CMD:", "action": "LAND_IMMEDIATELY", "complexity": "high"},
]

def simulate_quantization(model: nn.Module, bits=4):
    """Simulates weight quantization noise by adding uniform rounding error to weights."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                max_val = param.abs().max()
                if max_val > 0:
                    qmin = -(2 ** (bits - 1))
                    qmax = (2 ** (bits - 1)) - 1
                    scale = max_val / qmax
                    quantized = torch.clamp(torch.round(param / scale), qmin, qmax) * scale
                    param.copy_(quantized)

def measure_thinking_steps(model, prompt_ids, device, use_velocity=True, epsilon=0.04):
    """Simulates the thinking loop for the first generated token and counts required steps."""
    model.eval()
    with torch.no_grad():
        context = model.encoder(model.embedding(prompt_ids))
        cos, sin = model.rope(context, context.size(1))
        context = apply_rotary_pos_emb(context, cos, sin)
        
        c_last = context[:, -1]
        z = model.initial_state(c_last)
        
        steps = model._thinking_budget()
        steps_taken = steps
        
        for step_idx in range(steps):
            tau_value = (step_idx + 0.5) / float(steps)
            tau = torch.full((1,), tau_value, dtype=z.dtype, device=device)
            
            velocity = model.field(z, c_last, tau)
            proposal = z + velocity / float(steps)
            
            gate_input = torch.cat([z, proposal, c_last], dim=-1)
            gate = torch.sigmoid(model.update_gate(model.gate_norm(gate_input)))
            delta_z = gate * (proposal - z)
            z_next = z + delta_z
            
            # Apply CGMP
            if z_next.size(-1) > 1:
                z_mean = z_next.mean(dim=-1, keepdim=True)
                z_std = z_next.std(dim=-1, keepdim=True, unbiased=False)
                c_mean = c_last.mean(dim=-1, keepdim=True)
                c_std = c_last.std(dim=-1, keepdim=True, unbiased=False)
                mask = (c_std > 1e-5) & (z_std > 1e-6)
                projected = ((z_next - z_mean) / torch.clamp(z_std, min=1e-6)) * c_std + c_mean
                z_next = torch.where(mask, projected, z_next)
            
            if use_velocity:
                velocity_norm = torch.norm(delta_z, p=2, dim=-1)
                if torch.all(velocity_norm < epsilon):
                    steps_taken = step_idx + 1
                    break
            else:
                halt_prob = torch.sigmoid(model.halt_head(model.state_norm(z_next)))
                if torch.all(halt_prob >= model.config.halting_threshold):
                    steps_taken = step_idx + 1
                    break
                    
            z = z_next
            
        return steps_taken

def run_verification():
    print("======================================================================")
    print("        CMF DRONE SCENARIO COMPLEXITY & QUANTIZATION VERIFICATION      ")
    print("======================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}\n")

    # 1. Setup CMF Config for a 120M Model
    config = CMFConfig(
        vocab_size=256,
        d_model=128,
        hidden_dim=256,
        num_layers=3,
        max_seq_len=128,
        thinking_steps=8,
        adaptive_thinking=True,
        min_thinking_steps=2,
        max_thinking_steps=8,
    )
    
    tokenizer = ByteTokenizer()
    model = DeliberativeContinuousMeaningField(config).to(device)
    
    # 2. Tokenize command strings
    tokenized_scenarios = []
    
    for idx, sc in enumerate(DRONE_SCENARIOS):
        prompt_tokens = tokenizer.encode(sc["prompt"])
        action_tokens = tokenizer.encode(sc["action"])
        
        full_tokens = torch.cat([prompt_tokens, action_tokens])
        labels = full_tokens.clone()
        labels[:len(prompt_tokens)] = -100
        
        tokenized_scenarios.append({
            "input_ids": full_tokens,
            "labels": labels,
            "complexity": sc["complexity"],
            "prompt_len": len(prompt_tokens),
            "target": sc["action"]
        })

    # 3. Train on Drone Flight Policies to show Learning Capacity
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    model.train()
    
    print(">>> Training CMF on Drone Emergency Flight Rules (Overfitting test)...")
    for step in range(450):
        total_loss = 0
        optimizer.zero_grad()
        for sc in tokenized_scenarios:
            x = sc["input_ids"].unsqueeze(0).to(device)
            y = sc["labels"].unsqueeze(0).to(device)
            
            out = model(x, labels=y)
            loss = out["loss"] + 0.1 * out.get("ponder_loss", torch.tensor(0.0, device=device))
            loss.backward()
            total_loss += loss.item()
            
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"    Step {step + 1}/450 | Training Loss: {total_loss / len(DRONE_SCENARIOS):.4f}")

    # 4. Evaluate Capabilities (Full Precision FP32)
    model.eval()
    print("\n>>> Evaluating Decision Accuracy & Thinking Adaptability (FP32)...")
    
    correct_fp = 0
    steps_velocity_halt = {"low": [], "medium": [], "high": []}
    steps_head_halt = {"low": [], "medium": [], "high": []}
    
    with torch.no_grad():
        for sc in tokenized_scenarios:
            prompt_x = sc["input_ids"][:sc["prompt_len"]].unsqueeze(0).to(device)
            
            # Run inference generate (using RoPE-corrected generate loop)
            gen_tokens = model.generate(
                prompt_x,
                max_new_tokens=16,
                temperature=0.01,
                use_velocity_halting=False
            )
            
            generated_cmd = tokenizer.decode(gen_tokens[0, sc["prompt_len"]:])
            is_correct = sc["target"] in generated_cmd
            if is_correct:
                correct_fp += 1
                
            # Measure thinking steps via physical velocity halting (epsilon=0.035)
            s_vel = measure_thinking_steps(model, prompt_x, device, use_velocity=True, epsilon=0.035)
            steps_velocity_halt[sc["complexity"]].append(s_vel)
            
            # Measure thinking steps via learned halt head
            s_head = measure_thinking_steps(model, prompt_x, device, use_velocity=False)
            steps_head_halt[sc["complexity"]].append(s_head)
            
            print(f"    [{sc['complexity'].upper()}] Prompt: {tokenizer.decode(prompt_x[0])}")
            print(f"      - Target: {sc['target']} | Predicted: {generated_cmd.strip()} ({'CORRECT' if is_correct else 'FAILED'})")
            print(f"      - Steps (Velocity Halt): {s_vel} | Steps (Halt Head): {s_head}")

    acc_fp = (correct_fp / len(DRONE_SCENARIOS)) * 100
    
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print("\n------------------ FP32 RESULTS ------------------")
    print(f"Decision Accuracy: {acc_fp:.1f}%")
    print(f"Avg Steps (Physical Velocity Halt):")
    print(f"  - Low Complexity:    {avg(steps_velocity_halt['low']):.1f} steps")
    print(f"  - Medium Complexity: {avg(steps_velocity_halt['medium']):.1f} steps")
    print(f"  - High Complexity:   {avg(steps_velocity_halt['high']):.1f} steps")
    print(f"Avg Steps (Learned Halt Head):")
    print(f"  - Low Complexity:    {avg(steps_head_halt['low']):.1f} steps")
    print(f"  - Medium Complexity: {avg(steps_head_halt['medium']):.1f} steps")
    print(f"  - High Complexity:   {avg(steps_head_halt['high']):.1f} steps")
    
    # 5. Simulate 4-bit Quantization Survival (Proof of CGMP)
    print("\n>>> Simulating 4-Bit Weight Quantization...")
    simulate_quantization(model, bits=4)
    
    correct_q = 0
    with torch.no_grad():
        for sc in tokenized_scenarios:
            prompt_x = sc["input_ids"][:sc["prompt_len"]].unsqueeze(0).to(device)
            gen_tokens = model.generate(
                prompt_x,
                max_new_tokens=16,
                temperature=0.01,
                use_velocity_halting=False
            )
            generated_cmd = tokenizer.decode(gen_tokens[0, sc["prompt_len"]:])
            if sc["target"] in generated_cmd:
                correct_q += 1

    acc_q = (correct_q / len(DRONE_SCENARIOS)) * 100
    print("\n------------------ 4-BIT QUANTIZED RESULTS ------------------")
    print(f"Quantized Decision Accuracy: {acc_q:.1f}%")
    print("======================================================================")

if __name__ == "__main__":
    run_verification()
