import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField
from cmf.presets import get_preset
from cmf.runtime import resolve_device

def evaluate_math_reward(completion: str, target: str) -> float:
    """
    Simulates a discrete external verifier (Compiler / Math Engine).
    If the model's generated logical trajectory concludes with the exact target,
    it receives a massive scalar reward, driving the continuous vector field 
    toward self-discovered algorithms instead of human imitation.
    """
    if not completion or not target:
        return -1.0
    completion_clean = completion.strip().lower()
    target_clean = target.strip().lower()
    
    # Check if the exact target logic string is embedded anywhere in the conclusion
    if target_clean in completion_clean:
        return 10.0
    return -2.0 # Negative penalty for hallucination / incorrect logic

class MathSelfPlayDataset(Dataset):
    def __init__(self, tokenizer, num_samples=1000):
        self.tokenizer = tokenizer
        self.samples = []
        # Procedurally generate arithmetic reasoning challenges
        for a in range(1, 20):
            for b in range(1, 20):
                self.samples.append({
                    "prompt": f"User: Calculate {a} + {b} and explain the steps.\nAssistant: <think>",
                    "target": str(a + b)
                })
        self.samples = self.samples[:num_samples]

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="infinity-reasoning-0.12b")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 CMF INFINITY: SELF-PLAY REINFORCEMENT LEARNING (GRPO)")
    print("=" * 70)

    device = resolve_device("auto")
    
    checkpoint_path = ROOT / "checkpoint_latest.pt"
    if not checkpoint_path.exists():
        print("Base checkpoint missing. Run pretraining or SFT first.")
        sys.exit(1)

    print("Loading Base Policy Model...")
    payload = torch.load(checkpoint_path, map_location="cpu")
    preset = get_preset(args.preset)
    config = CMFConfig(**preset.config.__dict__)
    
    model = DeliberativeContinuousMeaningField(config)
    
    # Load state dict without strict enforcement since RL might add its own projection heads
    state_dict = payload.get("model", payload)
    clean_state_dict = {}
    for k, v in state_dict.items():
        key = k.replace("_orig_mod.module.", "").replace("module.", "")
        clean_state_dict[key] = v
        
    model.load_state_dict(clean_state_dict, strict=False)
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    dataset = MathSelfPlayDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("\n--- Initiating Algorithmic Self-Play ---")
    
    for step, batch in enumerate(dataloader):
        if step >= args.steps:
            break
            
        prompts = batch["prompt"]
        targets = batch["target"]
        
        # 1. Rollout Generation (Model explores logical trajectories)
        model.eval()
        trajectories = []
        log_probs = []
        
        with torch.no_grad():
            for p in prompts:
                input_ids = torch.tensor([tokenizer.encode(p)], device=device)
                
                # We use the dynamic physics engine to generate the rollout
                generated = model.generate(
                    input_ids, 
                    max_new_tokens=40,
                    temperature=0.7, 
                    use_velocity_halting=True
                )
                
                text_out = tokenizer.decode(generated[0][input_ids.size(1):].tolist())
                trajectories.append((input_ids, generated, text_out))
        
        # 2. Reward Verification (External Engine checking logic)
        rewards = torch.tensor([evaluate_math_reward(t[2], tgt) for t, tgt in zip(trajectories, targets)], device=device)
        
        # 3. Group Relative Policy Optimization (GRPO) Step
        # Standardize rewards across the batch to reduce variance without needing a Critic model
        if rewards.std() > 0:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = torch.zeros_like(rewards)
            
        model.train()
        optimizer.zero_grad()
        
        loss = 0.0
        for i, (input_ids, generated, _) in enumerate(trajectories):
            # Compute log probabilities of the generated sequence under current policy
            outputs = model(generated[:, :-1])
            logits = outputs["logits"]
            
            # Extract target token probabilities
            target_ids = generated[:, 1:].unsqueeze(-1)
            token_log_probs = F.log_softmax(logits, dim=-1).gather(-1, target_ids).squeeze(-1)
            
            # Policy gradient loss scaled by external advantage
            trajectory_loss = - (token_log_probs.sum() * advantages[i])
            loss += trajectory_loss
            
        loss = loss / len(prompts)
        loss.backward()
        
        # Protect gradients from explosion during massive reward signals
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            avg_reward = rewards.mean().item()
            print(f"Step {step} | Avg Reward: {avg_reward:+.2f} | Policy Loss: {loss.item():.4f}")
            print(f"   Sample Rollout: {trajectories[0][2][:50]}... -> Extracted Logic: {targets[0]}")

    print("\n--- Self-Play Logic Calibration Complete ---")
    out_path = ROOT / "checkpoint_rlhf.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved verified reasoning weights to {out_path}")

if __name__ == "__main__":
    main()
