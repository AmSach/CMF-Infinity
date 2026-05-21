import torch
import sys
from pathlib import Path

# Add current dir to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from cmf.model import DeliberativeContinuousMeaningField, ParallelContinuousMeaningField

ckpt_path = "checkpoint_latest(1).pt"
print(f"Loading checkpoint {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

state_dict = checkpoint["model"]
# clean keys
clean_state_dict = {k.replace("_orig_mod.module.", "").replace("module.", ""): v for k, v in state_dict.items()}

# Let's check keys containing 'router' or 'deliberation' or 'memory'
reasoning_keys = [k for k in clean_state_dict.keys() if any(w in k.lower() for w in ["router", "deliberation", "memory", "thinking", "velocity"])]
print("Reasoning-related keys found in checkpoint:")
for k in reasoning_keys:
    print(f"  {k}: shape={list(clean_state_dict[k].shape)}")

config = checkpoint["config"]
print("Instantiating models to check compatibility:")
try:
    p_model = ParallelContinuousMeaningField(config)
    p_keys = set(p_model.state_dict().keys())
    missing_p = p_keys - set(clean_state_dict.keys())
    unexpected_p = set(clean_state_dict.keys()) - p_keys
    print(f"Parallel CMF: missing keys={len(missing_p)}, unexpected keys={len(unexpected_p)}")
except Exception as e:
    print("Parallel CMF instantiation failed:", e)

try:
    d_model = DeliberativeContinuousMeaningField(config)
    d_keys = set(d_model.state_dict().keys())
    missing_d = d_keys - set(clean_state_dict.keys())
    unexpected_d = set(clean_state_dict.keys()) - d_keys
    print(f"Deliberative CMF: missing keys={len(missing_d)}, unexpected keys={len(unexpected_d)}")
    if len(missing_d) > 0:
        print("Missing Deliberative keys:", missing_d)
    if len(unexpected_d) > 0:
        print("Unexpected Deliberative keys:", unexpected_d)
except Exception as e:
    print("Deliberative CMF instantiation failed:", e)
