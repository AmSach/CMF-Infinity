import torch
import sys
from pathlib import Path

# Add current dir to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

ckpt_path = "checkpoint_latest(1).pt"
print(f"Loading checkpoint metadata from {ckpt_path} using weights_only=False...")
try:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("Checkpoint keys:", list(checkpoint.keys()))
    
    if "config" in checkpoint:
        print("Config details:", checkpoint["config"])
        if hasattr(checkpoint["config"], "__dict__"):
            print("Config dict:", checkpoint["config"].__dict__)
    
    if "step" in checkpoint:
        print("Step:", checkpoint["step"])
    if "epoch" in checkpoint:
        print("Epoch:", checkpoint["epoch"])
    if "opt" in checkpoint:
        print("Opt key exists:", True)
    if "optimizer" in checkpoint:
        print("Optimizer key exists:", True)
    
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        print(f"Model state dict contains {len(state_dict)} keys.")
        # Print first few keys
        for i, (k, v) in enumerate(state_dict.items()):
            if i < 20:
                print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
            else:
                break
except Exception as e:
    import traceback
    traceback.print_exc()
