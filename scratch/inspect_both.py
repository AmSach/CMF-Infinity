import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

for ckpt_name in ["checkpoint_latest.pt", "checkpoint_latest(1).pt"]:
    print(f"\n===== Inspecting {ckpt_name} =====")
    if not Path(ckpt_name).exists():
        print("File does not exist.")
        continue
    try:
        ckpt = torch.load(ckpt_name, map_location="cpu", weights_only=False)
        print("Keys:", list(ckpt.keys()))
        if "training" in ckpt:
            print("Training details:", ckpt["training"])
        if "config" in ckpt:
            print("Config details:", ckpt["config"])
        if "model" in ckpt:
            print(f"Model parameters: {sum(p.numel() for p in ckpt['model'].values())}")
    except Exception as e:
        print(f"Error loading {ckpt_name}:", e)
