import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

ckpt = torch.load("checkpoint_latest(1).pt", map_location="cpu", weights_only=False)
if "training" in ckpt:
    print("Training metadata:", ckpt["training"])
else:
    print("No training metadata in checkpoint.")
