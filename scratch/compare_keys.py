import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

c1 = torch.load("checkpoint_latest.pt", map_location="cpu", weights_only=False)
c2 = torch.load("checkpoint_latest(1).pt", map_location="cpu", weights_only=False)

s1 = c1["model"]
s2 = c2["model"]

# Clean keys
s1_clean = {k.replace("_orig_mod.module.", "").replace("module.", ""): v for k, v in s1.items()}
s2_clean = {k.replace("_orig_mod.module.", "").replace("module.", ""): v for k, v in s2.items()}

keys1 = set(s1_clean.keys())
keys2 = set(s2_clean.keys())

print(f"Checkpoint 1 clean keys count: {len(keys1)}")
print(f"Checkpoint 2 clean keys count: {len(keys2)}")

print("\nKeys in checkpoint 2 (new) but not in checkpoint 1:")
for k in (keys2 - keys1):
    print(f"  {k}: shape={list(s2_clean[k].shape)}")

print("\nKeys in checkpoint 1 but not in checkpoint 2 (new):")
for k in (keys1 - keys2):
    print(f"  {k}: shape={list(s1_clean[k].shape)}")

print("\nShape discrepancies between matching keys:")
for k in (keys1 & keys2):
    if s1_clean[k].shape != s2_clean[k].shape:
        print(f"  {k}: {list(s1_clean[k].shape)} vs {list(s2_clean[k].shape)}")
