import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.test_compositional_reasoning import generate_hop_batch, VAL_COLORS, ENTITIES

device = "cuda" if torch.cuda.is_available() else "cpu"
x, y = generate_hop_batch(2, 3, distance=5)
print("x:", x)
print("y:", y)
for i in range(2):
    print(f"Sequence {i}:")
    tokens = x[i].tolist()
    print("Tokens:", tokens)
    print("Target:", y[i, -1].item())
