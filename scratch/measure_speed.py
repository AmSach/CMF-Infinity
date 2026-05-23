import sys, time, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn.functional as F
from cmf.model_v2 import PersistentStateCMF, CMFv2Config

dev = "cpu"
config = CMFv2Config(vocab_size=50, d_model=64, hidden_dim=128, num_slots=8, thinking_steps=4)
model = PersistentStateCMF(config).to(dev)

# Compile model
print("Compiling model...")
compiled_model = torch.compile(model)

optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=1e-3)

x = torch.randint(0, 50, (32, 30), device=dev)
y = torch.randint(0, 2, (32,), device=dev) + 30

# Warmup compiled model
print("Warming up compiled model...")
out = compiled_model(x)
loss = F.cross_entropy(out["logits"][:, -1, :], y)
loss.backward()

# Timing compiled model
print("Timing compiled model...")
start = time.time()
for _ in range(50):
    optimizer.zero_grad()
    out = compiled_model(x)
    loss = F.cross_entropy(out["logits"][:, -1, :], y)
    loss.backward()
    optimizer.step()
end = time.time()
print(f"Compiled model time for 50 steps: {end - start:.4f}s")
