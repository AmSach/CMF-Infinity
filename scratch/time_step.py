import sys, time, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

ckpt = torch.load("checkpoint_latest_new.pt", map_location="cpu", weights_only=False)
cfg_dict = ckpt["config"] if isinstance(ckpt["config"], dict) else ckpt["config"].__dict__.copy()
mem_keys = [k for k in ckpt["model"] if "field.memory" in k and "bank" not in k]
if mem_keys:
    cfg_dict["num_memory_anchors"] = ckpt["model"][mem_keys[0]].shape[0]

config = CMFConfig(**cfg_dict)
model = DeliberativeContinuousMeaningField(config).to(device)
model.train()

x = torch.randint(0, 1000, (64, 64), device=device)
y = torch.randint(0, 1000, (64, 64), device=device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Warmup with autocast
print("Warming up...")
with torch.amp.autocast("cuda", dtype=torch.float16):
    out = model(x, labels=y)
    loss = out["loss"]
loss.backward()

# Measure
print("Measuring...")
t0 = time.time()
for i in range(5):
    opt.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(x, labels=y)
        loss = out["loss"]
    loss.backward()
    opt.step()
t1 = time.time()
print(f"Average time per step with FP16 Autocast: {(t1-t0)/5 * 1000:.2f} ms")
