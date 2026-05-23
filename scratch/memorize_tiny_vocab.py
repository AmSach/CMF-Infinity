import sys, time, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define small config with tiny vocab for rapid training
config = CMFConfig(
    vocab_size=1000,
    d_model=128,
    hidden_dim=256,
    num_layers=4,
    max_seq_len=64,
    thinking_steps=8,
    adaptive_thinking=False,
    num_memory_anchors=16,
    field_depth=2
)

print("\nInitializing tiny CMF model...")
model = DeliberativeContinuousMeaningField(config).to(device)
model.train()

# Compute parameter count
params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {params:,}")

# Generate 1 batch: 64 sequences, length 64, vocab 1000
batch_size = 64
seq_len = 64
torch.manual_seed(42)
input_ids = torch.randint(0, 1000, (batch_size, seq_len + 1), device=device)
x = input_ids[:, :-1]
y = input_ids[:, 1:]

print(f"Batch shape: x={x.shape}, y={y.shape}")

# High LR for quick memorization
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

steps = 400
t0 = time.time()

for step in range(steps + 1):
    opt.zero_grad()
    
    with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
        out = model(x, labels=y)
        loss = out["loss"]
        
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    opt.step()
    
    if step % 20 == 0 or step == steps:
        elapsed = time.time() - t0
        print(f"  Step {step:3d}/{steps} | Loss: {loss.item():.6f} | Elapsed: {elapsed:.2f}s")
        
    if loss.item() < 0.001:
        print(f"\n[SUCCESS] Loss successfully drove below 0.001 at step {step}!")
        break

final_loss = loss.item()
print("\n" + "="*80)
print(f"MEMORIZATION TEST RESULT:")
print(f"Final Loss: {final_loss:.6f}")
if final_loss < 0.01:
    print("VERDICT: PASS (Gradients healthy, recurrence stable, normalization correct)")
else:
    print("VERDICT: FAIL (Optimization/convergence failed)")
print("="*80)
