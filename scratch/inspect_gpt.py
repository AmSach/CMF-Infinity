import sys, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.test_compositional_reasoning import generate_hop_batch, TinyGPTLM, VAL_COLORS, VOCAB_SIZE

device = "cuda" if torch.cuda.is_available() else "cpu"

gpt_model = TinyGPTLM(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    nhead=4,
    num_layers=2,
    hidden_dim=128,
    max_seq_len=5000,
).to(device)

# Generate a 1-hop batch
x, y, target_colors = generate_hop_batch(1, 1, distance=5)
print("Input sequence x:", x[0].tolist())
print("Target labels y :", y[0].tolist())

# Train for 200 steps
opt = torch.optim.AdamW(gpt_model.parameters(), lr=1e-3)
for step in range(200 + 1):
    gpt_model.train()
    opt.zero_grad()
    bx, by, b_colors = generate_hop_batch(8, 1, distance=5)
    out = gpt_model(bx)
    loss = torch.nn.functional.cross_entropy(
        out["logits"][:, :-1].reshape(-1, VOCAB_SIZE),
        by[:, :-1].reshape(-1),
        ignore_index=-100
    )
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.4f}")

# Evaluate and print prediction at each position
gpt_model.eval()
with torch.no_grad():
    out = gpt_model(x)
    logits = out["logits"][0]  # [L, V]
    preds = torch.argmax(logits, dim=-1).tolist()
    
    print("\nToken-by-token comparison:")
    print(f"{'Index':<6} | {'Input Token':<12} | {'Label':<6} | {'Prediction':<10} | {'Correct?':<8}")
    print("-" * 55)
    for t in range(len(x[0]) - 1):
        input_token = x[0, t].item()
        label = y[0, t].item()
        pred = preds[t]
        correct = "Yes" if pred == label else "No"
        label_str = str(label) if label != -100 else "IGNORE"
        print(f"{t:<6d} | {input_token:<12d} | {label_str:<6} | {pred:<10d} | {correct:<8}")
