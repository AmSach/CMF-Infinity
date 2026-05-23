import sys, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.test_compositional_reasoning import generate_hop_batch, TinyGPTLM, DeliberativeContinuousMeaningField, CMFConfig, VAL_COLORS, VOCAB_SIZE

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing TinyGPT...")
gpt_model = TinyGPTLM(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    nhead=4,
    num_layers=4,
    hidden_dim=128,
    max_seq_len=5000,
).to(device)

print("Training TinyGPT on 1-hop transitive retrieval with batch_size=64...")
opt = torch.optim.AdamW(gpt_model.parameters(), lr=1e-3)
for step in range(1500 + 1):
    gpt_model.train()
    opt.zero_grad()
    bx, by, b_colors = generate_hop_batch(64, 1, distance=0)
    out = gpt_model(bx)
    loss = torch.nn.functional.cross_entropy(
        out["logits"][:, :-1].reshape(-1, VOCAB_SIZE),
        by[:, :-1].reshape(-1),
        ignore_index=-100
    )
    loss.backward()
    opt.step()
    if step % 250 == 0:
        gpt_model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(10):
                x, y, target_colors = generate_hop_batch(1, 1, distance=0)
                prefix = x[:, :-1]
                out_eval = gpt_model(prefix)
                logits = out_eval["logits"][0, -1, VAL_COLORS].tolist()
                pred = torch.argmax(out_eval["logits"][0, -1, :]).item()
                is_correct = (pred == target_colors[0].item())
                if is_correct:
                    correct += 1
                if step == 1500 or step == 0:
                    print(f"Trial {i} | Seq: {x[0].tolist()} | Target: {target_colors[0].item()} | Pred: {pred} | Logits: {dict(zip(VAL_COLORS, logits))}")
        print(f"Step {step:4d} | Loss: {loss.item():.4f} | 1-Hop Eval Accuracy: {correct/10:.1%}")
