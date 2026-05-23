import sys, time, torch, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField
from cmf.baselines import TinyGPTLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Task configuration
VOCAB_SIZE = 100
KEY_A = 0
EQUALS = 1
VAL_COLORS = [2, 3, 4, 5]  # blue, red, green, yellow
GARBAGE_MIN = 6
GARBAGE_MAX = 59
QUERY_A = 60

def generate_batch(batch_size, distance):
    seq_len = distance + 4
    x = torch.randint(GARBAGE_MIN, GARBAGE_MAX + 1, (batch_size, seq_len), device=device)
    y = torch.full((batch_size, seq_len), -100, device=device)
    for b in range(batch_size):
        val = VAL_COLORS[torch.randint(0, len(VAL_COLORS), (1,)).item()]
        x[b, 0] = KEY_A
        x[b, 1] = EQUALS
        x[b, 2] = val
        x[b, -1] = QUERY_A
        y[b, -1] = val
    return x, y

def train_model_randomized(model, lr=1e-3, steps=400):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    for step in range(steps + 1):
        # Randomize distance between 50 and 250 for each batch
        dist = torch.randint(50, 250, (1,)).item()
        x, y = generate_batch(16, dist)
        opt.zero_grad()
        
        with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
            out = model(x)
            logits = out["logits"]
            shift_logits = logits[:, -1, :]
            shift_labels = y[:, -1]
            loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if step % 50 == 0:
            with torch.no_grad():
                preds = torch.argmax(shift_logits, dim=-1)
                acc = (preds == shift_labels).float().mean().item()
            print(f"  Step {step:3d} (dist={dist:3d}) | Loss: {loss.item():.4f} | Training Acc: {acc:.0%}")

def evaluate_extrapolation(model, distances, num_trials=30):
    model.eval()
    accuracies = []
    
    for d in distances:
        correct = 0
        total = 0
        for _ in range(num_trials):
            x, y = generate_batch(1, d)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
                    out = model(x)
                    logits = out["logits"]
                    pred = torch.argmax(logits[0, -1, :]).item()
                    label = y[0, -1].item()
                    if pred == label:
                        correct += 1
                    total += 1
            del x, y, out, logits
            if device == "cuda":
                torch.cuda.empty_cache()
                    
        acc = correct / total
        accuracies.append(acc)
        print(f"  Distance: {d:<5d} | Accuracy: {acc:.1%}")
        
    return accuracies

print("Initializing models...")
cmf_config = CMFConfig(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    hidden_dim=128,
    num_layers=4,
    max_seq_len=35000,
    adaptive_thinking=False,
    thinking_steps=4,
    use_global_memory_router=True,
)
cmf_model = DeliberativeContinuousMeaningField(cmf_config).to(device)

gpt_model = TinyGPTLM(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    nhead=4,
    num_layers=2,
    hidden_dim=128,
    max_seq_len=35000,
).to(device)

print("\n" + "="*60)
print("TRAINING CMF MODEL WITH RANDOMIZED DISTANCES [50, 250]")
print("="*60)
train_model_randomized(cmf_model, lr=1e-3, steps=400)

print("\n" + "="*60)
print("TRAINING TINY-GPT MODEL WITH RANDOMIZED DISTANCES [50, 250]")
print("="*60)
train_model_randomized(gpt_model, lr=1e-3, steps=400)

distances = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]

print("\n" + "="*60)
print("EVALUATING CMF MODEL EXTRAPOLATION")
print("="*60)
cmf_accs = evaluate_extrapolation(cmf_model, distances)

print("\n" + "="*60)
print("EVALUATING TINY-GPT MODEL EXTRAPOLATION")
print("="*60)
gpt_accs = evaluate_extrapolation(gpt_model, distances)

print("\n" + "="*60)
print("RETRIEVAL ACCURACY VS DISTANCE COMPARISON (RANDOMIZED TRAINING)")
print("="*60)
print(f"{'Distance':<10} | {'CMF Model (O(1) Field)':<23} | {'Tiny-GPT (Transformer)':<23}")
print("-" * 60)
for d, c_acc, g_acc in zip(distances, cmf_accs, gpt_accs):
    print(f"{d:<10d} | {c_acc:<23.1%} | {g_acc:<23.1%}")
print("="*60)
