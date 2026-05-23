"""
Honest evaluation of the new CMF checkpoint.
Tests:
  1. Can it overfit ONE sentence? (memorization capacity test)
  2. Does loss decrease when we train on tiny data for a few steps? (trainability)
  3. Does adding more solver steps actually lower loss? (the core CMF claim)
  4. Generation quality / repetition / coherence
  5. Vocab distribution: is it stuck in a degenerate mode?
"""
import sys, io, math, torch, tiktoken
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField

CKPT_NEW  = Path("e:/CMF/checkpoint_latest_new.pt")   # step 2040, new architecture
CKPT_OLD  = Path("e:/CMF/checkpoint_latest(1).pt")    # step 22780, old architecture

enc = tiktoken.get_encoding("gpt2")

# -------------------------------------------------------------------
def load_model(path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg_src = ckpt["config"]
    cfg_dict = cfg_src if isinstance(cfg_src, dict) else cfg_src.__dict__.copy()
    # detect memory anchor size mismatch
    sd = ckpt["model"]
    mem_keys = [k for k in sd if "field.memory" in k and "bank" not in k]
    if mem_keys:
        actual = sd[mem_keys[0]].shape[0]
        cfg_dict["num_memory_anchors"] = actual
    config = CMFConfig(**cfg_dict)
    model = DeliberativeContinuousMeaningField(config)
    clean = {}
    for k, v in sd.items():
        k2 = k.replace("_orig_mod.module.", "").replace("module.", "")
        clean[k2] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print(f"  [WARN] {len(missing)} missing keys in {path.name}")
    model.eval()
    return model

def token_loss(model, text, device="cpu"):
    model.to(device)
    ids = torch.tensor([enc.encode(text)], dtype=torch.long, device=device)
    if ids.size(1) < 2:
        return float("nan")
    with torch.no_grad():
        out = model(ids[:, :-1], labels=ids[:, 1:])
    return out["loss"].item()

def generate(model, prompt, max_new=40, temp=0.8, device="cpu"):
    model.to(device)
    ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, temperature=temp)
    return enc.decode(out[0].tolist())

def repetition_rate(text):
    words = text.split()
    if len(words) < 2: return 0.0
    counts = Counter(words)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(words)

def solver_step_loss(model, text, steps_list, device="cpu"):
    """Test the core CMF claim: more thinking steps = lower loss."""
    model.to(device)
    ids = torch.tensor([enc.encode(text)], dtype=torch.long, device=device)
    results = {}
    
    # Store original values
    orig_adaptive = model.config.adaptive_thinking
    orig_steps = model.config.thinking_steps
    
    # Disable adaptive thinking so thinking_steps is actually used
    model.config.adaptive_thinking = False
    
    for s in steps_list:
        model.config.thinking_steps = s
        with torch.no_grad():
            out = model(ids[:, :-1], labels=ids[:, 1:])
        results[s] = out["loss"].item()
        
    # Restore original values
    model.config.adaptive_thinking = orig_adaptive
    model.config.thinking_steps = orig_steps
    return results

def overfit_one_sentence(model, sentence, steps=50, lr=1e-3, device="cpu"):
    """Can the model memorize one sentence with gradient updates?"""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ids = torch.tensor([enc.encode(sentence)], dtype=torch.long, device=device)
    losses = []
    for i in range(steps):
        opt.zero_grad()
        out = model(ids[:, :-1], labels=ids[:, 1:])
        loss = out["loss"]
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses[0], losses[-1]

# -------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print("=" * 60)

print("\nLoading NEW checkpoint (step 2040, new architecture)...")
new_model = load_model(CKPT_NEW)
print("Loading OLD checkpoint (step 22780, old architecture)...")
old_model = load_model(CKPT_OLD)

# -------------------------------------------------------------------
print("\n[TEST 1] Loss on simple structured sentences")
print("-" * 60)
test_sentences = [
    "The capital of France is Paris.",
    "One plus one equals two.",
    "def hello(): print('hello')",
    "The quick brown fox jumps over the lazy dog.",
]
for s in test_sentences:
    new_l = token_loss(new_model, s, DEVICE)
    old_l = token_loss(old_model, s, DEVICE)
    better = "NEW better" if new_l < old_l else "OLD better"
    print(f"  [{better}]  new={new_l:.3f}  old={old_l:.3f}  | {s[:50]}")

# -------------------------------------------------------------------
print("\n[TEST 2] Does more thinking steps reduce loss? (core CMF claim)")
print("-" * 60)
test_text = "The capital of France is Paris."
steps_results = solver_step_loss(new_model, test_text, [1, 2, 4, 8, 16], DEVICE)
for s, l in steps_results.items():
    print(f"  thinking_steps={s:2d}  =>  loss={l:.4f}")
monotone = list(steps_results.values()) == sorted(steps_results.values(), reverse=True)
print(f"  Loss monotonically decreasing with steps? {'YES' if monotone else 'NO (claim not holding)'}")

# -------------------------------------------------------------------
print("\n[TEST 3] Generation quality & repetition")
print("-" * 60)
prompts = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time,",
    "The meaning of life is",
]
for p in prompts:
    new_gen = generate(new_model, p, max_new=40, temp=0.8, device=DEVICE)
    new_rep = repetition_rate(new_gen)
    old_gen = generate(old_model, p, max_new=40, temp=0.8, device=DEVICE)
    old_rep = repetition_rate(old_gen)
    print(f"\n  Prompt: \"{p}\"")
    print(f"  NEW (rep={new_rep:.0%}): {new_gen[:100]}")
    print(f"  OLD (rep={old_rep:.0%}): {old_gen[:100]}")

# -------------------------------------------------------------------
print("\n[TEST 4] Vocab diversity (top-10 tokens predicted on fixed prompt)")
print("-" * 60)
probe = "The meaning of life is"
probe_ids = torch.tensor([enc.encode(probe)], dtype=torch.long)
new_model.to(DEVICE)
with torch.no_grad():
    out = new_model(probe_ids.to(DEVICE))
    logits = out["logits"][0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top10 = torch.topk(probs, 10)
print("  Top 10 tokens from new model on probe prompt:")
for prob, idx in zip(top10.values.tolist(), top10.indices.tolist()):
    tok = enc.decode([idx])
    print(f"    {tok!r:<20} p={prob:.4f}")

# -------------------------------------------------------------------
print("\n[TEST 5] Can it memorize? (overfit 1 sentence for 50 steps)")
print("-" * 60)
sentence = "The capital of France is Paris."
# Run on new_model since it's the last test and we don't need to preserve it
start_l, end_l = overfit_one_sentence(new_model, sentence, steps=50, device=DEVICE)
dropped = start_l - end_l
print(f"  Start loss: {start_l:.4f}  |  End loss: {end_l:.4f}  |  Drop: {dropped:.4f}")
print(f"  Can it memorize? {'YES' if end_l < 2.0 else 'PARTIAL' if dropped > 1.0 else 'NO'}")

print("\n" + "=" * 60)
print("VERDICT SUMMARY")
print("=" * 60)
print(f"1. Monotonic Step Scaling: {'PASS' if monotone else 'FAIL'}")
print(f"2. Memorization Capacity:  {'PASS' if end_l < 2.0 else 'FAIL'}")
print("=" * 60)
