import sys, time, torch, math
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.model_v2 import PersistentStateCMF, CMFv2Config

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
torch.manual_seed(42)
if device == "cpu":
    torch.set_num_threads(1)

# =====================================================================
# DATA GENERATOR: SEQUENTIAL KEY-DOOR MAZE TRACKER
# =====================================================================
# Vocabulary details
# Pos 0 to 9: 10 to 19
# Found Key Red/Blue: 20, 21
# Move East/West: 22, 23
# Encounter Door Red/Blue: 24, 25
# Query Openable: 26
# Yes/No (Targets): 30, 31
# Padding: 0

VOCAB_SIZE = 50
YES = 30
NO = 31

def generate_maze_sequence(batch_size, length):
    x = torch.zeros(batch_size, length, dtype=torch.long, device=device)
    y = torch.full((batch_size, length), -100, dtype=torch.long, device=device)
    
    pos = torch.randint(0, 10, (batch_size,), device=device)
    x[:, 0] = 10 + pos
    
    key_colors = torch.randint(0, 2, (batch_size,), device=device)
    door_colors = torch.randint(0, 2, (batch_size,), device=device)
    
    decisions = torch.randint(0, 2, (batch_size, length), device=device)
    actions = torch.where(decisions == 0, 22, 23)
    x[:, 1:-2] = actions[:, 1:-2]
    
    key_step = length // 3
    x[:, key_step] = 20 + key_colors
    
    x[:, length - 2] = 24 + door_colors
    x[:, length - 1] = 26
    
    targets = torch.where(key_colors == door_colors, YES, NO)
    y[:, -1] = targets
    return x, y

# =====================================================================
# GRU BASELINE MODEL
# =====================================================================
class StatefulGRUBaseline(nn.Module):
    def __init__(self, vocab_size=50, d_model=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        out, _ = self.gru(emb)
        logits = self.output(out)
        return {"logits": logits}

# =====================================================================
# INITIALIZE MODELS
# =====================================================================
print("Initializing CMF v2 and GRU models...")
cmf_config = CMFv2Config(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    hidden_dim=128,
    num_layers=4,
    num_slots=8,
    thinking_steps=4,
)
cmf_model = PersistentStateCMF(cmf_config).to(device)
gru_model = StatefulGRUBaseline(vocab_size=VOCAB_SIZE, d_model=64).to(device)

# =====================================================================
# TRAINING PIPELINE (Streaming Step-by-Step Loss)
# =====================================================================
print("\n" + "="*80)
print("TRAINING MODELS IN STEP-BY-STEP STREAMING MODE")
print("="*80)

cmf_opt = torch.optim.AdamW(cmf_model.parameters(), lr=1e-3, weight_decay=0.01)
gru_opt = torch.optim.AdamW(gru_model.parameters(), lr=1e-3, weight_decay=0.01)

steps = 1000
batch_size = 512 if device == "cuda" else 32

for step in range(steps + 1):
    # Length of training sequences (fixed length 25 for fast stable convergence)
    length = 25
    x, y = generate_maze_sequence(batch_size, length)
    
    # Train CMF v2
    cmf_model.train()
    cmf_opt.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
        out = cmf_model(x)
        # We only compute loss at the final query token
        loss = F.cross_entropy(out["logits"][:, -1, :], y[:, -1])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(cmf_model.parameters(), 1.0)
    cmf_opt.step()
    
    # Train GRU
    gru_model.train()
    gru_opt.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
        out_gru = gru_model(x)
        loss_gru = F.cross_entropy(out_gru["logits"][:, -1, :], y[:, -1])
    loss_gru.backward()
    torch.nn.utils.clip_grad_norm_(gru_model.parameters(), 1.0)
    gru_opt.step()
    
    if step % 100 == 0:
        print(f"  Step {step:4d} | CMFv2 Loss: {loss.item():.4f} | GRU Loss: {loss_gru.item():.4f}")

# =====================================================================
# EVALUATION: STREAMING WORLD MODELING & LONG-HORIZON CONSISTENCY
# =====================================================================
print("\n" + "="*80)
print("EVALUATING STREAMING LONG-HORIZON CONSISTENCY (ACCURACY VS SEQUENCE LENGTH)")
print("="*80)

# We evaluate on sequence lengths 20, 50, and 100!
# Note: For length 100, the key is found around step 33, and the door is encountered at step 98.
# The model must persist the key identity across 65 random-walk steps of interference
# without any historical context or KV cache storage!

lengths_to_test = [20, 50, 100]
num_trials = 100

print(f"{'Length':<10} | {'CMF v2 Accuracy':<18} | {'GRU Accuracy':<15}")
print("-" * 50)

cmf_model.eval()
gru_model.eval()

for l in lengths_to_test:
    cmf_correct = 0
    gru_correct = 0
    
    for _ in range(num_trials):
        x, y = generate_maze_sequence(1, l)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
                # Run CMF v2 step-by-step to enforce streaming isolation (no sequence caching)
                z, M = cmf_model.init_state(1, device)
                for t in range(l):
                    x_t = x[:, t]
                    logits, z, M = cmf_model.step_forward(x_t, z, M)
                c_pred = torch.argmax(logits[0]).item()
                
                # Run GRU
                out_gru = gru_model(x)
                g_pred = torch.argmax(out_gru["logits"][0, -1]).item()
                
                label = y[0, -1].item()
                if c_pred == label: cmf_correct += 1
                if g_pred == label: gru_correct += 1
                
        if device == "cuda": torch.cuda.empty_cache()
        
    print(f"{l:<10d} | {cmf_correct/num_trials:<18.1%} | {gru_correct/num_trials:<15.1%}")
print("="*80)
