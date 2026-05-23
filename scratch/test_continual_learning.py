import sys, time, torch, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField
from cmf.baselines import TinyGPTLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =====================================================================
# VOCABULARY AND TASK DEFINITIONS
# =====================================================================
VOCAB_SIZE = 150
# Special/query tokens
KEY_A = 0
EQUALS = 1
QUESTION_MARK = 3

# Target Colors
VAL_COLORS = [4, 5, 6, 7]  # blue, red, green, yellow
GARBAGE_TOKENS = list(range(10, 100))

# Task Entities
TASK1_ENTITIES = [100, 101, 102]  # John, James, Jake
TASK2_ENTITIES = [110, 111, 112]  # Mary, Mike, Mark
TASK3_ENTITIES = [120, 121, 122]  # Sarah, Sam, Sally

def get_continual_batch(batch_size, entities, distance=50):
    """
    Format: [E_1, EQUALS, C_1, E_2, EQUALS, C_2, GARBAGE..., Query_Entity, QUESTION_MARK]
    """
    x_list = []
    y_list = []
    for _ in range(batch_size):
        # Pick 2 distinct entities from the active set
        selected_ents = torch.randperm(len(entities))[:2]
        ent1 = entities[selected_ents[0].item()]
        ent2 = entities[selected_ents[1].item()]
        
        # Pick 2 distinct colors
        selected_colors = torch.randperm(len(VAL_COLORS))[:2]
        c1 = VAL_COLORS[selected_colors[0].item()]
        c2 = VAL_COLORS[selected_colors[1].item()]
        
        # Generate distractors
        distractors = torch.randint(GARBAGE_TOKENS[0], GARBAGE_TOKENS[-1] + 1, (distance,))
        
        # Query one of the two entities
        query_idx = torch.randint(0, 2, (1,)).item()
        query_ent = ent1 if query_idx == 0 else ent2
        target_color = c1 if query_idx == 0 else c2
        
        x = torch.cat([
            torch.tensor([ent1, EQUALS, c1, ent2, EQUALS, c2]),
            distractors,
            torch.tensor([query_ent, QUESTION_MARK])
        ])
        
        y = torch.full_like(x, -100)
        y[-1] = target_color
        x_list.append(x)
        y_list.append(y)
        
    return torch.stack(x_list).to(device), torch.stack(y_list).to(device)

# =====================================================================
# MODELS INITIALIZATION
# =====================================================================
print("Initializing models...")
cmf_config = CMFConfig(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    hidden_dim=128,
    num_layers=4,
    max_seq_len=5000,
    adaptive_thinking=False,
    thinking_steps=4,
)
# Disable RoPE for pure semantic field
import cmf.model
cmf.model.apply_rotary_pos_emb = lambda x, cos, sin: x

cmf_model = DeliberativeContinuousMeaningField(cmf_config).to(device)
gpt_model = TinyGPTLM(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    nhead=4,
    num_layers=2,
    hidden_dim=128,
    max_seq_len=5000,
).to(device)

# =====================================================================
# EVALUATION METRICS
# =====================================================================
def eval_accuracy(model, entities, num_trials=30):
    model.eval()
    correct = 0
    for _ in range(num_trials):
        x, y = get_continual_batch(1, entities, distance=50)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
                out = model(x)
                pred = torch.argmax(out["logits"][0, -1, :]).item()
                label = y[0, -1].item()
                if pred == label:
                    correct += 1
    return correct / num_trials

# =====================================================================
# SEQUENTIAL TRAINING LOOP (Task 1 -> Task 2 -> Task 3)
# =====================================================================
cmf_opt = torch.optim.AdamW(cmf_model.parameters(), lr=1e-3)
gpt_opt = torch.optim.AdamW(gpt_model.parameters(), lr=1e-3)

phases = [
    ("PHASE 1: Training Task 1 (John, James, Jake)", TASK1_ENTITIES),
    ("PHASE 2: Training Task 2 (Mary, Mike, Mark)", TASK2_ENTITIES),
    ("PHASE 3: Training Task 3 (Sarah, Sam, Sally)", TASK3_ENTITIES),
]

steps_per_phase = 200

print("\n" + "="*80)
print("CONTINUAL LEARNING ACCURACY MONITOR (NO REPLAY BUFFER)")
print("="*80)
print(f"{'Phase / Step':<25} | {'CMF Task1':<10} | {'CMF Task2':<10} | {'CMF Task3':<10} || {'GPT Task1':<10} | {'GPT Task2':<10} | {'GPT Task3':<10}")
print("-" * 105)

for phase_idx, (phase_name, phase_entities) in enumerate(phases):
    print(f"\n--- {phase_name} ---")
    for step in range(steps_per_phase + 1):
        x, y = get_continual_batch(16, phase_entities, distance=50)
        
        # Train CMF
        cmf_model.train()
        cmf_opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
            out = cmf_model(x)
            loss = torch.nn.functional.cross_entropy(out["logits"][:, -1, :], y[:, -1])
        loss.backward()
        cmf_opt.step()
        
        # Train GPT
        gpt_model.train()
        gpt_opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
            out = gpt_model(x)
            loss_gpt = torch.nn.functional.cross_entropy(out["logits"][:, -1, :], y[:, -1])
        loss_gpt.backward()
        gpt_opt.step()
        
        # Evaluate every 100 steps
        if step % 100 == 0:
            c1 = eval_accuracy(cmf_model, TASK1_ENTITIES)
            c2 = eval_accuracy(cmf_model, TASK2_ENTITIES)
            c3 = eval_accuracy(cmf_model, TASK3_ENTITIES)
            
            g1 = eval_accuracy(gpt_model, TASK1_ENTITIES)
            g2 = eval_accuracy(gpt_model, TASK2_ENTITIES)
            g3 = eval_accuracy(gpt_model, TASK3_ENTITIES)
            
            print(f"Step {step:3d} (P{phase_idx+1}) | {c1:<9.1%} | {c2:<9.1%} | {c3:<9.1%} || {g1:<9.1%} | {g2:<9.1%} | {g3:<9.1%}")

print("="*105)
