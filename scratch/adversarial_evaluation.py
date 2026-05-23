import sys, time, torch, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField
from cmf.baselines import TinyGPTLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =====================================================================
# VOCABULARY AND TOKEN DEFINITIONS
# =====================================================================
VOCAB_SIZE = 150
# Special/query tokens
KEY_A = 0
EQUALS = 1
QUERY_A = 2
QUESTION_MARK = 3

# Values (Target colors)
VAL_COLORS = [4, 5, 6, 7]  # blue, red, green, yellow

# Semantically similar colors/distractors (for saturation test)
SEMANTIC_DISTRACTORS = list(range(8, 50))  # navy, teal, sky, orange, pink, etc.
# Neutral garbage tokens
GARBAGE_TOKENS = list(range(50, 140))

# Entity tokens (for interference test)
ENTITIES = list(range(140, 150))  # John, James, Jake, Mary, etc.

# Helper function to generate sequences
def get_distractor_seq(distance, semantic=False):
    """
    Format: [KEY_A, EQUALS, VAL, DISTRACTORS..., QUERY_A, QUESTION_MARK]
    """
    val = VAL_COLORS[torch.randint(0, len(VAL_COLORS), (1,)).item()]
    dist_pool = SEMANTIC_DISTRACTORS if semantic else GARBAGE_TOKENS
    
    # Generate distractors
    distractors = torch.randint(dist_pool[0], dist_pool[-1] + 1, (distance,))
    
    x = torch.cat([
        torch.tensor([KEY_A, EQUALS, val]),
        distractors,
        torch.tensor([QUERY_A, QUESTION_MARK])
    ]).unsqueeze(0).to(device)
    
    y = torch.full_like(x, -100)
    y[0, -1] = val  # predict target at the very end
    return x, y

def get_interference_seq(num_entities):
    """
    Format: [E_1, EQUALS, C_1, E_2, EQUALS, C_2, ..., E_N, EQUALS, C_N, QUERY_E_k, QUESTION_MARK]
    Predicts: C_k
    """
    selected_entities = torch.randperm(len(ENTITIES))[:num_entities]
    colors = torch.randint(0, len(VAL_COLORS), (num_entities,))
    
    x_list = []
    for i in range(num_entities):
        x_list.extend([ENTITIES[selected_entities[i].item()], EQUALS, VAL_COLORS[colors[i].item()]])
        
    # Query a random entity from the set
    query_idx = torch.randint(0, num_entities, (1,)).item()
    query_entity = ENTITIES[selected_entities[query_idx].item()]
    target_color = VAL_COLORS[colors[query_idx].item()]
    
    x_list.extend([query_entity, QUESTION_MARK])
    x = torch.tensor(x_list).unsqueeze(0).to(device)
    
    y = torch.full_like(x, -100)
    y[0, -1] = target_color
    return x, y

# =====================================================================
# MODELS INITIALIZATION
# =====================================================================
print("Initializing models...")
cmf_config = CMFConfig(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    hidden_dim=128,
    num_layers=4,
    max_seq_len=20000,
    adaptive_thinking=False,
    thinking_steps=4,
)
# Disable RoPE to test pure semantic field
import cmf.model
cmf.model.apply_rotary_pos_emb = lambda x, cos, sin: x

cmf_model = DeliberativeContinuousMeaningField(cmf_config).to(device)
gpt_model = TinyGPTLM(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    nhead=4,
    num_layers=2,
    hidden_dim=128,
    max_seq_len=20000,
).to(device)

# =====================================================================
# TRAINING ON MULTI-TASK MIXTURE (Short Context)
# =====================================================================
print("\n" + "="*80)
print("TRAINING CMF AND TINY-GPT ON MIXED MEMORY TASKS (SHORT CONTEXT)")
print("="*80)

def train_models(steps=600):
    cmf_opt = torch.optim.AdamW(cmf_model.parameters(), lr=1e-3, weight_decay=0.01)
    gpt_opt = torch.optim.AdamW(gpt_model.parameters(), lr=1e-3, weight_decay=0.01)
    
    for step in range(steps + 1):
        # Mix of distractor and interference sequences (short lengths)
        task_type = torch.randint(0, 3, (1,)).item()
        if task_type == 0:
            # Distractor sequence (length 100-200)
            x, y = get_distractor_seq(torch.randint(100, 200, (1,)).item(), semantic=False)
        elif task_type == 1:
            # Semantic distractor sequence (length 100-200)
            x, y = get_distractor_seq(torch.randint(100, 200, (1,)).item(), semantic=True)
        else:
            # Interference sequence (2 to 4 entities)
            x, y = get_interference_seq(torch.randint(2, 5, (1,)).item())
            
        # Train CMF
        cmf_model.train()
        cmf_opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
            out = cmf_model(x)
            loss = torch.nn.functional.cross_entropy(out["logits"][:, -1, :], y[:, -1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cmf_model.parameters(), 1.0)
        cmf_opt.step()
        
        # Train GPT
        gpt_model.train()
        gpt_opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
            out = gpt_model(x)
            loss_gpt = torch.nn.functional.cross_entropy(out["logits"][:, -1, :], y[:, -1])
        loss_gpt.backward()
        torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), 1.0)
        gpt_opt.step()
        
        if step % 100 == 0:
            print(f"  Step {step:4d} | CMF Loss: {loss.item():.4f} | GPT Loss: {loss_gpt.item():.4f}")

train_models()

# =====================================================================
# ADVERSARIAL TEST 1: DISTRACTOR SATURATION (Neutral vs Semantic Distractors)
# =====================================================================
print("\n" + "="*80)
print("ADVERSARIAL TEST 1: DISTRACTOR SATURATION (ACCURACY VS CONTEXT LENGTH)")
print("="*80)

distances = [200, 1000, 4000, 8000]
num_trials = 30

print(f"{'Distance':<10} | {'CMF Neutral':<12} | {'CMF Semantic':<12} | {'GPT Neutral':<12} | {'GPT Semantic':<12}")
print("-" * 70)

for d in distances:
    cmf_neut_correct = 0
    cmf_sem_correct = 0
    gpt_neut_correct = 0
    gpt_sem_correct = 0
    
    cmf_model.eval()
    gpt_model.eval()
    
    for _ in range(num_trials):
        # 1. Neutral distractor
        x, y = get_distractor_seq(d, semantic=False)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
                c_out = cmf_model(x)
                g_out = gpt_model(x)
                
                c_pred = torch.argmax(c_out["logits"][0, -1, :]).item()
                g_pred = torch.argmax(g_out["logits"][0, -1, :]).item()
                label = y[0, -1].item()
                
                if c_pred == label: cmf_neut_correct += 1
                if g_pred == label: gpt_neut_correct += 1
                
        # 2. Semantic distractor
        x, y = get_distractor_seq(d, semantic=True)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
                c_out = cmf_model(x)
                g_out = gpt_model(x)
                
                c_pred = torch.argmax(c_out["logits"][0, -1, :]).item()
                g_pred = torch.argmax(g_out["logits"][0, -1, :]).item()
                label = y[0, -1].item()
                
                if c_pred == label: cmf_sem_correct += 1
                if g_pred == label: gpt_sem_correct += 1
                
        # Reclaim memory
        if device == "cuda":
            torch.cuda.empty_cache()
            
    c_neut_acc = cmf_neut_correct / num_trials
    c_sem_acc = cmf_sem_correct / num_trials
    g_neut_acc = gpt_neut_correct / num_trials
    g_sem_acc = gpt_sem_correct / num_trials
    
    print(f"{d:<10d} | {c_neut_acc:<12.1%} | {c_sem_acc:<12.1%} | {g_neut_acc:<12.1%} | {g_sem_acc:<12.1%}")

# =====================================================================
# ADVERSARIAL TEST 2: INTERFERENCE STACKING
# =====================================================================
print("\n" + "="*80)
print("ADVERSARIAL TEST 2: INTERFERENCE STACKING (ACCURACY VS ENTITY DENSITY)")
print("="*80)

entities_counts = [2, 4, 6, 8, 10]

print(f"{'Entities':<10} | {'CMF Accuracy':<15} | {'GPT Accuracy':<15}")
print("-" * 45)

for count in entities_counts:
    cmf_correct = 0
    gpt_correct = 0
    
    for _ in range(num_trials):
        x, y = get_interference_seq(count)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
                c_out = cmf_model(x)
                g_out = gpt_model(x)
                
                c_pred = torch.argmax(c_out["logits"][0, -1, :]).item()
                g_pred = torch.argmax(g_out["logits"][0, -1, :]).item()
                label = y[0, -1].item()
                
                if c_pred == label: cmf_correct += 1
                if g_pred == label: gpt_correct += 1
                
        if device == "cuda":
            torch.cuda.empty_cache()
            
    c_acc = cmf_correct / num_trials
    g_acc = gpt_correct / num_trials
    
    print(f"{count:<10d} | {c_acc:<15.1%} | {g_acc:<15.1%}")

# =====================================================================
# ADVERSARIAL TEST 3: STATE PERTURBATION ROBUSTNESS
# =====================================================================
print("\n" + "="*80)
print("ADVERSARIAL TEST 3: STATE PERTURBATION (ACCURACY VS NOISE MAGNITUDE)")
print("="*80)

noise_stds = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
print(f"{'Noise Std':<12} | {'CMF Accuracy':<15}")
print("-" * 32)

# Subclass / Patch the CMF model forward pass to inject noise to z during deliberation
for std in noise_stds:
    correct = 0
    # Temporary patch in model's forward
    original_run = cmf_model.forward
    
    def patched_forward(input_ids, labels=None, **kwargs):
        # We run the standard forward pass but intercept the return_states flag or field forward
        # For simplicity, let's just run standard model, but inject noise directly into the field net
        # by patching model.field.proposal or adding noise to z_accum inside forward.
        # Actually, we can hook into field's forward pass!
        orig_field_forward = cmf_model.field.forward
        def noisy_field_forward(z, context, tau, goal=None):
            # Inject noise to z
            z_noisy = z + torch.randn_like(z) * std
            return orig_field_forward(z_noisy, context, tau, goal)
        cmf_model.field.forward = noisy_field_forward
        res = original_run(input_ids, labels=labels, **kwargs)
        cmf_model.field.forward = orig_field_forward
        return res
        
    cmf_model.forward = patched_forward
    
    # Evaluate on short distractor sequences
    for _ in range(num_trials):
        x, y = get_distractor_seq(100, semantic=False)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
                out = cmf_model(x)
                pred = torch.argmax(out["logits"][0, -1, :]).item()
                label = y[0, -1].item()
                if pred == label:
                    correct += 1
                    
    cmf_model.forward = original_run
    acc = correct / num_trials
    print(f"{std:<12.3f} | {acc:<15.1%}")

print("="*80)
