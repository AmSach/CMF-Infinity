import sys, time, torch, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField
from cmf.baselines import TinyGPTLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

VOCAB_SIZE = 180
KEY_A = 0
EQUALS = 1
QUERY_A = 2
QUESTION_MARK = 3

VAL_COLORS = [4, 5, 6, 7]  # target colors
GARBAGE_TOKENS = list(range(10, 120))  # neutral distractors
ENTITIES = list(range(120, 179))  # Johns, Marys, etc.

def get_distractor_seq(distance):
    val = VAL_COLORS[torch.randint(0, len(VAL_COLORS), (1,)).item()]
    distractors = torch.randint(GARBAGE_TOKENS[0], GARBAGE_TOKENS[-1] + 1, (distance,))
    x = torch.cat([
        torch.tensor([KEY_A, EQUALS, val]),
        distractors,
        torch.tensor([QUERY_A, QUESTION_MARK])
    ]).unsqueeze(0).to(device)
    y = torch.full_like(x, -100)
    y[0, -1] = val
    return x, y

def get_interference_seq(num_entities):
    selected_entities = torch.randperm(len(ENTITIES))[:num_entities]
    colors = torch.randint(0, len(VAL_COLORS), (num_entities,))
    x_list = []
    for i in range(num_entities):
        x_list.extend([ENTITIES[selected_entities[i].item()], EQUALS, VAL_COLORS[colors[i].item()]])
    query_idx = torch.randint(0, num_entities, (1,)).item()
    query_entity = ENTITIES[selected_entities[query_idx].item()]
    target_color = VAL_COLORS[colors[query_idx].item()]
    x_list.extend([query_entity, QUESTION_MARK])
    x = torch.tensor(x_list).unsqueeze(0).to(device)
    y = torch.full_like(x, -100)
    y[0, -1] = target_color
    return x, y

# Initialize models
print("Initializing models...")
cmf_config = CMFConfig(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    hidden_dim=128,
    num_layers=4,
    max_seq_len=25000,
    adaptive_thinking=False,
    thinking_steps=4,
    use_global_memory_router=False,  # Rely purely on recurrent deliberation
)
# Disable RoPE for pure semantic field comparisons
import cmf.model
cmf.model.apply_rotary_pos_emb = lambda x, cos, sin: x

cmf_model = DeliberativeContinuousMeaningField(cmf_config).to(device)
gpt_model = TinyGPTLM(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    nhead=4,
    num_layers=4,
    hidden_dim=128,
    max_seq_len=25000,
).to(device)

# Training loop
print("\n" + "="*80)
print("TRAINING MODELS ON A JOINT RETRIEVAL & INTERFERENCE DATASET")
print("="*80)
cmf_opt = torch.optim.AdamW(cmf_model.parameters(), lr=1e-3, weight_decay=0.01)
gpt_opt = torch.optim.AdamW(gpt_model.parameters(), lr=1e-3, weight_decay=0.01)

steps = 1500
for step in range(steps + 1):
    task_type = torch.randint(0, 2, (1,)).item()
    if task_type == 0:
        x, y = get_distractor_seq(torch.randint(50, 150, (1,)).item())
    else:
        x, y = get_interference_seq(torch.randint(2, 6, (1,)).item())
        
    cmf_model.train()
    cmf_opt.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
        out = cmf_model(x)
        loss = torch.nn.functional.cross_entropy(out["logits"][:, -1, :], y[:, -1])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(cmf_model.parameters(), 1.0)
    cmf_opt.step()
    
    gpt_model.train()
    gpt_opt.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
        out_gpt = gpt_model(x)
        loss_gpt = torch.nn.functional.cross_entropy(out_gpt["logits"][:, -1, :], y[:, -1])
    loss_gpt.backward()
    torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), 1.0)
    gpt_opt.step()
    
    if step % 250 == 0:
        print(f"  Step {step:4d} | CMF Loss: {loss.item():.4f} | GPT Loss: {loss_gpt.item():.4f}")

# =====================================================================
# EVALUATION: DISTRACTOR SATURATION (UP TO 10,000 DISTRACTORS)
# =====================================================================
print("\n" + "="*80)
print("BRUTAL TEST 1: DISTRACTOR SATURATION UP TO 10,000 DISTRACTORS")
print("="*80)
distances = [100, 1000, 5000, 10000]
num_trials = 50

print(f"{'Distance':<10} | {'CMF Accuracy':<15} | {'GPT Accuracy':<15}")
print("-" * 45)

cmf_model.eval()
gpt_model.eval()

for d in distances:
    cmf_correct = 0
    gpt_correct = 0
    for _ in range(num_trials):
        x, y = get_distractor_seq(d)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
                c_out = cmf_model(x)
                g_out = gpt_model(x)
                c_pred = torch.argmax(c_out["logits"][0, -1, :]).item()
                g_pred = torch.argmax(g_out["logits"][0, -1, :]).item()
                label = y[0, -1].item()
                if c_pred == label: cmf_correct += 1
                if g_pred == label: gpt_correct += 1
        if device == "cuda": torch.cuda.empty_cache()
    print(f"{d:<10d} | {cmf_correct/num_trials:<15.1%} | {gpt_correct/num_trials:<15.1%}")

# =====================================================================
# EVALUATION: LATENT CAPACITY SATURATION (INTERFERENCE DENSITY)
# =====================================================================
print("\n" + "="*80)
print("BRUTAL TEST 2: LATENT CAPACITY SATURATION (ENTITY DENSITY)")
print("="*80)
densities = [2, 4, 8, 12, 16]

print(f"{'Entities':<10} | {'CMF Accuracy':<15} | {'GPT Accuracy':<15}")
print("-" * 45)

for density in densities:
    cmf_correct = 0
    gpt_correct = 0
    for _ in range(num_trials):
        x, y = get_interference_seq(density)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
                c_out = cmf_model(x)
                g_out = gpt_model(x)
                c_pred = torch.argmax(c_out["logits"][0, -1, :]).item()
                g_pred = torch.argmax(g_out["logits"][0, -1, :]).item()
                label = y[0, -1].item()
                if c_pred == label: cmf_correct += 1
                if g_pred == label: gpt_correct += 1
        if device == "cuda": torch.cuda.empty_cache()
    print(f"{density:<10d} | {cmf_correct/num_trials:<15.1%} | {gpt_correct/num_trials:<15.1%}")

# =====================================================================
# EVALUATION: ATTENTION ENTROPY COLLAPSE ANALYSIS
# =====================================================================
print("\n" + "="*80)
print("BRUTAL TEST 3: ATTENTION ENTROPY COLLAPSE (SHANNON ENTROPY)")
print("="*80)

# We will measure the Shannon entropy of attention distributions at the query token.
# For CMF, we measure entropy of anchoring attention across steps 1 -> 4.
# For Tiny-GPT, we measure entropy of self-attention at query position across layers 1 -> 4.

def calculate_entropy(probs):
    probs = torch.clamp(probs, min=1e-9)
    entropy = -torch.sum(probs * torch.log2(probs), dim=-1)
    return entropy.mean().item()

cmf_entropies = [[] for _ in range(4)]
gpt_entropies = [[] for _ in range(4)]

for _ in range(50):
    x, y = get_distractor_seq(500)
    
    # 1. Capture CMF anchoring attention weights across steps
    recorded_weights = []
    
    orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    def patched_sdpa(*args, **kwargs):
        q = args[0] if len(args) > 0 else kwargs['query']
        k = args[1] if len(args) > 1 else kwargs['key']
        scale = kwargs.get('scale', None)
        if scale is None and len(args) > 6:
            scale = args[6]
        if scale is None:
            scale = 1.0 / math.sqrt(q.size(-1))
        is_causal = kwargs.get('is_causal', False)
        if not is_causal and len(args) > 5:
            is_causal = args[5]
            
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        if is_causal:
            sz = attn_weights.size(-1)
            mask = torch.triu(torch.full((sz, sz), float("-inf"), device=q.device), diagonal=1)
            attn_weights = attn_weights + mask
        attn_probs = torch.softmax(attn_weights, dim=-1)
        recorded_weights.append(attn_probs[0, :, -1, :].detach().cpu())
        return orig_sdpa(*args, **kwargs)
        
    torch.nn.functional.scaled_dot_product_attention = patched_sdpa
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
            _ = cmf_model(x)
    torch.nn.functional.scaled_dot_product_attention = orig_sdpa
    
    # Calculate CMF entropy at each step
    for step_idx, weights in enumerate(recorded_weights[:4]):
        cmf_entropies[step_idx].append(calculate_entropy(weights))
        
    # 2. Capture Tiny-GPT attention weights across layers
    recorded_gpt_weights = []
    def patched_gpt_sdpa(*args, **kwargs):
        q = args[0] if len(args) > 0 else kwargs['query']
        k = args[1] if len(args) > 1 else kwargs['key']
        scale = kwargs.get('scale', None)
        if scale is None and len(args) > 6:
            scale = args[6]
        if scale is None:
            scale = 1.0 / math.sqrt(q.size(-1))
        is_causal = kwargs.get('is_causal', False)
        if not is_causal and len(args) > 5:
            is_causal = args[5]
            
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        if is_causal:
            sz = attn_weights.size(-1)
            mask = torch.triu(torch.full((sz, sz), float("-inf"), device=q.device), diagonal=1)
            attn_weights = attn_weights + mask
        attn_probs = torch.softmax(attn_weights, dim=-1)
        recorded_gpt_weights.append(attn_probs[0, :, -1, :].detach().cpu())
        return orig_sdpa(*args, **kwargs)
        
    torch.nn.functional.scaled_dot_product_attention = patched_gpt_sdpa
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
            _ = gpt_model(x)
    torch.nn.functional.scaled_dot_product_attention = orig_sdpa
    
    # Calculate GPT entropy at each layer
    for layer_idx, weights in enumerate(recorded_gpt_weights[:4]):
        gpt_entropies[layer_idx].append(calculate_entropy(weights))

avg_cmf_entropies = [sum(e)/len(e) for e in cmf_entropies]
avg_gpt_entropies = [sum(e)/len(e) for e in gpt_entropies]

print(f"{'Model / Thinking Stage':<30} | {'Average Shannon Entropy (bits)':<30}")
print("-" * 65)
for step_idx in range(4):
    print(f"CMF Deliberation Step {step_idx+1:<2d}      | {avg_cmf_entropies[step_idx]:.4f} bits")
print("-" * 65)
for layer_idx in range(4):
    print(f"Tiny-GPT Layer {layer_idx+1:<2d}             | {avg_gpt_entropies[layer_idx]:.4f} bits")
print("="*80)
