import sys, time, torch, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField
from cmf.baselines import TinyGPTLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =====================================================================
# VOCABULARY AND COMPOSITION DATA GENERATION
# =====================================================================
VOCAB_SIZE = 150
# Special tokens
KEY_A = 0
EQUALS = 1
QUESTION_MARK = 3

# Values (Colors)
VAL_COLORS = [4, 5, 6, 7]  # blue, red, green, yellow
GARBAGE_TOKENS = list(range(10, 80))

# Entities
ENTITIES = list(range(80, 120))  # A, B, C, D...

def generate_hop_batch(batch_size, num_hops, distance=10):
    """
    Format: Context bindings in random order, query, QUESTION_MARK, target_color.
    Returns: x (complete sequence), y (aligned next-token labels), target_color.
    """
    x_list = []
    y_list = []
    colors_list = []
    for _ in range(batch_size):
        # Pick distinct entities
        selected_ents = torch.randperm(len(ENTITIES))
        ents = [ENTITIES[selected_ents[i].item()] for i in range(len(ENTITIES))]
        
        # We need distinct colors for distractor bindings
        color_perms = torch.randperm(len(VAL_COLORS))
        target_color = VAL_COLORS[color_perms[0].item()]
        
        # Build all bindings blocks
        blocks = []
        
        # Target chain E1 = Color
        blocks.append([ents[2], EQUALS, target_color])
        
        # Target chain E_k = E_{k-1}
        target_ents = ents[2 : 2 + num_hops]
        for k in range(1, num_hops):
            blocks.append([target_ents[k], EQUALS, target_ents[k-1]])
            
        # Distractor bindings
        for i in range(2):
            distractor_ent = ents[i]
            distractor_color = VAL_COLORS[color_perms[i+1].item()]
            blocks.append([distractor_ent, EQUALS, distractor_color])
            
        # Shuffle the blocks
        perm = torch.randperm(len(blocks))
        shuffled_blocks = [blocks[perm[i].item()] for i in range(len(blocks))]
        
        # Assemble sequence and labels
        x_tokens = []
        y_labels = []
        for block in shuffled_blocks:
            x_tokens.extend(block)
            # Label for Ent is EQUALS, label for EQUALS is ignored (value definition), label for Val is ignored
            y_labels.extend([EQUALS, -100, -100])
            
        # Fixed garbage padding
        distractors = torch.randint(GARBAGE_TOKENS[0], GARBAGE_TOKENS[-1] + 1, (distance,)).tolist()
        x_tokens.extend(distractors)
        y_labels.extend([-100] * len(distractors))
        
        # Query and answer
        x_tokens.extend([target_ents[-1], QUESTION_MARK, target_color])
        # Label for Query_Ent is QUESTION_MARK, label for QUESTION_MARK is target_color, label for target_color is ignored
        y_labels.extend([QUESTION_MARK, target_color, -100])
        
        x = torch.tensor(x_tokens, dtype=torch.long)
        y = torch.tensor(y_labels, dtype=torch.long)
        
        x_list.append(x)
        y_list.append(y)
        colors_list.append(target_color)
        
    return torch.stack(x_list).to(device), torch.stack(y_list).to(device), torch.tensor(colors_list).to(device)

def main():
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
        use_global_memory_router=False,
    )
    # Disable RoPE for pure semantic field
    import cmf.model
    cmf.model.apply_rotary_pos_emb = lambda x, cos, sin: x

    cmf_model = DeliberativeContinuousMeaningField(cmf_config).to(device)
    gpt_model = TinyGPTLM(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        nhead=4,
        num_layers=4,
        hidden_dim=128,
        max_seq_len=5000,
    ).to(device)

    # =====================================================================
    # TRAINING (Trained on 1-hop and 2-hop tasks only)
    # =====================================================================
    print("\n" + "="*80)
    print("TRAINING CMF AND GPT ON 1-HOP AND 2-HOP TRANSITIVE RETRIEVAL")
    print("="*80)

    cmf_opt = torch.optim.AdamW(cmf_model.parameters(), lr=1e-3)
    gpt_opt = torch.optim.AdamW(gpt_model.parameters(), lr=1e-3)

    for step in range(2500 + 1):
        # Randomly select 1-hop or 2-hop sequence
        hops = torch.randint(1, 3, (1,)).item()
        x, y, target_colors = generate_hop_batch(64, hops, distance=10)
        
        # Train CMF
        cmf_model.train()
        cmf_opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
            out = cmf_model(x)
            # Align logits[:, :-1] with y[:, :-1]
            loss = torch.nn.functional.cross_entropy(
                out["logits"][:, :-1].reshape(-1, VOCAB_SIZE),
                y[:, :-1].reshape(-1),
                ignore_index=-100
            )
        loss.backward()
        cmf_opt.step()
        
        # Train GPT
        gpt_model.train()
        gpt_opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
            out = gpt_model(x)
            loss_gpt = torch.nn.functional.cross_entropy(
                out["logits"][:, :-1].reshape(-1, VOCAB_SIZE),
                y[:, :-1].reshape(-1),
                ignore_index=-100
            )
        loss_gpt.backward()
        gpt_opt.step()
        
        if step % 250 == 0:
            print(f"  Step {step:4d} | CMF Loss: {loss.item():.4f} | GPT Loss: {loss_gpt.item():.4f}")

    # =====================================================================
    # EVALUATION (Testing Extrapolation to 3-Hop Tasks with Solver Budget Scaling)
    # =====================================================================
    print("\n" + "="*80)
    print("EVALUATING TRANSITIVE RETRIEVAL EXTRAPOLATION (ACCURACY VS REASONING HOPS)")
    print("="*80)

    num_trials = 100

    # Evaluate GPT (which has fixed feedforward depth)
    gpt_model.eval()
    gpt_results = {}
    for hops in [1, 2, 3]:
        correct = 0
        for _ in range(num_trials):
            x, y, target_colors = generate_hop_batch(1, hops, distance=20)
            prefix = x[:, :-1]
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
                    out = gpt_model(prefix)
                    pred = torch.argmax(out["logits"][0, -1, :]).item()
                    if pred == target_colors[0].item():
                        correct += 1
        gpt_results[hops] = correct / num_trials

    # Evaluate CMF at different thinking steps
    cmf_model.eval()
    cmf_results = {}
    thinking_budgets = [1, 2, 4, 8]

    for budget in thinking_budgets:
        cmf_model.config.thinking_steps = budget
        cmf_results[budget] = {}
        for hops in [1, 2, 3]:
            correct = 0
            for _ in range(num_trials):
                x, y, target_colors = generate_hop_batch(1, hops, distance=20)
                prefix = x[:, :-1]
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
                        out = cmf_model(prefix)
                        pred = torch.argmax(out["logits"][0, -1, :]).item()
                        if pred == target_colors[0].item():
                            correct += 1
            cmf_results[budget][hops] = correct / num_trials

    # Print comparison table
    print(f"{'Model / Budget':<20} | {'1-Hop Accuracy':<15} | {'2-Hop Accuracy':<15} | {'3-Hop (OOD) Accuracy':<20}")
    print("-" * 80)
    print(f"{'Tiny-GPT (Baseline)':<20} | {gpt_results[1]:<15.1%} | {gpt_results[2]:<15.1%} | {gpt_results[3]:<20.1%}")
    print("-" * 80)
    for budget in thinking_budgets:
        print(f"CMF (Steps={budget:<2d})       | {cmf_results[budget][1]:<15.1%} | {cmf_results[budget][2]:<15.1%} | {cmf_results[budget][3]:<20.1%}")
    print("="*80)

    # Plotting
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    # Plot GPT
    plt.plot([1, 2, 3], [gpt_results[1], gpt_results[2], gpt_results[3]], 
             label="Tiny-GPT (Baseline)", marker='o', linestyle='--', color='#e06666', linewidth=2)
             
    # Plot CMF budgets
    colors = {1: '#3c78d8', 2: '#6fa8dc', 4: '#8e7cc3', 8: '#674ea7'}
    for budget in thinking_budgets:
        accs = [cmf_results[budget][1], cmf_results[budget][2], cmf_results[budget][3]]
        plt.plot([1, 2, 3], accs, label=f"CMF (Steps={budget})", 
                 marker='s', color=colors[budget], linewidth=2.5)
                 
    plt.title("Transitive Retrieval Extrapolation: Accuracy vs Reasoning Hops", fontsize=14, fontweight='bold')
    plt.xlabel("Reasoning Hops", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks([1, 2, 3], ["1-Hop", "2-Hop", "3-Hop (OOD)"])
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='lower left')
    
    # Save plot to artifacts
    artifacts_dir = Path("C:/Users/amans/.gemini/antigravity/brain/50eff4ca-c18b-4ea4-ac8a-f257390d674b/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plot_path = artifacts_dir / "compositional_reasoning_results.webp"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved results plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
