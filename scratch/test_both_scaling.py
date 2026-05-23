import sys, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.test_compositional_reasoning import generate_hop_batch, TinyGPTLM, DeliberativeContinuousMeaningField, CMFConfig, VAL_COLORS, VOCAB_SIZE

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing models...")
cmf_config = CMFConfig(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    hidden_dim=128,
    num_layers=4,
    max_seq_len=5000,
    adaptive_thinking=False,
    thinking_steps=4,
    use_global_memory_router=True,  # Enable learnable global attention routing!
)
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

cmf_opt = torch.optim.AdamW(cmf_model.parameters(), lr=1e-3)
gpt_opt = torch.optim.AdamW(gpt_model.parameters(), lr=1e-3)

print("Training both models on 1-hop transitive retrieval with batch_size=64 and distance=10 (seq_len=22)...")
for step in range(2000 + 1):
    # Train CMF
    cmf_model.train()
    cmf_opt.zero_grad()
    bx, by, b_colors = generate_hop_batch(64, 1, distance=10)
    out_cmf = cmf_model(bx)
    loss_cmf = torch.nn.functional.cross_entropy(
        out_cmf["logits"][:, :-1].reshape(-1, VOCAB_SIZE),
        by[:, :-1].reshape(-1),
        ignore_index=-100
    )
    loss_cmf.backward()
    cmf_opt.step()

    # Train GPT
    gpt_model.train()
    gpt_opt.zero_grad()
    out_gpt = gpt_model(bx)
    loss_gpt = torch.nn.functional.cross_entropy(
        out_gpt["logits"][:, :-1].reshape(-1, VOCAB_SIZE),
        by[:, :-1].reshape(-1),
        ignore_index=-100
    )
    loss_gpt.backward()
    gpt_opt.step()

    if step % 250 == 0:
        # Evaluate CMF and GPT
        cmf_model.eval()
        gpt_model.eval()
        correct_cmf = 0
        correct_gpt = 0
        with torch.no_grad():
            for _ in range(100):
                x, y, target_colors = generate_hop_batch(1, 1, distance=10)
                prefix = x[:, :-1]
                
                # CMF eval
                out_eval_cmf = cmf_model(prefix)
                pred_cmf = torch.argmax(out_eval_cmf["logits"][0, -1, :]).item()
                if pred_cmf == target_colors[0].item():
                    correct_cmf += 1
                
                # GPT eval
                out_eval_gpt = gpt_model(prefix)
                pred_gpt = torch.argmax(out_eval_gpt["logits"][0, -1, :]).item()
                if pred_gpt == target_colors[0].item():
                    correct_gpt += 1
                    
        print(f"Step {step:4d} | CMF Loss: {loss_cmf.item():.4f} (Acc: {correct_cmf/100:.1%}) | GPT Loss: {loss_gpt.item():.4f} (Acc: {correct_gpt/100:.1%})")
