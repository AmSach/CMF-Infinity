import torch
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.model import CMFConfig, ContinuousMeaningField, ParallelContinuousMeaningField
from cmf.baselines import TinyGPTLM
from cmf.tokenizer import SimpleBPETokenizer
from cmf.data import cyclic_lm_batches

def run_showdown():
    device = torch.device("cpu")
    print(f"--- The Final Showdown: CMF vs Transformer (GPT) ---")
    
    # 1. Dataset (Fair & Balanced)
    corpus = (
        "Fact: paris is france. Fact: tokyo is japan. Fact: london is uk. "
        "Fact: alice is bob. Fact: bob is developer. Q: what is alice? A: developer. "
        "The DNA molecule is a double helix. Python is code. "
    ) * 100
    
    tokenizer = SimpleBPETokenizer(vocab_size=300)
    tokenizer.train(corpus)
    encoded = tokenizer.encode(corpus)
    
    # 2. Parameter-Matched Models (~800k)
    cmf_config = CMFConfig(vocab_size=300, d_model=128, hidden_dim=256, num_layers=2)
    cmf = ParallelContinuousMeaningField(cmf_config).to(device)
    gpt = TinyGPTLM(vocab_size=300, d_model=128, num_layers=4, hidden_dim=256).to(device)
    
    print(f"CMF Params: {sum(p.numel() for p in cmf.parameters())}")
    print(f"GPT Params: {sum(p.numel() for p in gpt.parameters())}")
    
    # 3. Fair Training (150 steps)
    cmf_opt = torch.optim.AdamW(cmf.parameters(), lr=1e-3)
    gpt_opt = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    
    batches = list(cyclic_lm_batches(encoded, seq_len=32, batch_size=8, num_batches=150))
    
    print("Training both models fairly...")
    for x, y in batches:
        # CMF step
        cmf_opt.zero_grad()
        c_loss = cmf(x, labels=y)["loss"]
        c_loss.backward()
        cmf_opt.step()
        
        # GPT step
        gpt_opt.zero_grad()
        g_loss = gpt(x, labels=y)["loss"]
        g_loss.backward()
        gpt_opt.step()
    
    # 4. Benchmarking Prompts
    prompts = [
        "Fact: paris is france. Q: what is paris? A: ",
        "Fact: alice is bob. Fact: bob is developer. Q: what is alice? A: ",
        "The DNA molecule is ",
    ]
    
    print("\n" + "="*50)
    print(f"{'PROMPT':<40} | {'CMF OUTPUT':<15} | {'GPT OUTPUT':<15}")
    print("-" * 80)
    
    cmf.eval()
    gpt.eval()
    
    for p in prompts:
        p_enc = tokenizer.encode(p).unsqueeze(0).to(device)
        
        # Simple greedy decode (1 token)
        with torch.no_grad():
            c_logits = cmf(p_enc)["logits"][:, -1]
            g_logits = gpt(p_enc)["logits"][:, -1]
            
            c_res = tokenizer.decode(torch.argmax(c_logits, dim=-1))
            g_res = tokenizer.decode(torch.argmax(g_logits, dim=-1))
            
        print(f"{p[:38]:<40} | {c_res[:13]:<15} | {g_res[:13]:<15}")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    run_showdown()
