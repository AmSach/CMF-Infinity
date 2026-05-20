import torch
import tiktoken
from pathlib import Path
import sys

# Add current dir to sys.path to import cmf
sys.path.append(str(Path(__file__).parent.resolve()))

from cmf.model import DeliberativeContinuousMeaningField

def generate_text(model, enc, prompt, max_new_tokens=50, temp=0.5):
    device = next(model.parameters()).device
    tokens = enc.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # Generate
    generated = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temp)
    out_text = enc.decode(generated[0].tolist())
    return out_text

def main():
    ckpt_path = r"C:\Users\amans\Downloads\checkpoint_latest(2).pt"
    print(f"Loading checkpoint from {ckpt_path}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    
    # Clean state dict keys (remove _orig_mod.module. prefix from torch.compile/DDP)
    state_dict = ckpt['model']
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.module.", "").replace("module.", "")
        clean_state_dict[new_key] = v
        
    print(f"Initializing model with config: {config}")
    # Using DeliberativeContinuousMeaningField since adaptive_thinking=True
    model = DeliberativeContinuousMeaningField(config)
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    
    enc = tiktoken.get_encoding("gpt2")
    
    prompts = [
        "The capital of France is ",
        "def fibonacci(n):",
        "Once upon a time in a distant galaxy,"
    ]
    
    print("\n--- Model Tests ---\n")
    for p in prompts:
        print(f"Prompt: {repr(p)}")
        with torch.no_grad():
            res = generate_text(model, enc, p, max_new_tokens=60, temp=0.6)
        print(f"Completion: {res}\n")

if __name__ == "__main__":
    main()
