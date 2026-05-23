import sys, io, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField

CKPT_PATH = Path("e:/CMF/checkpoint_latest_new.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_config():
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    cfg_src = ckpt["config"]
    cfg_dict = cfg_src if isinstance(cfg_src, dict) else cfg_src.__dict__.copy()
    sd = ckpt["model"]
    mem_keys = [k for k in sd if "field.memory" in k and "bank" not in k]
    if mem_keys:
        cfg_dict["num_memory_anchors"] = sd[mem_keys[0]].shape[0]
    return CMFConfig(**cfg_dict)

def run_memorization_test(model_type="fresh", lr=1e-3, steps=300):
    print("\n" + "="*80)
    print(f"RUNNING MEMORIZATION TEST: {model_type.upper()} MODEL")
    print(f"Device: {device} | Learn Rate: {lr} | Steps: {steps}")
    print("="*80)
    
    config = get_config()
    
    # Instantiate model
    if model_type == "fresh":
        model = DeliberativeContinuousMeaningField(config)
    else:
        # Load from checkpoint
        model = DeliberativeContinuousMeaningField(config)
        ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
        clean = {}
        for k, v in ckpt["model"].items():
            k2 = k.replace("_orig_mod.module.", "").replace("module.", "")
            clean[k2] = v
        model.load_state_dict(clean, strict=False)
        
    model = model.to(device)
    model.train()
    
    # Generate 1 batch: 64 sequences, length 64
    batch_size = 64
    seq_len = 64
    tiny_vocab_size = 1000
    
    torch.manual_seed(42)
    # Generate random tokens in [0, tiny_vocab_size - 1]
    input_ids = torch.randint(0, tiny_vocab_size, (batch_size, seq_len + 1), device=device)
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    print(f"Batch shape: x={x.shape}, y={y.shape} | Vocab Range: [0, {tiny_vocab_size}]")
    
    initial_loss = None
    for step in range(steps + 1):
        opt.zero_grad()
        out = model(x, labels=y)
        loss = out["loss"]
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        opt.step()
        
        if step == 0:
            initial_loss = loss.item()
            
        if step % 20 == 0 or step == steps:
            print(f"  Step {step:3d}/{steps} | Loss: {loss.item():.6f}")
            
        if loss.item() < 1e-4:
            print(f"\n[SUCCESS] Loss drove below 1e-4 at step {step}!")
            break
            
    final_loss = loss.item()
    print("-"*80)
    print(f"Initial Loss: {initial_loss:.6f} | Final Loss: {final_loss:.6f}")
    if final_loss < 0.01:
        print("VERDICT: PASS (Perfect Memorization)")
    else:
        print("VERDICT: FAIL (Underfitting / Optimization Failure)")
    print("="*80)

if __name__ == "__main__":
    # Test a fresh model first
    run_memorization_test(model_type="fresh", lr=1e-3, steps=300)
    # Test the loaded model to see if it can also memorize
    run_memorization_test(model_type="loaded", lr=1e-3, steps=300)
