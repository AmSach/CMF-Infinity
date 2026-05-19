import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from cmf import CMFConfig, DeliberativeContinuousMeaningField
from cmf.runtime import resolve_device

def main():
    device = resolve_device("cuda")
    
    # 1. Load baseline GPT-2 Small
    print("Loading GPT-2 Small baseline...")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    gpt2_model.eval()

    # 2. Load CMF 120M with strict key cleanup
    print("Loading CMF 120M pre-trained model and cleaning weight keys...")
    checkpoint_path = ROOT / "checkpoint_latest.pt"
    if not checkpoint_path.exists():
        checkpoint_path = ROOT / "cmf_120m_weights.pt"
        
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload["model"]
    
    # Clean compile prefixes
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        key = k
        while True:
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]
            elif key.startswith("module."):
                key = key[len("module."):]
            else:
                break
        cleaned_state_dict[key] = v
        
    from cmf.presets import get_preset
    preset = get_preset("infinity-reasoning-0.12b")
    config = CMFConfig(**preset.config.__dict__)
    
    cmf_model = DeliberativeContinuousMeaningField(config).to(device)
    cmf_model.load_state_dict(cleaned_state_dict, strict=True)
    cmf_model.eval()
    print("Pre-trained CMF 120M weights loaded with 100% strict matching success!")

    # List of test prompts
    prompts = [
        "Q: 2+3=? A:",
        "Q: 41+2=? A:",
        "Fact: alice is bob. Fact: bob is developer. Q: what is alice? A:",
        "Q: what is paris? A:",
        "System: You are CMF Infinity.\nUser: write a mathematical equation for meaning field.\nAssistant:"
    ]

    report_lines = []
    report_lines.append("======================================================================")
    report_lines.append(" ACTIVE PROMPT SIDE-BY-SIDE SHOWN DOWN")
    report_lines.append(" CMF 120M (Mathematical Cures Active) vs GPT-2 Small (124M)")
    report_lines.append("======================================================================")
    
    for idx, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {idx+1}/{len(prompts)}...")
        
        # --- GPT-2 Generation ---
        gpt2_input = torch.tensor([gpt2_tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            gpt2_output = gpt2_model.generate(
                gpt2_input,
                max_new_tokens=32,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                pad_token_id=gpt2_tokenizer.eos_token_id
            )
        gpt2_text = gpt2_tokenizer.decode(gpt2_output[0, gpt2_input.size(1):])
        
        # --- CMF 120M Generation ---
        cmf_input = torch.tensor([gpt2_tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            cmf_output = cmf_model.generate(
                cmf_input,
                max_new_tokens=32,
                temperature=0.8,
                top_k=50,
                sharp_memory_scale=0.25 # Steer active!
            )
        cmf_text = gpt2_tokenizer.decode(cmf_output[0, cmf_input.size(1):])

        # Clean outputs for printing
        gpt2_clean = gpt2_text.replace('\n', ' ').strip()
        cmf_clean = cmf_text.replace('\n', ' ').strip()

        report_lines.append(f"\n--- PROMPT {idx+1}: \"{prompt}\" ---")
        report_lines.append(f"  GPT-2 Small: {gpt2_clean}")
        report_lines.append(f"  CMF 120M:    {cmf_clean}")

    report_lines.append("\n======================================================================")
    
    report_text = "\n".join(report_lines)
    Path("records").mkdir(exist_ok=True)
    Path("records/active_prompts_completions.txt").write_text(report_text, encoding="utf-8")
    
    print("\nBenchmark and prompt comparison complete! Saved to records/active_prompts_completions.txt")

if __name__ == "__main__":
    main()
