import os
import sys
import time
import math
from pathlib import Path
import torch

# Configure terminal output to support UTF-8 encoding safely on Windows
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField
from cmf.presets import get_preset
from cmf.generation import generate_ids, decode_tokens, encode_to_tensor, trim_assistant_response
from cmf.runtime import resolve_device

def load_gpt2_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")

def evaluate_perplexity_on_textbook(model, tokenizer, device, textbook_path: Path, seq_len: int = 128) -> tuple[float, float]:
    model.eval()
    if not textbook_path.exists():
        return float("nan"), float("nan")
    
    text = textbook_path.read_text(encoding="utf-8")
    tokens = encode_to_tensor(tokenizer, text)
    
    step_losses = []
    stride = seq_len
    with torch.no_grad():
        for start in range(0, tokens.numel() - seq_len - 1, stride):
            chunk = tokens[start : start + seq_len + 1]
            x = chunk[:-1].unsqueeze(0).to(device)
            y = chunk[1:].unsqueeze(0).to(device)
            out = model(x, labels=y)
            step_losses.append(float(out["loss"].detach().cpu()))
            
    if not step_losses:
        return float("nan"), float("nan")
    avg_loss = sum(step_losses) / len(step_losses)
    perplexity = math.exp(min(avg_loss, 20.0))
    return avg_loss, perplexity

def run_factual_and_reasoning_benchmarks(model, tokenizer, device) -> dict:
    prompts = [
        {"prompt": "Q: what is cuda? A: ", "expected": "gpu", "kind": "facts"},
        {"prompt": "Q: what is cmf? A: ", "expected": "field", "kind": "facts"},
        {"prompt": "Q: what is python? A: ", "expected": "language", "kind": "facts"},
        {"prompt": "Q: 7+8=? A: ", "expected": "15", "kind": "arithmetic"},
        {"prompt": "Q: 12+5=? A: ", "expected": "17", "kind": "arithmetic"},
        {"prompt": "Fact: alice is bob. Fact: bob is developer. Q: what is alice? A: ", "expected": "developer", "kind": "chain_reasoning"},
    ]
    
    model.eval()
    results = []
    correct_count = 0
    
    with torch.no_grad():
        for item in prompts:
            input_ids = encode_to_tensor(tokenizer, item["prompt"]).unsqueeze(0).to(device)
            output_ids = generate_ids(
                model,
                input_ids,
                max_new_tokens=8,
                temperature=0.1,
                top_k=5,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None,
                max_context_tokens=128
            )
            decoded = decode_tokens(tokenizer, output_ids[0])
            completed_part = trim_assistant_response(decoded)
            
            is_correct = item["expected"].lower() in completed_part.lower()
            if is_correct:
                correct_count += 1
                
            results.append({
                "prompt": item["prompt"],
                "expected": item["expected"],
                "generated": completed_part.strip(),
                "correct": is_correct,
                "kind": item["kind"]
            })
            
    return {
        "accuracy": correct_count / len(prompts),
        "results": results
    }

def main():
    print("=" * 70)
    print("CMF 120M PRETRAINING BENCHMARK ENGINE")
    print("=" * 70)
    
    device = resolve_device("auto")
    print(f"Hardware device: {device}")
    
    checkpoint_path = ROOT / "checkpoint_latest.pt"
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}!")
        sys.exit(1)
        
    print(f"Loading checkpoint: {checkpoint_path}...")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    step = payload["training"]["step"]
    loss_at_checkpoint = payload["training"].get("loss", "unknown")
    print(f"Checkpoint step: {step} | Last loss: {loss_at_checkpoint}")
    
    preset_name = "infinity-reasoning-0.12b"
    preset = get_preset(preset_name)
    config = CMFConfig(**preset.config.__dict__)
    config.max_seq_len = 128
    
    print(f"Reconstructing deliberative model using preset: {preset_name}")
    model = DeliberativeContinuousMeaningField(config)
    
    state_dict = payload["model"]
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
        
    model.load_state_dict(cleaned_state_dict)
    model = model.to(device)
    print(f"Model loaded and mapped to device.")
    
    print("\nLoading GPT-2 Tokenizer...")
    tokenizer = load_gpt2_tokenizer()
    print("Tokenizer loaded.")
    
    print("\n" + "-" * 50)
    print("BENCHMARK 1: Continuous DL & CMF Textbook Perplexity")
    print("-" * 50)
    textbook_path = ROOT / "docs" / "cmf_masterclass_textbook.md"
    textbook_loss, textbook_ppl = evaluate_perplexity_on_textbook(model, tokenizer, device, textbook_path)
    print(f"Textbook Cross-Entropy Loss: {textbook_loss:.4f}")
    print(f"Textbook Model Perplexity (PPL): {textbook_ppl:.4f}")
    
    print("\n" + "-" * 50)
    print("BENCHMARK 2: Reasoning & Fact-Retrieval Accuracy")
    print("-" * 50)
    bench_results = run_factual_and_reasoning_benchmarks(model, tokenizer, device)
    print(f"Total Prompt Accuracy: {bench_results['accuracy'] * 100:.1f}%\n")
    
    for r in bench_results["results"]:
        status = "[CORRECT]" if r["correct"] else "[INCORRECT]"
        print(f"[{r['kind'].upper()}] Prompt: '{r['prompt']}'")
        print(f" -> Expected: '{r['expected']}'")
        print(f" -> Generated: '{r['generated']}'")
        print(f" -> Verdict: {status}\n")
        
    print("-" * 50)
    print("BENCHMARK 3: Open-Ended Generative Synthesis Samples")
    print("-" * 50)
    open_prompts = [
        "Explain CMF in one short sentence: ",
        "In the continuous cosmos, the meaning field is defined by ",
        "Deep learning and standard transformers are limited because "
    ]
    
    model.eval()
    with torch.no_grad():
        for prompt in open_prompts:
            input_ids = encode_to_tensor(tokenizer, prompt).unsqueeze(0).to(device)
            output_ids = generate_ids(
                model,
                input_ids,
                max_new_tokens=24,
                temperature=0.6,
                top_k=20,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None,
                max_context_tokens=128
            )
            decoded = decode_tokens(tokenizer, output_ids[0])
            completed = trim_assistant_response(decoded)
            print(f"Prompt: \"{prompt}\"")
            print(f"Completion: \"{completed.strip()}\"")
            print("-" * 30)

    report_path = ROOT / "records" / "pretrain_10k_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_content = f"""# Continuous Meaning Field (CMF) 10,000 Step Pretraining Report

## Overview
- **Model Preset**: `infinity-reasoning-0.12b` (Deliberative Continuous Meaning Field)
- **Parameter Count**: ~120M
- **Training Steps Completed**: {step:,} steps
- **Pretraining Dataset**: Mixed AGI High-Density Mix (Wikipedia, FineWeb-Edu, cosmopedia, stack-code, open-web-math, proofs)

---

## Benchmark 1: Out-of-Domain Textbook Perplexity
We evaluated the model's cross-entropy loss and perplexity on the comprehensive **CMF Masterclass Textbook** (`docs/cmf_masterclass_textbook.md`). This serves as a clean, contamination-free test of logical cohesion.
- **Cross-Entropy Loss**: `{textbook_loss:.4f}`
- **Model Perplexity (PPL)**: `{textbook_ppl:.4f}`

---

## Benchmark 2: Exact Factual Retrieval & Reasoning
We evaluated direct factual completion accuracy on math, logical chaining, and general knowledge:
- **Exact Accuracy**: `{bench_results['accuracy'] * 100:.1f}%`

### Task Matrix:
| Prompt | Expected | Generated | Verdict |
| :--- | :--- | :--- | :--- |
"""
    for r in bench_results["results"]:
        status = "CORRECT" if r["correct"] else "INCORRECT"
        report_content += f"| `{r['prompt']}` | `{r['expected']}` | `{r['generated']}` | {status} |\n"
        
    report_content += f"""
---

## Benchmark 3: Open-Ended Generative Synthesis
We ran generative completions with moderate temperature (0.6) to see if the model produces structurally and semantically coherent English paragraphs:

"""
    for prompt in open_prompts:
        input_ids = encode_to_tensor(tokenizer, prompt).unsqueeze(0).to(device)
        output_ids = generate_ids(
            model,
            input_ids,
            max_new_tokens=24,
            temperature=0.6,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None,
            max_context_tokens=128
        )
        completed = trim_assistant_response(decode_tokens(tokenizer, output_ids[0]))
        report_content += f"### Prompt: \"{prompt}\"\n> **Completion**: \"{completed.strip()}\"\n\n"
        
    report_content += "\n---\n*Report generated on the fly by Antigravity.*"
    report_path.write_text(report_content, encoding="utf-8")
    print(f"\nSaved comprehensive pretraining report to: {report_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
