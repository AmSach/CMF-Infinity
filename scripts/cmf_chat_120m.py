import os
import sys
import time
from pathlib import Path
import torch

# Prevent console character mapping errors on Windows
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
from cmf.generation import sample_next_token, apply_repetition_penalty, encode_to_tensor, decode_tokens
from cmf.runtime import resolve_device

def main():
    print("=" * 70)
    print("🌌 CMF INFINITY 120M INTERACTIVE TERMINAL ENGINE")
    print("=" * 70)
    print("Initializing model...")

    device = resolve_device("auto")
    checkpoint_path = ROOT / "checkpoint_latest.pt"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}!")
        print("Please verify the checkpoint is in the CMF root directory.")
        sys.exit(1)

    print(f"Loading checkpoint weights: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    step = payload["training"]["step"]
    print(f"Weights loaded successfully (Trained to step {step:,}).")

    preset_name = "infinity-reasoning-0.12b"
    preset = get_preset(preset_name)
    config = CMFConfig(**preset.config.__dict__)
    config.max_seq_len = 1024  # Allow longer context for interactive testing

    print(f"Building model architecture: {preset.display_name}")
    model = DeliberativeContinuousMeaningField(config)

    # Clean keys and load state dict
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
    model.eval()
    print(f"Model mapped to hardware target: {device}")

    print("Loading GPT-2 Tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("Tokenizer loaded.")

    print("\n" + "=" * 70)
    print("🔮 INTERACTIVE CHAT ENVIRONMENT READY!")
    print("=" * 70)
    print("Instructions:")
    print("1. Type your prompt and press Enter. The model will stream its completion in real-time.")
    print("2. Type 'exit' to quit.")
    print("3. Type 'temp <value>' (e.g. 'temp 0.1') to change temperature (0.1 = precise, 0.7 = creative).")
    print("=" * 70 + "\n")

    temperature = 0.5
    repetition_penalty = 1.1

    while True:
        try:
            user_input = input("\n>>> ").strip()
            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("Exiting...")
                break

            if user_input.lower().startswith("temp "):
                try:
                    val = float(user_input.split(" ")[1])
                    if 0.0 < val <= 2.0:
                        temperature = val
                        print(f"Temperature successfully updated to: {temperature}")
                    else:
                        print("Temperature must be between 0.1 and 2.0")
                except ValueError:
                    print("Invalid temperature value format.")
                continue

            # Standard language model completion run
            print(f"\n[CMF-120M (Step {step:,}) Generating... Temp={temperature}]")
            print("-" * 50)
            
            # Print the prompt to start the line
            sys.stdout.write(user_input)
            sys.stdout.flush()

            input_ids = encode_to_tensor(tokenizer, user_input).unsqueeze(0).to(device)
            generated = input_ids
            
            max_new_tokens = 512
            eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    model_input = generated
                    if generated.size(1) > config.max_seq_len:
                        model_input = generated[:, -config.max_seq_len:]
                        
                    output = model(model_input)
                    logits = output["logits"][:, -1]
                    logits = apply_repetition_penalty(logits, generated, repetition_penalty)
                    
                    next_token = sample_next_token(
                        logits,
                        temperature=temperature,
                        top_k=20,
                        top_p=0.9,
                    )
                    
                    # Decoded text token to stream to stdout
                    token_text = decode_tokens(tokenizer, next_token[0])
                    sys.stdout.write(token_text)
                    sys.stdout.flush()
                    
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    if eos_token_id is not None and int(next_token[0, 0]) == eos_token_id:
                        break
                        
            print("\n" + "-" * 50)

        except KeyboardInterrupt:
            print("\nGeneration cancelled by user.")
        except Exception as e:
            print(f"\nError during generation: {e}")

if __name__ == "__main__":
    main()
