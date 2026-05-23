import os
import torch
import tiktoken
from pathlib import Path

# Ensure we can import the CMF package
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from cmf.model import DeliberativeContinuousMeaningField
from cmf.config import CMFConfig

# Paths to checkpoints
BASELINE_PATH = Path('checkpoint_latest.pt')
TINY_PATH = Path('checkpoint_latest(1).pt')  # checkpoint from Downloads (step 2040)

# Tiny dataset – a few short sentences
DATASET_PATH = Path('scratch') / 'tiny_dataset.txt'
if not DATASET_PATH.exists():
    DATASET_PATH.write_text('''
The quick brown fox jumps over the lazy dog.
Ada Lovelace wrote the first algorithm.
Python is a popular programming language.
Artificial intelligence can reason.
The sky is blue during the day.
'''.strip(), encoding='utf-8')

# Load tokenizer (gpt2 compatible)
enc = tiktoken.get_encoding('gpt2')

def load_model_from_checkpoint(ckpt_path: Path):
    # Load checkpoint and adjust config if needed
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    # Extract config dict from checkpoint (may be a CMFConfig instance or dict)
    config_source = ckpt['config']
    config_dict = config_source if isinstance(config_source, dict) else config_source.__dict__
    # The tiny checkpoint was trained with 32 memory anchors, but current default is 64.
    # If the checkpoint's state dict expects 32, adjust the config accordingly.
    if any('field.memory' in k for k in ckpt['model'].keys()):
        # Detect size mismatch by checking a known param shape length (32 vs 64)
        sample_key = next(k for k in ckpt['model'].keys() if 'field.memory' in k)
        if ckpt['model'][sample_key].shape[0] == 32:
            config_dict['num_memory_anchors'] = 32
    config = CMFConfig(**config_dict)
    model = DeliberativeContinuousMeaningField(config)
    # Clean state dict keys
    state_dict = ckpt['model']
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.module.', '').replace('module.', '')
        clean_state_dict[new_key] = v
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    return model

def compute_average_loss(model, tokens):
    device = torch.device('cpu')
    model.to(device)
    with torch.no_grad():
        x = tokens[:-1].unsqueeze(0).to(device)
        y = tokens[1:].unsqueeze(0).to(device)
        out = model(x, labels=y)
        loss = out['loss'].item()
    return loss

def evaluate_checkpoint(ckpt_path, name):
    print(f'--- Evaluating {name} checkpoint ({ckpt_path.name}) ---')
    model = load_model_from_checkpoint(ckpt_path)
    # Load dataset
    text = DATASET_PATH.read_text(encoding='utf-8')
    token_ids = torch.tensor(enc.encode(text), dtype=torch.long)
    # Compute loss on entire dataset (single forward pass)
    avg_loss = compute_average_loss(model, token_ids)
    # Use ASCII hyphen to avoid encoding issues on Windows
    print(f'Average cross-entropy loss on tiny dataset: {avg_loss:.4f}')
    # Simple generation test for semantic continuity
    prompt = "The quick brown fox"
    input_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        gen_ids = model.generate(input_ids, max_new_tokens=30, temperature=0.7)
    generated = enc.decode(gen_ids[0].tolist())
    print('Generation (semantic continuity test):')
    print(generated)
    print()

if __name__ == '__main__':
    # Baseline evaluation
    evaluate_checkpoint(BASELINE_PATH, 'Baseline')
    # Tiny checkpoint evaluation
    evaluate_checkpoint(TINY_PATH, 'Tiny (new)')
