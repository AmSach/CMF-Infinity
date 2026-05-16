import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.model import CMFConfig, FastParallelContinuousMeaningField
from cmf.runtime import resolve_device

def generate_actual_cmf_image():
    print("--- REAL CMF IMAGE GENERATION ---")
    device = resolve_device("auto")
    
    # 1. Setup CMF Config for a 32x32 image (1024 pixels)
    config = CMFConfig(
        vocab_size=256,
        d_model=64,
        hidden_dim=128,
        num_layers=2,
        max_seq_len=1024
    )
    model = FastParallelContinuousMeaningField(config).to(device)
    
    # 2. Create Target Pattern: A White Cross
    target = torch.zeros((32, 32), device=device)
    target[14:18, :] = 1.0 # Horizontal bar
    target[:, 14:18] = 1.0 # Vertical bar
    target_flat = (target.flatten() * 255).long() # (1024,)
    
    # 3. Training Loop (Memorizing the Field)
    # We use a dummy input and train the model to output the cross pattern
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    dummy_input = torch.zeros((1, 1024), dtype=torch.long, device=device)
    
    print("Training CMF to learn the 'Meaning Field' of a cross...")
    for step in range(150):
        optimizer.zero_grad()
        out = model(dummy_input, labels=target_flat.unsqueeze(0))
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            
    # 4. ACTUAL GENERATION
    # We ask the model to generate the image from its learned field
    print("Generating image from CMF latent flow...")
    with torch.no_grad():
        gen_out = model(dummy_input)
        logits = gen_out["logits"] # (1, 1024, 256)
        # Take the most likely intensity for each pixel
        gen_pixels = torch.argmax(logits, dim=-1).squeeze(0) # (1024,)
        
    # 5. Save the Output
    gen_np = gen_pixels.cpu().numpy().reshape(32, 32).astype(np.uint8)
    img = Image.fromarray(gen_np)
    output_path = "cmf_generated_output.png"
    img.save(output_path)
    print(f"Actual CMF generated image saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_actual_cmf_image()
