import torch
import torch.nn as nn
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.model import CMFConfig, ContinuousMeaningField
from cmf.tokenizer import SimpleBPETokenizer
from cmf.checkpointing import save_model_package
from cmf.runtime import resolve_device

def train_basic_grammar():
    print("--- CMF GRAMMAR FLASH-TRAINING ---")
    device = resolve_device("auto")
    
    config = CMFConfig(
        vocab_size=1000, 
        d_model=128, hidden_dim=256, num_layers=8,
        adaptive_steps=True
    )
    
    grammar_data = [
        "Hello, I am the CMF Infinity engine.",
        "The field is continuous and fluid.",
        "Logic is the foundation of intelligence.",
        "I can reason through complex paths.",
        "The sky is blue and the grass is green.",
        "AI is moving beyond transformers.",
        "CMF solves the quadratic bottleneck."
    ] * 100
    
    # Train Tokenizer
    tokenizer = SimpleBPETokenizer(vocab_size=1000)
    full_text = " ".join(grammar_data)
    tokenizer.train(full_text)
    
    model = ContinuousMeaningField(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Encode a training block
    tokens = [int(t) for t in tokenizer.encode(full_text[:1024])]
    x = torch.tensor([tokens[:-1]], device=device, dtype=torch.long)
    y = torch.tensor([tokens[1:]], device=device, dtype=torch.long)
    
    print("Teaching the model English structure...")
    for step in range(300):
        optimizer.zero_grad()
        out = model(x, labels=y)
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            
    # Save the "Grammar Aware" weights
    torch.save(model.state_dict(), "infinity_weights_grammar.pt")
    save_model_package(
        ROOT / "records" / "checkpoints" / "infinity_grammar.package.pt",
        model,
        model_type="continuous_cmf",
        config=config,
        tokenizer=tokenizer,
        training={"steps": 300, "corpus": "local grammar demo"},
    )
    # Also save the tokenizer state
    import pickle
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
        
    print("Success: Weights saved to infinity_weights_grammar.pt")

if __name__ == "__main__":
    train_basic_grammar()
