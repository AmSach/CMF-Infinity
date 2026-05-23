import sys, time, torch, math
from pathlib import Path
from transformers import GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load GPT2 tokenizer
print("Loading GPT2 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

validation_text = (
    "In mathematics, a continuous function is a function that does not have any abrupt changes in value, "
    "known as discontinuities. More precisely, sufficiently small changes in the input of a continuous function "
    "result in arbitrarily small changes in its output. If the function is not continuous, it is said to be discontinuous. "
    "Continuity of functions is one of the core concepts of mathematical analysis and topology, which refers to the "
    "stability of properties under deformation. Many physical processes, such as the motion of a planet or the "
    "cooling of a cup of coffee, are continuous."
)

tokens = tokenizer.encode(validation_text)
batch_size = 4
seq_len = 64
inputs_list = []
labels_list = []
for i in range(batch_size):
    start = i * 10
    chunk = tokens[start : start + seq_len + 1]
    inputs_list.append(torch.tensor(chunk[:-1], dtype=torch.long))
    labels_list.append(torch.tensor(chunk[1:], dtype=torch.long))
    
x = torch.stack(inputs_list).to(device)
y = torch.stack(labels_list).to(device)

# Load the trained checkpoint
print("Loading model checkpoint...")
ckpt = torch.load("checkpoint_latest_new.pt", map_location="cpu", weights_only=False)
cfg_src = ckpt["config"]
cfg_dict = cfg_src if isinstance(cfg_src, dict) else cfg_src.__dict__.copy()
sd = ckpt["model"]

# Ensure memory anchors config matches checkpoint
mem_keys = [k for k in sd if "field.memory" in k and "bank" not in k]
if mem_keys:
    cfg_dict["num_memory_anchors"] = sd[mem_keys[0]].shape[0]

# Enable adaptive thinking
cfg_dict["adaptive_thinking"] = True
config = CMFConfig(**cfg_dict)

model = DeliberativeContinuousMeaningField(config).to(device)
clean = {}
for k, v in sd.items():
    k2 = k.replace("_orig_mod.module.", "").replace("module.", "")
    clean[k2] = v
model.load_state_dict(clean, strict=False)
model.eval()

# Test different halting thresholds
thresholds = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]

print("\n" + "="*80)
print("EVALUATING MODEL LOSS & STEPS VS HALTING THRESHOLD (ADAPTIVE THINKING)")
print("="*80)
print(f"{'Halt Threshold':<15} | {'Loss':<12} | {'Average Steps':<15} | {'Halt Mean':<12}")
print("-" * 60)

for th in thresholds:
    model.config.halting_threshold = th
    
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
            out = model(x, labels=y)
            loss = out["loss"].item()
            steps = out["thinking_steps"].item()
            halt_mean = out["halt_mean"].item()
            
    print(f"{th:<15.2f} | {loss:<12.6f} | {steps:<15.2f} | {halt_mean:<12.4f}")

print("="*80)
