import sys, io, math, torch, tiktoken
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf.config import CMFConfig
from cmf.model import DeliberativeContinuousMeaningField

CKPT_NEW = Path("e:/CMF/checkpoint_latest_new.pt")
enc = tiktoken.get_encoding("gpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"

test_sentences = [
    "The capital of France is Paris.",
    "One plus one equals two.",
    "def hello(): print('hello')",
    "The quick brown fox jumps over the lazy dog.",
]

def load_clean_model():
    ckpt = torch.load(str(CKPT_NEW), map_location="cpu", weights_only=False)
    cfg_src = ckpt["config"]
    cfg_dict = cfg_src if isinstance(cfg_src, dict) else cfg_src.__dict__.copy()
    sd = ckpt["model"]
    mem_keys = [k for k in sd if "field.memory" in k and "bank" not in k]
    if mem_keys:
        cfg_dict["num_memory_anchors"] = sd[mem_keys[0]].shape[0]
    config = CMFConfig(**cfg_dict)
    model = DeliberativeContinuousMeaningField(config)
    clean = {}
    for k, v in sd.items():
        k2 = k.replace("_orig_mod.module.", "").replace("module.", "")
        clean[k2] = v
    model.load_state_dict(clean, strict=False)
    model.eval()
    return model.to(device)

def get_loss(model, text, steps):
    model.config.adaptive_thinking = False
    model.config.thinking_steps = steps
    ids = torch.tensor([enc.encode(text)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(ids[:, :-1], labels=ids[:, 1:])
    return out["loss"].item()

model = load_clean_model()
steps_list = [1, 2, 4, 8, 16, 32]

print("="*90)
print(f"{'Sentence':<50} | " + " | ".join(f"S={s:<2d}" for s in steps_list))
print("="*90)

sent_losses = {s: [] for s in steps_list}
for text in test_sentences:
    row = []
    for s in steps_list:
        l = get_loss(model, text, s)
        row.append(f"{l:.4f}")
        sent_losses[s].append(l)
    print(f"{text[:50]:<50} | " + " | ".join(row))

print("="*90)
avg_row = []
for s in steps_list:
    avg = sum(sent_losses[s]) / len(sent_losses[s])
    avg_row.append(f"{avg:.4f}")
print(f"{'AVERAGE':<50} | " + " | ".join(avg_row))
print("="*90)
