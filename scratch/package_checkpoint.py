import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.model import DeliberativeContinuousMeaningField
from cmf.presets import get_preset
from cmf.config import CMFConfig
from cmf.checkpointing import save_model_package

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = ROOT / "checkpoint_latest(1).pt"
    package_out = ROOT / "records" / "checkpoints" / "cmf_120m_pretrained_latest.package.pt"
    
    print(f"Loading raw checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    preset = get_preset("infinity-reasoning-0.12b")
    config = CMFConfig(**preset.config.__dict__)
    
    print("Initializing model...")
    model = DeliberativeContinuousMeaningField(config)
    
    state_dict = ckpt["model"]
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
    model.to(device)
    model.eval()
    
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print(f"Packaging and saving model package to {package_out}...")
    save_model_package(
        package_out,
        model,
        model_type="deliberative_cmf",
        config=config,
        tokenizer=tokenizer,
        tokenizer_name="gpt2",
        training=ckpt.get("training", {}),
        extra={"notes": "Pretrained 120M Deliberative CMF checkpoint at step 22,780, ~1.49B tokens."}
    )
    print("Packaging complete!")

if __name__ == "__main__":
    main()
