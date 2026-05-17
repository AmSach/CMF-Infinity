import os
import sys
import subprocess
from pathlib import Path

# Resolve absolute root of CMF repository
ROOT = Path(__file__).resolve().parents[1]

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def run(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    if not is_colab():
        print("This script is intended to be run inside a Google Colab notebook.")
        return

    # 1. Mount Google Drive for persistent checkpoints if not already mounted
    if not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except Exception as exc:
            print(f"Could not mount drive interactively: {exc}")
            print("Please mount drive from the Colab sidebar or using a cell command instead.")
    
    drive_root = Path("/content/drive/MyDrive/CMF_Training_2B")
    drive_root.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = drive_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    package_path = drive_root / "cmf_2b_reasoning.package.pt"

    # 2. Install dependencies
    print("--- Installing CMF dependencies ---")
    run(f"pip install -e {str(ROOT)}[scale,vision,power]")
    run("pip install ninja")

    # 3. Parallel Tokenization (Fast Path)
    data_dir = Path("/content/fineweb_edu_2b")
    target_tokens = 200_000_000 # 200M tokens is perfect for a 4-hour Colab budget
    
    if not data_dir.exists() or not (data_dir / "manifest.json").exists():
        print(f"--- Preparing {target_tokens:,} tokens using Parallel Tokenizer ---")
        run(f"python {str(ROOT / 'scripts' / 'prepare_hf_token_parallel.py')} "
            f"--dataset HuggingFaceTB/smollm-corpus "
            f"--dataset-name fineweb-edu-dedup "
            f"--target-tokens {target_tokens} "
            f"--shard-tokens 25000000 "
            f"--output-dir {str(data_dir)} "
            f"--append-eos")

    # 4. Start Deep 2B Reasoning Training
    print("--- Starting Deep 2B Reasoning Training on Colab ---")
    # Using single-GPU distributed framework with gradient checkpointing
    run(f"torchrun --nproc_per_node=1 {str(ROOT / 'scripts' / 'train_distributed.py')} "
        f"--preset infinity-reasoning-2b "
        f"--token-cache-dir {str(data_dir)} "
        f"--micro-batch-size 2 " # Can be 2 or 4 depending on T4 vs L4/A100 VRAM
        f"--grad-accum 16 "
        f"--lr 1e-4 "
        f"--seq-len 512 "
        f"--amp "
        f"--tf32 "
        f"--gradient-checkpointing "
        f"--log-every 1 "
        f"--save-every 250 "
        f"--package-out {str(package_path)} "
        f"--checkpoint-dir {str(checkpoint_dir)}")

if __name__ == "__main__":
    main()
