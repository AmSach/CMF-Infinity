import os
import sys
import subprocess
from pathlib import Path

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

    # 1. Mount Google Drive for persistent checkpoints
    from google.colab import drive
    drive.mount('/content/drive')
    
    drive_root = Path("/content/drive/MyDrive/CMF_Training")
    drive_root.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = drive_root / "0.5b_extensive_checkpoint.pt"
    package_path = drive_root / "0.5b_extensive_final.package.pt"

    # 2. Install dependencies
    print("--- Installing CMF dependencies ---")
    run("pip install -e .[scale,vision,power]")
    run("pip install ninja")

    # 3. Training Configuration
    # We use a larger micro-batch-size for Colab (assuming L4 or A100)
    # Target: Full FineWeb-Edu training
    print("--- Starting Extensive Training on FineWeb-Edu ---")
    
    train_cmd = [
        "python", "scripts/train_large_scale.py",
        "--preset", "infinity-0.5b",
        "--dataset", "HuggingFaceTB/smollm-corpus",
        "--dataset-name", "fineweb-edu-dedup",
        "--steps", "1000000", # High step count for extensive training
        "--micro-batch-size", "8", # Increased for Colab VRAM
        "--grad-accum", "8",
        "--lr", "2e-4",
        "--amp",
        "--tf32",
        "--compile",
        "--checkpoint", str(checkpoint_path),
        "--package-out", str(package_path),
        "--save-every", "500",
        "--log-every", "10"
    ]
    
    run(" ".join(train_cmd))

if __name__ == "__main__":
    main()
