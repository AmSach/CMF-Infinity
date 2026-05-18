import os
import sys
import subprocess
from pathlib import Path

# Resolve root path of the project absolute
ROOT = Path(__file__).resolve().parents[1]

def run(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    # 1. Install dependencies
    print("--- Installing dependencies ---")
    run([sys.executable, "-m", "pip", "install", "-e", f"{str(ROOT)}[scale,vision,power]"])
    run([sys.executable, "-m", "pip", "install", "ninja"])

    # 2. Parallel Tokenization (Fast Path)
    data_dir = ROOT / "records" / "data" / "cmf_hybrid_agi_cache"
    data_dir = ROOT / "records" / "data" / "cmf_hybrid_agi_cache"
    target_tokens = 200_000_000_000 # 200 Billion tokens for absolute hyper-saturation of the 120M model weights
    
    # Kill any zombie tokenizer processes from aborted runs to prevent file locks/contention
    try:
        subprocess.run(["pkill", "-f", "prepare_hf_token_parallel.py"], capture_output=True)
        subprocess.run(["pkill", "-f", "prepare_hybrid_datasets.py"], capture_output=True)
    except Exception:
        pass
    
    if not data_dir.exists() or not (data_dir / "manifest.json").exists():
        # Count existing shards to reassure the user
        existing_shards = sorted(data_dir.glob("tokens_*.pt")) if data_dir.exists() else []
        if existing_shards:
            print(f"\n--- [INFO] Found {len(existing_shards)} existing token shards already on disk! ---")
            print(f"--- The tokenizer will resume downloading and parsing starting at Shard {len(existing_shards)} ---\n")
        else:
            print(f"\n--- Preparing {target_tokens:,} tokens using Parallel Interleaved Mixer (BACKGROUND PROCESS) ---\n")
        
        # Launch tokenization as a background process redirecting stdout/stderr to a dedicated log file
        log_file = ROOT / "records" / "tokenizer_output.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_file, "w", encoding="utf-8")
        
        tok_proc = subprocess.Popen([
            sys.executable, str(ROOT / "scripts" / "prepare_hybrid_datasets.py"),
            "--target-tokens", str(target_tokens),
            "--shard-tokens", "25000000",
            "--output-dir", str(data_dir),
            "--append-eos"
        ], stdout=log_handle, stderr=subprocess.STDOUT)
        
        print(f"--- Parallel Interleaved Mixer launched in background. ---")
        print(f"--- All downloader/tokenizer output is logged to: {log_file} ---")
        print("--- You can view the live download/tokenizer progress anytime by running: !tail -n 20 /kaggle/working/records/tokenizer_output.log ---\n")
    else:
        print("\n--- [INFO] Full dataset tokenization manifest already exists. Skipping downloader. ---\n")
        tok_proc = None
 
 
    # 3. Start High-Speed 120M Reasoning Training (2x T4)
    print("--- Starting High-Speed 120M Reasoning Training (2x T4) ---")
    # For a 120M model, we can increase the micro-batch size to 4
    # and reduce grad-accum to 8 to keep the effective batch size at 64 (4 * 8 * 2 = 64)
    # while dramatically cutting sequential loop overhead and speeding up steps!
    run([
        "torchrun",
        "--nproc_per_node=2",
        str(ROOT / "scripts" / "train_distributed.py"),
        "--preset", "infinity-reasoning-0.12b",
        "--token-cache-dir", str(data_dir),
        "--micro-batch-size", "32",
        "--grad-accum", "2",
        "--lr", "1.5e-4", # Adjusted learning rate for stable 120M convergence
 
        "--seq-len", "512",
        "--steps", "3051757", # (200,000,000,000 / 65,536) matches the exact 200 Billion token budget!
        "--amp",
        "--tf32",
        "--gradient-checkpointing",
        "--compile",
        "--delete-consumed-shards",
        "--log-every", "1",
        "--save-every", "10",
        "--package-out", str(ROOT / "records" / "checkpoints" / "cmf_120m_reasoning.package.pt"),
        "--checkpoint-dir", str(ROOT / "records" / "checkpoints" / "cmf_120m_steps")
    ])

    if tok_proc is not None:
        print("Waiting for background parallel tokenizer to finish...")
        tok_proc.wait()

if __name__ == "__main__":
    main()
