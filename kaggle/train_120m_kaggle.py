import os
import sys
import subprocess
from pathlib import Path

# Resolve root path of the project absolute
ROOT = Path(__file__).resolve().parents[1]

def run(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def cleanup_disk_space():
    print("\n--- [Space Saver Auto-Pilot] Scanning for redundant files to free up disk space... ---")
    
    # 1. Clean up old runs directory
    runs_dir = ROOT / "records" / "runs"
    if runs_dir.exists():
        freed = 0
        deleted_dirs = 0
        for item in runs_dir.iterdir():
            if item.is_dir():
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                try:
                    import shutil
                    shutil.rmtree(item)
                    freed += size
                    deleted_dirs += 1
                except Exception as e:
                    print(f"Warning: Failed to delete old run dir {item.name}: {e}")
        if deleted_dirs > 0:
            print(f"Successfully cleaned up {deleted_dirs} obsolete run directories, freeing {freed / (1024*1024*1024):.2f} GB!")
            
    # 1.5 Migrate previous checkpoints if they exist to the root directory
    old_nested = ROOT / "records" / "checkpoints" / "cmf_120m_steps" / "checkpoint_latest.pt"
    old_flat = ROOT / "records" / "checkpoints" / "checkpoint_latest.pt"
    new_root_ckpt = ROOT / "checkpoint_latest.pt"
    
    if old_nested.exists() and not new_root_ckpt.exists():
        try:
            import shutil
            shutil.move(str(old_nested), str(new_root_ckpt))
            print(f"\n--- [Migration] Successfully migrated nested checkpoint to root: {new_root_ckpt} ---\n")
        except Exception as migration_err:
            print(f"Warning: Failed to migrate nested checkpoint: {migration_err}")
            
    if old_flat.exists() and not new_root_ckpt.exists():
        try:
            import shutil
            shutil.move(str(old_flat), str(new_root_ckpt))
            print(f"\n--- [Migration] Successfully migrated flat checkpoint to root: {new_root_ckpt} ---\n")
        except Exception as migration_err:
            print(f"Warning: Failed to migrate flat checkpoint: {migration_err}")

    # 2. Clean up redundant checkpoints in records/checkpoints
    ckpt_dir = ROOT / "records" / "checkpoints"
    if ckpt_dir.exists():
        freed = 0
        deleted_files = 0
        # Preserve: the active latest checkpoint file, active steps directory, and the final target package
        preserve_names = {"checkpoint_latest.pt", "cmf_120m_steps", "cmf_120m_reasoning.package.pt"}
        for item in ckpt_dir.iterdir():
            if item.name in preserve_names:
                continue
            if item.is_file() and item.suffix in {".pt", ".package.pt"}:
                size = item.stat().st_size
                try:
                    item.unlink()
                    freed += size
                    deleted_files += 1
                except Exception as e:
                    print(f"Warning: Failed to delete redundant checkpoint {item.name}: {e}")
        if deleted_files > 0:
            print(f"Successfully deleted {deleted_files} redundant checkpoints, freeing {freed / (1024*1024*1024):.2f} GB!")

    # 3. Clean up obsolete datasets in records/data
    data_dir_root = ROOT / "records" / "data"
    if data_dir_root.exists():
        freed = 0
        deleted_dirs = 0
        # Preserve: only the active dataset cache folder used by Kaggle pretraining
        preserve_names = {"cmf_hybrid_agi_cache"}
        for item in data_dir_root.iterdir():
            if item.is_dir() and item.name not in preserve_names:
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                try:
                    import shutil
                    shutil.rmtree(item)
                    freed += size
                    deleted_dirs += 1
                except Exception as e:
                    print(f"Warning: Failed to delete old dataset dir {item.name}: {e}")
        if deleted_dirs > 0:
            print(f"Successfully cleaned up {deleted_dirs} obsolete dataset directories, freeing {freed / (1024*1024*1024):.2f} GB!")

    print("--- [Space Saver Auto-Pilot] Scan complete. Output storage is optimized. ---\n")

def auto_import_weights():
    checkpoint_path = ROOT / "checkpoint_latest.pt"
    print("\n--- [Git LFS Auto-Import] Checking checkpoint weights... ---")
    
    needs_pull = False
    if not checkpoint_path.exists():
        print("checkpoint_latest.pt does not exist locally. Will trigger Git LFS pull.")
        needs_pull = True
    elif checkpoint_path.stat().st_size < 1000:
        print("Found checkpoint_latest.pt but it appears to be a Git LFS pointer file (< 1KB).")
        needs_pull = True
    else:
        print(f"Verified checkpoint_latest.pt is present and valid ({checkpoint_path.stat().st_size / (1024*1024):.2f} MB).")
        return
        
    if needs_pull:
        print("Running 'git lfs install' and 'git lfs pull' to download the 612MB weights file...")
        try:
            subprocess.run(["git", "lfs", "install"], check=True, cwd=str(ROOT))
            subprocess.run(["git", "lfs", "pull"], check=True, cwd=str(ROOT))
            
            if checkpoint_path.exists() and checkpoint_path.stat().st_size >= 1000:
                print(f"Success! Downloaded checkpoint_latest.pt ({checkpoint_path.stat().st_size / (1024*1024):.2f} MB).")
            else:
                print("Warning: git lfs pull executed but checkpoint_latest.pt is still missing or a pointer.")
        except Exception as e:
            print(f"Error executing Git LFS commands: {e}")
            print("Please ensure git-lfs is installed in the system. Trying to install git-lfs via apt...")
            try:
                subprocess.run(["apt-get", "update"], check=True)
                subprocess.run(["apt-get", "install", "-y", "git-lfs"], check=True)
                subprocess.run(["git", "lfs", "install"], check=True, cwd=str(ROOT))
                subprocess.run(["git", "lfs", "pull"], check=True, cwd=str(ROOT))
                if checkpoint_path.exists() and checkpoint_path.stat().st_size >= 1000:
                    print(f"Success after apt-get! checkpoint_latest.pt downloaded ({checkpoint_path.stat().st_size / (1024*1024):.2f} MB).")
            except Exception as apt_err:
                print(f"Could not install git-lfs via apt-get: {apt_err}")
                print("Please download the weights manually or configure Git LFS in your environment.")

def main():
    # 0. Clean up space first to prevent disk full issues
    cleanup_disk_space()
    
    # 0.5 Auto-import weights if Git LFS checked out a pointer
    auto_import_weights()

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
            "--append-eos",
            "--max-ahead", "5"
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
        "--warmup-steps", "1000", # Fast warmup for 1k steps to allow maximum decay time
        "--min-lr-ratio", "0.05",
        "--empty-cache-every", "100",
 
        "--seq-len", "512",
        "--steps", "15000", # Complete cosine decay exactly at step 15,000 for ultimate quality
        "--amp",
        "--tf32",
        "--gradient-checkpointing",
        "--compile",
        "--delete-consumed-shards",
        "--log-every", "1",
        "--save-every", "5",
        "--package-out", str(ROOT / "cmf_120m_reasoning.package.pt"),
        "--checkpoint-dir", str(ROOT)
    ])

    if tok_proc is not None:
        print("Waiting for background parallel tokenizer to finish...")
        tok_proc.wait()

    # 4. Start Supervised Fine-Tuning (SFT) Alignment automatically
    print("\n--- [CMF-v2 Alignment] Starting Supervised Fine-Tuning (SFT) Alignment ---")
    run([
        sys.executable,
        str(ROOT / "scripts" / "train_sft_v2.py"),
        "--base-checkpoint", "checkpoint_latest.pt",
        "--out-package", str(ROOT / "cmf_120m_reasoning.package.pt"),
        "--lr", "3e-5",
        "--epochs", "5",
        "--batch-size", "4",
        "--solver-method", "symplectic"
    ])
    print("--- [CMF-v2 Alignment] SFT Alignment complete! Aligned package saved to cmf_120m_reasoning.package.pt ---")

if __name__ == "__main__":
    main()
