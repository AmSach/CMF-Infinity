import os
import shutil
from pathlib import Path

def main():
    print("--- CMF Infinity Space Cleanup Script ---")
    ROOT = Path(__file__).resolve().parents[1]
    checkpoint_dir = ROOT / "records" / "checkpoints" / "cmf_1.2b_steps"
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory {checkpoint_dir} does not exist. Nothing to clean up.")
        return

    # Files to keep
    keep_files = {"checkpoint_latest.pt"}
    
    deleted_count = 0
    bytes_freed = 0
    
    print("\nScanning for redundant milestone checkpoints...")
    for item in checkpoint_dir.iterdir():
        if item.is_file():
            if item.name.startswith("checkpoint_step_") and item.suffix == ".pt":
                file_size = item.stat().st_size
                print(f"Found redundant milestone checkpoint: {item.name} ({file_size / (1024*1024*1024):.2f} GB)")
                try:
                    item.unlink()
                    deleted_count += 1
                    bytes_freed += file_size
                except Exception as e:
                    print(f"Error deleting {item.name}: {e}")
                    
    # Scan main checkpoints directory
    main_ckpt_dir = ROOT / "records" / "checkpoints"
    if main_ckpt_dir.exists():
        for item in main_ckpt_dir.iterdir():
            if item.is_file() and item.name.startswith("checkpoint_step_") and item.suffix == ".pt":
                file_size = item.stat().st_size
                print(f"Found redundant milestone checkpoint in main dir: {item.name} ({file_size / (1024*1024*1024):.2f} GB)")
                try:
                    item.unlink()
                    deleted_count += 1
                    bytes_freed += file_size
                except Exception as e:
                    print(f"Error deleting {item.name}: {e}")

    gb_freed = bytes_freed / (1024 * 1024 * 1024)
    print(f"\nCleanup complete! Deleted {deleted_count} milestone checkpoints.")
    print(f"Successfully freed up {gb_freed:.2f} GB of disk space.")
    print("Safe to resume training now. All necessary files (checkpoint_latest.pt, dataset shards) were preserved.")

if __name__ == "__main__":
    main()
