import os
import sys
import subprocess

def run(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    # 1. Install dependencies
    print("--- Installing dependencies ---")
    # We use -e .[scale,vision,power] to install the local package with all optional extras
    run([sys.executable, "-m", "pip", "install", "-e", ".[scale,vision,power]"])

    # 2. Ensure data directory exists
    # The training batch file expects records/data/chinchilla_gpt2_120m as the source.
    base_data_dir = "records/data/chinchilla_gpt2_120m"
    if not os.path.exists(base_data_dir):
        print(f"--- Data directory {base_data_dir} not found. Preparing shards from FineWeb-Edu... ---")
        # Prepares 100M tokens from FineWeb-Edu (the default in prepare_hf_token_shards.py)
        run([
            sys.executable, "scripts/prepare_hf_token_shards.py",
            "--target-tokens", "100000000",
            "--output-dir", base_data_dir
        ])

    # 3. Create token cache snapshot (equivalent to first command in .bat)
    print("--- Creating token cache snapshot ---")
    run([
        sys.executable, "-u", "scripts/snapshot_token_cache.py",
        "--source-dir", base_data_dir,
        "--output-dir", "records/data/chinchilla_gpt2_100m_snapshot",
        "--target-tokens", "100000000",
        "--overwrite"
    ])

    # 4. Run training (equivalent to second command in .bat)
    print("--- Starting training ---")
    # Note: On Kaggle, you might want to increase micro-batch-size if using P100/T4, 
    # but we keep it identical to your .bat file as requested.
    run([
        sys.executable, "-u", "scripts/run_rtx4050_chinchilla.py",
        "--phase", "train",
        "--target-tokens", "100000000",
        "--data-dir", "records/data/chinchilla_gpt2_100m_snapshot",
        "--seq-len", "128",
        "--micro-batch-size", "4",
        "--grad-accum", "4",
        "--steps", "48828",
        "--save-every", "500",
        "--log-every", "1"
    ])

if __name__ == "__main__":
    main()
