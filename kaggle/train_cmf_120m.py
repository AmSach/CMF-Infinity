from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KAGGLE_INPUT = Path("/kaggle/input")
DEFAULT_WORKDIR = Path(os.environ.get("CMF_WORKDIR", "/kaggle/working/cmf_120m"))
if not Path("/kaggle").exists():
    DEFAULT_WORKDIR = ROOT / "records" / "kaggle_run"


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def discover_package() -> Path | None:
    candidates = [
        DEFAULT_WORKDIR / "checkpoints" / "cmf_120m_kaggle.package.pt",
        ROOT / "pretrained" / "cmf_120m_gate.package.pt",
        ROOT / "records" / "checkpoints" / "cmf_120m_gate.package.pt",
    ]
    if KAGGLE_INPUT.exists():
        candidates.extend(KAGGLE_INPUT.rglob("cmf_120m_gate.package.pt"))
        candidates.extend(KAGGLE_INPUT.rglob("cmf_reasoning_120m_gate.package.pt"))
        candidates.extend(KAGGLE_INPUT.rglob("*.package.pt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def discover_token_cache() -> Path | None:
    candidates = [
        ROOT / "data" / "smollm_fineweb_edu_gpt2_2m.pt",
        ROOT / "records" / "data" / "smollm_fineweb_edu_gpt2_2m.pt",
    ]
    if KAGGLE_INPUT.exists():
        candidates.extend(KAGGLE_INPUT.rglob("*token*.pt"))
        candidates.extend(KAGGLE_INPUT.rglob("*gpt2*.pt"))
        candidates.extend(KAGGLE_INPUT.rglob("*smollm*.pt"))
    for candidate in candidates:
        if candidate.name.endswith(".package.pt"):
            continue
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle launcher for CMF Infinity 120M training.")
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preset", default="infinity-0.12b", choices=["infinity-0.12b", "infinity-reasoning-0.12b"])
    parser.add_argument("--init-package", type=Path, help="Optional CMF package to continue from.")
    parser.add_argument("--auto-init-package", action="store_true", help="Use the first uploaded CMF package found.")
    parser.add_argument("--token-cache", type=Path, help="Optional prebuilt token cache.")
    parser.add_argument("--dataset", default="HuggingFaceTB/smollm-corpus")
    parser.add_argument("--dataset-name", default="fineweb-edu-dedup")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("CMF_MAX_TOKENS", "2000000")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("CMF_STEPS", "1000")))
    parser.add_argument("--seq-len", type=int, default=int(os.environ.get("CMF_SEQ_LEN", "128")))
    parser.add_argument("--micro-batch-size", type=int, default=int(os.environ.get("CMF_MICRO_BATCH", "1")))
    parser.add_argument("--grad-accum", type=int, default=int(os.environ.get("CMF_GRAD_ACCUM", "16")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("CMF_LR", "0.0003")))
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--package-name", default="cmf_120m_kaggle.package.pt")
    args = parser.parse_args()

    data_dir = args.workdir / "data"
    ckpt_dir = args.workdir / "checkpoints"
    data_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    token_cache = args.token_cache or discover_token_cache()
    if token_cache is None:
        token_cache = data_dir / f"{args.dataset.replace('/', '_')}_{args.tokenizer_name}_{args.max_tokens}.pt"
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "prepare_hf_token_cache.py"),
                "--dataset",
                args.dataset,
                "--dataset-name",
                args.dataset_name,
                "--split",
                args.split,
                "--text-column",
                args.text_column,
                "--tokenizer-name",
                args.tokenizer_name,
                "--max-tokens",
                str(args.max_tokens),
                "--output",
                str(token_cache),
            ]
        )
    else:
        print(f"Using token cache: {token_cache}", flush=True)

    init_package = args.init_package
    if init_package is None and args.auto_init_package:
        init_package = discover_package()
    if init_package:
        print(f"Continuing from package: {init_package}", flush=True)

    train_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_large_scale.py"),
        "--device",
        args.device,
        "--preset",
        args.preset,
        "--token-cache",
        str(token_cache),
        "--seq-len",
        str(args.seq_len),
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--steps",
        str(args.steps),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--checkpoint",
        str(ckpt_dir / "cmf_120m_train.pt"),
        "--package-out",
        str(ckpt_dir / args.package_name),
        "--save-every",
        str(args.save_every),
        "--log-every",
        str(args.log_every),
        "--tf32",
        "--fused-adamw",
    ]
    if not args.no_amp:
        train_cmd.append("--amp")
    if args.compile:
        train_cmd.append("--compile")
    if init_package:
        train_cmd.extend(["--init-package", str(init_package)])

    run(train_cmd)
    print(f"Saved package: {ckpt_dir / args.package_name}", flush=True)


if __name__ == "__main__":
    main()
