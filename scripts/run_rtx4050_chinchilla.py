from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.presets import estimate_cmf_parameters, get_preset


DEFAULT_DATASET = "HuggingFaceTB/smollm-corpus"
DEFAULT_DATASET_NAME = "fineweb-edu-dedup"
DEFAULT_INIT_PACKAGE = ROOT / "records" / "checkpoints" / "cmf_120m_gate.package.pt"


class GpuMonitor:
    def __init__(self, path: Path, interval_sec: float) -> None:
        self.path = path
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.interval_sec <= 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "timestamp,name,memory_total_mb,memory_used_mb,temperature_c,power_w\n",
            encoding="utf-8",
        )
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def _loop(self) -> None:
        query = "name,memory.total,memory.used,temperature.gpu,power.draw"
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--query-gpu={query}",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
                if line:
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{line}\n")
            except Exception as exc:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},monitor_error,{exc}\n")
            self.stop_event.wait(self.interval_sec)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_logged(cmd: list[str], *, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n# started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("+ " + " ".join(cmd) + "\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
        log.write(f"# exited {code} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)


def command_text(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def run_environment_snapshot(run_dir: Path, env: dict[str, str]) -> None:
    commands = {
        "python": [sys.executable, "--version"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
        "nvidia_smi": ["nvidia-smi"],
    }
    for name, cmd in commands.items():
        path = run_dir / f"{name}.txt"
        try:
            result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
            path.write_text(result.stdout + result.stderr, encoding="utf-8")
        except Exception as exc:
            path.write_text(str(exc), encoding="utf-8")


def configure_local_caches(env: dict[str, str]) -> None:
    cache_root = ROOT / "records" / "runtime_cache"
    hf_home = cache_root / "huggingface"
    temp_dir = cache_root / "tmp"
    torch_home = cache_root / "torch"
    for path in [hf_home, hf_home / "datasets", hf_home / "hub", temp_dir, torch_home]:
        path.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(hf_home)
    env["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    env["TRANSFORMERS_CACHE"] = str(hf_home / "hub")
    env["TORCH_HOME"] = str(torch_home)
    env["TMP"] = str(temp_dir)
    env["TEMP"] = str(temp_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="RTX 4050 local Chinchilla-scaling trainer for CMF 120M.")
    parser.add_argument("--preset", default="infinity-0.12b", choices=["infinity-0.12b", "infinity-reasoning-0.12b"])
    parser.add_argument("--tokens-per-param", type=float, default=20.0)
    parser.add_argument("--target-tokens", type=int, help="Override Chinchilla target tokens.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "records" / "data" / "chinchilla_gpt2_120m")
    parser.add_argument("--shard-tokens", type=int, default=25_000_000)
    parser.add_argument("--token-dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--steps", type=int, help="Override optimizer steps. Default covers target tokens once.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-package", type=Path)
    parser.add_argument("--no-init-package", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, help="Resume optimizer/model state from an existing training checkpoint.")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--phase", choices=["all", "download", "train"], default="all")
    parser.add_argument("--resume-download", action="store_true", default=True)
    parser.add_argument("--overwrite-download", action="store_true")
    parser.add_argument("--resume-training", action="store_true", default=True)
    parser.add_argument("--random-batches", action="store_true", help="Sample cache windows randomly instead of sequentially.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--gpu-monitor-interval", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preset = get_preset(args.preset)
    param_count = estimate_cmf_parameters(preset.config, model_type=preset.model_type)
    target_tokens = args.target_tokens or int(param_count * args.tokens_per_param)
    tokens_per_step = args.seq_len * args.micro_batch_size * args.grad_accum
    steps = args.steps or math.ceil(target_tokens / tokens_per_step)
    train_tokens = steps * tokens_per_step

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (ROOT / "records" / "runs" / f"rtx4050_chinchilla_{timestamp}")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "cmf_120m_train.pt"
    package_path = checkpoint_dir / "cmf_120m_chinchilla.package.pt"
    init_package = args.init_package
    if init_package is None and not args.no_init_package and DEFAULT_INIT_PACKAGE.exists():
        init_package = DEFAULT_INIT_PACKAGE

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    configure_local_caches(env)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "preset": preset.to_dict(),
        "parameter_count": param_count,
        "tokens_per_param": args.tokens_per_param,
        "chinchilla_target_tokens": target_tokens,
        "seq_len": args.seq_len,
        "micro_batch_size": args.micro_batch_size,
        "grad_accum": args.grad_accum,
        "tokens_per_optimizer_step": tokens_per_step,
        "optimizer_steps": steps,
        "planned_train_tokens": train_tokens,
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "data_dir": str(args.data_dir),
        "shard_tokens": args.shard_tokens,
        "token_dtype": args.token_dtype,
        "init_package": str(init_package) if init_package else None,
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "checkpoint": str(checkpoint_path),
        "package_out": str(package_path),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "plan.json", plan)
    run_environment_snapshot(run_dir, env)

    download_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_hf_token_shards.py"),
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
        "--target-tokens",
        str(target_tokens),
        "--shard-tokens",
        str(args.shard_tokens),
        "--output-dir",
        str(args.data_dir),
        "--dtype",
        args.token_dtype,
    ]
    if args.resume_download:
        download_cmd.append("--resume")
    if args.overwrite_download:
        download_cmd.append("--overwrite")

    train_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_large_scale.py"),
        "--device",
        args.device,
        "--preset",
        args.preset,
        "--token-cache-dir",
        str(args.data_dir),
        "--seq-len",
        str(args.seq_len),
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--steps",
        str(steps),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--checkpoint",
        str(checkpoint_path),
        "--package-out",
        str(package_path),
        "--save-every",
        str(args.save_every),
        "--log-every",
        str(args.log_every),
        "--tf32",
        "--fused-adamw",
        "--cache-batches-per-shard",
        "1024",
    ]
    if not args.no_amp:
        train_cmd.append("--amp")
    if not args.random_batches:
        train_cmd.append("--sequential-cache-batches")
    if args.compile:
        train_cmd.append("--compile")
    if init_package:
        train_cmd.extend(["--init-package", str(init_package)])
    if args.resume_checkpoint:
        train_cmd.extend(["--resume", str(args.resume_checkpoint)])
    elif args.resume_training and checkpoint_path.exists():
        train_cmd.extend(["--resume", str(checkpoint_path)])

    commands = {
        "download": command_text(download_cmd),
        "train": command_text(train_cmd),
    }
    write_json(run_dir / "commands.json", commands)
    (run_dir / "commands.ps1").write_text(
        "\n".join([f"# {name}\n{cmd}" for name, cmd in commands.items()]) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(plan, indent=2, sort_keys=True))
    print(f"Run directory: {run_dir}")
    if args.dry_run:
        print("Dry run only; no download or training started.")
        return

    monitor = GpuMonitor(run_dir / "gpu.csv", args.gpu_monitor_interval)
    monitor.start()
    try:
        if args.phase in {"all", "download"}:
            run_logged(download_cmd, log_path=run_dir / "download.log", env=env)
        if args.phase in {"all", "train"}:
            run_logged(train_cmd, log_path=run_dir / "train.log", env=env)
    finally:
        monitor.stop()

    write_json(
        run_dir / "result.json",
        {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "package_out": str(package_path),
            "checkpoint": str(checkpoint_path),
            "run_dir": str(run_dir),
        },
    )
    print(f"Done. Package: {package_path}")
    print(f"Logs: {run_dir}")


if __name__ == "__main__":
    main()
