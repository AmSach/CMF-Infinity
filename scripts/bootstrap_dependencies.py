from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PYTHON_PACKAGES = {
    "torch": "torch",
    "pytest": "pytest",
    "transformers": "transformers",
    "datasets": "datasets",
    "nvidia-ml-py": "pynvml",
    "numpy": "numpy",
    "pillow": "PIL",
    "ninja": "ninja",
}

NATIVE_TOOLS = {
    "msvc_cl": "cl",
    "cuda_nvcc": "nvcc",
    "ninja_exe": "ninja",
}


def module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def install_python_packages(packages: list[str]) -> None:
    if not packages:
        return
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    subprocess.run(cmd, cwd=ROOT, check=True)


def check() -> dict:
    scripts_dir = Path(sys.executable).resolve().parent

    def find_command(command: str) -> str | None:
        found = shutil.which(command)
        if found:
            return found
        candidate = scripts_dir / (f"{command}.exe" if not command.endswith(".exe") else command)
        if candidate.exists():
            return str(candidate)
        return None

    python = {
        package: {
            "module": module,
            "available": module_available(module),
        }
        for package, module in PYTHON_PACKAGES.items()
    }
    native = {
        label: {
            "command": command,
            "path": find_command(command),
            "available": find_command(command) is not None,
        }
        for label, command in NATIVE_TOOLS.items()
    }
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python_executable": sys.executable,
        "python": python,
        "native": native,
        "native_install_notes": {
            "msvc_cl": "Install Microsoft Visual Studio Build Tools with the C++ workload.",
            "cuda_nvcc": "Install an NVIDIA CUDA Toolkit version compatible with the active PyTorch CUDA build.",
            "ninja_exe": "Installed by the Python package 'ninja' for this venv, but a system executable may still require PATH refresh.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check or install CMF Infinity dependencies.")
    parser.add_argument("--install-python", action="store_true")
    parser.add_argument("--json-out", type=Path, default=ROOT / "records" / "dependency_status.json")
    args = parser.parse_args()

    before = check()
    missing_python = [
        package
        for package, info in before["python"].items()
        if not info["available"] and package != "torch"
    ]
    if args.install_python and missing_python:
        install_python_packages(missing_python)
    after = check()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(after, indent=2, sort_keys=True))
    missing_native = [name for name, info in after["native"].items() if not info["available"]]
    if missing_native:
        print("\nMissing native tools:", ", ".join(missing_native))
        print("These are system-level dependencies; install them before building cmf_cuda.")


if __name__ == "__main__":
    main()
