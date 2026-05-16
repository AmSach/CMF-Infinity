from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.fast_integrator import euler_integrate_precomputed


def find_command(command: str) -> str | None:
    found = shutil.which(command)
    if found:
        return found
    candidate = Path(sys.executable).resolve().parent / f"{command}.exe"
    if candidate.exists():
        return str(candidate)
    return None


def main() -> None:
    records = ROOT / "records"
    records.mkdir(parents=True, exist_ok=True)
    extension_spec = importlib.util.find_spec("cmf_cuda")
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "extension_available": extension_spec is not None,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "toolchain": {
            "cl": find_command("cl"),
            "nvcc": find_command("nvcc"),
            "ninja": find_command("ninja"),
        },
        "checks": {},
    }

    z0 = torch.randn(2, 8)
    velocity = torch.randn(2, 4, 8)
    expected = euler_integrate_precomputed(z0, velocity, dt=0.25, use_extension=False)
    result["checks"]["python_reference_finite"] = bool(torch.isfinite(expected).all().item())

    if extension_spec is not None:
        actual = euler_integrate_precomputed(z0, velocity, dt=0.25, use_extension=True)
        result["checks"]["cpu_extension_max_abs_error"] = float((actual - expected).abs().max())
        if torch.cuda.is_available():
            z0_cuda = z0.cuda()
            velocity_cuda = velocity.cuda()
            expected_cuda = euler_integrate_precomputed(z0_cuda, velocity_cuda, dt=0.25, use_extension=False)
            actual_cuda = euler_integrate_precomputed(z0_cuda, velocity_cuda, dt=0.25, use_extension=True)
            torch.cuda.synchronize()
            result["checks"]["cuda_extension_max_abs_error"] = float(
                (actual_cuda - expected_cuda).abs().max().cpu()
            )
    else:
        result["build_note"] = (
            "cmf_cuda is not importable. On this Windows host, building requires "
            "Microsoft Visual C++ Build Tools and, for CUDA kernels, a matching CUDA toolkit."
        )

    json_path = records / "cpp_extension_status.json"
    md_path = records / "cpp_extension_status.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text("# C++/CUDA Extension Status\n\n```json\n" + json.dumps(result, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
