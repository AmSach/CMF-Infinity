from __future__ import annotations

import os
import shutil
from pathlib import Path

from setuptools import setup


def build_extensions():
    scripts_dir = Path(__import__("sys").executable).resolve().parent
    if (scripts_dir / "ninja.exe").exists():
        os.environ["PATH"] = f"{scripts_dir}{os.pathsep}" + os.environ.get("PATH", "")
    try:
        import torch
        from torch.utils.cpp_extension import (
            BuildExtension,
            CppExtension,
            CUDAExtension,
            CUDA_HOME,
        )
    except Exception:
        return [], {}

    define_macros = []
    sources = ["cpp/cmf_extension.cpp"]
    extension_cls = CppExtension

    nvcc_available = shutil.which("nvcc") is not None
    if (
        CUDA_HOME is not None
        and nvcc_available
        and Path("cpp/cmf_cuda_kernel.cu").exists()
        and torch.cuda.is_available()
    ):
        define_macros.append(("WITH_CUDA", None))
        sources.append("cpp/cmf_cuda_kernel.cu")
        extension_cls = CUDAExtension

    cxx_flags = ["/O2"] if os.name == "nt" else ["-O3"]
    extra_compile_args = {"cxx": cxx_flags}
    if extension_cls is CUDAExtension:
        extra_compile_args["nvcc"] = ["-O3"]

    ext_modules = [
        extension_cls(
            name="cmf_cuda",
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules, {"build_ext": BuildExtension.with_options(use_ninja=False)}


ext_modules, cmdclass = build_extensions()

setup(
    name="continuous-meaning-field",
    version="0.1.0",
    packages=["cmf"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
