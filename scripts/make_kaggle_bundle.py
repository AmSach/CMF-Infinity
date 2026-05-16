from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / "cmf_kaggle_bundle.zip"
INCLUDE_DIRS = ["cmf", "scripts", "tests", "docs", "paper", "kaggle"]
INCLUDE_FILES = ["README.md", "HANDOFF.md", "pyproject.toml", "setup.py"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".pytest_tmp", ".venv", "build", "dist", "downloads"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".obj", ".exp", ".lib"}


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_DIRS:
        return True
    if path.suffix in SKIP_SUFFIXES:
        return True
    return False


def add_path(zf: zipfile.ZipFile, path: Path, arcname: Path) -> None:
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file() and not should_skip(child.relative_to(ROOT)):
                zf.write(child, child.relative_to(ROOT))
    elif path.exists() and not should_skip(path.relative_to(ROOT)):
        zf.write(path, arcname)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Kaggle-ready CMF source bundle.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-package", type=Path, help="Optional .package.pt to include under pretrained/.")
    parser.add_argument("--include-token-cache", type=Path, help="Optional token cache to include under data/.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for directory in INCLUDE_DIRS:
            add_path(zf, ROOT / directory, Path(directory))
        for filename in INCLUDE_FILES:
            add_path(zf, ROOT / filename, Path(filename))
        if args.include_package:
            zf.write(args.include_package, Path("pretrained") / args.include_package.name)
        if args.include_token_cache:
            zf.write(args.include_token_cache, Path("data") / args.include_token_cache.name)

    size_mb = args.output.stat().st_size / 1024**2
    print(f"Wrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
