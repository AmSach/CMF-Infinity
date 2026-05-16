from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.presets import PRESETS, get_preset


def main() -> None:
    parser = argparse.ArgumentParser(description="Print CMF Infinity preset plans.")
    parser.add_argument("--preset", default="all")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.preset == "all":
        data = {name: preset.to_dict() for name, preset in PRESETS.items()}
    else:
        preset = get_preset(args.preset)
        data = {preset.name: preset.to_dict()}
    text = json.dumps(data, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
