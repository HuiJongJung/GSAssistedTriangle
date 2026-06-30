from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = (
    "A_baseline_triangle_only",
    "B_ours_mixed_triangle_gs",
    "C_ours_converted_triangle_only",
)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect(experiment_root: Path) -> dict:
    rows = []
    for variant_dir in sorted(p for p in experiment_root.iterdir() if p.is_dir()):
        if variant_dir.name not in VARIANTS:
            continue
        summary = load_json(variant_dir / "summary.json")
        results = load_json(variant_dir / "results.json")
        rows.append(
            {
                "variant": variant_dir.name,
                "summary": summary,
                "results": results,
                "path": str(variant_dir),
            }
        )
    return {"experiment_root": str(experiment_root), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = collect(args.experiment_root)
    text = json.dumps(data, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
