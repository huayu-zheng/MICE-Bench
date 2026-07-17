"""Attach a model's generated images to a fresh metadata JSON."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension", default=".png")
    args = parser.parse_args()
    records = json.loads(args.metadata.read_text(encoding="utf-8"))
    missing = []
    for item in records:
        image = args.images / f"{item['case_id']}{args.extension}"
        if not image.is_file():
            missing.append(str(image))
            continue
        item.setdefault("result", {})[args.model_name] = str(image.resolve())
    if missing:
        print(f"Missing {len(missing)} outputs; first: {missing[0]}")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

