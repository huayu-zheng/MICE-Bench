"""Validate MICE-Bench metadata and referenced image paths."""

import argparse
import json
from collections import Counter
from pathlib import Path


def validate(metadata: Path, root: Path) -> tuple[int, list[str]]:
    records = json.loads(metadata.read_text(encoding="utf-8"))
    errors: list[str] = []
    ids = [item.get("case_id") for item in records]
    duplicate_ids = [key for key, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        errors.append(f"duplicate case_id values: {duplicate_ids[:10]}")
    for item in records:
        case_id = item.get("case_id", "<missing>")
        if not item.get("prompts", {}).get("en"):
            errors.append(f"{case_id}: missing prompts.en")
        for source in item.get("sources", []):
            path = root / source.get("path", "")
            if not path.is_file():
                errors.append(f"{case_id}: missing {path}")
    return len(records), errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    jobs = [
        (args.data_root / "create.json", args.data_root / "create"),
        (args.data_root / "edit.json", args.data_root),
    ]
    total = 0
    all_errors: list[str] = []
    for metadata, root in jobs:
        if not metadata.is_file():
            all_errors.append(f"missing metadata: {metadata}")
            continue
        count, errors = validate(metadata, root)
        total += count
        all_errors.extend(errors)
        print(f"{metadata}: {count} cases, {len(errors)} errors")
    if all_errors:
        print("\n".join(all_errors[:100]))
        print(f"Validation failed with {len(all_errors)} error(s).")
        return 1
    print(f"Validation passed: {total} cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

