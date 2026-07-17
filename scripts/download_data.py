"""Download the MICE-Bench dataset snapshot from Hugging Face."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="MICE-Bench/MICE-Bench")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output,
    )
    print(f"Dataset downloaded to {args.output.resolve()}")


if __name__ == "__main__":
    main()

