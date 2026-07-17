"""Smoke-test the UniPercept deployment used by MICE-Bench creation Q3."""

import argparse
import sys
from pathlib import Path


UNIPERCEPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = UNIPERCEPT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.creation.Q3_evaluate import LocalUniPerceptInferencer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=PROJECT_ROOT / "docs" / "assets" / "MICE-teaser.png",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=UNIPERCEPT_ROOT / "ckpt" / "UniPercept",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-flash-attn", action="store_true")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"test image not found: {args.image}")

    inferencer = LocalUniPerceptInferencer(
        unipercept_root=UNIPERCEPT_ROOT,
        model_path=args.model_path,
        device=args.device,
        use_flash_attn=not args.no_flash_attn,
    )
    scores = inferencer.reward(str(args.image))
    print(f"Image: {args.image}")
    print(f"IAA (aesthetics):       {scores['iaa']:.4f}")
    print(f"IQA (quality):          {scores['iqa']:.4f}")
    print(f"ISTA (structure/texture): {scores['ista']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
