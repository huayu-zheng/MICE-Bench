import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "edit.json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "evaluation_results"


def run_command(
    name: str, command: List[str], environment: dict[str, str], dry_run: bool
) -> None:
    print(f"\n{'=' * 72}")
    print(f"Running {name}")
    print(" ".join(command))
    print("=" * 72)
    if not dry_run:
        subprocess.run(command, check=True, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run edit Q1, Q2, Q3, and Q4 evaluations in one command."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument(
        "--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument("--vlm-model", default="gemini-3-pro-preview")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-q1", action="store_true")
    parser.add_argument("--skip-q2", action="store_true")
    parser.add_argument("--skip-q4", action="store_true")
    parser.add_argument("--skip-q3", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input JSON not found: {args.input}")
    if not args.dataset_root.is_dir():
        parser.error(f"dataset root not found: {args.dataset_root}")
    if not args.dry_run and not args.api_key:
        parser.error("provide --api-key or set OPENAI_API_KEY")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.skip_q1 and args.skip_q2 and args.skip_q3 and args.skip_q4:
        parser.error("all evaluations are skipped")

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if args.api_key:
        environment["OPENAI_API_KEY"] = args.api_key
    environment["OPENAI_BASE_URL"] = args.base_url
    python = sys.executable

    commands = []
    if not args.skip_q1:
        commands.append(
            (
                "Q1 prompt-image alignment",
                [
                    python,
                    str(SCRIPT_DIR / "Q1_evaluate.py"),
                    "--input",
                    str(args.input),
                    "--dataset-root",
                    str(args.dataset_root),
                    "--output-dir",
                    str(args.output_root / "Q1"),
                    "--model",
                    args.vlm_model,
                    "--workers",
                    str(args.workers),
                ],
            )
        )
    if not args.skip_q2:
        commands.append(
            (
                "Q2 concept preservation",
                [
                    python,
                    str(SCRIPT_DIR / "Q2_evaluate.py"),
                    "--metadata-file",
                    str(args.input),
                    "--dataset-root",
                    str(args.dataset_root),
                    "--output-dir",
                    str(args.output_root / "Q2"),
                    "--vlm-model",
                    args.vlm_model,
                    "--num-workers",
                    str(args.workers),
                    "--end",
                    "0",
                ],
            )
        )
    if not args.skip_q3:
        commands.append(
            (
                "Q3 non-edited region preservation",
                [
                    python,
                    str(SCRIPT_DIR / "Q3_evaluate.py"),
                    "--input",
                    str(args.input),
                    "--dataset-root",
                    str(args.dataset_root),
                    "--output-dir",
                    str(args.output_root / "Q3"),
                    "--model",
                    args.vlm_model,
                    "--workers",
                    str(args.workers),
                ],
            )
        )
    if not args.skip_q4:
        commands.append(
            (
                "Q4 physical realism",
                [
                    python,
                    str(SCRIPT_DIR / "Q4_evaluate.py"),
                    "--input",
                    str(args.input),
                    "--dataset-root",
                    str(args.dataset_root),
                    "--output-dir",
                    str(args.output_root / "Q4"),
                    "--model",
                    args.vlm_model,
                    "--workers",
                    str(args.workers),
                ],
            )
        )

    try:
        for name, command in commands:
            run_command(name, command, environment, args.dry_run)
    except subprocess.CalledProcessError as error:
        print(f"\nEvaluation failed with exit code {error.returncode}: {error.cmd}")
        return error.returncode

    print("\nAll requested evaluations completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
