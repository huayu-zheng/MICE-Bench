import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "create.json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "create"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "evaluation_results"
DEFAULT_UNIPERCEPT_ROOT = PROJECT_ROOT / "unipercept"
DEFAULT_UNIPERCEPT_MODEL_PATH = DEFAULT_UNIPERCEPT_ROOT / "ckpt" / "UniPercept"


def validate_creation_schema(input_file: Path) -> None:
    """Validate the flattened PF_prompt/CC_prompt creation metadata contract."""
    try:
        records = json.loads(input_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read creation metadata: {error}") from error
    if not isinstance(records, list):
        raise ValueError("creation metadata must be a top-level JSON list")
    invalid_pf = [
        item.get("case_id", f"index_{index}")
        for index, item in enumerate(records)
        if not isinstance(item, dict) or not isinstance(item.get("PF_prompt"), str)
    ]
    invalid_cc = [
        item.get("case_id", f"index_{index}")
        for index, item in enumerate(records)
        if not isinstance(item, dict) or not isinstance(item.get("CC_prompt"), dict)
    ]
    if invalid_pf:
        raise ValueError(f"missing or invalid PF_prompt: {invalid_pf[:10]}")
    if invalid_cc:
        raise ValueError(f"missing or invalid CC_prompt: {invalid_cc[:10]}")


def run_command(
    name: str, command: List[str], environment: dict[str, str], dry_run: bool
) -> None:
    print(f"\n{'=' * 72}")
    print(f"Running {name}")
    print(" ".join(command))
    print("=" * 72)
    if dry_run:
        return
    subprocess.run(command, check=True, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PF, CC, IA, and PR creation evaluations in one command."
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
    parser.add_argument(
        "--unipercept-root", type=Path, default=DEFAULT_UNIPERCEPT_ROOT
    )
    parser.add_argument(
        "--unipercept-model-path", type=Path, default=DEFAULT_UNIPERCEPT_MODEL_PATH
    )
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-pf", action="store_true")
    parser.add_argument("--skip-cc", action="store_true")
    parser.add_argument("--skip-ia", action="store_true")
    parser.add_argument("--skip-pr", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input JSON not found: {args.input}")
    try:
        validate_creation_schema(args.input)
    except ValueError as error:
        parser.error(str(error))
    if not args.dataset_root.is_dir():
        parser.error(f"dataset root not found: {args.dataset_root}")
    if not args.skip_ia and not args.unipercept_root.is_dir():
        parser.error(f"UniPercept source not found: {args.unipercept_root}")
    if not args.dry_run and not args.skip_ia and not args.unipercept_model_path.exists():
        parser.error(f"UniPercept model path not found: {args.unipercept_model_path}")
    if (
        not args.dry_run
        and not (args.skip_pf and args.skip_cc and args.skip_pr)
        and not args.api_key
    ):
        parser.error("provide --api-key or set OPENAI_API_KEY")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    environment = os.environ.copy()
    if args.api_key:
        environment["OPENAI_API_KEY"] = args.api_key
    environment["OPENAI_BASE_URL"] = args.base_url

    commands = []
    if not args.skip_pf:
        commands.append(
            (
                "PF prompt following (PF_prompt)",
                [
                    python,
                    str(SCRIPT_DIR / "PF_evaluate.py"),
                    "--input",
                    str(args.input),
                    "--dataset-root",
                    str(args.dataset_root),
                    "--output-dir",
                    str(args.output_root / "PF"),
                    "--model",
                    args.vlm_model,
                    "--workers",
                    str(args.workers),
                ],
            )
        )
    if not args.skip_cc:
        commands.append(
            (
                "CC concept consistency (CC_prompt)",
                [
                    python,
                    str(SCRIPT_DIR / "CC_evaluate.py"),
                    "--metadata-file",
                    str(args.input),
                    "--dataset-root",
                    str(args.dataset_root),
                    "--output-dir",
                    str(args.output_root / "CC"),
                    "--vlm-model",
                    args.vlm_model,
                    "--num-workers",
                    str(args.workers),
                    "--end",
                    "0",
                ],
            )
        )
    if not args.skip_ia:
        commands.append(
            (
                "IA UniPercept aesthetics/quality/structure",
                [
                    python,
                    str(SCRIPT_DIR / "IA_evaluate.py"),
                    "--input",
                    str(args.input),
                    "--dataset-root",
                    str(args.dataset_root),
                    "--output-dir",
                    str(args.output_root / "IA"),
                    "--unipercept-root",
                    str(args.unipercept_root),
                    "--model-path",
                    str(args.unipercept_model_path),
                    "--device",
                    args.device,
                ],
            )
        )
        if args.no_flash_attn:
            commands[-1][1].append("--no-flash-attn")
    if not args.skip_pr:
        commands.append(
            (
                "PR physical realism",
                [
                    python,
                    str(SCRIPT_DIR / "PR_evaluate.py"),
                    "--input",
                    str(args.input),
                    "--dataset-root",
                    str(args.dataset_root),
                    "--output-dir",
                    str(args.output_root / "PR"),
                    "--model",
                    args.vlm_model,
                    "--workers",
                    str(args.workers),
                ],
            )
        )

    if not commands:
        parser.error("all creation evaluations are skipped")

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
