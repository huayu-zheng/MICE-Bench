import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_FILE = PROJECT_ROOT / "data" / "edit.json"
DEFAULT_EVALUATION_ROOT = Path(__file__).resolve().parent / "evaluation_results"
DEFAULT_CSV_DIR = Path(__file__).resolve().parent / "benchmark_results"

SCORE_COLUMNS = {
    "q1": "IF",
    "q2": "CC",
    "q3": "NERC",
    "q4": "PR",
}


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read JSON list {path}: {error}") from error
    if not isinstance(data, list):
        raise ValueError(f"JSON must contain a top-level list: {path}")
    return [item for item in data if isinstance(item, dict)]


def load_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def save_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def average_by_key(
    values: Iterable[Tuple[Tuple[str, str], float]]
) -> Dict[Tuple[str, str], float]:
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for key, score in values:
        grouped.setdefault(key, []).append(score)
    return {
        key: sum(scores) / len(scores)
        for key, scores in grouped.items()
        if scores
    }


def metadata_key(
    record: Dict[str, Any], metadata: Dict[str, Dict[str, Any]]
) -> Tuple[str, str]:
    case_id = record.get("case_id")
    source = metadata.get(case_id, record) if isinstance(case_id, str) else record
    return (
        str(source.get("task_type") or "").strip(),
        str(source.get("task_subtype") or "").strip(),
    )


def model_evaluation(record: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    evaluations = record.get("evaluations")
    if not isinstance(evaluations, dict):
        return {}
    evaluation = evaluations.get(model_name)
    return evaluation if isinstance(evaluation, dict) else {}


def q1_scores(
    records: List[Dict[str, Any]],
    model_name: str,
    metadata: Dict[str, Dict[str, Any]],
) -> Dict[Tuple[str, str], float]:
    values = []
    for record in records:
        score = to_float(model_evaluation(record, model_name).get("score"))
        if score is not None:
            values.append((metadata_key(record, metadata), score))
    return average_by_key(values)


def q2_scores(
    records: List[Dict[str, Any]], metadata: Dict[str, Dict[str, Any]]
) -> Dict[Tuple[str, str], float]:
    values = []
    for record in records:
        evaluations = record.get("evaluations")
        if not isinstance(evaluations, dict):
            continue
        answers = [
            value.get("answer", "").strip().lower()
            for key, value in evaluations.items()
            if key.startswith("question_") and isinstance(value, dict)
        ]
        valid_answers = [answer for answer in answers if answer in {"yes", "no"}]
        if valid_answers:
            score = sum(answer == "yes" for answer in valid_answers) / len(valid_answers) * 10.0
            values.append((metadata_key(record, metadata), score))
    return average_by_key(values)


def q3_scores(
    records: List[Dict[str, Any]],
    model_name: str,
    metadata: Dict[str, Dict[str, Any]],
) -> Dict[Tuple[str, str], float]:
    values = []
    for record in records:
        score = to_float(
            model_evaluation(record, model_name).get("preservation_score")
        )
        if score is not None:
            task_type, _ = metadata_key(record, metadata)
            values.append(((task_type, ""), score))
    return average_by_key(values)


def q4_scores(
    records: List[Dict[str, Any]],
    model_name: str,
    metadata: Dict[str, Dict[str, Any]],
) -> Dict[Tuple[str, str], float]:
    return q1_scores(records, model_name, metadata)


def evaluation_path(root: Path, score_name: str, model_name: str) -> Path:
    if score_name == "q1":
        filename = f"processed_metadata_with_verifications.match_{model_name}.json"
    elif score_name == "q2":
        filename = f"{model_name}_evaluated.json"
    elif score_name == "q3":
        filename = f"processed_metadata_with_verifications.Q3_{model_name}.json"
    else:
        filename = f"processed_metadata_with_verifications.match_{model_name}.json"
    return root / score_name.upper() / filename


def csv_path(csv_dir: Path, model_name: str) -> Path:
    return csv_dir / f"MICE_Edit_Result_{model_name}.csv"


def initialize_csv(path: Path, metadata_records: List[Dict[str, Any]]) -> None:
    """Create a deterministic editing result template from metadata."""
    fieldnames = ["Task Type", "Sub-task"]
    unique_rows = {
        (
            str(record.get("task_type") or "").strip(),
            str(record.get("task_subtype") or "").strip(),
        )
        for record in metadata_records
    }
    rows = [
        {"Task Type": task_type, "Sub-task": task_subtype}
        for task_type, task_subtype in sorted(unique_rows)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(path, rows, fieldnames)


def apply_score(
    rows: List[Dict[str, str]],
    fieldnames: List[str],
    score_name: str,
    averages: Dict[Tuple[str, str], float],
) -> None:
    column = SCORE_COLUMNS[score_name]
    if column not in fieldnames:
        fieldnames.append(column)
    for row in rows:
        task_type = str(row.get("Task Type") or "").strip()
        task_subtype = str(row.get("Sub-task") or "").strip()
        key = (task_type, "") if score_name == "q3" else (task_type, task_subtype)
        score = averages.get(key)
        if score is not None:
            row[column] = f"{score:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate edit Q1-Q4 evaluation JSON files into per-model CSV files."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_FILE)
    parser.add_argument(
        "--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT
    )
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument(
        "--scores",
        nargs="+",
        choices=tuple(SCORE_COLUMNS),
        default=list(SCORE_COLUMNS),
    )
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    metadata_records = load_json_list(args.metadata)
    metadata = {
        record["case_id"]: record
        for record in metadata_records
        if isinstance(record.get("case_id"), str)
    }
    discovered_models = sorted(
        {
            model_name
            for record in metadata_records
            if isinstance(record.get("result"), dict)
            for model_name in record["result"]
        }
    )
    models = args.models or discovered_models
    if not models:
        parser.error("no model keys found in edit.json")

    updated = 0
    for model_name in models:
        target_csv = csv_path(args.csv_dir, model_name)
        if not target_csv.is_file():
            initialize_csv(target_csv, metadata_records)
            print(f"Created CSV template: {target_csv}")
        rows, fieldnames = load_csv(target_csv)
        if not fieldnames:
            print(f"Skipping {model_name}; CSV has no header: {target_csv}")
            continue

        applied = []
        for score_name in args.scores:
            source_json = evaluation_path(
                args.evaluation_root, score_name, model_name
            )
            if not source_json.is_file():
                print(f"Skipping {score_name.upper()} for {model_name}; JSON not found: {source_json}")
                continue
            records = load_json_list(source_json)
            if score_name == "q1":
                averages = q1_scores(records, model_name, metadata)
            elif score_name == "q2":
                averages = q2_scores(records, metadata)
            elif score_name == "q3":
                averages = q3_scores(records, model_name, metadata)
            else:
                averages = q4_scores(records, model_name, metadata)
            apply_score(rows, fieldnames, score_name, averages)
            applied.append(score_name.upper())

        if applied:
            save_csv(target_csv, rows, fieldnames)
            updated += 1
            print(f"Updated {target_csv} with {', '.join(applied)}")

    print(f"Updated CSV files: {updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
