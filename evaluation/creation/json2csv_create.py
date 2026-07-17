import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_FILE = PROJECT_ROOT / "data" / "create.json"
DEFAULT_EVALUATION_ROOT = Path(__file__).resolve().parent / "evaluation_results"
DEFAULT_CSV_DIR = Path(__file__).resolve().parent / "benchmark_results"

SCORE_COLUMNS = {
    "q1": ("PF",),
    "q2": ("CC",),
    "q3": ("IA", "IQ", "IS"),
    "q4": ("PR",),
}

HEADER_CANDIDATES = {
    "count": ("Number of Images", "Image Count", "Num Images", "Images"),
    "combination": ("Combination", "combination"),
    "subclass": (
        "Sub-class Comb",
        "Sub-class Combination",
        "Sub-class",
        "sub_class_comb",
    ),
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


def normalize_count(value: Any) -> str:
    try:
        return str(int(float(str(value).strip())))
    except (TypeError, ValueError):
        return str(value or "").strip()


def find_header(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {field.strip().lower(): field for field in fieldnames}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    return None


def grouping_headers(fieldnames: Sequence[str]) -> Dict[str, str]:
    headers = {
        name: header
        for name, candidates in HEADER_CANDIDATES.items()
        if (header := find_header(fieldnames, candidates)) is not None
    }
    if "combination" not in headers:
        expected = ", ".join(HEADER_CANDIDATES["combination"])
        raise ValueError(f"CSV must contain a combination column: {expected}")
    return headers


def record_key(record: Dict[str, Any], headers: Dict[str, str]) -> Tuple[str, ...]:
    values = []
    if "count" in headers:
        sources = record.get("sources")
        values.append(normalize_count(len(sources) if isinstance(sources, list) else ""))
    values.append(str(record.get("combination") or "").strip())
    if "subclass" in headers:
        values.append(str(record.get("sub_class_comb") or "").strip())
    return tuple(values)


def row_key(row: Dict[str, str], headers: Dict[str, str]) -> Tuple[str, ...]:
    values = []
    if "count" in headers:
        values.append(normalize_count(row.get(headers["count"])))
    values.append(str(row.get(headers["combination"]) or "").strip())
    if "subclass" in headers:
        values.append(str(row.get(headers["subclass"]) or "").strip())
    return tuple(values)


def source_record(
    record: Dict[str, Any], metadata: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    case_id = record.get("case_id")
    if isinstance(case_id, str) and case_id in metadata:
        return metadata[case_id]
    return record


def average_by_key(
    values: Iterable[Tuple[Tuple[str, ...], float]]
) -> Dict[Tuple[str, ...], float]:
    grouped: Dict[Tuple[str, ...], List[float]] = {}
    for key, score in values:
        grouped.setdefault(key, []).append(score)
    return {key: sum(scores) / len(scores) for key, scores in grouped.items() if scores}


def model_score(record: Dict[str, Any], model_name: str) -> Optional[float]:
    evaluations = record.get("evaluations")
    if not isinstance(evaluations, dict):
        return None
    evaluation = evaluations.get(model_name)
    if not isinstance(evaluation, dict):
        return None
    return to_float(evaluation.get("score"))


def scalar_scores(
    records: List[Dict[str, Any]],
    model_name: str,
    metadata: Dict[str, Dict[str, Any]],
    headers: Dict[str, str],
) -> Dict[Tuple[str, ...], float]:
    values = []
    for record in records:
        score = model_score(record, model_name)
        if score is not None:
            values.append((record_key(source_record(record, metadata), headers), score))
    return average_by_key(values)


def q2_scores(
    records: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, Any]],
    headers: Dict[str, str],
) -> Dict[Tuple[str, ...], float]:
    values = []
    for record in records:
        evaluations = record.get("evaluations")
        if not isinstance(evaluations, dict):
            continue
        answers = [
            str(value.get("answer") or "").strip().lower()
            for value in evaluations.values()
            if isinstance(value, dict)
        ]
        valid = [answer for answer in answers if answer in {"yes", "no"}]
        if valid:
            score = sum(answer == "yes" for answer in valid) / len(valid) * 10.0
            values.append((record_key(source_record(record, metadata), headers), score))
    return average_by_key(values)


def q3_scores(
    records: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, Any]],
    headers: Dict[str, str],
) -> Dict[str, Dict[Tuple[str, ...], float]]:
    metrics = {"IA": "iaa", "IQ": "iqa", "IS": "ista"}
    result = {}
    for column, field in metrics.items():
        values = []
        for record in records:
            score = to_float(record.get(field))
            if score is not None:
                values.append((record_key(source_record(record, metadata), headers), score))
        result[column] = average_by_key(values)
    return result


def evaluation_path(root: Path, score_name: str, model_name: str) -> Path:
    if score_name in {"q1", "q4"}:
        filename = f"processed_metadata_with_verifications.match_{model_name}.json"
    else:
        filename = f"{model_name}_evaluated.json" if score_name == "q2" else f"{model_name}.json"
    return root / score_name.upper() / filename


def csv_path(csv_dir: Path, model_name: str) -> Path:
    return csv_dir / f"MICE_Create_Result_{model_name}.csv"


def initialize_csv(path: Path, metadata_records: List[Dict[str, Any]]) -> None:
    """Create a deterministic creation result template from metadata."""
    fieldnames = ["Number of Images", "Combination", "Sub-class Comb"]
    unique_rows = {
        (
            len(record.get("sources", []))
            if isinstance(record.get("sources"), list)
            else 0,
            str(record.get("combination") or "").strip(),
            str(record.get("sub_class_comb") or "").strip(),
        )
        for record in metadata_records
    }
    rows = [
        {
            "Number of Images": count,
            "Combination": combination,
            "Sub-class Comb": subclass,
        }
        for count, combination, subclass in sorted(
            unique_rows, key=lambda item: (item[0], item[1], item[2])
        )
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(path, rows, fieldnames)


def apply_columns(
    rows: List[Dict[str, str]],
    fieldnames: List[str],
    headers: Dict[str, str],
    averages: Dict[str, Dict[Tuple[str, ...], float]],
) -> None:
    for column in averages:
        if column not in fieldnames:
            fieldnames.append(column)
    for row in rows:
        key = row_key(row, headers)
        for column, values in averages.items():
            score = values.get(key)
            if score is not None:
                precision = 4 if column in {"IA", "IQ", "IS"} else 2
                row[column] = f"{score:.{precision}f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate create Q1-Q4 evaluation JSON files into per-model CSV files."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_FILE)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
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
        parser.error("no model keys found in create.json")

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
        headers = grouping_headers(fieldnames)

        applied = []
        for score_name in args.scores:
            source_json = evaluation_path(args.evaluation_root, score_name, model_name)
            if not source_json.is_file():
                print(
                    f"Skipping {score_name.upper()} for {model_name}; "
                    f"JSON not found: {source_json}"
                )
                continue
            records = load_json_list(source_json)
            if score_name == "q2":
                column_values = {"CC": q2_scores(records, metadata, headers)}
            elif score_name == "q3":
                column_values = q3_scores(records, metadata, headers)
            else:
                column = SCORE_COLUMNS[score_name][0]
                column_values = {
                    column: scalar_scores(records, model_name, metadata, headers)
                }
            apply_columns(rows, fieldnames, headers, column_values)
            applied.append(score_name.upper())

        if applied:
            save_csv(target_csv, rows, fieldnames)
            updated += 1
            print(f"Updated {target_csv} with {', '.join(applied)}")

    print(f"Updated CSV files: {updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
