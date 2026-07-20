"""Offline contract tests for MICE-Bench evaluators and CSV aggregators.

No model checkpoint, GPU, network request, or API key is required.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.creation import PF_evaluate as create_pf
from evaluation.creation import CC_evaluate as create_cc
from evaluation.creation import PR_evaluate as create_pr
from evaluation.editing import IF_evaluate as edit_if
from evaluation.editing import NERC_evaluate as edit_nerc
from evaluation.editing import PR_evaluate as edit_pr

def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return next(csv.DictReader(file))


class FakePromptFollowingScorer:
    def __init__(self, result_type: type, score: int = 82):
        self.result_type = result_type
        self.score = score

    def evaluate_match_score(self, description: str, image_path: str):
        return self.result_type(description, "simulated", self.score, image_path, "fake")


class FakePhysicalRealismScorer:
    def evaluate_background_physical_realism(self, image_path: str):
        return {"score": 76, "reasoning": "simulated"}


class FakeNERCScorer:
    def evaluate(self, **_: object):
        return {"preservation_score": 88, "reasoning": "simulated"}


def install_fake_openai() -> None:
    response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content="Explanation: simulated\nAnswer: yes"
                )
            )
        ]
    )
    completions = types.SimpleNamespace(create=lambda **_: response)
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    module = types.ModuleType("openai")
    module.OpenAI = lambda **_: client
    sys.modules["openai"] = module


def test_evaluator_contracts(root: Path) -> None:
    image = root / "images" / "sample.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), (120, 80, 40)).save(image)

    create_record = {
        "case_id": "create_case",
        "PF_prompt": "a simulated image",
        "sources": [{"path": str(image)}],
        "CC_prompt": {"image1": "Does it match?"},
        "result": {"fake_model": str(image)},
    }
    edit_record = {
        "case_id": "edit_case",
        "IF_prompt": "an edited simulated image",
        "prompts": {"en": "Edit the image."},
        "sources": [{"path": str(image)}, {"path": str(image)}],
        "CC_prompt": {"image1": "Does it match?"},
        "result": {"fake_model": str(image)},
    }

    result = create_pf.process_single_record(
        dict(create_record), FakePromptFollowingScorer(create_pf.MatchScoreResult), "fake_model", root
    )
    assert result["evaluations"]["fake_model"]["score"] == 82

    result = edit_if.process_single_record(
        dict(edit_record), FakePromptFollowingScorer(edit_if.MatchScoreResult), "fake_model", root
    )
    assert result["evaluations"]["fake_model"]["score"] == 82

    result = create_pr.process_single_record(
        dict(create_record), FakePhysicalRealismScorer(), "fake_model", root
    )
    assert result["evaluations"]["fake_model"]["score"] == 76

    result = edit_pr.process_single_record(
        dict(edit_record), FakePhysicalRealismScorer(), "fake_model", root
    )
    assert result["evaluations"]["fake_model"]["score"] == 76

    result = edit_nerc.process_single_record(
        dict(edit_record), FakeNERCScorer(), "fake_model", root
    )
    assert result["evaluations"]["fake_model"]["preservation_score"] == 88

    install_fake_openai()
    cc_result = create_cc.process_task(
        (
            "create_case",
            "fake_model",
            str(image),
            create_record["CC_prompt"],
            {"image1": str(image)},
            "fake-key",
            "https://example.invalid/v1",
            "fake-vlm",
            1,
            0.0,
            True,
        )
    )
    assert cc_result is not None
    assert cc_result["evaluations"]["question_1"]["answer"] == "yes"


def test_creation_aggregation(root: Path) -> None:
    metadata = [
        {
            "case_id": case_id,
            "combination": "identity_style",
            "sub_class_comb": "person|oil",
            "sources": [{}, {}],
            "result": {"fake_model": "unused.png"},
        }
        for case_id in ("c1", "c2")
    ]
    metadata_path = root / "create.json"
    write_json(metadata_path, metadata)
    evaluation_root = root / "create_eval"
    csv_dir = root / "create_csv"
    write_json(
        evaluation_root / "PF" / "processed_metadata_with_verifications.match_fake_model.json",
        [
            {"case_id": "c1", "evaluations": {"fake_model": {"score": 80}}},
            {"case_id": "c2", "evaluations": {"fake_model": {"score": 60}}},
        ],
    )
    write_json(
        evaluation_root / "CC" / "fake_model_evaluated.json",
        [
            {"case_id": "c1", "evaluations": {"question_1": {"answer": "yes"}}},
            {"case_id": "c2", "evaluations": {"question_1": {"answer": "no"}}},
        ],
    )
    write_json(
        evaluation_root / "IA" / "fake_model.json",
        [
            {"case_id": "c1", "iaa": 60, "iqa": 70, "ista": 40},
            {"case_id": "c2", "iaa": 80, "iqa": 90, "ista": 60},
        ],
    )
    write_json(
        evaluation_root / "PR" / "processed_metadata_with_verifications.match_fake_model.json",
        [
            {"case_id": "c1", "evaluations": {"fake_model": {"score": 90}}},
            {"case_id": "c2", "evaluations": {"fake_model": {"score": 70}}},
        ],
    )
    target = csv_dir / "MICE_Create_Result_fake_model.csv"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "evaluation" / "creation" / "json2csv_create.py"),
            "--metadata", str(metadata_path),
            "--evaluation-root", str(evaluation_root),
            "--csv-dir", str(csv_dir),
        ],
        check=True,
    )
    row = read_csv_row(target)
    assert row["PF"] == "70.00" and row["CC"] == "5.00"
    assert row["IA"] == "70.0000" and row["IQ"] == "80.0000"
    assert row["IS"] == "50.0000" and row["PR"] == "80.00"


def test_editing_aggregation(root: Path) -> None:
    metadata = [
        {
            "case_id": case_id,
            "task_type": "ADD",
            "task_subtype": "add_object",
            "result": {"fake_model": "unused.png"},
        }
        for case_id in ("e1", "e2")
    ]
    metadata_path = root / "edit.json"
    write_json(metadata_path, metadata)
    evaluation_root = root / "edit_eval"
    csv_dir = root / "edit_csv"
    write_json(
        evaluation_root / "IF" / "processed_metadata_with_verifications.match_fake_model.json",
        [
            {"case_id": "e1", "evaluations": {"fake_model": {"score": 80}}},
            {"case_id": "e2", "evaluations": {"fake_model": {"score": 60}}},
        ],
    )
    write_json(
        evaluation_root / "CC" / "fake_model_evaluated.json",
        [
            {"case_id": "e1", "evaluations": {"question_1": {"answer": "yes"}}},
            {"case_id": "e2", "evaluations": {"question_1": {"answer": "no"}}},
        ],
    )
    write_json(
        evaluation_root / "NERC" / "processed_metadata_with_verifications.NERC_fake_model.json",
        [
            {"case_id": "e1", "evaluations": {"fake_model": {"preservation_score": 90}}},
            {"case_id": "e2", "evaluations": {"fake_model": {"preservation_score": 70}}},
        ],
    )
    write_json(
        evaluation_root / "PR" / "processed_metadata_with_verifications.match_fake_model.json",
        [
            {"case_id": "e1", "evaluations": {"fake_model": {"score": 60}}},
            {"case_id": "e2", "evaluations": {"fake_model": {"score": 40}}},
        ],
    )
    target = csv_dir / "MICE_Edit_Result_fake_model.csv"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "evaluation" / "editing" / "json2csv_edit.py"),
            "--metadata", str(metadata_path),
            "--evaluation-root", str(evaluation_root),
            "--csv-dir", str(csv_dir),
        ],
        check=True,
    )
    row = read_csv_row(target)
    assert row["IF"] == "70.00" and row["CC"] == "5.00"
    assert row["NERC"] == "80.00" and row["PR"] == "50.00"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mice_eval_test_") as directory:
        root = Path(directory)
        test_evaluator_contracts(root)
        test_creation_aggregation(root)
        test_editing_aggregation(root)
    print("All offline evaluation contract tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
