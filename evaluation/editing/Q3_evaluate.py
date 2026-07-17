import argparse
import base64
import concurrent.futures
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "edit.json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "evaluation_results" / "Q3"


def resolve_path(
    path_text: Union[str, Path, None], dataset_root: Path
) -> Optional[Path]:
    if not path_text:
        return None
    path = Path(str(path_text).strip())
    return path if path.is_absolute() else dataset_root / path


def encode_image(image_path: Path, max_size: int = 1024) -> str:
    try:
        with Image.open(image_path) as image:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.thumbnail((max_size, max_size))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except (OSError, ValueError) as error:
        print(f"Image encoding failed for {image_path}: {error}", flush=True)
        return ""


def get_edit_instruction(record: Dict[str, Any]) -> str:
    prompts = record.get("prompts")
    return prompts.get("en", "") if isinstance(prompts, dict) else ""


def get_original_and_reference_paths(
    record: Dict[str, Any], dataset_root: Path
) -> tuple[Optional[Path], List[Path]]:
    sources = record.get("sources")
    if not isinstance(sources, list) or not sources:
        return None, []

    first_source = sources[0] if isinstance(sources[0], dict) else {}
    original_path = resolve_path(first_source.get("path"), dataset_root)
    reference_paths = []
    for source in sources[1:]:
        if not isinstance(source, dict):
            continue
        reference_path = resolve_path(source.get("path"), dataset_root)
        if reference_path:
            reference_paths.append(reference_path)
    return original_path, reference_paths


class NonEditedRegionScorer:
    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def evaluate(
        self,
        original_image_path: Path,
        edited_image_path: Path,
        reference_image_paths: List[Path],
        editing_instruction: str,
    ) -> Optional[Dict[str, Any]]:
        image_paths = [*reference_image_paths, original_image_path, edited_image_path]
        image_labels = [
            *(f"Reference Image {index}" for index in range(1, len(reference_image_paths) + 1)),
            "Original Image",
            "Edited Image",
        ]
        encoded_images = [encode_image(path) for path in image_paths]
        if not all(encoded_images):
            return None

        reference_note = (
            "Reference Images are provided before the Original Image."
            if reference_image_paths
            else "No external reference image is provided for this task."
        )

        prompt = f"""# Non-Edited Region Consistency Evaluation Criteria

## System Role
You are an expert Image Editing Quality Evaluator specialized in "Non-Edited Area Preservation" (Background Consistency). Your goal is to assess whether the parts of an image that should *not* have been changed remain identical to the original image.

## Task Description
You will be provided with:
1. Reference Images: Visual references used for the editing task, such as the object to be added or the replacement target. {reference_note}
2. Original Image: The source image before editing.
3. Edited Image: The final result.
4. Editing Instruction: Text description of the desired edit.

Editing Instruction:
{editing_instruction}

Your task is to calculate a Preservation Score (0-100) based strictly on how well the non-edited areas are preserved.

## Evaluation Steps
Please process the input in the following order:
1. Analyze Intent: Read the Editing Instruction and look at Reference Images to understand what represents the Foreground/Edited Area and what represents the Background/Non-Edited Area.
2. Visual Comparison: Compare Original Image and Edited Image. Ignore the intended edited object/region, and focus on all areas that should remain unchanged.
3. Scoring Criteria:
   - 90-100: Perfect preservation. Non-edited areas are pixel-perfect or perceptually identical to the Original Image.
   - 80-90: High quality. Minor, barely noticeable lighting shifts or compression artifacts in the background.
   - 60-79: Acceptable but flawed. Noticeable changes in background texture, lighting, or minor distortions, but semantic content is intact.
   - 40-59: Poor. Background details are lost, blurred, or colors have significantly shifted.
   - 0-39: Failure. Severe hallucinations, background structure collapsed, or the edit affected the whole image globally when it should have been local.

## Output Format
Output ONLY a valid JSON object with exactly these keys:
{{
  "preservation_score": 0-100 integer,
  "reasoning": "Explain the score within 100 words."
}}"""

        content = [{"type": "text", "text": prompt}]
        for label, encoded_image in zip(image_labels, encoded_images):
            content.extend(
                [
                    {"type": "text", "text": f"({label})"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                            "detail": "high",
                        },
                    },
                ]
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content or "")
        except Exception as error:
            print(f"API request failed: {error}", flush=True)
            return None

        if not isinstance(result, dict):
            print("API returned invalid JSON", flush=True)
            return None

        score = result.get("preservation_score")
        if isinstance(score, str):
            try:
                score = int(float(score.strip()))
            except ValueError:
                score = None
        reasoning = result.get("reasoning", "")
        if not isinstance(score, int) or not isinstance(reasoning, str):
            print(f"Unexpected Q3 output format: {result}", flush=True)
            return None
        return {
            "preservation_score": max(0, min(100, score)),
            "reasoning": reasoning,
        }


def process_single_record(
    record: Dict[str, Any],
    scorer: NonEditedRegionScorer,
    model_name: str,
    dataset_root: Path,
) -> Dict[str, Any]:
    case_id = record.get("case_id", "")
    print(f"Processing case: {case_id} - {model_name}", flush=True)
    evaluations = record.setdefault("evaluations", {})
    if model_name in evaluations:
        return record

    result_map = record.get("result")
    if not isinstance(result_map, dict):
        evaluations[model_name] = {"error": "result is not an object"}
        return record

    generated_path = resolve_path(result_map.get(model_name), dataset_root)
    original_path, reference_paths = get_original_and_reference_paths(
        record, dataset_root
    )
    instruction = get_edit_instruction(record)

    if not generated_path:
        evaluations[model_name] = {"error": "Missing generated image path"}
        return record
    if not original_path:
        evaluations[model_name] = {"error": "Missing original image path"}
        return record
    if not instruction:
        evaluations[model_name] = {"error": "Missing editing instruction"}
        return record

    missing_paths = [
        str(path)
        for path in [original_path, generated_path, *reference_paths]
        if not path.is_file()
    ]
    if missing_paths:
        evaluations[model_name] = {
            "error": "Image path not found",
            "missing_paths": missing_paths,
            "image_path": str(generated_path),
        }
        return record

    evaluation = scorer.evaluate(
        original_image_path=original_path,
        edited_image_path=generated_path,
        reference_image_paths=reference_paths,
        editing_instruction=instruction,
    )
    if evaluation:
        evaluations[model_name] = {
            **evaluation,
            "image_path": str(generated_path),
            "original_image_path": str(original_path),
            "reference_image_paths": [str(path) for path in reference_paths],
        }
    else:
        evaluations[model_name] = {
            "error": "Evaluation failed",
            "image_path": str(generated_path),
        }
    return record


def load_existing(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def process_dataset(
    input_file: Path,
    output_dir: Path,
    api_key: str,
    base_url: str,
    model: str,
    num_threads: int,
    dataset_root: Path,
) -> None:
    data = json.loads(input_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a top-level list")

    model_names = sorted(
        {
            model_name
            for record in data
            if isinstance(record.get("result"), dict)
            for model_name in record["result"]
        }
    )
    if not model_names:
        raise ValueError("No model keys found in any result object.")

    output_dir.mkdir(parents=True, exist_ok=True)

    def worker(record: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        scorer = NonEditedRegionScorer(api_key, base_url, model)
        return process_single_record(dict(record), scorer, model_name, dataset_root)

    for model_name in model_names:
        output_file = output_dir / f"processed_metadata_with_verifications.Q3_{model_name}.json"
        processed_data = load_existing(output_file)
        processed_ids = {
            item.get("case_id")
            for item in processed_data
            if isinstance(item, dict) and item.get("case_id")
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {}
            for index, record in enumerate(data, 1):
                case_id = record.get("case_id", f"idx_{index}")
                result_map = record.get("result")
                if not isinstance(result_map, dict) or model_name not in result_map:
                    continue
                if case_id in processed_ids:
                    continue
                futures[executor.submit(worker, record, model_name)] = case_id

            for future in concurrent.futures.as_completed(futures):
                case_id = futures[future]
                try:
                    updated_record = future.result()
                except Exception as error:
                    print(f"[{model_name}] Failed {case_id}: {error}", flush=True)
                    updated_record = dict(
                        next(
                            (
                                item
                                for item in data
                                if item.get("case_id") == case_id
                            ),
                            {},
                        )
                    )
                    updated_record.setdefault("evaluations", {})[model_name] = {
                        "error": str(error)
                    }

                processed_data.append(updated_record)
                processed_ids.add(case_id)
                save_json(output_file, processed_data)
                print(f"[{model_name}] Saved {case_id}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate non-edited region preservation with a vision API."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument(
        "--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument("--model", default="gemini-3-pro-preview")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not args.api_key:
        parser.error("provide --api-key or set OPENAI_API_KEY")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    process_dataset(
        input_file=args.input,
        output_dir=args.output_dir,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        num_threads=args.workers,
        dataset_root=args.dataset_root,
    )
    print("All processing completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
