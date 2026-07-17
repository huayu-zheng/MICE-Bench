import os
import argparse
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from io import BytesIO
from PIL import Image
import concurrent.futures


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "create.json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "create"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "evaluation_results" / "Q4"


def encode_image(image_path: Union[str, Path], max_size: int = 1024) -> str:
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((max_size, max_size))
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Image encoding failed {image_path}: {e}")
        return ""


class ImageMatchScorer:

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gemini-3-flash-preview",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def _call_api_with_vision(
        self,
        text_prompt: str,
        images: List[str],
        is_json: bool = True,
        temperature: float = 0.7,
    ) -> Any:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)

            content_list = [{"type": "text", "text": text_prompt}]

            for i, b64_image in enumerate(images):
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    }
                )
                content_list.append({"type": "text", "text": f"(image {i + 1})"})

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content_list}],
                temperature=temperature,
                response_format={"type": "json_object"} if is_json else None,
            )

            content = response.choices[0].message.content
            return json.loads(content) if is_json else content

        except Exception as e:
            print(f"API call failed: {e}")
            return None

    def evaluate_background_physical_realism(self, image_path: str) -> Optional[Dict[str, Any]]:
        print(f"Evaluating image: {image_path}")

        image_b64 = encode_image(image_path)
        if not image_b64:
            print(f"Failed to encode image: {image_path}")
            return None

        evaluation_prompt = """# Role
You are an expert AI Physics & Visual Logic Auditor. Your goal is to evaluate AI-generated images for their adherence to physical laws, material properties, and logical consistency. You must adapt your evaluation criteria based on the content of the image.

# Evaluation Dimensions (Broad Scope)

1. **Entity Integrity & Topology (Crucial)**
   - **Focus:** Object separation and solidity.
   - **Check:** Are objects distinct? Look for "fusion" or "melding" artifacts (e.g., a hand merging into a cup, hair melting into clothes). Objects should not behave like liquid unless they are liquid.

2. **Materiality & Light Interaction**
   - **Focus:** How light interacts with surfaces.
   - **Check:** Does skin look like skin (subsurface scattering) or plastic? Does metal reflect? Are shadows consistent with a global light source? If mirrors/water exist, are reflections geometrically correct?

3. **Spatial & Geometric Logic**
   - **Focus:** 3D space construction in a 2D image.
   - **Check:** Perspective lines, vanishing points, and relative scale (e.g., a cat shouldn't be larger than a car). Do objects rest naturally on surfaces, or do they float/clip through the ground?

4. **Contextual & Causal Consistency**
   - **Focus:** Cause and effect within the environment.
   - **Check:** If it's raining, is it wet? If there is wind, is there motion? Are shadows cast in the correct direction relative to the light?

5. **Structural & Functional Plausibility**
   - **Focus:** Internal logic of objects/beings.
   - **Check:** Anatomy (finger count, joint bending). Engineering (do buildings/vehicles look structurally sound? do chairs have legs?).

# Scoring Rubric (0-100)

- **90-100 (Physically Accurate):** Flawless logic. Perfect object separation, accurate lighting physics, and correct scale.
- **70-89 (High Consistency):** Generally follows laws of physics. Minor artifacts allowed in background textures, but main objects are solid and distinct.
- **40-69 (Noticeable Flaws):** Visible violations. Minor object fusion (blending edges), inconsistent shadows, or slight perspective distortion.
- **0-39 (Physically Broken):** Severe hallucinations. Objects melting into each other, impossible geometry, severe anatomical errors, or chaotic lighting.

# Output Format (JSON)
Output ONLY a JSON object with exactly these keys:
{
  "score": 0-100 integer,
  "reasoning": "brief reasons"
}"""

        result = self._call_api_with_vision(
            text_prompt=evaluation_prompt,
            images=[image_b64],
            is_json=True,
            temperature=0.3,
        )
        if not result or not isinstance(result, dict):
            print("API returned invalid result")
            return None

        score = result.get("score")
        reasoning = result.get("reasoning")
        if isinstance(score, str) and score.isdigit():
            score = int(score)
        if not isinstance(score, int) or not isinstance(reasoning, str):
            print("API returned unexpected format")
            return None

        return {"score": score, "reasoning": reasoning}



def process_single_record(
    record: Dict[str, Any],
    scorer: ImageMatchScorer,
    model_name: str,
    dataset_root: Path,
) -> Dict[str, Any]:
    case_id = record.get("case_id", "")
    print(f"Processing case: {case_id}")

    if "evaluations" not in record:
        record["evaluations"] = {}

    if model_name in record["evaluations"]:
        print(f"Skip {case_id} - {model_name}, evaluation exists")
        return record

    results_map = record.get("result", {})
    if model_name in results_map:
        image_path = results_map[model_name]
        if image_path:
            path = Path(image_path)
            image_path = str(path if path.is_absolute() else dataset_root / path)

        if not image_path:
            print(f"Empty image path: {image_path}")
            record["evaluations"][model_name] = {
                "error": "Empty image path",
                "image_path": image_path,
            }
            return record
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            record["evaluations"][model_name] = {
                "error": "Image path not found",
                "image_path": image_path,
            }
            return record

        evaluation_result = scorer.evaluate_background_physical_realism(image_path)

        if evaluation_result:
            record["evaluations"][model_name] = evaluation_result
            print(f"Completed {case_id} - {model_name} evaluation")
        else:
            print(f"Evaluation failed for {case_id} - {model_name}")
    else:
        print(f"Case {case_id} has no result path for model {model_name}")

    return record


def _load_existing(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    return []


def process_mige_dataset_for_verification_multi(
    input_file: str,
    output_dir: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gemini-3-pro-preview",
    num_threads: int = 4,
    dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
):
    print(f"Start processing dataset: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a top-level list")

    models_to_evaluate = sorted(
        {
            model_name
            for record in data
            if isinstance(record.get("result"), dict)
            for model_name in record["result"]
        }
    )

    if not models_to_evaluate:
        raise ValueError("No model keys found in any result object.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(dataset_root)

    def _worker(record: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        scorer = ImageMatchScorer(api_key=api_key, base_url=base_url, model=model)
        return process_single_record(dict(record), scorer, model_name, dataset_root)

    for model_name in models_to_evaluate:
        out_file = output_path / f"processed_metadata_with_verifications.match_{model_name}.json"
        existing = _load_existing(out_file)
        processed_ids = {r.get("case_id") for r in existing if r.get("case_id")}
        processed_data: List[Dict[str, Any]] = list(existing)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {}
            for idx, record in enumerate(data, 1):
                case_id = record.get("case_id", f"idx_{idx}")
                result_map = record.get("result")
                if not isinstance(result_map, dict) or model_name not in result_map:
                    continue
                if case_id in processed_ids:
                    print(f"[{model_name}] Skip processed: {case_id}")
                    continue
                futures[executor.submit(_worker, record, model_name)] = case_id

            for future in concurrent.futures.as_completed(futures):
                case_id = futures[future]
                try:
                    updated_record = future.result()
                except Exception as e:
                    print(f"[{model_name}] Failed {case_id}: {e}")
                    updated_record = dict(
                        next((r for r in data if r.get("case_id") == case_id), {})
                    )

                processed_data.append(updated_record)
                processed_ids.add(case_id)
                out_file.write_text(
                    json.dumps(processed_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"[{model_name}] Saved {case_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate physical realism (Q4).")
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

    process_mige_dataset_for_verification_multi(
        input_file=str(args.input),
        output_dir=str(args.output_dir),
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        num_threads=args.workers,
        dataset_root=args.dataset_root,
    )
    print("All done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
