import argparse
import json
import os
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from io import BytesIO
from PIL import Image
from dataclasses import dataclass
import concurrent.futures

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "edit.json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "evaluation_results" / "IF"


def encode_image(image_path: Union[str, Path], max_size: int = 1024) -> str:
    try:
        with Image.open(image_path) as img:
            
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((max_size, max_size))
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Image encoding failed: {image_path}: {e}")
        return ""

@dataclass
class MatchScoreResult:
    detailed_description: str  
    reasoning: str  
    match_score: int  
    image_path: str  
    model_name: str  

class ImageMatchScorer:

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gemini-3-flash-preview"
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def _call_api_with_vision(
        self,
        text_prompt: str,
        images: List[str],
        is_json: bool = True,
        temperature: float = 0.7
    ) -> Any:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
           
            content_list = [{"type": "text", "text": text_prompt}]
  
            for b64_image in images:
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                })
                content_list.append({
                    "type": "text",
                    "text": f"(Image)"
                })
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content_list}],
                temperature=temperature,
                response_format={"type": "json_object"} if is_json else None
            )

            content = response.choices[0].message.content
            return json.loads(content) if is_json else content

        except Exception as e:
            print(f"API call failed: {e}")
            return None

    def evaluate_match_score(
        self,
        detailed_description: str,
        image_path: str
    ) -> Optional[MatchScoreResult]:
        print(f"Evaluating image: {image_path}")
                
        model_name = Path(image_path).parent.name
                
        image_b64 = encode_image(image_path)
        if not image_b64:
            print(f"Failed to encode image: {image_path}")
            return None
       
        match_prompt = f"""
        You are an expert AI Visual Evaluator specializing in assessing text-to-image generation quality. Your task is to evaluate the semantic alignment between the provided input image and the Text Prompt.

**Text Prompt:**
"{detailed_description}"

**Evaluation Steps:**
1.  **Object Detection:** Are all subjects/objects mentioned in the text prompt present in the image?
2.  **Attribute Check:** Do the objects have the correct colors, shapes, textures, and counts?
3.  **Spatial Relations & Actions:** Are the objects interacting correctly or positioned as described?
4.  **Style & Aesthetics:** Does the image match the requested artistic style (e.g., photorealistic, sketch, oil painting)?

**Scoring Rubric (0-100):**
- **90-100 (Excellent):** Flawless alignment. All objects, attributes, relations, and styles are perfectly represented.
- **70-89 (Good):** The main subject is correct, but there are minor discrepancies in background details, non-essential attributes, or slight style deviations.
- **40-69 (Fair):** The main subject is present but distorted, or key attributes (color, action) are wrong. The prompt is recognizable but flawed.
- **0-39 (Poor):** Severe hallucinations, missing main subjects, or completely wrong style. The image is irrelevant to the text.

**Output Format:**
Provide your response in a strict JSON format.
- "explanation": A concise analysis (under 100 words) detailing what matches and what is missing or wrong.
- "score": An integer between 0 and 100 based on the rubric.

**Response Requirements:**
- Output **ONLY** the JSON object. Do not include markdown backticks (```json) or introductory text.

**JSON Template:**
{{"explanation": "...", "score": ...}}"""
       
        result = self._call_api_with_vision(
            text_prompt=match_prompt,
            images=[image_b64],
            is_json=True,
            temperature=0.3  
        )
        if not result or not isinstance(result, dict):
            print("The API returned an invalid result")
            return None
        
        try:
            match_score_result = MatchScoreResult(
                detailed_description=detailed_description,
                reasoning=result.get("explanation", ""),
                match_score=int(result.get("score", 0)),
                image_path=image_path,
                model_name=model_name
            )
            return match_score_result
        except Exception as e:
            print(f"Failed to parse the match score: {e}")
            return None

def process_single_record(
    record: Dict[str, Any],
    scorer: ImageMatchScorer,
    model_name: str,
    dataset_root: Path,
) -> Dict[str, Any]:
    case_id = record.get("case_id", "")
    print(f"Processing case: {case_id}")
       
    detailed_description = record.get("IF_prompt", "")
    
    if not detailed_description:
        print(f"Case {case_id} has no IF_prompt")
        return record
    
    if "evaluations" not in record:
        record["evaluations"] = {}

    if model_name in record["evaluations"]:
        print(f"Skipping {case_id} - {model_name}; evaluation already exists")
        return record

    results_map = record.get("result", {})
    if model_name in results_map:
        image_path = results_map[model_name]
        if image_path:
            path = Path(image_path)
            image_path = str(path if path.is_absolute() else dataset_root / path)
        
        if not image_path:
            print(f"Image path is empty: {image_path}")
            record["evaluations"][model_name] = {
                "explanation": "Empty image path",
                "score": 0,
                "image_path": image_path
            }
            return record
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            record["evaluations"][model_name] = {
                "explanation": "Image path not found",
                "score": 0,
                "image_path": image_path
            }
            return record
        
        evaluation = scorer.evaluate_match_score(detailed_description, image_path)

        if evaluation:
            
            record["evaluations"][model_name] = {
                "explanation": evaluation.reasoning,
                "score": evaluation.match_score,
                "image_path": evaluation.image_path
            }
            print(f"Completed {case_id} - {model_name}; score: {evaluation.match_score}")
        else:
            print(f"Evaluation {case_id} - {model_name} failed")
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


def process_mige_dataset_for_evaluation(
    input_file: str,
    output_dir: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gemini-3-pro-preview",
    num_threads: int = 4,
    dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
):
    print(f"Processing dataset: {input_file}")

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

    print(f"Evaluation models ({len(models_to_evaluate)}): {models_to_evaluate}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(dataset_root)

    def _worker(idx: int, record: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        scorer = ImageMatchScorer(api_key=api_key, base_url=base_url, model=model)
        print(f"\n[{model_name}] Progress: {idx+1}/{len(data)}")
        return process_single_record(dict(record), scorer, model_name, dataset_root)

    for model_name in models_to_evaluate:
        out_file = output_path / f"processed_metadata_with_verifications.match_{model_name}.json"
        existing = _load_existing(out_file)
        processed_ids = {r.get("case_id") for r in existing if r.get("case_id")}
        processed_data: List[Dict[str, Any]] = list(existing)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {}
            for i, record in enumerate(data):
                case_id = record.get("case_id", f"idx_{i+1}")
                result_map = record.get("result")
                if not isinstance(result_map, dict) or model_name not in result_map:
                    continue
                if case_id in processed_ids:
                    continue
                futures[executor.submit(_worker, i, record, model_name)] = case_id

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
    parser = argparse.ArgumentParser(description="Evaluate instruction following (IF).")
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

    process_mige_dataset_for_evaluation(
        input_file=str(args.input),
        output_dir=str(args.output_dir),
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        num_threads=args.workers,
        dataset_root=args.dataset_root,
    )
    print("All processing completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
