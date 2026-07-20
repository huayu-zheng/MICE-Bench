from __future__ import annotations

import argparse
import base64
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import io


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "create.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "evaluation_results" / "CC"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "create"


def encode_image_to_base64(image_path: Path, max_size: int = 1024) -> str:
    try:
        with Image.open(image_path) as img:
            
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            
            if max_size and (img.width > max_size or img.height > max_size):
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_bytes = buffered.getvalue()
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"[ERROR] Image encoding failed {image_path}: {e}", flush=True)
        return ""


def _sort_image_keys(keys: List[str]) -> List[str]:
    def k_num(k: str) -> Tuple[int, str]:
        
        try:
            return int(k.replace("image", "")), k
        except Exception:
            return 10**9, k
    return sorted(keys, key=k_num)


def _has_valid_evaluations(evaluations: Dict[str, Any]) -> bool:
    if not evaluations or len(evaluations) == 0:
        return False
    
    for question_result in evaluations.values():
        answer = question_result.get("answer") if isinstance(question_result, dict) else None
        if answer in ("yes", "no"):
            return True
    
    return False


def call_vlm_for_evaluation(
    client: OpenAI,
    model: str,
    reference_image_path: Path,
    generated_image_path: Path,
    question: str,
    max_retries: int = 3,
    temperature: float = 0.0,
    include_explanation: bool = True,
) -> Optional[Dict[str, str]]:
    
    ref_b64 = encode_image_to_base64(reference_image_path)
    gen_b64 = encode_image_to_base64(generated_image_path)
    
    if not ref_b64 or not gen_b64:
        return None
    
    
    if include_explanation:
        prompt = f"""You are evaluating concept preservation in an image creation task.

You will see two images:
1. Reference Image (left): The source of the visual concept
2. Generated Output Image (right): Should incorporate the concept from the reference

Question: {question}

Use a slightly lenient rule:
- If the generated image mostly preserves the intended visual concept from the reference (even if some details or colors differ), answer "yes".
- Only answer "no" when the concept is clearly missing or obviously wrong.

Please compare the two images and provide:
1. A brief explanation of your observation
2. Your answer: "yes" or "no" (lowercase)

Format your response as:
Explanation: [your explanation]
Answer: [yes or no]"""
    else:
        prompt = f"""You are evaluating concept preservation in an image creation task.

You will see two images:
1. Reference Image (left): The source of the visual concept
2. Generated Output Image (right): Should incorporate the concept from the reference

Question: {question}

Please compare the two images and answer ONLY "yes" or "no" (lowercase, no punctuation, no explanation).

Answer:"""
    
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            content_list = [
                {
                    "type": "image_url",
                    "image_url": {"url": ref_b64, "detail": "high"}
                },
                {
                    "type": "text",
                    "text": "(This is the reference image)"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": gen_b64, "detail": "high"}
                },
                {
                    "type": "text",
                    "text": "(This is the generated output image)"
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content_list}],
                temperature=temperature,
            )
            
            response_text = (resp.choices[0].message.content or "").strip()
            
            
            answer = None
            explanation = ""
            
            if include_explanation:
                
                lines = response_text.split("\n")
                for line in lines:
                    line_lower = line.lower().strip()
                    if line_lower.startswith("explanation:"):
                        explanation = line.split(":", 1)[1].strip() if ":" in line else ""
                    elif line_lower.startswith("answer:"):
                        answer_part = line.split(":", 1)[1].strip() if ":" in line else line_lower
                        if answer_part.startswith("yes"):
                            answer = "yes"
                        elif answer_part.startswith("no"):
                            answer = "no"
                
                
                if answer is None:
                    response_lower = response_text.lower()
                    if response_lower.startswith("yes") or ("yes" in response_lower and "no" not in response_lower[:10]):
                        answer = "yes"
                    elif response_lower.startswith("no") or "no" in response_lower[:10]:
                        answer = "no"
            else:
                
                response_lower = response_text.lower()
                if response_lower.startswith("yes") or ("yes" in response_lower and "no" not in response_lower[:10]):
                    answer = "yes"
                elif response_lower.startswith("no") or "no" in response_lower[:10]:
                    answer = "no"
            
            if answer:
                return {"answer": answer, "explanation": explanation}
            else:
                print(f"[WARN] Unable to parse response: {response_text[:100]}", flush=True)
                return None
                
        except Exception as e:
            last_err = e
            print(f"[ERROR] VLM request error (attempt {attempt}/{max_retries}): {e}", flush=True)
            time.sleep(0.8 * attempt)
    
    print(f"[ERROR] VLM request failed: {last_err}", flush=True)
    return None


def process_task(args_tuple: Tuple) -> Optional[Dict[str, Any]]:
    (
        case_id,
        model_name,
        generated_image_path,
        cc_prompt,
        source_map,
        api_key,
        base_url,
        vlm_model,
        max_retries,
        temperature,
        include_explanation,
    ) = args_tuple
    
    try:
        
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        
        
        generated_image_path = str(generated_image_path).strip()
        generated_path = Path(generated_image_path).resolve()
        
        if not generated_path.exists():
            print(f"[ERROR] {case_id} - {model_name}: Generated Image not found ({generated_path})", flush=True)
            return None
        
        
        evaluations = {}
        sorted_keys = _sort_image_keys(list(cc_prompt.keys()))
        
        for q_idx, image_key in enumerate(sorted_keys, 1):
            question_key = f"question_{q_idx}"
            question = cc_prompt[image_key]
            reference_path = source_map.get(image_key)
            
            if not reference_path or not Path(reference_path).exists():
                evaluations[question_key] = {
                    "answer": None,
                    "explanation": f"Reference image not found: {reference_path}",
                }
                continue
            
            eval_result = call_vlm_for_evaluation(
                client=client,
                model=vlm_model,
                reference_image_path=Path(reference_path),
                generated_image_path=generated_path,
                question=question,
                max_retries=max_retries,
                temperature=temperature,
                include_explanation=include_explanation,
            )
            
            if eval_result:
                evaluations[question_key] = {
                    "answer": eval_result.get("answer", "unknown"),
                    "explanation": eval_result.get("explanation", ""),
                }
            else:
                evaluations[question_key] = {
                    "answer": None,
                    "explanation": "VLM evaluation failed",
                }
        
        return {
            "case_id": case_id,
            "model_name": model_name,
            "evaluations": evaluations,
        }
        
    except Exception as e:
        print(f"[ERROR] {case_id} - {model_name}: Processing error: {e}", flush=True)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate generated images and write one JSON file per model.")
    parser.add_argument(
        "--metadata-file",
        "--metadata_file",
        type=str,
        default=DEFAULT_METADATA_PATH,
        help="Path to create.json containing CC_prompt and result.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for per-model JSON files.",
    )
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root used to resolve relative source paths.",
    )
    parser.add_argument(
        "--api-key",
        "--api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="API Key",
    )
    parser.add_argument(
        "--base-url",
        "--base_url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="API Base URL",
    )
    parser.add_argument(
        "--vlm-model",
        "--vlm_model",
        type=str,
        default="gemini-3-pro-preview",
        help="VLM model used for evaluation.",
    )
    parser.add_argument(
        "--max-retries",
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries per question.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; use 0.0 for deterministic answers.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N cases; 0 means all.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based start index.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=612,
        help="Exclusive end index; 0 means the end.",
    )
    parser.add_argument(
        "--case-id",
        "--case_id",
        type=str,
        default="",
        help="Process only this case_id; empty means all.",
    )
    parser.add_argument(
        "--no-explanation",
        "--no_explanation",
        action="store_true",
        help="Return only yes/no without explanations.",
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        type=int,
        default=10,
        help="Number of worker processes.",
    )

    args = parser.parse_args()
    
    if not args.api_key:
        raise RuntimeError("Provide --api-key or set OPENAI_API_KEY.")
    
    
    metadata_path = Path(args.metadata_file)
    print(f"Loading metadata: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
    
    if not isinstance(metadata_list, list):
        raise ValueError(f"Expected JSON array in {metadata_path}")
    
    total_cases_before_filter = len(metadata_list)
    
    
    if args.case_id:
        metadata_list = [c for c in metadata_list if c.get("case_id") == args.case_id]
    
    
    if args.start > 0 or args.end > 0:
        if args.end > 0:
            metadata_list = metadata_list[args.start:args.end]
        else:
            metadata_list = metadata_list[args.start:]
        print(f"Selected range: from {args.start} to {args.end if args.end > 0 else len(metadata_list) + args.start} cases")
    elif args.limit and args.limit > 0:
        metadata_list = metadata_list[: args.limit]
    
    print(f"Total cases: {total_cases_before_filter}, selected: {len(metadata_list)}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    
    print(f"Output directory: {output_dir}")
    print(f"Dataset root: {dataset_root}")
    
    
    per_model_records: Dict[str, Dict[str, Dict[str, Any]]] = {}
    per_model_processed: Dict[str, set] = {}

    model_names = sorted(
        {
            model_name
            for record in metadata_list
            if isinstance(record.get("result"), dict)
            for model_name in record["result"]
        }
    )
    if not model_names:
        raise ValueError("No model keys found in any result object.")

    print("\nLoading existing result files for resume...")
    for model_name in model_names:
        output_path = output_dir / f"{model_name}_evaluated.json"
        existing_data = []
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    raise ValueError("output JSON must contain a list")
                print(f"  Loaded {model_name}_evaluated.json: {len(existing_data)} entries")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                print(f"  Warning: Unable to read {output_path}: {error}")
                existing_data = []

        per_model_records[model_name] = {
            entry["case_id"]: entry
            for entry in existing_data
            if isinstance(entry, dict) and entry.get("case_id")
        }
        per_model_processed[model_name] = {
            entry["case_id"]
            for entry in existing_data
            if isinstance(entry, dict)
            and entry.get("case_id")
            and _has_valid_evaluations(entry.get("evaluations", {}))
        }
    
    
    tasks = []
    for record in metadata_list:
        case_id = record.get("case_id", "")
        if not case_id:
            print("[WARN] Skipping record without case_id")
            continue
        
        cc_prompt = record.get("CC_prompt", {})
        if not cc_prompt:
            print(f"[WARN] {case_id}: missing CC_prompt; skipped")
            continue
        
        result_map = record.get("result", {})
        if not isinstance(result_map, dict):
            print(f"[WARN] {case_id}: result is not an object; skipped")
            continue
        
        sources = record.get("sources", [])
        
        source_map = {}
        for i, src in enumerate(sources, 1):
            image_key = f"image{i}"
            rel_path = src.get("path", "")
            if rel_path:
                source_map[image_key] = str(dataset_root / rel_path)
        
        for model_name, generated_image_path in result_map.items():
            
            if case_id in per_model_processed.get(model_name, set()):
                continue
            
            generated_path = Path(str(generated_image_path))
            if not generated_path.is_absolute():
                generated_path = dataset_root / generated_path

            tasks.append((
                case_id,
                model_name,
                str(generated_path),
                cc_prompt,
                source_map,
                args.api_key,
                args.base_url,
                args.vlm_model,
                args.max_retries,
                args.temperature,
                not args.no_explanation,
            ))
    
    if not tasks:
        print("No pending tasks.")
        return 0
    
    total_tasks = len(tasks)
    print(f"\nTotal tasks: {total_tasks}")
    print(f"Starting {args.num_workers} worker processes...\n")
    
    
    completed = 0
    with Pool(processes=args.num_workers) as pool:
        results = pool.imap_unordered(process_task, tasks)
        
        for result in results:
            if result is None:
                continue
            
            completed += 1
            case_id = result["case_id"]
            model_name = result["model_name"]
            evaluations = result["evaluations"]
            
            print(f"[{completed}/{total_tasks}] Completed: {model_name} - {case_id} ({completed*100//total_tasks}%)", flush=True)
            
            per_model_records.setdefault(model_name, {})[case_id] = {
                "case_id": case_id,
                "evaluations": evaluations,
            }

            output_path = output_dir / f"{model_name}_evaluated.json"
            entries_list = list(per_model_records[model_name].values())
            temp_path = output_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(entries_list, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, output_path)
    
    print(f"\nAll tasks completed. Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
