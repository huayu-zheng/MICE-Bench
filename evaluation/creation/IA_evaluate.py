import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "create.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "evaluation_results" / "IA"
DEFAULT_UNIPERCEPT_ROOT = PROJECT_ROOT / "unipercept"
DEFAULT_MODEL_PATH = DEFAULT_UNIPERCEPT_ROOT / "ckpt" / "UniPercept"


class LocalUniPerceptInferencer:
    """Run MICE creation metrics with the UniPercept source bundled in this repo."""

    METRICS = {
        "iaa": "aesthetics",
        "iqa": "quality",
        "ista": "structure and texture richness",
    }

    def __init__(
        self,
        unipercept_root: Path,
        model_path: Path,
        device: str,
        use_flash_attn: bool,
    ) -> None:
        source_root = unipercept_root / "src"
        if not source_root.is_dir():
            raise FileNotFoundError(
                f"UniPercept source directory not found: {source_root}. "
                "Keep the bundled unipercept directory at the repository root or "
                "pass --unipercept-root."
            )
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"UniPercept checkpoint not found: {model_path}. Download "
                "Thunderbolt215215/UniPercept from Hugging Face into this directory "
                "or pass --model-path."
            )

        sys.path.insert(0, str(source_root))
        try:
            import torch
            import torchvision.transforms as transforms
            from PIL import Image, ImageFile
            from torchvision.transforms.functional import InterpolationMode
            from transformers import AutoTokenizer
            from internvl.model.internvl_chat.modeling_unipercept import (
                InternVLChatModel,
            )
        except ImportError as error:
            raise RuntimeError(
                "UniPercept dependencies are unavailable. Follow the UniPercept "
                "deployment instructions in README.md and install "
                "unipercept/requirements-mice.txt."
            ) from error

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

        self.torch = torch
        self.Image = Image
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda image: image.convert("RGB")
                    if image.mode != "RGB"
                    else image
                ),
                transforms.Resize(
                    (448, 448), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        print(f"Loading UniPercept checkpoint from {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, use_fast=False
        )
        self.model = InternVLChatModel.from_pretrained(
            str(model_path),
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn and self.device.type == "cuda",
        ).eval().to(self.device)
        self.generation_config = {"max_new_tokens": 32, "do_sample": False}

    def reward(self, image_path: str) -> Dict[str, float]:
        with self.Image.open(image_path) as image:
            pixel_values = self.transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)

        scores: Dict[str, float] = {}
        with self.torch.inference_mode():
            for key, description in self.METRICS.items():
                scores[key] = float(
                    self.model.score(
                        self.device,
                        self.tokenizer,
                        pixel_values,
                        self.generation_config.copy(),
                        description,
                    )
                )
        return scores


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute image aesthetics/quality/structure (IA/IQ/IS) with UniPercept."
    )
    parser.add_argument(
        "--input",
        "--metadata",
        dest="metadata",
        default=DEFAULT_INPUT_FILE,
        help="Path to processed metadata JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write per-model JSON outputs.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "create",
        help="Base directory for relative image paths in create.json.",
    )
    parser.add_argument(
        "--unipercept-root",
        type=Path,
        default=DEFAULT_UNIPERCEPT_ROOT,
        help="Path to the bundled UniPercept repository (default: unipercept).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local UniPercept checkpoint directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (e.g., cuda or cpu).",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable FlashAttention (use this when flash-attn is not installed).",
    )

    args = parser.parse_args()

    with open(args.metadata, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    inferencer = LocalUniPerceptInferencer(
        unipercept_root=args.unipercept_root.resolve(),
        model_path=args.model_path,
        device=args.device,
        use_flash_attn=not args.no_flash_attn,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    per_model_records: Dict[str, Dict[str, Dict[str, Any]]] = {}
    per_model_processed: Dict[str, set] = {}

    model_names = sorted(
        {
            model_name
            for record in records
            if isinstance(record.get("result"), dict)
            for model_name in record["result"]
        }
    )
    if not model_names:
        raise ValueError("No model keys found in any result object.")

    print("Loading existing result files for resume...")
    for model_name in model_names:
        output_path = os.path.join(args.output_dir, f"{model_name}.json")
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_entries = json.load(f)
                    per_model_records[model_name] = {
                        entry["case_id"]: entry for entry in existing_entries
                    }
                    per_model_processed[model_name] = {
                        entry["case_id"] for entry in existing_entries
                    }
                    print(f"  Loaded {model_name}.json: {len(existing_entries)} entries")
            except Exception as e:
                print(f"  Warning: Unable to read {output_path}: {e}")
                per_model_records[model_name] = {}
                per_model_processed[model_name] = set()
        else:
            per_model_records[model_name] = {}
            per_model_processed[model_name] = set()
    total_tasks = sum(
        len(record.get("result", {}))
        for record in records
        if isinstance(record.get("result"), dict)
    )
    processed_count = sum(len(processed) for processed in per_model_processed.values())
    print(f"\nTotal tasks: {total_tasks}, processed: {processed_count}, pending: {total_tasks - processed_count}\n")
    current_task = 0

    for record in records:
        result_map = record.get("result", {})
        if not isinstance(result_map, dict):
            continue

        case_id = record.get("case_id", "")
        pool_names = record.get("pool_names", {})

        for model_name, result_path in result_map.items():
            current_task += 1
            if case_id in per_model_processed.get(model_name, set()):
                print(f"[{current_task}/{total_tasks}] Skipping processed: {model_name} - {case_id}")
                continue
            print(f"[{current_task}/{total_tasks}] Processing: {model_name} - {case_id} ({current_task*100//total_tasks}%)")
            
            entry = {
                "case_id": case_id,
                "pool_names": pool_names,
                "result_path": result_path
            }

            if result_path and not Path(result_path).is_absolute():
                result_path = str(args.dataset_root / result_path)
                entry["result_path"] = result_path
            
            if not result_path:
                entry["iaa"] = None
                entry["iqa"] = None
                entry["ista"] = None
            elif not os.path.exists(result_path):
                print(f"  Warning: Image not found: {result_path}")
                entry["iaa"] = None
                entry["iqa"] = None
                entry["ista"] = None
            else:
                try:
                    reward = inferencer.reward(result_path)
                    entry["iaa"] = reward["iaa"]
                    entry["iqa"] = reward["iqa"]
                    entry["ista"] = reward["ista"]
                    print(f"  Completed: iaa={entry['iaa']:.4f}, iqa={entry['iqa']:.4f}, ista={entry['ista']:.4f}")
                except Exception as e:
                    print(f"  Error: processing failed: {e}")
                    entry["iaa"] = None
                    entry["iqa"] = None
                    entry["ista"] = None
            
            
            per_model_records.setdefault(model_name, {})[case_id] = entry
            per_model_processed.setdefault(model_name, set()).add(case_id)
            
            
            output_path = os.path.join(args.output_dir, f"{model_name}.json")
            entries_list = list(per_model_records[model_name].values())
            temp_path = f"{output_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(entries_list, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, output_path)
    
    print(f"\nAll tasks completed. Results saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
