# Dataset format

MICE-Bench metadata is distributed as UTF-8 JSON arrays. Paths are relative to the dataset root. Do not infer semantics from directory names; use the metadata fields.

## Creation records

```json
{
  "case_id": "appearance_identity_material_texture_024",
  "pool_names": {"image1": "appearance", "image2": "identity"},
  "combination": "appearance_identity_material_texture",
  "sources": [{"path": "3/example/input_1.png"}],
  "prompts": {"short": "...", "en": "..."},
  "PF_prompt": "...",
  "CC_prompt": {"image1": "..."},
  "result": {}
}
```

`pool_names` maps prompt image labels to concept categories. `PF_prompt` is the text-only prompt used by creation PF evaluation. `CC_prompt` maps each reference image to its creation CC question.

## Editing records

```json
{
  "case_id": "2_add_appearance_clothes_1",
  "task_type": "ADD",
  "task_subtype": "add_appearance_clothes",
  "sources": [
    {"path": "edit/2/example/input_1.png"},
    {"path": "edit/2/example/input_2.png"}
  ],
  "prompts": {"short": "...", "en": "..."},
  "IF_prompt": "...",
  "CC_prompt": {"image1": "...", "image2": "..."},
  "result": {}
}
```

For editing, `input_1` is the target image and subsequent inputs are visual references unless a task's metadata states otherwise. `IF_prompt` is the text-only prompt used by the IF metric, while `CC_prompt` contains the per-reference concept-consistency questions used by CC.

## Model outputs

Evaluators discover systems from keys in `result`:

```json
"result": {
  "my_model": "/absolute/or/dataset-relative/path/to/output.png"
}
```

Use `scripts/prepare_submission.py` to populate this field without modifying the canonical metadata.

## Integrity

Run `python scripts/validate_dataset.py` after downloading. It checks required metadata, unique IDs, and all referenced files. The final public snapshot must contain the paper's 3,119 records (1,872 creation and 1,247 editing).
