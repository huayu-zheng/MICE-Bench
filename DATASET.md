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
  "caption_result": {"converted_prompt": "..."},
  "verification_questions": {"image1": "..."},
  "result": {}
}
```

`pool_names` maps prompt image labels to concept categories. `caption_result.converted_prompt` replaces visual placeholders with descriptions for prompt-following evaluation. `verification_questions` supports concept-level consistency scoring.

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
  "verification_questions": {"image1": "...", "image2": "..."},
  "result": {}
}
```

For editing, `input_1` is the target image and subsequent inputs are visual references unless a task's metadata states otherwise.

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

