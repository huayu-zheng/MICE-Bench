# Evaluation

The creation and editing tracks share four evaluator stages, but Q3 differs by track.

| Stage | Creation | Editing | Backend |
|---|---|---|---|
| Q1 | Prompt following (PF) | Instruction following (IF) | vision API |
| Q2 | Concept consistency (CC) | Concept consistency (CC) | vision API |
| Q3 | Aesthetics, quality, structure (IA/IQ/IS) | Non-edited region consistency (NERC) | UniPercept / vision API |
| Q4 | Physical realism (PR) | Physical realism (PR) | vision API |

Scores are written per model and can be resumed. Generated model names are discovered from each record's `result` mapping; `--model` or `--vlm-model` selects the scoring model, not the system being benchmarked.

## Creation Q3: local UniPercept

Creation Q3 does not use the vision API. It imports the bundled implementation from `unipercept/src`, loads the official checkpoint from `unipercept/ckpt/UniPercept`, and calls UniPercept's native `model.score(...)` interface for aesthetics, quality, and structure/texture.

Install its environment and checkpoint by following [the main deployment guide](../README.md#deploy-unipercept-for-creation-q3). Then run:

```bash
python evaluation/creation/Q3_evaluate.py \
  --input outputs/my_model/create.json \
  --dataset-root data/create
```

Useful options are `--model-path`, `--unipercept-root`, `--device cuda:0`, and `--no-flash-attn`. Each model is written to `evaluation/creation/evaluation_results/Q3/<model>.json`; existing case IDs are skipped when the command is resumed.

Creation aggregation:

```bash
python evaluation/creation/json2csv_create.py \
  --metadata outputs/my_model/create.json \
  --evaluation-root evaluation/creation/evaluation_results
```

Editing aggregation:

```bash
python evaluation/editing/json2csv_edit.py \
  --metadata outputs/my_model/edit.json \
  --evaluation-root evaluation/editing/evaluation_results
```

If a per-model CSV does not yet exist, `json2csv_create.py` or `json2csv_edit.py` creates its template automatically from the metadata. Creation is grouped by image count, concept combination, and subclass combination; editing is grouped by task type and subtype. Q1/Q4 and editing Q3 retain the authoritative evaluators' raw 0–100 scores, while Q2 concept consistency is the fraction of `yes` answers mapped to 0–10. Creation Q3 retains UniPercept's raw outputs.

Run the complete offline contract test without an API key, model, GPU, or network access:

```bash
python evaluation/self_test.py
```

Run each script with `--help` for its output layout and optional overrides. API calls may incur cost; test on a small metadata subset before a full benchmark run.
