# Evaluation

Creation uses PF/CC/IA/PR, while editing uses IF/CC/NERC/PR.

| Creation | Editing | Metric | Backend |
|---|---|---|---|
| PF | IF | Prompt/instruction following | vision API |
| CC | CC | Concept consistency | vision API |
| IA | NERC | IA/IQ/IS for creation; non-editing region consistency for editing | UniPercept / vision API |
| PR | PR | Physical realism | vision API |

Scores are written per model and can be resumed. Generated model names are discovered from each record's `result` mapping; `--model` or `--vlm-model` selects the scoring model, not the system being benchmarked.

## Creation IA: local UniPercept

Creation IA does not use the vision API. It imports the bundled implementation from `unipercept/src`, loads the official checkpoint from `unipercept/ckpt/UniPercept`, and calls UniPercept's native `model.score(...)` interface for aesthetics, quality, and structure/texture.

Install its environment and checkpoint by following [the main deployment guide](../README.md#deploy-unipercept-for-creation-ia). Then run:

```bash
python evaluation/creation/IA_evaluate.py \
  --input outputs/my_model/create.json \
  --dataset-root data/create
```

Useful options are `--model-path`, `--unipercept-root`, `--device cuda:0`, and `--no-flash-attn`. Each model is written to `evaluation/creation/evaluation_results/IA/<model>.json`; existing case IDs are skipped when the command is resumed.

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

If a per-model CSV does not yet exist, `json2csv_create.py` or `json2csv_edit.py` creates its template automatically from the metadata. Creation is grouped by image count, concept combination, and subclass combination; editing is grouped by task type and subtype. Creation PF/PR and editing IF/PR/NERC retain raw 0–100 scores. Creation and editing CC map the fraction of `yes` answers to 0–10. Creation IA retains UniPercept's raw outputs.

Run the complete offline contract test without an API key, model, GPU, or network access:

```bash
python evaluation/self_test.py
```

Run each script with `--help` for its output layout and optional overrides. API calls may incur cost; test on a small metadata subset before a full benchmark run.
