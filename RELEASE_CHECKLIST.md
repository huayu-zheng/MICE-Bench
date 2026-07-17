# Release checklist

- [ ] Replace the provisional GitHub organization and Pages URLs if the final owner is not `MICE-Bench`.
- [ ] Create `MICE-Bench/MICE-Bench` on Hugging Face and upload the final data snapshot.
- [ ] Confirm the public snapshot has 1,872 creation and 1,247 editing records.
- [ ] Add per-image provenance/license metadata to the Hugging Face dataset card.
- [ ] Confirm the final paper/arXiv URL and add it to README, project page, and `CITATION.cff`.
- [ ] Confirm that MIT is the intended code license with all contributors.
- [ ] Enable GitHub Pages with `/docs` on the default branch.
- [ ] Run `python scripts/validate_dataset.py` against a fresh download.
- [ ] Run both evaluation launchers with `--dry-run` and one small end-to-end sample.

The local pre-release metadata available during repository preparation contains 1,798 creation and 1,215 editing records (3,013 total), while the paper reports 3,119. Do not publish those local JSON files as the final Hugging Face release until the missing 106 records are reconciled.

