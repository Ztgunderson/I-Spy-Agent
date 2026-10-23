# Migrated Stage 0 / Stage 1 results

These artifacts were produced by the original CLEF probe in the parent `FineTunedProjects`
repository at source commit `fddd43f` and copied here unchanged:

- `stage0_rows.json` — one row per generated example with `family`, `label`, `split`,
  `p_covert`, and `p_none`.
- `stage1_schemes.pkl` — per-schema overall and per-family AUC for the Stage 1 subset.

`clef_stage0_results.ipynb` reads these files and plots them without loading a model. The
numbers are **not** evidence of transfer: the `tight` schema was authored after observing the
`baseline` failure on `env_split`, so its AUC 1.00 is schema-fitting, not a held-out result.

Stage 2/3 evaluations write `stage23_metrics.csv` and `stage23_scores.jsonl` into this
directory. Do not edit the migrated files in place; regenerate them with the probe notebook or
`python -m experiment.clef_unseen_attack.probe`.
