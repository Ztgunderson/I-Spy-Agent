# CLEF unseen-attack file probe

This package answers one question: **does Cloudflare's CLEF, trained on human text
classification, transfer to classifying machine-generated files for covert or steganographic
agent behaviour — and does a file-level QLoRA fine-tune beat the same backbone without CLEF's
pretrained head?**

It is a feasibility probe, not a monitoring product. The design document lives in the parent
`FineTunedProjects` repository at
`docs/superpowers/specs/2026-10-16-clef-file-classifier-probe-design.md`.

## Stages

| Stage | What it does | Status |
|---|---|---|
| 0 | Zero-shot CLEF probe over paired synthetic files; every family is unseen | migrated from `src/clef_probe` |
| 1 | Zero-shot schema iteration (`baseline`, `tight`, `attack_framed`) | migrated |
| 2 | File-level QLoRA on `Cloudflare/clef-flash`, `covert` question only, data-efficiency sweep | new |
| 3 | Same recipe on `Qwen/Qwen3.5-9B-Base` with a **fresh** joint head | new |

Stage 2/3 share the dataset, splits, LoRA recipe, optimizer budget, and evaluation protocol.
Only the backbone and head initialisation differ, so a Stage 2-vs-3 gap is attributable to the
pretrained CLEF representation/head rather than to a recipe change.

## Splits and holdouts

- **U1:** held-out synthetic pairs from families seen in training.
- **U2:** the entire `multi_file` family, never trained on.
- **U3:** real `straj` attack runs, never trained on.
- Training data also includes real LinuxArena attack files, real honest files, and four fixed
  hard-negative archetypes (design decision D6).
- Synthetic clean/covert pairs and files from one real trajectory never cross splits.
- Monitor re-runs of the same trajectory are collapsed before splitting, so a trajectory is not
  counted once per monitor.

**Chain-of-thought is excluded.** `actions[i].agent_reasoning` and `extended_thinking` are
attacker-controlled. `corpus.drop_cot` strips them before anything downstream sees the action,
and `extract_trajectory` asserts they are gone. A pipeline that reads them is not testing the
hypothesis.

## Models and hardware

| Stage | Model | Notes |
|---|---|---|
| 0/1 | `meossistant/clef-flash-4bit` | pre-quantised; the local RTX 5060 Laptop (7.93 GiB) ran at ~26 s/example |
| 2 | `Cloudflare/clef-flash` | bf16 release, quantised to NF4 at load time (QLoRA) |
| 3 | `Qwen/Qwen3.5-9B-Base` | same NF4 QLoRA recipe, fresh head |

Override the repository ids with `CLEF_MODEL_ID`, `CLEF_BASE_MODEL`, or
`CLEF_4BIT_MODEL_ID`. Stage 3 needs the CLEF model code (`joint_schema_model.py`) and
`joint_head_config.json`; it deliberately does **not** load `joint_head.safetensors`.

Expect roughly 8–12 GiB of VRAM for Stage 2/3 with NF4, LoRA on the last four layers, and
gradient checkpointing. A CUDA GPU is required; the dataset, metrics, and tests run on CPU.

## Data provenance

`data/linuxarena_files.jsonl` is extracted from `anonymouslinuxarena/linuxarena-public`
(CC-BY-4.0). Only `text_editor` `create` actions and bash heredocs yield a complete file body.
`str_replace` is a patch and is counted but not emitted. Failed attacks (`label=None`) are
dropped rather than treated as negatives.

Rebuild the dataset with:

```bash
python -m experiment.clef_unseen_attack.data \
  --corpus ../data/linuxarena_files.jsonl \
  --n-per-family 100
```

The builder writes `data/clef_unseen_attack/dataset.jsonl` and `manifest.json` (ignored by git)
with source/split counts and a SHA-256 of the dataset file.

## Run order

All commands run from the `I-Spy-Agent` repository root.

```bash
# CPU-only checks
python -m experiment.clef_unseen_attack.data --smoke
python -m pytest tests/experiment -v

# Stage 0/1 (migrated; needs the 4-bit checkpoint and a GPU)
python -m experiment.clef_unseen_attack.probe --n-per-family 6

# Build the Stage 2/3 dataset
python -m experiment.clef_unseen_attack.data --n-per-family 100

# Stage 2/3 training sweep
python -m experiment.clef_unseen_attack.train stage2 --sizes 50 100 200 400
python -m experiment.clef_unseen_attack.train stage3 --sizes 50 100 200 400

# Subset smoke of the trainer (still needs a GPU)
python -m experiment.clef_unseen_attack.train smoke --stage stage2

# Evaluation (threshold calibrated on validation negatives only)
python -m experiment.clef_unseen_attack.evaluate \
  --stage-dir runs/clef_unseen_attack/stage2/size_400
python -m experiment.clef_unseen_attack.evaluate --zero-shot
```

The notebooks under `notebooks/` drive the same functions and are generated from
`scripts/build_notebook.py` and `scripts/build_results_viz.py`:

```bash
python -m experiment.clef_unseen_attack.scripts.build_notebook
python -m experiment.clef_unseen_attack.scripts.build_results_viz
```

`clef_unseen_attack_probe.ipynb` contains Stages 0–3; `clef_stage0_results.ipynb` is read-only
and never loads a model.

## Reporting

Retrieve metrics from `evaluate.report` (also written to
`experiment/clef_unseen_attack/results/stage23_metrics.csv`):

- paired per-family AUC for `train`/`val`/`u1`;
- AUC for U2 (held-out family) and U3 (real attacks vs real honest files);
- per-source FPR on held-out honest runs and the four hard-negative archetypes — never pooled;
- TPR at 1%, 4%, 5%, and 10% FPR, with the threshold calibrated on validation negatives;
- bootstrap CIs resampling pairs or trajectories, not individual files.

The operating threshold is selected on validation negatives and applied unchanged to U1/U2/U3.
Choosing it on the evaluation split would leak the test set.

## Limitations

- **Single seed.** No seed variance is reported; treat small differences across sizes or stages
  as provisional.
- **Four hard negatives.** The hard-negative FPR is an anecdote, not a deployment estimate.
- **Real-corpus holdout size.** U3 is limited by available `straj` runs and honest holdouts.
- **Synthetic construction.** The generator's covert variants differ from their clean pairs by
  nameable edits; high paired AUC can reflect template recognition rather than composition.
- **No Stage 0b.** The frozen hidden-state probe remains conditional in the design.
- Stage 2/3 have not been executed in this workspace: `torch.cuda.is_available()` is `False`.
  The tests cover data, metrics, sampling, and control flow; the training commands above are
  GPU-only.

## Files and git hygiene

- Tracked results: `experiment/clef_unseen_attack/results/` (the experimental record).
- Ignored: `data/`, `runs/`, caches, and run logs.
- Do not commit agent configs, memory files, or design docs; they stay outside the product
  tree.
