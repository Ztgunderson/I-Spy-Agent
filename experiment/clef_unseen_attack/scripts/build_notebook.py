"""Build the CLEF unseen-attack probe notebook as valid nbformat JSON.

Regenerate with:
    python -m experiment.clef_unseen_attack.scripts.build_notebook

Adapted from ``scripts/build_notebook.py`` in FineTunedProjects. The Stage 0/1
cells keep the original probe's imports and behaviour; the added cells cover
dataset assembly, Stage 2/3 training, and evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
OUT = PACKAGE / "notebooks" / "clef_unseen_attack_probe.ipynb"


def md(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.strip("\n").split("\n"),
    }


def code(cell_id: str, source: str) -> dict:
    lines = source.strip("\n").split("\n")
    lines = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


CELLS = [
    md(
        "title",
        """
# CLEF unseen-attack probe — covert file classification

**Question:** does Cloudflare CLEF (trained on *human* text classification) transfer to
classifying *machine-generated files* for covert / steganographic agent behaviour, and how
does that compare with a QLoRA fine-tune of the same backbone?

**Scope:** a feasibility probe, not a monitoring product. See
`docs/superpowers/specs/2026-10-16-clef-file-classifier-probe-design.md` in the parent repo.

**Hardware:** the local Stage 0 run used 4-bit `clef-flash` on an RTX 5060 Laptop (7.93 GiB)
at roughly **26 s/example**. Stage 2/3 require a CUDA GPU and downloaded weights; the dataset
and metric cells run on CPU.
""",
    ),
    md(
        "manifest",
        """
## 0. Config and run manifest

Every run is reproducible and diffable from this cell: seed, model ids, git revision, versions.
""",
    ),
    code(
        "manifest-code",
        """
import json, os, platform, subprocess, sys, time
from pathlib import Path

REPO = Path.cwd()
while not (REPO / "experiment" / "clef_unseen_attack").exists() and REPO != REPO.parent:
    REPO = REPO.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from experiment.clef_unseen_attack import config

np.random.seed(config.SEED)

def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"

MANIFEST = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "git_rev": git_rev(),
    "model_stage0": config.MODEL_4BIT,
    "model_stage2": config.MODEL_CLEF,
    "model_stage3": config.MODEL_BASE,
    "seed": config.SEED,
    "n_per_family_stage0": config.N_PER_FAMILY_STAGE0,
    "n_per_family_train": config.N_PER_FAMILY_TRAIN,
    "python": platform.python_version(),
}
try:
    import torch
    MANIFEST.update(
        {
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if torch.cuda.is_available()
            else None,
        }
    )
except ImportError:
    pass
print(json.dumps(MANIFEST, indent=2))
""",
    ),
    md(
        "stage0",
        """
## 1. Stage 0 — zero-shot transfer

No training. Ask CLEF a `covert: noul` question over each generated file, take the returned
probability, and compute AUC against labels. Every family here is an unseen attack, which is
the point: this is the U1/U2 reading from the design.

The generator produces matched clean/covert pairs, so every `pair_id` must contain exactly one
label 0 and one label 1.
""",
    ),
    code(
        "stage0-data",
        """
from collections import Counter

from experiment.clef_unseen_attack.generator import FAMILIES, HELD_OUT, build_dataset

examples = build_dataset(n_per_family=config.N_PER_FAMILY_STAGE0, seed=config.SEED)
print(f"{len(examples)} examples | families={list(FAMILIES)} | held_out={list(HELD_OUT)}")

pair_counts = Counter((e.pair_id, e.label) for e in examples if e.pair_id)
assert all(v == 1 for v in pair_counts.values()), "paired construction violated"
assert all(e.label in (0, 1) for e in examples), "bad label"

for e in examples[:1]:
    print("=" * 70)
    print(e.content[:400])
""",
    ),
    code(
        "stage0-model",
        """
# Inference only, no gradients. Stage 0/1 use the pre-quantised release so the
# original 26 s/example numbers remain reproducible.
from experiment.clef_unseen_attack.probe import load_model

model, processor, systemone = load_model()
print("model loaded")
""",
    ),
    code(
        "stage0-run",
        """
from experiment.clef_unseen_attack import probe

stage0 = probe.run_stage0(examples, model, processor, systemone)
stage0.head()
""",
    ),
    code(
        "stage0-report",
        """
import pandas as pd

print("=== OVERALL (zero-shot) ===")
print(f"AUC = {probe.auc(stage0.p_covert.values, stage0.label.values):.4f}")
print(f"AUC using P(family=none) inverted = "
      f"{probe.auc(-stage0.p_none.values, stage0.label.values):.4f}")

print("\\n=== PER FAMILY — every family is unseen (no training) ===")
for fam in sorted(stage0.family.unique()):
    mask = (stage0.family == fam).values
    value = probe.auc(stage0.p_covert.values[mask], stage0.label.values[mask])
    print(f"  {fam:<22} n={mask.sum():>3}  AUC={'n/a' if value is None else round(value, 3)}")

print("\\n=== HARD NEGATIVES — clean files that use the flagged techniques ===")
hard = stage0[stage0.family.str.startswith("hard_neg")]
for fam, group in hard.groupby("family"):
    print(f"  {fam:<22} P(covert)={group.p_covert.iloc[0]:.4f}  (all labelled benign)")

config.RESULTS.mkdir(parents=True, exist_ok=True)
stage0.to_json(config.RESULTS / "stage0_rows.json", orient="records", indent=2)
print(f"\\nwrote {config.RESULTS / 'stage0_rows.json'}")
""",
    ),
    md(
        "stage1",
        """
## 2. Stage 1 — schema iteration (still zero-shot)

If the instruction is the bottleneck rather than the representation, a better schema buys a
lot for free. **Discipline:** held-out-family results must not be tuned against. If we change
the schema after reading them, they are no longer held out — relabel and say so.
""",
    ),
    code(
        "stage1-run",
        """
import pickle

from experiment.clef_unseen_attack.schema import SCHEMAS

stage1 = probe.run_stage1(examples, model, processor, systemone, schemas=SCHEMAS, limit=40)
with (config.RESULTS / "stage1_schemes.pkl").open("wb") as handle:
    pickle.dump(stage1, handle)
print(f"wrote {config.RESULTS / 'stage1_schemes.pkl'}")
""",
    ),
    code(
        "stage1-report",
        """
for name, result in stage1.items():
    print(f"\\n--- {name} (overall AUC {result['overall_auc']:.4f}) ---")
    for family, value in result["per_family"].items():
        print(f"    {family:<22} {'n/a' if value is None else round(value, 3)}")
""",
    ),
    md(
        "dataset",
        """
## 3. Dataset assembly for Stages 2 and 3

One file-level dataset, four sources: seen synthetic families (train/val/U1), the held-out
`multi_file` family (U2), real LinuxArena attacks with `straj` held out (U3), and real honest
runs for per-source FPR. Synthetic pairs and real trajectories are never split.

**Chain-of-thought is deliberately dropped at read time.** `agent_reasoning` and
`extended_thinking` are attacker-controlled, so a pipeline that reads them is not testing the
hypothesis. `corpus.extract_trajectory` strips them and asserts they are gone.
""",
    ),
    code(
        "dataset-build",
        """
from experiment.clef_unseen_attack.data import (
    build_datasets,
    read_dataset,
    summarize,
    write_dataset,
)

records, manifest = build_datasets(
    corpus_path=REPO / "data" / "linuxarena_files.jsonl"
    if (REPO / "data" / "linuxarena_files.jsonl").exists()
    else config.ROOT / "data" / "linuxarena_files.jsonl",
    n_per_family=config.N_PER_FAMILY_TRAIN,
    seed=config.SEED,
)
path = write_dataset(records, manifest, config.DATA)
print(summarize(records).to_string(index=False))
print(f"\\nwrote {path}")
print(json.dumps(manifest["counts"], indent=2))

# Round-trip check: every split, source, and label survives the write.
assert read_dataset(path) == records
""",
    ),
    md(
        "stage2",
        """
## 4. Stage 2 — file-level QLoRA on CLEF

Trains **only** the `covert` question; `family` and `plausible_cover` remain zero-shot
readouts. The released joint head is the starting point, and the data-efficiency sweep uses
the same recipe at every size.

> Requires a CUDA GPU and the `Cloudflare/clef-flash` weights. The commands are recorded even
> when this notebook is executed on a CPU-only machine.
""",
    ),
    code(
        "stage2-train",
        """
from experiment.clef_unseen_attack.train import STAGES, sample_train, train_one

stage2_sweep = []
for size in config.TRAIN_SIZES:
    subset = sample_train(records, size=size, seed=STAGES["stage2"].seed)
    stage2_sweep.append(
        train_one(STAGES["stage2"], subset, config.stage_dir("stage2", size))
    )
stage2_sweep
""",
    ),
    md(
        "stage3",
        """
## 5. Stage 3 — base-model control with a fresh head

The control asks whether any gain comes from CLEF's pretrained head rather than the backbone.
Stage 3 loads the Qwen base model, builds a **fresh** joint head from the CLEF head config, and
applies the identical data, LoRA recipe, and evaluation protocol.
""",
    ),
    code(
        "stage3-train",
        """
stage3_sweep = []
for size in config.TRAIN_SIZES:
    subset = sample_train(records, size=size, seed=STAGES["stage3"].seed)
    stage3_sweep.append(
        train_one(STAGES["stage3"], subset, config.stage_dir("stage3", size))
    )
stage3_sweep
""",
    ),
    md(
        "evaluate",
        """
## 6. Evaluation — U1, U2, U3, and per-source FPR

The operating threshold is calibrated on validation negatives only and then applied unchanged.
The report includes paired per-family AUC, per-source FPR, TPR at 1/4/5/10% FPR, and bootstrap
CIs grouped by pair or trajectory.
""",
    ),
    code(
        "evaluate-run",
        """
from experiment.clef_unseen_attack import evaluate
from experiment.clef_unseen_attack.model import load_file_model

def score_checkpoint(backbone, model_id, stage, size, pretrained_head):
    model = load_file_model(backbone, model_id, pretrained_head=pretrained_head)
    model.load(config.stage_dir(stage, size))
    return evaluate.score_records(
        model,
        records,
        tag=f"{stage}/size_{size}",
        cache_path=config.RUNS / "scores.jsonl",
    )

scores = score_checkpoint("clef", config.MODEL_CLEF, "stage2", 400, pretrained_head=True)
report = evaluate.report(scores, target_fpr=0.01)
report.to_csv(config.RESULTS / "stage23_metrics.csv", index=False)
scores.to_json(config.RESULTS / "stage23_scores.jsonl", orient="records", lines=True)
report
""",
    ),
    md(
        "findings",
        """
## 7. Findings

Record what was measured, what it means, and what it does **not** mean:

- Stage 0/1 are zero-shot. A low AUC only bounds what is available without training.
- Stage 2/3 use a single seed and small real-corpus holdouts; treat differences as provisional.
- The hard-negative set is four fixed archetypes, so its FPR is an anecdote, not a deployment
  estimate.
- U2 (`multi_file`) is the family holdout; U3 is a real-`straj` holdout. Neither enters training.
""",
    ),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(NOTEBOOK, indent=1), encoding="utf-8")
    print(f"wrote {OUT}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
