"""Build the Stage 0/1 results notebook (read-only; no model load).

Regenerate with:
    python -m experiment.clef_unseen_attack.scripts.build_results_viz

The notebook reads only ``experiment/clef_unseen_attack/results/``. It never
imports the model stack, so the original run can be inspected on a CPU-only
machine.
"""

from __future__ import annotations

import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
OUT = PACKAGE / "notebooks" / "clef_stage0_results.ipynb"


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
# Stage 0 / Stage 1 results — CLEF covert-file probe

Visualisation of a completed run migrated from `FineTunedProjects` (source commit
`fddd43f`). **No model is loaded here** — this notebook only reads
`experiment/clef_unseen_attack/results/` and plots. To re-measure, use
`clef_unseen_attack_probe.ipynb`.

## Read this before the numbers

The headline numbers below are **not** evidence that CLEF detects covert files. Two problems:

1. **The `tight` schema was written after watching `baseline` fail.** `baseline` scores
   `env_split` at AUC **0.00** — perfectly inverted. The `tight` instruction was then authored
   to describe that exact pattern in words. Getting 1.00 afterwards is circular, not a
   discovery. The design doc (§9a) explicitly forbids tuning a schema against a family and then
   reporting that family as held out.

2. **AUC 1.00 on paired synthetic data is the expected artifact.** The generator has six
   hand-written templates, and each covert variant differs from its clean pair by a consistent,
   nameable edit. A classifier can separate those perfectly while learning nothing about
   composition. Per family that is 6 positives vs 6 negatives.

What these numbers *do* support: the baseline model is not useless, but it is unreliable and
family-dependent, and it actively inverts on the single most important family.
""",
    ),
    code(
        "load",
        """
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path.cwd()
while not (REPO / "experiment" / "clef_unseen_attack").exists() and REPO != REPO.parent:
    REPO = REPO.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

RESULTS = REPO / "experiment" / "clef_unseen_attack" / "results"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

FAMILIES = ["env_split", "char_array", "encoded", "dead_code", "typosquat", "multi_file"]
COVERT, BENIGN = "#c44e52", "#4c72b0"
""",
    ),
    md("s0", "\n## Stage 0 — baseline schema, per family\n"),
    code(
        "s0-chart",
        """
stage0 = pd.read_json(RESULTS / "stage0_rows.json")
stage0["kind"] = stage0.label.map({0: "benign", 1: "covert"})

fams = [f for f in FAMILIES if f in set(stage0.family)]
piv = stage0.pivot_table(
    index="family", columns="kind", values="p_covert", aggfunc="mean"
).reindex(fams)

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(piv))
ax.bar(x - 0.2, piv["benign"], 0.4, label="benign (label 0)", color=BENIGN)
ax.bar(x + 0.2, piv["covert"], 0.4, label="covert (label 1)", color=COVERT)
ax.set_xticks(x)
ax.set_xticklabels(piv.index, rotation=20, ha="right")
ax.set_ylabel("mean P(covert)")
ax.set_title("Stage 0 — baseline schema: does the covert file score higher?")
ax.legend()
plt.tight_layout()
plt.show()

piv["separates?"] = np.where(piv["covert"] > piv["benign"], "yes", "NO — inverted")
display(piv.round(4))
""",
    ),
    code(
        "s0-notes",
        """
print("Read this table, not the single AUC number.")
print()
print("  char_array, encoded  -> covert scores clearly higher. Real separation.")
print("  multi_file           -> weak but correct direction.")
print("  dead_code, typosquat -> no separation; both near the floor.")
print("  env_split            -> INVERTED. The covert file scores LOWER than the clean one.")
print()
print("The overall AUC averages families where some work, some do not, and one is backwards.")
""",
    ),
    md("s1", "\n## Stage 1 — schema comparison (40-example subset)\n"),
    code(
        "s1-chart",
        """
with (RESULTS / "stage1_schemes.pkl").open("rb") as handle:
    s1 = pickle.load(handle)

rows = []
for schema, value in s1.items():
    for family, auc_value in value["per_family"].items():
        if auc_value is not None:
            rows.append({"schema": schema, "family": family, "auc": auc_value})
s1df = pd.DataFrame(rows)

piv1 = s1df.pivot(index="family", columns="schema", values="auc").reindex(
    [f for f in FAMILIES if f in set(s1df.family)]
)

fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(piv1))
for i, schema in enumerate(piv1.columns):
    ax.bar(x + (i - 1) * 0.27, piv1[schema], 0.27, label=schema)
ax.axhline(0.5, color="grey", ls="--", lw=1)
ax.text(len(piv1) - 0.5, 0.52, "chance", color="grey", fontsize=9, va="bottom", ha="right")
ax.set_xticks(x)
ax.set_xticklabels(piv1.index)
ax.set_ylim(-0.15, 1.15)
ax.set_ylabel("AUC")
ax.set_title("Stage 1 — per-family AUC by schema (dashed line = chance)")
ax.legend()
plt.tight_layout()
plt.show()

display(piv1.round(3))
""",
    ),
    code(
        "s1-notes",
        """
print("The `baseline` row is the honest one to look at.")
print()
print("  baseline     overall %.3f" % s1["baseline"]["overall_auc"])
for family, value in s1["baseline"]["per_family"].items():
    if value is not None:
        flag = "  <-- perfectly inverted" if value < 0.1 else (
            "  <-- chance" if abs(value - 0.5) < 0.01 else ""
        )
        print("      %-14s %.3f%s" % (family, value, flag))
print()
print("tight and attack_framed hit 1.00 because their instructions were authored to")
print("describe the failing pattern in words, after baseline was observed to miss it.")
print("That is schema-fitting, not transfer. See the header cell.")
""",
    ),
    md("limits", "\n## What this run does and does not establish\n"),
    code(
        "limits-table",
        """
table = pd.DataFrame(
    [
        ["Baseline is not random", "env_split inverted, dead_code/typosquat at chance", "supported by stage 0"],
        ["Family-dependent reliability", "some families separate, others do not", "supported"],
        ["tight/attack_framed transfer", "AUC 1.00", "NOT supported - schema was tuned after seeing the failure"],
        ["Generalises to unseen attacks", "-", "untested; stages 2/3 address this"],
        ["Any false-positive rate", "-", "NOT estimable; 4 hard negatives is an anecdote"],
        ["Deployable threshold", "-", "no; scores cluster near 0.05 with heavy overlap"],
    ],
    columns=["claim", "evidence", "status"],
)
display(table)
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
