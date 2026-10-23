"""Stage 0 and Stage 1: zero-shot CLEF probing over generated covert files.

This is the migrated version of ``src/clef_probe/probe.py`` in the root
FineTunedProjects repository. The model-loading path is unchanged so the
published Stage 0/1 numbers remain reproducible. The scoring loops are
factored into functions that accept an injected ``systemone`` callback, which
lets the test suite run without a GPU.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import config
from .generator import Example, build_dataset
from .metrics import auc
from .schema import SCHEMA, SCHEMAS, SCHEMA_ATTACK, SCHEMA_TIGHT

__all__ = [
    "SCHEMA",
    "SCHEMAS",
    "SCHEMA_ATTACK",
    "SCHEMA_TIGHT",
    "compose_state",
    "auc",
    "load_model",
    "load_model_4bit",
    "run_stage0",
    "run_stage1",
    "_extract_score",
]


def compose_state(ex: Example) -> str:
    return f"PURPOSE: {ex.purpose}\n---\n{ex.content}"


def load_model_4bit():
    """Load the pre-quantised Stage 0/1 checkpoint and the System One callable."""
    from dotenv import load_dotenv

    load_dotenv()
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=config.MODEL_4BIT, token=os.getenv("HF_TOKEN"))
    sys.path.insert(0, path)
    from joint_schema_model import load_release_model, systemone

    model, processor = load_release_model(path, device="cuda")
    return model, processor, systemone


# The original name is kept for notebook cells that still call load_model().
load_model = load_model_4bit


def _extract_score(ans) -> float:
    """Pull a scalar suspiciousness from whatever shape CLEF returns."""
    if isinstance(ans, (int, float)):
        return float(ans)
    if isinstance(ans, dict):
        for key in ("yes", "true", "Yes", "True"):
            if key in ans and isinstance(ans[key], (int, float)):
                return float(ans[key])
        for key in ("probability", "score", "value", "p"):
            if key in ans and isinstance(ans[key], (int, float)):
                return float(ans[key])
        for value in ans.values():
            if isinstance(value, (int, float)):
                return float(value)
        if "probability_distribution" in ans:
            distribution = ans["probability_distribution"]
            if isinstance(distribution, dict):
                for key, value in distribution.items():
                    if key.lower() in ("yes", "true") and isinstance(value, (int, float)):
                        return float(value)
    try:
        return float(ans)
    except (TypeError, ValueError):
        return 0.5


def run_stage0(
    examples: list[Example],
    model,
    processor,
    systemone,
    schema: dict | None = None,
    progress_every: int = 10,
) -> "pd.DataFrame":
    """Score every example with the baseline schema and return one row each.

    ``systemone`` is injected so tests can supply a deterministic fake. The
    returned columns match ``results/stage0_rows.json`` from the original
    probe: family, label, split, p_covert, p_none.
    """
    import pandas as pd

    schema = schema or SCHEMA
    rows = []
    started = time.time()
    for i, example in enumerate(examples):
        response = systemone(
            model,
            processor,
            {"model": "clef-flash", "state": compose_state(example), "questions": schema},
        )
        answers = response["answers"]
        family_probs = answers.get("family", {}).get("probabilities", {}) or {}
        rows.append(
            {
                "family": example.family,
                "label": int(example.label),
                "split": example.split,
                "p_covert": _extract_score(answers["covert"]),
                "p_none": family_probs.get("none", float("nan")),
            }
        )
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - started
            print(
                f"  {i + 1}/{len(examples)}  {elapsed:.0f}s elapsed, "
                f"{elapsed / (i + 1):.1f}s/example",
                flush=True,
            )
    return pd.DataFrame(rows)


def run_stage1(
    examples: list[Example],
    model,
    processor,
    systemone,
    schemas: dict | None = None,
    limit: int = 40,
) -> dict:
    """Score a subset under each schema and return overall/per-family AUC."""
    import pandas as pd

    schemas = schemas or SCHEMAS
    subset = examples[: min(limit, len(examples))]
    results: dict[str, dict] = {}
    for name, schema in schemas.items():
        scores = []
        for example in subset:
            response = systemone(
                model,
                processor,
                {
                    "model": "clef-flash",
                    "state": compose_state(example),
                    "questions": schema,
                },
            )
            scores.append(_extract_score(response["answers"]["covert"]))

        frame = pd.DataFrame(
            {
                "family": [example.family for example in subset],
                "label": [example.label for example in subset],
                "split": [example.split for example in subset],
                "p_covert": scores,
            }
        )
        results[name] = {
            "overall_auc": auc(frame.p_covert.values, frame.label.values),
            "per_family": {
                family: auc(
                    frame.p_covert[(frame.family == family).values].values,
                    frame.label[(frame.family == family).values].values,
                )
                for family in sorted(frame.family.unique())
            },
            "rows": frame.to_dict("records"),
        }
        print(f"{name:<14} overall AUC={results[name]['overall_auc']:.4f}")
    return results


def _write_artifact(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", indent=2)


def main() -> None:
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-family", type=int, default=config.N_PER_FAMILY_STAGE0)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--stage1-limit", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=config.RESULTS)
    args = parser.parse_args()

    config.ensure_dirs()
    examples = build_dataset(n_per_family=args.n_per_family, seed=args.seed)
    print(
        f"dataset: {len(examples)} examples "
        f"(n_per_family={args.n_per_family}, seed={args.seed})",
        flush=True,
    )

    model, processor, systemone = load_model_4bit()
    stage0 = run_stage0(examples, model, processor, systemone)
    _write_artifact(stage0, args.output_dir / "stage0_rows.json")
    print(f"AUC = {auc(stage0.p_covert.values, stage0.label.values):.4f}")
    print(f"wrote {args.output_dir / 'stage0_rows.json'}")

    stage1 = run_stage1(
        examples, model, processor, systemone, limit=args.stage1_limit
    )
    with (args.output_dir / "stage1_schemes.pkl").open("wb") as handle:
        pickle.dump(stage1, handle)
    print(f"wrote {args.output_dir / 'stage1_schemes.pkl'}")


if __name__ == "__main__":
    main()
