"""Evaluate Stage 2/3 checkpoints on seen, held-out-family, and held-out-real splits.

The report follows the design's fixed-FPR discipline: the operating threshold
is calibrated on validation negatives, then applied unchanged to U1/U2/U3.
Per-source FPR is never pooled.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, metrics
from .data import FileRecord, read_dataset
from .model import load_file_model
from .schema import SCHEMA, TRAIN_SCHEMA


def calibrate_threshold(negative_scores, target_fpr: float) -> float:
    """Threshold that keeps at most ``target_fpr`` of validation negatives above it.

    Uses the order statistic rather than a quantile because scores are often
    tied. ``floor(target_fpr * n)`` negatives may sit at or above the cut, and
    the threshold is placed just above the next value so that ties cannot push
    the realised FPR past the requested one.
    """
    negatives = np.sort(np.asarray(negative_scores, dtype=float).reshape(-1))
    n = len(negatives)
    if n == 0 or target_fpr <= 0:
        return float("inf")
    if target_fpr >= 1.0:
        return float("-inf")
    allowed = int(np.floor(target_fpr * n))
    if allowed >= n:
        return float("-inf")
    return float(np.nextafter(negatives[n - allowed - 1], np.inf))


def _probability(answer: dict | None, option: str = "true") -> float:
    if not answer:
        return float("nan")
    return float(answer.get(option, float("nan")))


def score_records(
    model,
    records: list[FileRecord],
    *,
    schema: dict | None = None,
    max_length: int = config.MAX_EVAL_LEN,
    cache_path: Path | str | None = None,
    tag: str = "eval",
) -> pd.DataFrame:
    """Score records, resuming from an append-only JSONL cache."""
    schema = schema or TRAIN_SCHEMA
    cache_path = Path(cache_path) if cache_path else config.RUNS / "scores.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[f"{row['tag']}:{row['qhash']}:{row['id']}"] = row

    rows = []
    handle = cache_path.open("a", encoding="utf-8")
    try:
        for i, record in enumerate(records):
            key = f"{tag}:{model.qhash}:{record.id}"
            if key not in done:
                started = time.perf_counter()
                probabilities = model.score(record, schema=schema, max_length=max_length)
                row = {
                    "tag": tag,
                    "qhash": model.qhash,
                    "backbone": model.backbone,
                    "id": record.id,
                    "p": probabilities,
                    "latency_s": round(time.perf_counter() - started, 3),
                    "tokens": len(
                        model.encode(record, schema=schema, max_length=max_length).input_ids
                    ),
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                done[key] = row
                if (i + 1) % 25 == 0:
                    print(f"  scored {i + 1}/{len(records)}", flush=True)

            probabilities = done[key]["p"]
            rows.append(
                {
                    "id": record.id,
                    "label": record.label,
                    "family": record.family,
                    "split": record.split,
                    "source": record.source,
                    "group_id": record.group_id,
                    "p_covert": _probability(probabilities.get("covert")),
                    "p_none": _probability(probabilities.get("family"), "none"),
                }
            )
    finally:
        handle.close()
    return pd.DataFrame(rows)


def _auc_row(frame: pd.DataFrame, eval_name: str, section: str = "auc") -> dict:
    stats = metrics.summarize(
        frame.p_covert.values, frame.label.values, groups=frame.group_id.values
    )
    return {
        "section": section,
        "eval": eval_name,
        "family": "all",
        **stats,
    }


def report(
    scores: pd.DataFrame,
    threshold: float | None = None,
    *,
    target_fpr: float = 0.01,
    fpr_points: tuple[float, ...] = config.FPR_POINTS,
) -> pd.DataFrame:
    """Build a long metrics table from a scored frame.

    Required columns: ``split``, ``label``, ``family``, ``source``,
    ``group_id``, ``p_covert``.
    """
    frame = scores.dropna(subset=["p_covert"]).copy()
    val_neg = frame[(frame.split == "val") & (frame.label == 0)]
    if threshold is None:
        threshold = calibrate_threshold(val_neg.p_covert.values, target_fpr)

    rows: list[dict] = []

    # Paired synthetic splits and the held-out real set.
    for split in ("train", "val", "u1", "u2"):
        subset = frame[frame.split == split]
        if subset.empty:
            continue
        rows.append(_auc_row(subset, split))
        for family, group in subset.groupby("family"):
            if family.startswith("hard_neg"):
                continue
            stats = metrics.summarize(
                group.p_covert.values, group.label.values, groups=group.group_id.values
            )
            stats["paired_auc"] = metrics.paired_auc(
                group.p_covert.values, group.label.values, group.group_id.values
            )
            rows.append(
                {
                    "section": "auc_family",
                    "eval": split,
                    "family": family,
                    **stats,
                }
            )

    # U3: held-out real attacks vs held-out real honest files.
    u3 = frame[frame.split == "u3"]
    honest = frame[frame.split == "honest"]
    if not u3.empty and not honest.empty:
        combined = pd.concat([u3, honest])
        rows.append(_auc_row(combined, "u3"))
    elif not u3.empty:
        rows.append(
            {
                "section": "auc",
                "eval": "u3",
                "family": "all",
                "auc": None,
                "ci_lo": None,
                "ci_hi": None,
                "n": len(u3),
                "pos": int(u3.label.sum()),
            }
        )

    # Per-source FPR at the calibrated threshold.
    for source, subset in (
        ("honest", honest),
        ("hard_neg", frame[frame.source == "hard_negative"]),
    ):
        if subset.empty:
            continue
        negatives = subset[subset.label == 0]
        rows.append(
            {
                "section": "fpr",
                "eval": source,
                "family": "all",
                "threshold": threshold,
                "fpr": metrics.fpr_at_threshold(
                    negatives.p_covert.values, negatives.label.values, threshold
                ),
                "n": len(negatives),
            }
        )

    # Fixed-FPR TPR, with the threshold calibrated on val negatives only.
    attack_sets = {
        "u1": frame[(frame.split == "u1") & (frame.label == 1)],
        "u2": frame[(frame.split == "u2") & (frame.label == 1)],
        "u3": u3[u3.label == 1],
    }
    negatives = val_neg
    for eval_name, positives in attack_sets.items():
        if positives.empty or negatives.empty:
            continue
        for point in fpr_points:
            cut = calibrate_threshold(negatives.p_covert.values, point)
            tpr = float((positives.p_covert.values >= cut).mean())
            rows.append(
                {
                    "section": "tpr_at_fpr",
                    "eval": eval_name,
                    "family": "all",
                    "target_fpr": point,
                    "threshold": cut,
                    "tpr": tpr,
                    "n": len(positives),
                }
            )
        rows.append(
            {
                "section": "tpr_at_threshold",
                "eval": eval_name,
                "family": "all",
                "target_fpr": target_fpr,
                "threshold": threshold,
                "tpr": float((positives.p_covert.values >= threshold).mean()),
                "n": len(positives),
            }
        )

    return pd.DataFrame(rows)


def _load_checkpoint(stage_dir: Path):
    cfg_path = stage_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"no config.json in {stage_dir}")
    cfg = json.loads(cfg_path.read_text())
    backbone = cfg["backbone"]
    model = load_file_model(
        backbone,
        cfg.get("model_dir"),
        pretrained_head=backbone == "clef",
        schema=TRAIN_SCHEMA,
    )
    model.load(stage_dir)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, default=None)
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--backbone", default="clef", help="zero-shot backbone (clef or d3)")
    parser.add_argument("--dataset", type=Path, default=config.DATA / "dataset.jsonl")
    parser.add_argument("--out-dir", type=Path, default=config.RESULTS)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--full-schema", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "u1", "u2", "u3", "honest"],
        help="splits to score; add 'train' for the in-sample AUC (slow, ~2.7k files)",
    )
    args = parser.parse_args()

    records = [record for record in read_dataset(args.dataset) if record.split in args.splits]
    if args.max_records:
        records = records[: args.max_records]
    schema = SCHEMA if args.full_schema else TRAIN_SCHEMA

    if args.zero_shot:
        model = load_file_model(
            args.backbone, config.resolved_model_id(args.backbone), pretrained_head=True,
            schema=schema,
        )
        tag = "zeroshot" if args.backbone == "clef" else f"zeroshot_{args.backbone}"
    else:
        if args.stage_dir is None:
            raise SystemExit("--stage-dir is required unless --zero-shot is set")
        model = _load_checkpoint(args.stage_dir)
        tag = f"{args.stage_dir.parent.name}/{args.stage_dir.name}"

    scores = score_records(
        model,
        records,
        schema=schema,
        cache_path=config.RUNS / "scores.jsonl",
        tag=tag,
    )
    metrics_frame = report(scores, target_fpr=args.target_fpr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # One pair of files per checkpoint so a sweep does not overwrite itself.
    name = config.sanitize_name(tag)
    scores_path = args.out_dir / f"stage23_scores_{name}.jsonl"
    scores.to_json(scores_path, orient="records", lines=True)
    metrics_path = args.out_dir / f"stage23_metrics_{name}.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    pd.set_option("display.width", 220)
    print(metrics_frame.to_string(index=False))
    print(f"wrote {metrics_path} and {scores_path}")


if __name__ == "__main__":
    main()
