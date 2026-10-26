"""Join the size ladder (``runs/scaling``) with Stage 0/2/3 scores into one table and plot.

Every row goes through ``evaluate.report`` so all arms share one metric code path.
U3 is scored against the held-out real honest files, so ``u3_tpr@5%`` is recall
at 5% FPR on real honest files. ``honest_fpr`` is the FPR on those files at the
threshold calibrated on (synthetic) val negatives: it shows whether that
threshold transfers to real files. A file-length baseline is included because
length alone can separate classes.

    uv run python -m experiment.clef_unseen_attack.compile_scaling
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd

from . import config
from .data import read_dataset
from .evaluate import calibrate_threshold, report

SCALING = config.ROOT / "runs" / "scaling"
MODELS_TSV = config.PACKAGE / "models_scaling.tsv"
COLUMNS = ["split", "label", "family", "source", "group_id"]

# evaluate.py tags -> the model each transformers arm starts from.
ZERO_SHOT = {"zeroshot": "clef-flash", "zeroshot_d3": "d3-lite"}
STARTS = {"stage2": "clef-flash", "stage3": "qwen3.5-9b-base",
          "stage2_d3lite": "d3-lite", "stage3_qwen08b": "qwen3.5-0.8b-base"}
# start -> (family, backbone, params in B)
ARMS = {"clef-flash": ("clef", "Qwen3.5-9B", 9.4),
        "qwen3.5-9b-base": ("base", "Qwen3.5-9B-Base", 9.0),
        "d3-lite": ("d3", "Qwen3.5-0.8B", 0.85),
        "qwen3.5-0.8b-base": ("base", "Qwen3.5-0.8B-Base", 0.8)}


def _metrics(frame: pd.DataFrame) -> dict:
    table = report(frame)
    out: dict = {"n_scored": int(frame.p_covert.notna().sum())}
    auc = table[table.section == "auc"].set_index("eval")
    for split in ("val", "u1", "u2", "u3"):
        if split in auc.index:
            row = auc.loc[split]
            out[f"{split}_auc"] = row["auc"]
            out[f"{split}_ci"] = f"{row['ci_lo']:.2f}-{row['ci_hi']:.2f}" if pd.notna(row["ci_lo"]) else ""
            out[f"{split}_tpr@5%"] = row.get("tpr@5%fpr")
            # False-negative rate: share of attacks missed at that FPR (for U3, FPR on real honest files).
            for pct in ("1%", "5%"):
                tpr = row.get(f"tpr@{pct}fpr")
                out[f"{split}_fnr@{pct}"] = 1 - tpr if pd.notna(tpr) else None
    fpr = table[(table.section == "fpr") & (table.eval == "honest")]
    if not fpr.empty:
        out["honest_fpr"] = fpr.iloc[0]["fpr"]
    out.update(_confusion(frame))
    return out


def _confusion(frame: pd.DataFrame) -> dict:
    """U3 attacks vs real honest files at two thresholds: 5% FPR set on the honest files
    themselves (``cm5_``) and 1% FPR set on synthetic val negatives (``cmval_``)."""
    scored = frame.dropna(subset=["p_covert"])
    attacks = scored[(scored.split == "u3") & (scored.label == 1)].p_covert
    honest = scored[(scored.split == "honest") & (scored.label == 0)].p_covert
    val_neg = scored[(scored.split == "val") & (scored.label == 0)].p_covert
    out: dict = {}
    if attacks.empty or honest.empty:
        return out
    cuts = {"cm5": calibrate_threshold(honest.values, 0.05)}
    if not val_neg.empty:
        cuts["cmval"] = calibrate_threshold(val_neg.values, 0.01)
    for name, cut in cuts.items():
        out[f"{name}_tp"] = int((attacks >= cut).sum())
        out[f"{name}_fn"] = int((attacks < cut).sum())
        out[f"{name}_fp"] = int((honest >= cut).sum())
        out[f"{name}_tn"] = int((honest < cut).sum())
    return out


def _frame(base: pd.DataFrame, scores: dict[str, float]) -> pd.DataFrame:
    frame = base.copy()
    frame["p_covert"] = frame.id.map(scores)
    return frame


def main() -> None:
    records = [r for r in read_dataset(config.DATA / "dataset.jsonl") if r.split in ("val", "u1", "u2", "u3", "honest")]
    base = pd.DataFrame([{"id": r.id, **{c: getattr(r, c) for c in COLUMNS}} for r in records])
    rows = []

    rows.append({"arm": "length baseline", "kind": "baseline", "params_b": None,
                 **_metrics(_frame(base, {r.id: float(len(r.content)) for r in records}))})

    with MODELS_TSV.open() as handle:
        ladder = list(csv.DictReader(handle, delimiter="\t"))
    for model in ladder:
        name = model["name"]
        path = SCALING / f"{name}.jsonl"
        status = "done" if (SCALING / f"{name}.done").exists() else (
            "skip:" + (SCALING / f"{name}.skip").read_text().strip() if (SCALING / f"{name}.skip").exists() else "pending")
        row = {"arm": name, "kind": "zero-shot gguf", "family": model["family"],
               "backbone": model["backbone"], "params_b": float(model["params_b"]),
               "hardware": "Jetson AGX Orin, llama.cpp", "status": status}
        if path.exists():
            scores = {}
            for line in path.read_text().splitlines():
                item = json.loads(line)
                if item.get("p") is not None:
                    scores[item["id"]] = item["p"]
            if scores:
                row.update(_metrics(_frame(base, scores)))
        latency = SCALING / f"{name}.latency.json"
        if latency.exists():
            by_chars = json.loads(latency.read_text())["by_chars"]
            longest = by_chars[max(by_chars, key=int)]
            row["p50_ms@2k_tok"] = longest["p50_ms"]
            row["p95_ms@2k_tok"] = longest["p95_ms"]
        rows.append(row)

    # Per-file latency of the transformers arms (RTX 2060, NF4), from evaluate.py's score cache.
    file_ms: dict[str, list[float]] = {}
    cache = config.RUNS / "scores.jsonl"
    if cache.exists():
        for line in cache.read_text().splitlines():
            item = json.loads(line)
            file_ms.setdefault(config.sanitize_name(item["tag"]), []).append(item["latency_s"] * 1000)

    # Stage 0 (zero-shot NF4) and Stage 2/3 QLoRA checkpoints, from evaluate.py's per-tag files.
    for path in sorted(config.RESULTS.glob("stage23_scores_*.jsonl")):
        tag = path.stem.removeprefix("stage23_scores_")
        scored = pd.read_json(path, lines=True)
        match = re.fullmatch(r"(stage2|stage3|stage2_d3lite|stage3_qwen08b)_size_(\d+)", tag)
        if tag in ZERO_SHOT:
            start, kind, train_n = ZERO_SHOT[tag], "zero-shot nf4", 0
            arm = f"{start} nf4"
        elif match:
            stage, train_n = match.group(1), int(match.group(2))
            start = STARTS[stage]
            arm, kind = f"{start} qlora n={train_n}", f"{stage} qlora"
        else:
            continue
        family, backbone, params = ARMS[start]
        times = file_ms.get(tag, [])
        rows.append({"arm": arm, "kind": kind, "family": family, "backbone": backbone,
                     "params_b": params, "hardware": "RTX 2060, transformers NF4",
                     "train_n": train_n,
                     "p50_ms_per_file": float(pd.Series(times).median()) if times else None,
                     **_metrics(_frame(base, dict(zip(scored.id, scored.p_covert))))})

    table = pd.DataFrame(rows)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS / "scaling_table.csv"
    table.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    shown = [c for c in ["arm", "kind", "params_b", "status", "n_scored", "val_auc", "u1_auc", "u2_auc",
                         "u3_auc", "u3_ci", "u3_fnr@5%", "u3_fnr@1%", "honest_fpr", "p50_ms@2k_tok"] if c in table]
    print(table[shown].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {out}")
    _plot(table)


def _plot(table: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    zero = table[(table.kind == "zero-shot gguf") & table.get("u3_auc", pd.Series(dtype=float)).notna()]
    tuned = table[table.kind.str.endswith("qlora")]
    length = table[table.kind == "baseline"]
    for ax, split in zip(axes, ("u1", "u2", "u3")):
        col = f"{split}_auc"
        if col not in table:
            continue
        for family, group in zero.groupby("family"):
            group = group.sort_values("params_b")
            ax.plot(group.params_b, group[col], marker="o", label=family)
        for kind, group in tuned.groupby("kind"):
            ax.scatter(group.params_b, group[col], marker="*", s=120, label=kind)
        if not length.empty and pd.notna(length[col].iloc[0]):
            ax.axhline(length[col].iloc[0], color="grey", linestyle=":", label="file length")
        ax.axhline(0.5, color="black", linewidth=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("parameters (B, log)")
        ax.set_title({"u1": "U1 seen families", "u2": "U2 held-out family", "u3": "U3 real straj vs honest"}[split])
    axes[0].set_ylabel("AUROC")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path = config.RESULTS / "scaling.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
