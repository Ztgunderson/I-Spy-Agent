"""Static figures for the Apart report: main story (fig1-fig4) and appendix (figA1-figA5).

Reads only result files written by the experiment: ``results/scaling_table.csv``
(``compile_scaling``), the per-file score caches (``runs/scaling/*.jsonl`` and
``results/stage23_scores_*.jsonl``), and the NARCBench player caches via
``stegowatch.narc_scaling``. Panels name what is plotted, never a claim; captions are
written in the report. Each figure is saved at dpi 200 and as a small ``_web.png`` copy.

    uv run --with matplotlib python -m experiment.clef_unseen_attack.make_figures
"""

from __future__ import annotations

import csv
import json
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from . import config
from .data import read_dataset
from .evaluate import calibrate_threshold

OUT = config.RESULTS / "figures"
SCALING = config.ROOT / "runs" / "scaling"
MODELS_TSV = config.PACKAGE / "models_scaling.tsv"

# Okabe-Ito, one colour per recipe family.
FAMILY_COLORS = {
    "d3": "#0072B2", "kev": "#E69F00", "clef": "#009E73", "pplx": "#CC79A7",
    "vega": "#56B4E9", "d1": "#D55E00", "aurelis": "#F0E442", "base": "#555555",
}
WRITER_COLORS = {"qwen3_32b": "#0072B2", "gpt_oss_20b": "#E69F00"}
WRITER_NAMES = {"qwen3_32b": "Qwen3-32B writer", "gpt_oss_20b": "GPT-OSS-20B writer"}
BACKBONES = [(0.8, "Qwen3.5\n0.8B"), (2.0, "Qwen3.5\n2B"), (4.0, "Qwen3.5\n4B"),
             (9.0, "Qwen3.5\n9B"), (27.0, "Qwen3.8\n27B")]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 100,
})


# ---------------------------------------------------------------- data


def scaling_table() -> pd.DataFrame:
    table = pd.read_csv(config.RESULTS / "scaling_table.csv")
    table["family"] = table["family"].fillna("")
    return table


def _arm_for_tag(tag: str) -> str | None:
    """Map an evaluate.py score-file tag to the arm name compile_scaling gives it."""
    if tag == "zeroshot":
        return "clef-flash nf4"
    match = re.search(r"(stage\d)(?:_([a-z0-9]+))?\D+(\d+)$", tag)
    if not match:
        return None
    stage, variant, n = match.group(1), match.group(2), int(match.group(3))
    start = {
        ("stage2", None): "clef-flash", ("stage3", None): "qwen3.5-9b-base",
        ("stage2", "d3lite"): "d3-lite", ("stage3", "qwen08b"): "qwen3.5-0.8b-base",
    }.get((stage, variant))
    return f"{start} qlora n={n}" if start else None


def u3_scores() -> pd.DataFrame:
    """Per-file P(covert) on U3 attacks and held-out honest files, one row per (arm, file)."""
    records = {r.id: r for r in read_dataset(config.DATA / "dataset.jsonl") if r.split in ("u3", "honest")}
    rows = [{"arm": "length baseline", "file": r.id, "attack": bool(r.label), "p": float(len(r.content))}
            for r in records.values()]
    with MODELS_TSV.open() as handle:
        ladder = [m["name"] for m in csv.DictReader(handle, delimiter="\t")]
    for name in ladder:
        if not (SCALING / f"{name}.done").exists():
            continue
        for line in (SCALING / f"{name}.jsonl").read_text().splitlines():
            item = json.loads(line)
            if item.get("p") is not None and item["id"] in records:
                rows.append({"arm": name, "file": item["id"], "attack": bool(records[item["id"]].label),
                             "p": float(item["p"])})
    for path in sorted(config.RESULTS.glob("stage23_scores_*.jsonl")):
        arm = _arm_for_tag(path.stem.removeprefix("stage23_scores_"))
        if arm is None:
            continue
        scored = pd.read_json(path, lines=True)
        for _, item in scored.iterrows():
            if item.id in records and pd.notna(item.p_covert):
                rows.append({"arm": arm, "file": item.id, "attack": bool(records[item.id].label),
                             "p": float(item.p_covert)})
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(["arm", "file"], keep="last")


def ci(text) -> tuple[float, float] | None:
    match = re.match(r"([\d.]+)-([\d.]+)", str(text))
    return (float(match.group(1)), float(match.group(2))) if match else None


def finetune_series(table: pd.DataFrame) -> dict[str, list[tuple[int, str]]]:
    """Starting model -> [(n, arm)], with n = 0 the zero-shot point of the same model."""
    series: dict[str, list[tuple[int, str]]] = {}
    for arm in table.arm:
        match = re.match(r"(.+) qlora n=(\d+)$", str(arm))
        if match:
            series.setdefault(match.group(1), []).append((int(match.group(2)), arm))
    zero = {"clef-flash": "clef-flash nf4", "d3-lite": "d3-lite"}
    for start, points in series.items():
        if start in zero and zero[start] in set(table.arm):
            points.append((0, zero[start]))
        points.sort()
    return series


def gate_cost(scores: pd.DataFrame, arm: str, catch: float) -> tuple[int, int] | None:
    """Honest files flagged at the highest gate that still catches ``catch`` of the attacks."""
    g = scores[scores.arm == arm]
    attacks = np.sort(g[g.attack].p.values)[::-1]
    honest = g[~g.attack].p.values
    if len(attacks) == 0:
        return None
    need = max(1, math.ceil(catch * len(attacks) - 1e-9))
    cut = attacks[need - 1]
    return int((honest >= cut).sum()), len(honest)


def det_curve(scores: pd.DataFrame, arm: str) -> tuple[np.ndarray, np.ndarray]:
    g = scores[scores.arm == arm]
    attacks, honest = g[g.attack].p.values, g[~g.attack].p.values
    cuts = np.r_[np.unique(g.p.values), np.inf]
    fpr = np.array([(honest >= c).mean() for c in cuts])
    fnr = np.array([(attacks < c).mean() for c in cuts])
    order = np.lexsort((fnr, fpr))
    return fpr[order], fnr[order]


def narc_frame():
    """NARCBench player view with every complete model cache as ``<model>__mean``."""
    from stegowatch import narc_scaling
    from stegowatch.evaluate import mean_len
    from stegowatch.load import VIEWS, load_runs

    df = pd.DataFrame(VIEWS["player"](load_runs()))
    df["length baseline__mean"] = -df.text.map(mean_len)
    models = narc_scaling.load_scores(df)
    return df, models, narc_scaling.ladder_meta()


# ---------------------------------------------------------------- drawing helpers


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}_web.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


def letter(ax, text: str) -> None:
    ax.text(-0.12, 1.06, text, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def backbone_bands(ax, top_labels: bool = True) -> None:
    for size, label in BACKBONES:
        ax.axvspan(size / 1.16, size * 1.16, color="#000000", alpha=0.05, lw=0)
        if top_labels:
            ax.text(size, 1.0, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=6, color="#666666")


def log_params_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlim(0.45, 34)
    ax.set_xticks([0.5, 1, 2, 4, 9, 27])
    ax.set_xticklabels(["0.5B", "1B", "2B", "4B", "9B", "27B"])
    ax.minorticks_off()
    ax.set_xlabel("parameters (log scale)")


def marker_style(kind: str, family: str) -> dict:
    color = FAMILY_COLORS.get(family, "#555555")
    if "qlora" in kind:
        return {"marker": "D", "color": color, "mfc": color, "ms": 5.5}
    if kind == "zero-shot nf4":
        return {"marker": "o", "color": color, "mfc": "white", "ms": 5.5}
    return {"marker": "o", "color": color, "mfc": color, "ms": 5.5}


def x_offset(kind: str) -> float:
    return 1.08 if "qlora" in kind else 0.93 if kind == "zero-shot nf4" else 1.0


def family_legend(families, extra=()) -> list:
    handles = [Line2D([], [], color=FAMILY_COLORS.get(f, "#555"), marker="o", lw=1.5, label=f)
               for f in families]
    handles += [Line2D([], [], color="#444", marker="o", mfc="#444", lw=0, label="zero-shot (GGUF)"),
                Line2D([], [], color="#444", marker="o", mfc="white", lw=0, label="zero-shot (NF4)"),
                Line2D([], [], color="#444", marker="D", lw=0, label="QLoRA fine-tuned")]
    return handles + list(extra)


def confusion_cell_grid(fig_title_rows, cells, ncols: int, name: str, title: str) -> None:
    """cells: list of (label, tp, fn, fp, tn)."""
    nrows = math.ceil(len(cells) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.75, nrows * 1.55))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(cells):]:
        ax.axis("off")
    for ax, (label, tp, fn, fp, tn) in zip(axes, cells):
        grid = np.array([[tp, fn], [fp, tn]], float)
        share = grid / grid.sum(axis=1, keepdims=True)
        good = np.array([[1, 0], [0, 1]], bool)
        for i in range(2):
            for j in range(2):
                color = "#009E73" if good[i, j] else "#D55E00"
                ax.add_patch(Rectangle((j, 1 - i), 1, 1, color=color, alpha=0.12 + 0.6 * share[i, j], lw=0))
                ax.text(j + 0.5, 1.5 - i, f"{int(grid[i, j])}", ha="center", va="center", fontsize=8)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["flagged", "passed"], fontsize=6)
        ax.set_yticks([1.5, 0.5])
        ax.set_yticklabels(["attack", "honest"], fontsize=6)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(label, fontsize=7)
    fig.suptitle(title, fontsize=8.5, y=1.0)
    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- figures


def fig1(table: pd.DataFrame) -> None:
    arms = table[table.kind.isin(["zero-shot gguf", "zero-shot nf4", "stage2 qlora"]) | table.kind.str.contains("qlora", na=False)]
    arms = arms[arms.u3_auc.notna() & arms.params_b.notna()]
    length = table[table.kind == "baseline"].iloc[0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, col, ylabel, lims in ((a, "u3_auc", "U3 AUROC", (0.3, 1.0)),
                                  (b, "u3_fnr@5%", "missed attacks (FNR) at 5% FPR", (-0.02, 1.05))):
        backbone_bands(ax)
        zs = arms[arms.kind == "zero-shot gguf"]
        for family, group in zs.groupby("family"):
            group = group.sort_values("params_b")
            ax.plot(group.params_b, group[col], color=FAMILY_COLORS.get(family, "#555"), lw=1.2, alpha=0.6)
        for _, row in arms.iterrows():
            style = marker_style(row.kind, row.family)
            x = row.params_b * x_offset(row.kind)
            if col == "u3_auc" and (c := ci(row.u3_ci)):
                ax.plot([x, x], c, color=style["color"], alpha=0.35, lw=1.5)
            ax.plot([x], [row[col]], ls="none", mec=style["color"], **style)
        ax.axhline(length[col], color="#888", ls="--", lw=1)
        ax.text(0.47, length[col], " file length", va="bottom", fontsize=6, color="#666")
        if col == "u3_auc":
            ax.axhline(0.5, color="#000", lw=0.5)
        log_params_axis(ax)
        ax.set_ylim(*lims)
        ax.set_ylabel(ylabel)
    a.set_title("U3: real attacks vs real honest files", pad=18)
    b.set_title("U3 at 5% FPR on real honest files", pad=18)
    letter(a, "a")
    letter(b, "b")
    families = [f for f in FAMILY_COLORS if f in set(arms.family)]
    b.legend(handles=family_legend(families), loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save(fig, "fig1_linuxarena_detection")


def fig2() -> None:
    narc = pd.read_csv(config.ROOT / "results" / "narc_scaling.csv")
    rows = narc[narc.arm == "mean"].copy()
    order = rows.drop_duplicates("model").sort_values(["params_b", "model"]).model.tolist()
    base = narc[narc.arm == "baseline"].set_index("writer")
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    width = 0.19
    writers = ["qwen3_32b", "gpt_oss_20b"]
    for i, model in enumerate(order):
        for j, writer in enumerate(writers):
            row = rows[(rows.model == model) & (rows.writer == writer)].iloc[0]
            for k, (value, lo, hi, hatch) in enumerate((
                    (row.auroc, row.ci_lo, row.ci_hi, None),
                    (row.auroc_lenmatched, row.lm_ci_lo, row.lm_ci_hi, "////"))):
                x = i + (j * 2 + k - 1.5) * width
                ax.bar(x, value, width * 0.92, color=WRITER_COLORS[writer], alpha=0.85 if not hatch else 0.35,
                       hatch=hatch, edgecolor=WRITER_COLORS[writer], lw=0.6)
                ax.errorbar(x, value, yerr=[[value - lo], [hi - value]], color="#333", lw=0.7, capsize=1.5)
    for writer in writers:
        ax.axhline(base.loc[writer, "auroc"], color=WRITER_COLORS[writer], ls="--", lw=1)
        ax.axhline(base.loc[writer, "auroc_lenmatched"], color=WRITER_COLORS[writer], ls=":", lw=1)
    ax.axhline(0.5, color="#000", lw=0.5)
    meta = rows.drop_duplicates("model").set_index("model")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{m}\n{meta.loc[m, 'params_b']:g}B" for m in order], fontsize=7)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("AUROC")
    ax.set_title("NARCBench-Stego, player view: is this player the counter?")
    handles = [Patch(color=WRITER_COLORS[w], label=WRITER_NAMES[w]) for w in writers]
    handles += [Patch(facecolor="white", edgecolor="#555", label="all players"),
                Patch(facecolor="white", edgecolor="#555", hatch="////", label="length-matched set"),
                Line2D([], [], color="#555", ls="--", label="message length, all players"),
                Line2D([], [], color="#555", ls=":", label="message length, length-matched")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    save(fig, "fig2_narcbench")


def fig3(table: pd.DataFrame, scores: pd.DataFrame) -> None:
    series = finetune_series(table)
    if not series:
        return
    colors = {"clef-flash": FAMILY_COLORS["clef"], "d3-lite": FAMILY_COLORS["d3"],
              "qwen3.5-9b-base": FAMILY_COLORS["base"], "qwen3.5-0.8b-base": "#999999"}
    ns = sorted({n for pts in series.values() for n, _ in pts})
    xpos = {n: i for i, n in enumerate(ns)}
    rows = table.set_index("arm")
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
    a, b, c, d = axes
    for start, points in series.items():
        color = colors.get(start, "#555")
        xs = [xpos[n] for n, _ in points]
        aucs = [rows.loc[arm, "u3_auc"] for _, arm in points]
        a.plot(xs, aucs, color=color, marker="D", lw=1.5, label=start)
        for x, (_, arm) in zip(xs, points):
            if (bounds := ci(rows.loc[arm, "u3_ci"])):
                a.plot([x, x], bounds, color=color, alpha=0.35, lw=1.5)
        for catch, ls in ((0.8, "-"), (0.9, "--")):
            costs = [gate_cost(scores, arm, catch) for _, arm in points]
            b.plot(xs, [c_[0] / c_[1] if c_ else np.nan for c_ in costs], color=color, marker="D", ls=ls, lw=1.5)
        fprs = []
        for _, arm in points:
            fp, tn = rows.loc[arm, "cmval_fp"], rows.loc[arm, "cmval_tn"]
            fprs.append(fp / (fp + tn) if pd.notna(fp) else np.nan)
        c.plot(xs, fprs, color=color, marker="D", lw=1.5)
    for ax in (a, b, c):
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels(["0\n(zero-shot)" if n == 0 else str(n) for n in ns])
        ax.set_xlabel("training files")
    a.set_ylim(0.5, 1.0)
    a.set_ylabel("U3 AUROC")
    a.set_title("U3 AUROC")
    b.set_ylim(0, 1.0)
    b.set_ylabel("share of honest files flagged")
    b.set_title("False alarms at fixed catch rate")
    b.legend(handles=[Line2D([], [], color="#555", ls="-", label="gate catches 80% of attacks"),
                      Line2D([], [], color="#555", ls="--", label="gate catches 90% of attacks")],
             loc="lower left", frameon=False)
    c.set_ylim(0, 1.05)
    c.set_ylabel("share of honest files flagged")
    c.set_title("Threshold set at 1% FPR on synthetic val")
    a.legend(loc="lower right", frameon=False)
    # (d) score distributions per arm
    arms = [arm for points in series.values() for _, arm in points]
    data, positions, facecolors = [], [], []
    for i, arm in enumerate(arms):
        g = scores[scores.arm == arm]
        for k, attack in enumerate((False, True)):
            values = np.log10(np.clip(g[g.attack == attack].p.values, 1e-6, None))
            data.append(values)
            positions.append(i * 3 + k)
            facecolors.append("#D55E00" if attack else "#56B4E9")
    parts = d.boxplot(data, positions=positions, widths=0.8, patch_artist=True, showfliers=False,
                      medianprops={"color": "#222"})
    for patch, color in zip(parts["boxes"], facecolors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    d.set_xticks([i * 3 + 0.5 for i in range(len(arms))])
    d.set_xticklabels([arm.replace(" qlora ", "\n").replace(" nf4", "\nzero-shot") for arm in arms],
                      fontsize=6)
    d.set_ylabel("log10 P(covert)")
    d.set_title("Score distributions, U3 attacks vs honest files")
    d.legend(handles=[Patch(color="#56B4E9", alpha=0.6, label="honest files"),
                      Patch(color="#D55E00", alpha=0.6, label="U3 attacks")], loc="lower right",
             frameon=False)
    for ax, l in zip(axes, "abcd"):
        letter(ax, l)
    fig.tight_layout()
    save(fig, "fig3_finetuning")


def fig4(table: pd.DataFrame) -> None:
    zs = table[(table.kind == "zero-shot gguf") & table.family.isin(["kev", "d3"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, col, title in ((a, "u3_auc", "U3: real held-out attacks vs real honest files"),
                           (b, "val_auc", "Synthetic validation split")):
        backbone_bands(ax)
        for family, group in zs.groupby("family"):
            group = group[group[col].notna()].sort_values("params_b")
            color = FAMILY_COLORS[family]
            ax.plot(group.params_b, group[col], color=color, marker="o", lw=1.5, label=family)
            for _, row in group.iterrows():
                ax.annotate(row.arm, (row.params_b, row[col]), textcoords="offset points", xytext=(4, 3),
                            fontsize=5.5, color=color)
            if col == "u3_auc":
                for _, row in group.iterrows():
                    if (bounds := ci(row.u3_ci)):
                        ax.plot([row.params_b] * 2, bounds, color=color, alpha=0.3, lw=1.5)
        ax.axhline(0.5, color="#000", lw=0.5)
        log_params_axis(ax)
        ax.set_title(title, pad=18)
    a.set_ylim(0.3, 1.0)
    a.set_ylabel("AUROC")
    a.legend(loc="lower left", frameon=False)
    letter(a, "a")
    letter(b, "b")
    save(fig, "fig4_scale")


def _ordered_arms(table: pd.DataFrame, prefix: str) -> list[pd.Series]:
    have = table[table[f"{prefix}_tp"].notna()]
    first = have[have.kind == "baseline"]
    zero = have[have.kind.str.startswith("zero-shot")].sort_values(["params_b", "arm"])
    tuned = have[have.kind.str.contains("qlora", na=False)].copy()
    tuned["n"] = tuned.arm.str.extract(r"n=(\d+)").astype(float)
    tuned = tuned.sort_values(["arm"]).sort_values(["n"], kind="stable")
    tuned = tuned.sort_values("arm", key=lambda s: s.str.replace(r" n=\d+", "", regex=True), kind="stable")
    return [row for frame in (first, zero, tuned) for _, row in frame.iterrows()]


def figA1_A2(table: pd.DataFrame) -> None:
    for prefix, name, title in (
            ("cm5", "figA1_confusion_honest5", "U3 (35 attacks) vs held-out honest files (319); threshold: 5% FPR on honest files"),
            ("cmval", "figA2_confusion_synthetic1", "U3 (35 attacks) vs held-out honest files (319); threshold: 1% FPR on synthetic val negatives")):
        cells = [(row.arm, row[f"{prefix}_tp"], row[f"{prefix}_fn"], row[f"{prefix}_fp"], row[f"{prefix}_tn"])
                 for row in _ordered_arms(table, prefix)]
        confusion_cell_grid(None, cells, 5, name, title)


def figA3(table: pd.DataFrame, scores: pd.DataFrame) -> None:
    arms = ["length baseline", "d3-lite", "kev-0.8b", "clef-flash nf4", "clef-flash qlora n=200"]
    tuned = table[table.arm.str.startswith("d3-lite qlora")]
    if not tuned.empty:
        arms.append(tuned.sort_values("u3_auc").arm.iloc[-1])
    styles = {"length baseline": {"color": "#888", "ls": "--"}}
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for arm in arms:
        if arm not in set(scores.arm):
            continue
        fpr, fnr = det_curve(scores, arm)
        family = table.set_index("arm").family.get(arm, "")
        style = styles.get(arm, {"color": FAMILY_COLORS.get(family, "#555"),
                                 "ls": "-" if "qlora" not in arm else (0, (4, 1.5))})
        ax.step(fpr, fnr, where="post", lw=1.6, label=arm, **style)
    for x in (0.01, 0.05):
        ax.axvline(x, color="#000", lw=0.5, ls=":")
        ax.text(x, 1.01, f"{x:.0%}", transform=ax.get_xaxis_transform(), ha="center", fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("FPR on real honest files")
    ax.set_ylabel("FNR on U3 attacks (missed)")
    ax.set_title("Missed attacks vs false alarms (U3)", pad=16)
    top = ax.secondary_xaxis("top", functions=(lambda f: f * 1000, lambda r: r / 1000))
    top.set_xlabel("honest files sent to review per 1,000", fontsize=7)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "figA3_fnr_vs_fpr")


def _heat(ax, matrix: pd.DataFrame, meta: dict, title: str) -> None:
    im = ax.imshow(matrix.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(matrix)))
    ax.set_xticklabels(matrix.columns, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=6)
    base = {m: meta.get(m, "").removesuffix("-Base") for m in matrix.index}
    for i, a in enumerate(matrix.index):
        for j, b in enumerate(matrix.columns):
            if i != j:
                ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5)
            if i != j and base[a] and base[a] == base[b]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#000", lw=1.2))
    ax.set_title(title)
    return im


def figA4(scores: pd.DataFrame) -> None:
    with MODELS_TSV.open() as handle:
        meta = {r["name"]: r for r in csv.DictReader(handle, delimiter="\t")}
    order_key = lambda m: (float(meta[m]["params_b"]), meta[m]["family"], m)
    u3 = scores[scores.attack & scores.arm.isin(meta.keys())]
    u3_models = sorted(set(u3.arm), key=order_key)
    wide = u3.pivot(index="file", columns="arm", values="p")[u3_models]
    rho_u3 = wide.corr(method="spearman")
    df, models, narc_meta = narc_frame()
    narc_models = sorted(models, key=lambda m: (float(narc_meta[m]["params_b"]), narc_meta[m]["family"], m))
    per_writer = []
    for _, g in df[df.y == 1].groupby("model"):
        per_writer.append(g[[f"{m}__mean" for m in narc_models]].corr(method="spearman").values)
    rho_narc = pd.DataFrame(np.mean(per_writer, axis=0), index=narc_models, columns=narc_models)
    backbone = {m: meta[m]["backbone"] for m in meta}
    backbone.update({m: narc_meta[m]["backbone"] for m in narc_models})
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 4.0))
    im = _heat(a, rho_u3, backbone, "U3 attack files: Spearman between monitors")
    _heat(b, rho_narc, backbone, "NARCBench counters: Spearman (mean of writers)")
    letter(a, "a")
    letter(b, "b")
    fig.colorbar(im, ax=[a, b], shrink=0.7, label="Spearman rho")
    fig.text(0.01, 0.01, "Outlined cells: same backbone (base and -Base variants treated as one).", fontsize=6)
    save(fig, "figA4_monitor_correlation")


def figA5() -> None:
    df, models, meta = narc_frame()
    arms = ["length baseline"] + sorted(models, key=lambda m: (float(meta[m]["params_b"]), m))
    cells = []
    for writer in ("qwen3_32b", "gpt_oss_20b"):
        g = df[df.model == writer]
        for arm in arms:
            s = g[f"{arm}__mean"].values
            y = g.y.values.astype(bool)
            cut = calibrate_threshold(s[~y], 0.05)
            flagged = s >= cut
            cells.append((f"{arm}\n{WRITER_NAMES[writer]}", int((flagged & y).sum()), int((~flagged & y).sum()),
                          int((flagged & ~y).sum()), int((~flagged & ~y).sum())))
    confusion_cell_grid(None, cells, len(arms), "figA5_narc_confusion",
                        "NARCBench player view (20 counters, 140 other players per writer); threshold: 5% FPR on non-counters")


def main() -> None:
    table = scaling_table()
    scores = u3_scores()
    fig1(table)
    fig2()
    fig3(table, scores)
    fig4(table)
    figA1_A2(table)
    figA3(table, scores)
    figA4(scores)
    figA5()
    for path in sorted(OUT.glob("*.png")):
        print(f"{path.stat().st_size / 1024:7.0f} KB  {path.name}")


if __name__ == "__main__":
    main()
