"""Rank metrics for the file-level covert classifier.

NumPy only: the original probe deliberately avoided scikit-learn, and keeping
that property makes the Stage 0/1 numbers reproducible in the root project's
environment. Ties are handled with average ranks, which is the Mann-Whitney
definition of AUC.
"""

from __future__ import annotations

import numpy as np

from .config import FPR_POINTS


def _as_arrays(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if len(scores) != len(labels):
        raise ValueError(f"score/label length mismatch: {len(scores)} != {len(labels)}")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary (0/1)")
    return scores, labels.astype(int)


def auc(scores, labels) -> float | None:
    """ROC AUC via the rank (Mann-Whitney U) identity.

    Returns ``None`` when either class is empty, which is the honest answer for
    a split that cannot be scored. Ties receive average ranks, so a constant
    score gives 0.5.
    """
    scores, labels = _as_arrays(scores, labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1

    rank_sum = ranks[labels == 1].sum()
    return float(
        (rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    )


def roc_curve(scores, labels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(fpr, tpr, thresholds)`` with one point per distinct score.

    The curve starts at ``(0, 0)`` and thresholds descend, matching the usual
    scikit-learn convention closely enough for fixed-FPR reporting.
    """
    scores, labels = _as_arrays(scores, labels)
    if len(scores) == 0:
        return np.array([0.0]), np.array([0.0]), np.array([np.inf])
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0]), np.array([0.0]), np.array([np.inf])

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)

    # Keep only the last index of each tie group.
    last = np.r_[np.where(np.diff(sorted_scores) != 0)[0], len(sorted_scores) - 1]
    fpr = np.r_[0.0, fps[last] / n_neg]
    tpr = np.r_[0.0, tps[last] / n_pos]
    thresholds = sorted_scores[last]
    return fpr, tpr, thresholds


def tpr_at_fpr(scores, labels, fpr_max: float) -> tuple[float, float]:
    """Best TPR at or below ``fpr_max`` and the threshold that achieves it."""
    fpr, tpr, thresholds = roc_curve(scores, labels)
    eligible = np.where(fpr <= fpr_max + 1e-12)[0]
    if len(eligible) == 0:
        return 0.0, float("inf")
    idx = eligible[np.argmax(tpr[eligible])]
    return float(tpr[idx]), float(thresholds[idx - 1]) if idx > 0 else float("inf")


def fpr_at_threshold(scores, labels, threshold: float) -> float | None:
    """False-positive rate among labelled negatives at a fixed threshold."""
    scores, labels = _as_arrays(scores, labels)
    neg = scores[labels == 0]
    if len(neg) == 0:
        return None
    return float((neg >= threshold).mean())


def paired_auc(scores, labels, groups) -> float | None:
    """AUC over matched pairs, computed as the mean pair-level win rate.

    ``groups`` must identify a clean/covert pair. Groups without both classes
    are ignored. A tie contributes 0.5, so the result has the same scale as
    ordinary AUC.
    """
    scores, labels = _as_arrays(scores, labels)
    if len(groups) != len(scores):
        raise ValueError("groups must have one entry per score")
    by_group: dict[object, list[tuple[float, int]]] = {}
    for score, label, group in zip(scores, labels, groups):
        by_group.setdefault(group, []).append((float(score), int(label)))

    wins = []
    for members in by_group.values():
        pos = [s for s, y in members if y == 1]
        neg = [s for s, y in members if y == 0]
        if not pos or not neg:
            continue
        # A pair has one positive and one negative by construction.
        for p in pos:
            for n in neg:
                wins.append(1.0 if p > n else 0.5 if p == n else 0.0)
    return float(np.mean(wins)) if wins else None


def bootstrap_auc(
    scores,
    labels,
    groups=None,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap AUROC, resampling groups when supplied.

    Group resampling matters because paired synthetic files and files from the
    same trajectory are not independent observations.
    """
    scores, labels = _as_arrays(scores, labels)
    point = auc(scores, labels)
    if point is None:
        return {"auc": None, "ci_lo": None, "ci_hi": None, "n": int(len(scores)),
                "pos": int(labels.sum())}

    rng = np.random.default_rng(seed)
    boots: list[float] = []
    if groups is not None:
        if len(groups) != len(scores):
            raise ValueError("groups must have one entry per score")
        unique = np.asarray(list(dict.fromkeys(groups)), dtype=object)
        for _ in range(n_boot):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([np.where(np.asarray(groups, dtype=object) == g)[0] for g in chosen])
            value = auc(scores[idx], labels[idx])
            if value is not None:
                boots.append(value)
    else:
        for _ in range(n_boot):
            idx = rng.integers(0, len(scores), len(scores))
            value = auc(scores[idx], labels[idx])
            if value is not None:
                boots.append(value)

    if boots:
        alpha = (1.0 - confidence) / 2.0
        lo, hi = np.percentile(boots, [100 * alpha, 100 * (1 - alpha)])
        lo, hi = float(lo), float(hi)
    else:
        lo = hi = None
    return {"auc": point, "ci_lo": lo, "ci_hi": hi, "n": int(len(scores)),
            "pos": int(labels.sum())}


def summarize(
    scores,
    labels,
    groups=None,
    fpr_points: tuple[float, ...] = FPR_POINTS,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """AUC, grouped CI, and TPR at each fixed FPR point."""
    out = bootstrap_auc(scores, labels, groups=groups, n_boot=n_boot, seed=seed)
    for point in fpr_points:
        tpr, threshold = tpr_at_fpr(scores, labels, point)
        pct = f"{round(point * 100):g}%"
        out[f"tpr@{pct}fpr"] = tpr
        out[f"threshold@{pct}fpr"] = threshold
    return out
