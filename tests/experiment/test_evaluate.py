import math

import pandas as pd

from experiment.clef_unseen_attack import evaluate


def _row(split, label, score, family="fam", group="g", source="synthetic"):
    return {
        "split": split,
        "label": label,
        "p_covert": score,
        "family": family,
        "group_id": group,
        "source": source,
    }


def _frame():
    rows = []
    for i in range(10):
        rows.append(_row("val", 0, 0.1, group=f"v{i}"))
    for i in range(5):
        rows.append(_row("val", 1, 0.8, group=f"vp{i}"))
    for split in ("train", "u1", "u2"):
        for i in range(5):
            rows.append(_row(split, 1, 0.8, group=f"{split}p{i}"))
            rows.append(_row(split, 0, 0.1, group=f"{split}n{i}"))
    for i in range(5):
        rows.append(_row("u3", 1, 0.8, group=f"u3p{i}"))
    for i in range(10):
        rows.append(_row("honest", 0, 0.1, group=f"h{i}", source="linuxarena_honest"))
    for i in range(4):
        rows.append(_row("train", 0, 0.2, family="hard_neg_config", group=f"hn{i}",
                         source="hard_negative"))
    return pd.DataFrame(rows)


def test_calibrate_threshold_uses_negatives_only():
    negatives = [i / 10 for i in range(10)]
    cut = evaluate.calibrate_threshold(negatives, 0.1)
    assert 0.8 < cut <= 0.9
    assert sum(score >= cut for score in negatives) == 1
    # Ties are broken conservatively: the threshold sits above the tied value,
    # so the realised FPR never exceeds the requested one.
    assert evaluate.calibrate_threshold([0.1] * 10, 0.1) > 0.1
    assert math.isinf(evaluate.calibrate_threshold([], 0.1))


def test_report_has_unseen_and_fpr_sections():
    frame = _frame()
    metrics = evaluate.report(frame, target_fpr=0.01)
    sections = set(metrics.section)
    assert {"auc", "auc_family", "fpr", "tpr_at_fpr", "tpr_at_threshold"}.issubset(sections)
    u3 = metrics[(metrics.section == "auc") & (metrics["eval"] == "u3")]
    assert len(u3) == 1
    assert u3.iloc[0]["auc"] == 1.0
    honest = metrics[(metrics.section == "fpr") & (metrics["eval"] == "honest")]
    assert len(honest) == 1
    assert honest.iloc[0]["fpr"] == 0.0
    tpr = metrics[(metrics.section == "tpr_at_fpr") & (metrics["eval"] == "u2")]
    assert len(tpr) >= 1
    assert tpr.iloc[0]["tpr"] == 1.0


def test_single_class_split_is_not_dropped():
    frame = pd.DataFrame(
        [
            _row("val", 0, 0.1, group="v0"),
            _row("u3", 1, 0.8, group="u0"),
        ]
    )
    metrics = evaluate.report(frame, target_fpr=0.01)
    u3 = metrics[(metrics.section == "auc") & (metrics["eval"] == "u3")]
    assert len(u3) == 1
    assert u3.iloc[0]["auc"] is None or pd.isna(u3.iloc[0]["auc"])
