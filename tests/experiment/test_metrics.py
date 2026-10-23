import numpy as np

from experiment.clef_unseen_attack import metrics


def test_auc_basic_and_ties():
    assert metrics.auc([0.1, 0.9], [0, 1]) == 1.0
    assert metrics.auc([0.1, 0.9], [1, 0]) == 0.0
    assert metrics.auc([0.5, 0.5], [0, 1]) == 0.5
    assert metrics.auc([0.5], [1]) is None


def test_tpr_at_fpr_never_exceeds_requested_fpr():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    tpr, threshold = metrics.tpr_at_fpr(scores, labels, 0.0)
    assert tpr == 1.0
    assert metrics.fpr_at_threshold(scores, labels, threshold) == 0.0

    # With the negatives ranked above every positive, the only FPR=0
    # operating point has TPR=0 and a threshold above the highest score.
    scores = [0.9, 0.8, 0.7, 0.6]
    labels = [0, 0, 1, 1]
    tpr, threshold = metrics.tpr_at_fpr(scores, labels, 0.0)
    assert tpr == 0.0
    assert threshold > max(scores)

    fpr, _, _ = metrics.roc_curve(scores, labels)
    assert fpr.max() <= 1.0


def test_paired_auc_respects_groups():
    scores = [0.9, 0.1, 0.8, 0.2]
    labels = [1, 0, 1, 0]
    groups = ["p1", "p1", "p2", "p2"]
    assert metrics.paired_auc(scores, labels, groups) == 1.0


def test_bootstrap_is_deterministic():
    scores = np.linspace(0, 1, 20)
    labels = np.array([0] * 10 + [1] * 10)
    groups = [f"g{i // 2}" for i in range(20)]
    first = metrics.bootstrap_auc(scores, labels, groups=groups, n_boot=50, seed=3)
    second = metrics.bootstrap_auc(scores, labels, groups=groups, n_boot=50, seed=3)
    assert first == second
    assert first["auc"] == 1.0
