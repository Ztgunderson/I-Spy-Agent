from experiment.clef_unseen_attack import data


def test_smoke_build_has_all_required_splits():
    summary = data.smoke_check()
    assert {"train", "val", "u1", "u2", "u3", "honest"}.issubset(set(summary.split))
