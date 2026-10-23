from dataclasses import replace

from experiment.clef_unseen_attack import probe
from experiment.clef_unseen_attack.generator import build_dataset


def _fake_systemone(model, processor, request):
    state = request["state"]
    score = 0.9 if "COVERT_PAYLOAD" in state else 0.1
    return {
        "answers": {
            "covert": {"true": score, "false": 1.0 - score},
            "family": {"probabilities": {"none": 0.2}},
        }
    }


def _examples():
    examples = build_dataset(n_per_family=2, seed=0)
    out = []
    for example in examples:
        marker = "COVERT_PAYLOAD " if example.label == 1 else "CLEAN "
        out.append(replace(example, content=marker + example.content))
    return out


def test_run_stage0_returns_one_row_per_example():
    examples = _examples()
    frame = probe.run_stage0(examples, None, None, _fake_systemone, progress_every=0)
    assert len(frame) == len(examples)
    assert set(["family", "label", "split", "p_covert", "p_none"]).issubset(frame.columns)
    assert probe.auc(frame.p_covert.values, frame.label.values) == 1.0


def test_run_stage1_returns_all_schemas():
    results = probe.run_stage1(
        _examples(), None, None, _fake_systemone, limit=4
    )
    assert set(results) == {"baseline", "tight", "attack_framed"}
    assert "overall_auc" in results["baseline"]
    assert "per_family" in results["baseline"]


def test_extract_score_shapes():
    assert probe._extract_score({"true": 0.7}) == 0.7
    assert probe._extract_score({"probability_distribution": {"yes": 0.4}}) == 0.4
    assert probe._extract_score(0.3) == 0.3
    assert probe._extract_score("not a score") == 0.5
