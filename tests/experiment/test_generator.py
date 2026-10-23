import re

from experiment.clef_unseen_attack.generator import HELD_OUT, build_dataset, hard_negatives


def test_paired_construction_stays_paired():
    examples = build_dataset(n_per_family=3, seed=0)
    pairs = {}
    for example in examples:
        if example.family.startswith("hard_neg"):
            continue
        pairs.setdefault(example.pair_id, []).append(example.label)
    assert pairs
    assert all(sorted(labels) == [0, 1] for labels in pairs.values())


def test_held_out_family_is_not_in_training_families():
    examples = build_dataset(n_per_family=2, seed=0)
    multi = [example for example in examples if example.family == "multi_file"]
    assert multi
    assert all(example.split == "held_out" for example in multi)
    assert HELD_OUT == ("multi_file",)


def test_all_ips_use_documentation_ranges():
    examples = build_dataset(n_per_family=5, seed=0)
    ip = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    for example in examples:
        for address in ip.findall(example.content):
            if address in {"0.0.0.0", "127.0.0.1"}:
                continue
            assert address.startswith(
                ("192.0.2.", "198.51.100.", "203.0.113.")
            ), address


def test_hard_negatives_are_benign():
    examples = hard_negatives(__import__("random").Random(0))
    assert len(examples) == 4
    assert all(example.label == 0 for example in examples)
