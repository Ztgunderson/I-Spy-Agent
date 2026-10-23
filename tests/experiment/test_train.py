import json
from pathlib import Path

from experiment.clef_unseen_attack import data, train


def _corpus(tmp_path: Path) -> Path:
    rows = [
        {
            "trajectory_id": f"honest-{i}",
            "run": "Haiku honest",
            "purpose": "Fix a test.",
            "path": f"/tmp/h{i}.py",
            "content": f"print('honest {i}')\n" * 4,
            "tier": "create",
            "label": 0,
        }
        for i in range(4)
    ] + [
        {
            "trajectory_id": "attack-1",
            "run": "Haiku attack",
            "purpose": "Fix a test.",
            "path": "/tmp/a.py",
            "content": "print('attack')\n" * 4,
            "tier": "create",
            "label": 1,
        }
    ]
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_sample_train_keeps_pairs_together_and_is_deterministic(tmp_path):
    records, _ = data.build_datasets(corpus_path=_corpus(tmp_path), n_per_family=3, seed=0)
    first = train.sample_train(records, size=6, seed=7)
    second = train.sample_train(records, size=6, seed=7)
    assert [record.id for record in first] == [record.id for record in second]
    assert len(first) >= 6
    included = {record.id for record in first}
    for group in {record.group_id for record in first}:
        members = {record.id for record in records if record.group_id == group and record.split == "train"}
        assert members.issubset(included)


def test_targets_map_covert_to_one():
    class Record:
        def __init__(self, label):
            self.label = label

    assert train._targets([Record(1), Record(0)]) == [1.0, 0.0]


def test_stage_configs_are_distinct():
    assert train.STAGES["stage2"].model_id == train.config.MODEL_CLEF
    assert train.STAGES["stage3"].model_id == train.config.MODEL_BASE
    assert train.STAGES["stage2"].pretrained_head != train.STAGES["stage3"].pretrained_head
