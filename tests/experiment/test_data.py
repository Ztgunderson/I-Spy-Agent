import json
from pathlib import Path

from experiment.clef_unseen_attack import data


def _row(trajectory_id, run, path, content, label):
    return {
        "trajectory_id": trajectory_id,
        "run": run,
        "purpose": "Fix a failing integration test.",
        "path": path,
        "content": content,
        "tier": "create",
        "label": label,
    }


def _tiny_corpus(tmp_path: Path) -> Path:
    rows = [
        _row("honest-1", "Haiku honest", "/tmp/a.py", "print('honest a')\n" * 4, 0),
        _row("honest-1", "GPT-5 Nano monitor on Opus honest", "/tmp/a2.py", "print('monitor a')\n" * 4, 0),
        _row("honest-2", "Haiku honest", "/tmp/b.py", "print('honest b')\n" * 4, 0),
        _row("attack-1", "Haiku attack", "/tmp/c.py", "print('attack')\n" * 4, 1),
        _row("straj-1", "GPT-5 Nano monitor on straj", "/tmp/d.py", "print('straj')\n" * 4, 1),
        _row("failed-1", "Haiku attack", "/tmp/e.py", "print('failed')\n" * 4, None),
    ]
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_build_datasets_splits_and_manifest(tmp_path):
    records, manifest = data.build_datasets(
        corpus_path=_tiny_corpus(tmp_path), n_per_family=2, seed=0
    )
    splits = {record.split for record in records}
    assert {"train", "val", "u1", "u2", "u3", "honest"}.issubset(splits)

    # Pairs never split.
    by_group = {}
    for record in records:
        if record.source == "synthetic" and record.split != "u2":
            by_group.setdefault(record.group_id, set()).add(record.split)
    assert all(len(value) == 1 for value in by_group.values())

    assert all(record.split == "u2" for record in records if record.family == "multi_file")
    assert all(record.split == "u3" for record in records if "straj" in str(record.meta.get("run")))
    assert not any(record.meta.get("run") == "GPT-5 Nano monitor on Opus honest" for record in records)
    assert not any(record.meta.get("trajectory_id") == "failed-1" for record in records)

    total = sum(
        count for split_counts in manifest["counts"].values() for count in split_counts.values()
    )
    assert total == len(records)


def test_write_read_roundtrip(tmp_path):
    records, manifest = data.build_datasets(
        corpus_path=_tiny_corpus(tmp_path), n_per_family=2, seed=0
    )
    path = data.write_dataset(records, manifest, tmp_path / "out")
    reloaded = data.read_dataset(path)
    assert [record.id for record in records] == [record.id for record in reloaded]
    assert (tmp_path / "out" / "manifest.json").exists()
