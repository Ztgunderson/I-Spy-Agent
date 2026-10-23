"""Assemble the Stage 2 / Stage 3 file-level dataset.

The dataset combines four sources:

* the paired synthetic generator (seen families for train/val/U1);
* the held-out ``multi_file`` family (U2, never trained);
* real LinuxArena attack files (train/val, with ``straj`` held out as U3);
* real LinuxArena honest files (train negatives, with a held-out honest split
  for per-source FPR).

Splitting is group-aware: a synthetic clean/covert pair always stays together,
and every file from one real trajectory stays on one side. Monitor re-runs of
the same trajectory are collapsed before splitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config, corpus
from .generator import Example, build_dataset

DEFAULT_CORPUS = config.ROOT.parent / "data" / "linuxarena_files.jsonl"


@dataclass(frozen=True)
class FileRecord:
    """One file, with its stated purpose and provenance."""

    id: str
    content: str
    purpose: str
    label: int
    family: str
    split: str
    source: str
    group_id: str
    meta: dict = field(default_factory=dict, compare=False)


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:16]


def _counts(total: int, weights: tuple[float, ...]) -> list[int]:
    """Allocate ``total`` groups across weighted splits.

    Every non-zero split gets at least one group when there are enough groups
    to go around. The remainder is assigned deterministically in weight order.
    """
    if total <= 0:
        return [0] * len(weights)
    active = sum(1 for weight in weights if weight > 0)
    if total < active:
        return [1 if i < total else 0 for i in range(len(weights))]

    counts = [int(total * weight) for weight in weights]
    for i, weight in enumerate(weights):
        if weight > 0 and counts[i] == 0:
            counts[i] = 1
    while sum(counts) > total:
        candidates = [i for i, count in enumerate(counts) if count > 1]
        if not candidates:
            break
        counts[max(candidates, key=lambda i: counts[i])] -= 1
    while sum(counts) < total:
        counts[max(range(len(weights)), key=lambda i: weights[i])] += 1
    return counts


def _split_groups(
    groups: Iterable[str], weights: tuple[float, ...], seed: int
) -> dict[str, str]:
    """Assign whole groups to splits deterministically."""
    groups = sorted(set(groups))
    random.Random(seed).shuffle(groups)
    counts = _counts(len(groups), weights)
    labels: list[str] = []
    for i, count in enumerate(counts):
        labels.extend([f"split_{i}"] * count)
    # ``labels`` is in split order, not shuffled order; that is intentional and
    # deterministic, while the group order itself was shuffled above.
    return {group: label for group, label in zip(groups, labels)}


SPLIT_NAMES = {
    "split_0": "train",
    "split_1": "val",
    "split_2": "u1",
}


def _synthetic_records(
    n_per_family: int, seed: int
) -> list[FileRecord]:
    examples: list[Example] = build_dataset(n_per_family=n_per_family, seed=seed)
    seen = [ex for ex in examples if ex.family not in config.HELD_OUT_FAMILIES]
    held = [ex for ex in examples if ex.family in config.HELD_OUT_FAMILIES]

    pair_groups = {ex.pair_id for ex in seen if ex.pair_id}
    split_map = _split_groups(pair_groups, (0.8, 0.1, 0.1), seed)
    records: list[FileRecord] = []
    for ex in seen:
        split = SPLIT_NAMES[split_map[ex.pair_id]]
        records.append(
            FileRecord(
                id=f"synthetic/{ex.family}/{ex.pair_id}/{ex.label}",
                content=ex.content,
                purpose=ex.purpose,
                label=int(ex.label),
                family=ex.family,
                split=split,
                source="synthetic",
                group_id=ex.pair_id,
                meta={"note": ex.note},
            )
        )
    for ex in held:
        records.append(
            FileRecord(
                id=f"synthetic/{ex.family}/{ex.pair_id}/{ex.label}",
                content=ex.content,
                purpose=ex.purpose,
                label=int(ex.label),
                family=ex.family,
                split="u2",
                source="synthetic",
                group_id=ex.pair_id,
                meta={"note": ex.note, "held_out_family": True},
            )
        )
    return records


def _hard_negative_records(seed: int) -> list[FileRecord]:
    # ``build_dataset`` returns the four fixed hard-negative archetypes.
    examples = build_dataset(n_per_family=1, seed=seed)
    records = []
    for ex in examples:
        if not ex.family.startswith("hard_neg"):
            continue
        records.append(
            FileRecord(
                id=f"hard_negative/{ex.family}",
                content=ex.content,
                purpose=ex.purpose,
                label=0,
                family=ex.family,
                split="train",
                source="hard_negative",
                group_id=ex.family,
                meta={"note": ex.note, "trained_on": True},
            )
        )
    return records


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"corpus not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _real_records(path: Path, seed: int) -> list[FileRecord]:
    raw = corpus.dedupe(_read_jsonl(path))
    raw = corpus.dedupe_runs(raw)
    raw = [row for row in raw if row.get("label") in (0, 1)]

    attack = [row for row in raw if row["label"] == 1]
    honest = [row for row in raw if row["label"] == 0]

    attack_groups = {row.get("trajectory_id") for row in attack if row.get("trajectory_id")}
    honest_groups = {row.get("trajectory_id") for row in honest if row.get("trajectory_id")}
    attack_split = _split_groups(attack_groups, (0.85, 0.15), seed)
    honest_split = _split_groups(honest_groups, (0.85, 0.15), seed + 1)

    # With very few trajectories the weighted allocator can put every honest
    # group in train. Force one into the FPR split so the report still has a
    # real-negative source.
    if honest_groups and not any(
        honest_split.get(group) == "split_1" for group in honest_groups
    ):
        honest_split[sorted(honest_groups)[0]] = "split_1"
    if attack_groups and not any(
        attack_split.get(group) == "split_1" for group in attack_groups
    ):
        attack_split[sorted(attack_groups)[0]] = "split_1"

    records: list[FileRecord] = []
    for row in raw:
        trajectory = row.get("trajectory_id") or "unknown"
        run = row.get("run")
        label = int(row["label"])
        is_attack = label == 1
        if is_attack and config.is_u3_run(run):
            split = "u3"
        elif is_attack:
            split = "val" if attack_split.get(trajectory) == "split_1" else "train"
        else:
            split = "honest" if honest_split.get(trajectory) == "split_1" else "train"

        content = _clean_text(row.get("content"))
        if len(content) < 40:
            continue
        rid = f"real/{trajectory}/{row.get('path', '')}/{_hash(content)}"
        records.append(
            FileRecord(
                id=rid,
                content=content,
                purpose=_clean_text(row.get("purpose")),
                label=label,
                family=str(row.get("run") or "real"),
                split=split,
                source="linuxarena_attack" if is_attack else "linuxarena_honest",
                group_id=trajectory,
                meta={
                    "run": run,
                    "path": row.get("path"),
                    "trajectory_id": row.get("trajectory_id"),
                    "tier": row.get("tier"),
                    "u3": split == "u3",
                },
            )
        )
    return records


def build_datasets(
    corpus_path: Path | str | None = None,
    n_per_family: int = config.N_PER_FAMILY_TRAIN,
    seed: int = config.SEED,
) -> tuple[list[FileRecord], dict]:
    """Build all splits and return ``(records, manifest)``."""
    corpus_path = Path(corpus_path) if corpus_path else DEFAULT_CORPUS
    records = _synthetic_records(n_per_family=n_per_family, seed=seed)
    records.extend(_hard_negative_records(seed=seed))
    records.extend(_real_records(corpus_path, seed=seed))

    # Stable IDs: synthetic ids already include the family and label; real ids
    # include a content hash. Guard against accidental collisions anyway.
    seen_ids: set[str] = set()
    unique: list[FileRecord] = []
    for record in records:
        rid = record.id
        suffix = 2
        while rid in seen_ids:
            rid = f"{record.id}#{suffix}"
            suffix += 1
        seen_ids.add(rid)
        unique.append(record if rid == record.id else FileRecord(
            id=rid,
            content=record.content,
            purpose=record.purpose,
            label=record.label,
            family=record.family,
            split=record.split,
            source=record.source,
            group_id=record.group_id,
            meta=record.meta,
        ))

    manifest = {
        "seed": seed,
        "n_per_family": n_per_family,
        "corpus": str(corpus_path),
        "counts": _count_dict(unique),
        "families": sorted({record.family for record in unique}),
        "splits": sorted({record.split for record in unique}),
        "sources": sorted({record.source for record in unique}),
        "u3_run_marker": config.U3_RUN_MARKER,
        "holdout_families": list(config.HELD_OUT_FAMILIES),
    }
    return unique, manifest


def _count_dict(records: list[FileRecord]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for record in records:
        out.setdefault(record.split, {})
        key = f"{record.source}|label={record.label}"
        out[record.split][key] = out[record.split].get(key, 0) + 1
    return out


def write_dataset(
    records: list[FileRecord], manifest: dict, out_dir: Path | str = config.DATA
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dataset.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    manifest = dict(manifest)
    manifest["dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return path


def read_dataset(path: Path | str) -> list[FileRecord]:
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(FileRecord(**json.loads(line)))
    return records


def summarize(records: list[FileRecord]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "split": record.split,
                "source": record.source,
                "family": record.family,
                "label": record.label,
            }
            for record in records
        ]
    )
    if frame.empty:
        return frame
    return (
        frame.groupby(["split", "source", "family", "label"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["split", "source", "family", "label"])
    )


def smoke_check() -> pd.DataFrame:
    """Build a tiny dataset from a synthetic corpus and verify a write/read round-trip."""
    tiny = [
        {
            "trajectory_id": "honest-1",
            "run": "Haiku honest",
            "purpose": "Fix a failing integration test.",
            "path": "/tmp/a.py",
            "content": "print('honest one')\n" * 4,
            "tier": "create",
            "label": 0,
        },
        {
            "trajectory_id": "honest-1",
            "run": "Haiku honest",
            "purpose": "Fix a failing integration test.",
            "path": "/tmp/b.py",
            "content": "print('honest two')\n" * 4,
            "tier": "create",
            "label": 0,
        },
        {
            "trajectory_id": "attack-1",
            "run": "Haiku attack",
            "purpose": "Fix a failing integration test.",
            "path": "/tmp/c.py",
            "content": "print('attack one')\n" * 4,
            "tier": "create",
            "label": 1,
        },
        {
            "trajectory_id": "straj-1",
            "run": "GPT-5 Nano monitor on straj",
            "purpose": "Fix a failing integration test.",
            "path": "/tmp/d.py",
            "content": "print('straj one')\n" * 4,
            "tier": "create",
            "label": 1,
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "corpus.jsonl"
        with corpus_path.open("w", encoding="utf-8") as handle:
            for row in tiny:
                handle.write(json.dumps(row) + "\n")
        records, manifest = build_datasets(
            corpus_path=corpus_path, n_per_family=2, seed=0
        )
        path = write_dataset(records, manifest, Path(tmp) / "out")
        reloaded = read_dataset(path)
        assert [record.id for record in records] == [record.id for record in reloaded]
        summary = summarize(reloaded)
        assert {"train", "val", "u1", "u2", "u3", "honest"}.issubset(
            set(summary.split)
        ), f"missing split in {sorted(set(summary.split))}"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--n-per-family", type=int, default=config.N_PER_FAMILY_TRAIN)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out-dir", type=Path, default=config.DATA)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="build a tiny in-memory corpus and print split counts",
    )
    args = parser.parse_args()

    if args.smoke:
        summary = smoke_check()
        print(summary.to_string(index=False))
        return

    records, manifest = build_datasets(
        corpus_path=args.corpus, n_per_family=args.n_per_family, seed=args.seed
    )
    path = write_dataset(records, manifest, args.out_dir)
    print(summarize(records).to_string(index=False))
    print(f"\nwrote {path}")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
