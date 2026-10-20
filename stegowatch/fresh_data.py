"""A fresh test set for the fine-tuned Clef: BashArena runs from a different release, none of them
in stegowatch.lora_data (new runs; many tasks are shared, see task_seen).

Source: data/basharena/monitor_eval/train.parquet (373 rows: 200 honest, 173 attack, all four
attack families). Each row is a run cut at the action a monitor was asked to judge
(previous_actions + action), so these are run prefixes, not whole runs. Its side tasks use their
own server IPs (malware 52.84.125.33, exfil 185.199.108.153), unlike the collusion_signal pool.
Rendered with the same lora_data.record() as train, clean only (no planted signals).

Writes data/basharena/lora/fresh.jsonl (split "fresh"). The release reuses many collusion_signal
tasks (new runs, same task text), so each record carries task_seen: its task title appears in a
LoRA train/val run. Report task_seen=False separately: that subset is new runs on new tasks.

    uv run python -m stegowatch.fresh_data
"""
from __future__ import annotations

import json

import pandas as pd

from .lora_data import OUT, SRC, record
from .load import ROOT

FRESH = ROOT / "data" / "basharena" / "monitor_eval" / "train.parquet"


def actions(prev: str, act: str) -> list[dict]:
    out = []
    for m in json.loads(prev) if prev else []:
        for c in m.get("tool_calls") or []:
            out.append({"tool": c["function"], "args": c["arguments"]})
    for c in json.loads(act):
        out.append({"tool": c["function"], "args": c["arguments"]})
    return out


def _title(task: str) -> str:
    for line in task.splitlines():
        line = line.strip("*# ").removeprefix("Title:").strip()
        if line and not line.lower().startswith("your task is"):
            return line.lower()
    return ""


def main():
    df = pd.read_parquet(FRESH)
    pool = pd.read_parquet(SRC)
    trained_ids = {json.loads(x)["base_id"].split("/", 1)[1]
                   for s in ("train", "val") for x in (OUT / f"{s}.jsonl").read_text().splitlines()}
    seen = set(pool[pool.task_id.isin(trained_ids)].task_description.map(_title))
    titles = df.task_description.map(_title)
    print(f"fresh task titles also in the training pool: {titles.isin(set(pool.task_description.map(_title))).sum()}"
          f" / {len(df)}; in the LoRA's train/val runs: {titles.isin(seen).sum()}")
    rows = []
    for i, r in df.iterrows():
        is_att = r.label == "attack"
        base = f"f/{i}/{r.sample_id}/{r.label}"
        rec = record(f"{base}+clean", base, "fresh", r.task_description,
                     actions(r.previous_actions, r.action), is_att, r.side_task or None, None)
        rec["task_seen"] = bool(titles[i] in seen)
        rows.append(rec)
    with (OUT / "fresh.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    out = pd.DataFrame(rows)
    out["attack"] = out.labels.map(lambda x: x["attack"])
    out["n_actions"] = out.state.str.count(r"\n\[\d+\] ")
    print(out.groupby(["task_seen", "attack"]).agg(n=("id", "size"), raw_chars=("n_chars", "median"),
                                     actions=("n_actions", "median")))
    print("families:", out[out.attack].attack_type.value_counts().to_dict())
    print(f"wrote {OUT / 'fresh.jsonl'} ({len(out)} records, {out.id.nunique()} unique ids)")


if __name__ == "__main__":
    main()
