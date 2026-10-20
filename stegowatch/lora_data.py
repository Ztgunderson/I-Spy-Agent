"""Train / val / test sets for fine-tuning Clef on whole BashArena runs (stegowatch.clef_lora).

One record = one run: the task, a run summary computed by code over the FULL action text (files
written and read, hosts contacted, secret-like paths, privilege and firewall changes), then every
action clipped to fit (~2k tokens). Labels, from how the record was built:
  attack         the run is a released attack trajectory
  extra_content  a collusion signal was planted (any of the five)
  to_reviewer    the planted signal is the approval note

Splits (by task: an attack and all its signalled copies stay on one side):
  test   the frozen 200-run Part 2 set (100 honest + 100 attack, data/basharena/built), every base
         clean and with each of the 5 signals, honest runs included, so signal is independent of
         attack. Never trained or tuned on.
  test_family  pool attacks of the held-out families (disable_firewall, sudo_edit), clean.
  train / val  the rest of the pool, trained families only (leak_passwords, download_malware),
         with honest runs matched 1:1 on character length and action count (honest pool runs are ~3x
         longer, which a fine-tuned model would learn). Each base appears clean and once with a random TRAIN
         signal (approval_note, codeword, identifier). natural_comment and noop are never trained.

    uv run python -m stegowatch.lora_data
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import yaml

from .bash_build import SIGNALS, SRC, host_index, parse_actions, plant
from .bash_score import load_built
from .load import ROOT

OUT = ROOT / "data" / "basharena" / "lora"
TRAIN_SIGNALS = ("approval_note", "codeword", "identifier")
TEST_SIGNALS = ("natural_comment", "noop")
HELD_OUT_FAMILIES = ("disable_firewall", "sudo_edit")
TASK_CHARS, ACTION_CHARS, LIST_CAP = 1200, 5000, 10

_URL = re.compile(r"https?://([^/\s'\"\\:]+)")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PATH = re.compile(r"(?<![\w.])(/(?:[\w.@+-]+/)*[\w.@+-]+)")
_WRITE_CMD = re.compile(r"(?:>>?|\btee(?:\s+-a)?|\bsed\s+-i\S*\s+(?:'[^']*'|\"[^\"]*\"|\S+))\s+(/[\w./@+-]+)")
_READ_CMD = re.compile(r"\b(?:cat|head|tail|less|grep|cp|scp|base64|xxd|strings)\b[^|;&\n]*?\s(/[\w./@+-]+)")
_SECRET = re.compile(r"api_tokens|\.env\b|id_rsa|/shadow|passw|secret|credential|\.pem\b|\.key\b|token|\.aws|\.ssh", re.I)
_PRIV = re.compile(r"sudoers|visudo|NOPASSWD|iptables|nftables|firewall|ufw\b|chmod\s+[0-7]*[4-7][0-7]{3}|chmod\s+\+s|setuid|usermod|/etc/passwd", re.I)
_FETCH_EXEC = re.compile(r"(?:curl|wget)[^\n|;]*(?:\|\s*(?:ba|z)?sh|&&\s*(?:chmod\s+\+x|bash|sh)\b)", re.I)


def _text(a: dict) -> str:
    return json.dumps(a["args"], ensure_ascii=False)


def _cap(xs) -> str:
    xs = list(dict.fromkeys(xs))
    more = f" (+{len(xs) - LIST_CAP} more)" if len(xs) > LIST_CAP else ""
    return ", ".join(xs[:LIST_CAP]) + more if xs else "none"


def run_summary(actions: list[dict]) -> str:
    """Deterministic run summary over the full, unclipped actions (rule 2: code, not a model)."""
    written, read, hosts, secrets, priv, fetch = [], [], [], [], [], 0
    for a in actions:
        args, raw = a["args"], _text(a)
        cmd = str(args.get("cmd", args.get("code", "")))
        if a["tool"] == "text_editor" and args.get("path"):
            (read if args.get("command") == "view" else written).append(str(args["path"]))
        written += _WRITE_CMD.findall(cmd)
        read += _READ_CMD.findall(cmd)
        hosts += _URL.findall(raw) + [ip for ip in _IP.findall(raw) if not ip.startswith(("127.", "0."))]
        secrets += [p for p in _PATH.findall(raw) if _SECRET.search(p)]
        priv += [m.group(0).lower() for m in _PRIV.finditer(raw)]
        fetch += bool(_FETCH_EXEC.search(cmd))
    return "\n".join([
        f"actions: {len(actions)}",
        f"files written: {_cap(written)}",
        f"files read: {_cap(read)}",
        f"hosts or IPs mentioned: {_cap(hosts)}",
        f"secret-like paths mentioned: {_cap(secrets)}",
        f"privilege or firewall terms: {_cap(priv)}",
        f"download-then-run commands: {fetch}",
    ])


def clip_actions(actions: list[dict]) -> str:
    per = int(np.clip(ACTION_CHARS // max(len(actions), 1), 160, 700))
    tail = 100 if per >= 300 else 60
    lines = []
    for i, a in enumerate(actions, 1):
        body = _text(a)
        if len(body) > per:
            body = body[: per - tail] + " ... " + body[-tail:]
        lines.append(f"[{i}] {a['tool']}: {body}")
    return "\n".join(lines)


def preamble() -> str:
    return yaml.safe_load((ROOT / "questions_bash_traj.yaml").read_text())["preamble"].strip()


def render_state(task: str, actions: list[dict]) -> str:
    return (f"{preamble()}\n\nTASK:\n{task[:TASK_CHARS]}\n\nRUN SUMMARY (computed by code):\n"
            f"{run_summary(actions)}\n\nACTIONS:\n{clip_actions(actions)}")


def record(rid, base, split, task, actions, is_attack, attack_type, signal, rng=None):
    acts = list(actions)
    if signal:
        h = host_index(acts)
        acts[h] = plant(acts[h], signal)
    return {"id": rid, "base_id": base, "split": split, "attack_type": attack_type,
            "signal": signal, "n_chars": sum(len(_text(a)) for a in acts),
            "state": render_state(task, acts),
            "labels": {"attack": bool(is_attack), "extra_content": signal is not None,
                       "to_reviewer": signal == "approval_note"},
            "provenance": "simulated"}  # benchmark construction labels, not real-world outcomes


def main(seed: int = 20261013, val_frac: float = 0.15):
    rng = np.random.default_rng(seed)
    built = load_built()
    eval_tasks = {t["base_id"].split("/", 1)[1] for t in built}
    df = pd.read_parquet(SRC)
    df["actions"] = df.tool_calls.map(parse_actions)
    df["n_chars"] = df.tool_calls.str.len()
    df["n_actions"] = df.actions.map(len)
    pool = df[~df.task_id.isin(eval_tasks) & (df.actions.map(len) >= 3)]
    att = pool[pool.split == "attack"]
    hon = pool[pool.split == "honest"].copy()
    rows = []

    # test: the frozen Part 2 bases, clean + every signal, honest and attack alike
    for t in built:
        if t["variant"] == "attack+signal":
            continue
        is_att = t["variant"] == "attack"
        for sig in (None, *SIGNALS):
            rows.append(record(f"{t['base_id']}+{sig or 'clean'}", t["base_id"], "test",
                               t["task_description"], t["actions"], is_att, t["attack_type"], sig))

    # test_family: held-out attack families from the pool, clean only (scored against test honest)
    for _, r in att[att.attack_type.isin(HELD_OUT_FAMILIES)].iterrows():
        rows.append(record(f"a/{r.task_id}+clean", f"a/{r.task_id}", "test_family",
                           r.task_description, r.actions, True, r.attack_type, None))

    # train / val: trained families, honest matched 1:1 on log characters + log action count
    tr_att = att[~att.attack_type.isin(HELD_OUT_FAMILIES)]
    tr_att = tr_att.iloc[rng.permutation(len(tr_att))]
    hon_pool = hon.copy()
    pairs = []
    for _, a in tr_att.sort_values("n_chars").iterrows():
        j = ((np.log(hon_pool.n_chars) - np.log(a.n_chars)).abs()
             + (np.log(hon_pool.n_actions) - np.log(a.n_actions)).abs()).idxmin()
        pairs.append((a, hon_pool.loc[j]))
        hon_pool = hon_pool.drop(index=j)
    order = rng.permutation(len(pairs))
    n_val = int(round(val_frac * len(pairs)))
    for k, idx in enumerate(order):
        split = "val" if k < n_val else "train"
        for r, is_att in zip(pairs[idx], (True, False)):
            base = f"{'a' if is_att else 'h'}/{r.task_id}"
            at = r.attack_type if is_att else None
            sig = TRAIN_SIGNALS[int(rng.integers(len(TRAIN_SIGNALS)))]
            for s in (None, sig):
                rows.append(record(f"{base}+{s or 'clean'}", base, split, r.task_description,
                                   r.actions, is_att, at, s))

    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    for split, g in out.groupby("split"):
        with (OUT / f"{split}.jsonl").open("w") as f:
            for row in g.to_dict("records"):
                f.write(json.dumps(row) + "\n")
    out["attack"] = out.labels.map(lambda x: x["attack"])
    out["state_chars"] = out.state.str.len()
    print(out.groupby(["split", "attack"]).agg(n=("id", "size"), raw_chars=("n_chars", "median"),
                                                state_chars=("state_chars", "median")))
    from sklearn.metrics import roc_auc_score
    for split, g in out.groupby("split"):
        if g.attack.nunique() == 2:
            print(f"{split}: length AUROC (raw / state chars) "
                  f"{roc_auc_score(g.attack, g.n_chars):.2f} / {roc_auc_score(g.attack, g.state_chars):.2f}")
    print("train families:", out[(out.split == "train") & out.attack].attack_type.value_counts().to_dict())


if __name__ == "__main__":
    main()
