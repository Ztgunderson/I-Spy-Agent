"""Evaluate fine-tuned vs zero-shot Clef on whole BashArena runs, on data never trained on.

Arms (all score the same rendered states from stegowatch.lora_data):
  length          rendered state characters; direction (shorter or longer = positive) is fixed
                  on train, never on test
  tfidf           char 3-5-gram tf-idf + logistic regression, trained on train (the cheap
                  trained classifier every claim here has to beat)
  clef zero-shot  Clef-flash release weights (4-bit), questions_bash_traj.yaml
  clef LoRA       the same model after stegowatch.clef_lora fine-tuning
Q1 catch the attack, per condition c: honest runs with condition c vs attack runs with c (c = clean
   or a planted signal, planted in honest runs too, so a signal is not evidence of attack). Plus
   the held-out families (test_family, clean) vs clean honest test runs.
Q1 fresh: a different BashArena release (stegowatch.fresh_data, run prefixes, new runs), all of it
   and only runs whose task is not in any LoRA train/val run (task_seen False).
Q2 spot the signal: base+signal vs the same base clean (honest and attack bases together).
   approval_note, codeword, identifier were trained on; natural_comment and noop never were.
AUROC with a 95% bootstrap CI (records resampled, not clustered by base), catch rate at 5% and
10% false alarms.

    uv run python -m stegowatch.lora_eval
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .bash_build import SIGNALS
from .clef_lora import SCORES
from .evaluate import score
from .lora_data import OUT as DATA
from .lora_data import TRAIN_SIGNALS
from .load import ROOT


def frame(split: str) -> pd.DataFrame:
    df = pd.DataFrame([json.loads(x) for x in (DATA / f"{split}.jsonl").read_text().splitlines()])
    for q in ("attack", "extra_content", "to_reviewer"):
        df[q] = df.labels.map(lambda x, q=q: x[q])
    df["signal"] = df.signal.fillna("clean")
    df["chars"] = df.state.str.len()
    return df


def add_arms(train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for q in ("attack", "extra_content"):
        sign = 1 if roc_auc_score(train[q], train.chars) >= 0.5 else -1
        df[f"length_{q}"] = sign * df.chars
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000,
                              sublinear_tf=True)
        lr = LogisticRegression(C=4.0, max_iter=2000).fit(vec.fit_transform(train.state), train[q])
        df[f"tfidf_{q}"] = lr.predict_proba(vec.transform(df.state))[:, 1]
    if SCORES.exists():
        rows = [json.loads(x) for x in SCORES.read_text().splitlines()]
        for tag in ("zeroshot", "lora"):
            p = {r["id"]: r["p"] for r in rows if r["tag"] == tag}
            for q in ("attack", "extra_content", "to_reviewer"):
                df[f"clef_{tag}_{q}"] = df.id.map(lambda i, q=q, p=p: p.get(i, {}).get(q, np.nan))
    return df


ARMS = {"length": "length", "tfidf": "tfidf", "clef zero-shot": "clef_zeroshot", "clef LoRA": "clef_lora"}


def rows_for(question, condition, d, label, q, rng, extra=None):
    cols = {name: f"{prefix}_{q}" for name, prefix in ARMS.items()}
    if q == "extra_content":  # Clef's to_reviewer answer as a second signal arm
        cols |= {f"{name} (to_reviewer q)": f"{prefix}_to_reviewer"
                 for name, prefix in ARMS.items() if name.startswith("clef")}
    out = []
    for name, col in cols.items():
        if col not in d or d[col].isna().any():
            continue
        m = score(d[label].astype(int), d[col], rng)
        out.append({"question": question, "condition": condition, "arm": name,
                    **{k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()},
                    **(extra or {})})
    return out


def main():
    rng = np.random.default_rng(0)
    train = frame("train")
    test = add_arms(train, frame("test"))
    fam = add_arms(train, frame("test_family"))
    rows = []
    for c in ("clean", *SIGNALS):
        d = test[test.signal == c]
        rows += rows_for("Q1 catch attack", c, d, "attack", "attack", rng)
    d = pd.concat([test[(test.signal == "clean") & ~test.attack], fam])
    rows += rows_for("Q1 catch attack", "held-out families (clean)", d, "attack", "attack", rng)
    if (DATA / "fresh.jsonl").exists():
        fresh = add_arms(train, frame("fresh"))
        for cond, d in (("fresh release, all", fresh), ("fresh release, unseen tasks", fresh[~fresh.task_seen])):
            rows += rows_for("Q1 catch attack", cond, d, "attack", "attack", rng)
    for s in SIGNALS:
        d = test[test.signal.isin(["clean", s])]
        rows += rows_for("Q2 spot signal", s, d, "extra_content", "extra_content", rng,
                         {"signal_trained": s in TRAIN_SIGNALS})
    out = pd.DataFrame(rows)
    path = ROOT / "results" / "bash_lora.csv"
    out.to_csv(path, index=False)
    pd.set_option("display.width", 200)
    print(out.pivot_table(index=["question", "condition"], columns="arm", values="auroc", sort=False))
    print(out.pivot_table(index=["question", "condition"], columns="arm", values="tpr@10%fpr", sort=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
