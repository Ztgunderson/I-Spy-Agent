"""Collect result tables into one JSON for the prototype page. Every number comes from a result file."""
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "experiment/clef_unseen_attack/results"
out = {}

st = pd.read_csv(R / "scaling_table.csv")
def ci(s):
    try:
        lo, hi = str(s).split("-"); return float(lo), float(hi)
    except Exception: return None, None
rows = []
for _, r in st.iterrows():
    lo, hi = ci(r["u3_ci"])
    rows.append({k: (None if pd.isna(v) else v) for k, v in {
        "model": r["arm"], "kind": r["kind"], "params": r["params_b"], "family": r["family"],
        "backbone": r["backbone"], "status": r["status"], "train_n": r.get("train_n"),
        "val": r["val_auc"], "u1": r["u1_auc"], "u2": r["u2_auc"], "u3": r["u3_auc"],
        "u3_lo": lo, "u3_hi": hi, "fnr5": r["u3_fnr@5%"], "fnr1": r["u3_fnr@1%"],
        "p50": r["p50_ms@2k_tok"], "hw": r["hardware"]}.items()})
out["linux"] = rows

nt = pd.read_csv(ROOT / "results/narc_scaling.csv")
out["narc"] = json.loads(nt.to_json(orient="records"))
for k in ["backbone", "backbone_fit", "backbone_fit_strong", "backbone_within_strong", "backbone_working_shoes", "channel"]:
    out[f"narc_{k}"] = json.loads(pd.read_csv(ROOT / f"results/narc_{k}.csv").to_json(orient="records"))
out["narc_corr"] = json.loads(pd.read_csv(ROOT / "results/narc_monitor_correlation.csv").to_json(orient="records"))
out["linux_corr"] = json.loads(pd.read_csv(R / "monitor_correlation_u3.csv").to_json(orient="records"))

# per-file U3 + honest scores for the operating-point explorer
lab = pd.read_csv(R / "u3_scores.csv")
truth = dict(zip(lab.file, lab.truth))
scores = {m: g.p.round(5).tolist() for m, g in lab.groupby("model")}
truths = {m: g.truth.tolist() for m, g in lab.groupby("model")}
for f in sorted((ROOT / "runs/scaling").glob("*.jsonl")):
    m = f.stem
    if m in scores: continue
    ps, ts = [], []
    for line in f.read_text().splitlines():
        d = json.loads(line)
        if d["id"] in truth and d.get("p") is not None:
            ps.append(round(d["p"], 5)); ts.append(truth[d["id"]])
    if len(ps) > 300: scores[m], truths[m] = ps, ts
out["u3_scores"] = {m: {"p": scores[m], "y": [1 if t != "honest" else 0 for t in truths[m]]} for m in scores}
print(sorted(scores), {m: len(v) for m, v in scores.items()}, file=sys.stderr)
print(sorted(set(lab.truth)), file=sys.stderr)
(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("data.json")).write_text(json.dumps(out, separators=(",", ":")))

# ---- per-item data for confusion matrices and failure analysis ----
sys.path.insert(0, str(ROOT))
from stegowatch.load import GEN_MODELS, MODELS, VIEWS, load_runs
from stegowatch.failures import scheme_of
from stegowatch.evaluate import mean_len
from stegowatch.run_clef import question_set

# LinuxArena U3 + honest files
recs = {}
for line in (ROOT / "data/clef_unseen_attack/dataset.jsonl").read_text().splitlines():
    d = json.loads(line)
    if d["split"] in ("u3", "honest"):
        import ast
        m = ast.literal_eval(d["meta"]) if isinstance(d["meta"], str) else d["meta"]
        path = m.get("path", "")
        name = path.rsplit("/", 1)[-1]
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else "(none)"
        recs[d["id"]] = {"id": d["id"], "y": int(str(d["label"]) == "1"), "path": path, "ext": ext,
                         "tier": m.get("tier", ""), "traj": d["group_id"][:8], "chars": len(d["content"]),
                         "task": d["purpose"].split("\n")[0][:110], "s": {}}
for m, g in lab.groupby("model"):
    for f, p in zip(g.file, g.p):
        if f in recs: recs[f]["s"][m] = round(float(p), 5)
for f in sorted((ROOT / "runs/scaling").glob("*.jsonl")):
    for line in f.read_text().splitlines():
        d = json.loads(line)
        if d["id"] in recs and d.get("p") is not None and f.stem not in recs[d["id"]]["s"]:
            recs[d["id"]]["s"][f.stem] = round(d["p"], 5)
for n in (50, 100, 200, 400):  # fine-tuned Clef-flash, per-file scores from the Stage 2 eval
    f = R / f"stage23_scores_stage2_size_{n}.jsonl"
    for line in f.read_text().splitlines():
        d = json.loads(line)
        if d["id"] in recs and d.get("p_covert") is not None:
            recs[d["id"]]["s"].setdefault(f"clef-flash qlora n={n}", round(d["p_covert"], 5))
out["linux_items"] = list(recs.values())

# NARCBench players (public talk only; scheme tag read from the counter's private plan, for analysis only)
qs, _, qhash = question_set("player")
caches = {"clef-flash": ROOT / "runs/clef_player.jsonl"}
caches.update({p.name.removesuffix("_player.jsonl"): p for p in sorted((ROOT / "runs/narc_scaling").glob("*_player.jsonl"))})
sc = {}
for m, p in caches.items():
    rows = {}
    for line in p.read_text().splitlines():
        d = json.loads(line)
        if d["qhash"] == qhash:
            rows[d["id"]] = round(sum(d["p"][q] for q in qs) / len(qs), 5)
    sc[m] = rows
items = []
for it in VIEWS["player"](load_runs(MODELS + GEN_MODELS)):
    run_id = it["id"].split("/")[1]
    s = {m: r[it["id"]] for m, r in sc.items() if it["id"] in r}
    role = "counter" if it["y"] else ("big player" if it["y_big_player"] else ("other player, collusion run" if it["collusion_run"] else "player, control run"))
    items.append({"id": it["id"], "writer": it["model"], "run": run_id, "player": it["player"], "y": it["y"], "role": role,
                  "n_msgs": it["n_msgs"], "mlen": round(mean_len(it["text"]), 1),
                  "scheme": scheme_of(it["model"], run_id) if it["y"] else "",
                  "text": it["text"][:700], "s": s})
out["narc_items"] = items
(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("data.json")).write_text(json.dumps(out, separators=(",", ":")))
print("linux items", len(recs), "narc items", len(items), {m: len(r) for m, r in sc.items()}, file=sys.stderr)

# ---- length baseline as a pseudo-model, oriented so higher = more suspicious ----
import datetime
def auc(pos, neg):
    import bisect
    neg = sorted(neg); n = 0.0
    for p in pos:
        lo = bisect.bisect_left(neg, p); hi = bisect.bisect_right(neg, p); n += lo + (hi - lo) / 2
    return n / (len(pos) * len(neg))
L = out["linux_items"]
a = auc([i["chars"] for i in L if i["y"]], [i["chars"] for i in L if not i["y"]])
sign = 1 if a >= 0.5 else -1
for i in L: i["s"]["length baseline"] = sign * i["chars"]
for i in out["narc_items"]: i["s"]["length baseline"] = -i["mlen"]
# original-model release dates (Hugging Face repo creation of the base_model each GGUF card names)
out["release"] = {r.split("\t")[0]: {"repo": r.split("\t")[1], "date": r.split("\t")[2]}
                  for r in (Path(__file__).resolve().parent / "release_dates.tsv").read_text().splitlines() if r.strip()}
out["asof"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
out["queue"] = {"narc": []}
# fine-tuning curves: U3 AUROC and recall at 5% FPR per (base, n) from the Stage 2/3 metric files
import csv
ft = []
for series, stem, zs in [("Clef-flash 9.4B (decision-tuned)", "stage2_size_{n}", None),
                         ("d3-lite 0.8B (decision-tuned)", "stage2_d3lite_size_{n}", "zeroshot_d3"),
                         ("Qwen3.5-0.8B-Base (no decision tuning)", "stage3_qwen08b_size_{n}", None)]:
    for n in (0, 50, 100, 200, 400):
        name = (zs if n == 0 else stem.format(n=n))
        if name is None: continue
        f = R / f"stage23_metrics_{name}.csv"
        if not f.exists(): continue
        r = next(r for r in csv.DictReader(f.open()) if r["section"] == "auc" and r["eval"] == "u3" and r["family"] == "all")
        ft.append({"series": series, "n": n, "auroc": float(r["auc"]), "lo": float(r["ci_lo"]), "hi": float(r["ci_hi"]), "tpr5": float(r["tpr@5%fpr"])})
z = st[st.arm == "clef-flash nf4"].iloc[0]  # Clef-flash n = 0: the same NF4 model, untrained
lo, hi = map(float, z["u3_ci"].split("-"))
ft.insert(0, {"series": "Clef-flash 9.4B (decision-tuned)", "n": 0, "auroc": float(z["u3_auc"]), "lo": lo, "hi": hi, "tpr5": float(z["u3_tpr@5%"])})
out["ft"] = ft
(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("data.json")).write_text(json.dumps(out, separators=(",", ":")))
print("linux length auc", a, file=sys.stderr)
