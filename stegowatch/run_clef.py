"""Score every item of a view with Clef-flash over `/v1/systemone` (SDK adapter, zero-shot).

Cache: runs/clef_<view>.jsonl, one line per item: id, question-set hash, model id, P(yes) per
question, latency, tokens (rule 3). Reruns only score what is missing. The server is shared:
at most 2 requests in flight (serve_clef.sh runs -np 2).

    uv run python -m stegowatch.run_clef --view table
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import yaml
from dlsdk.adapters.decision.systemone_http import SystemOneHTTP
from dlsdk.core.types import DecisionRequest, NoulQ

from .load import MODELS, ROOT, VIEWS, load_runs

MODEL_ID = "clef-flash-q4_k_m"


def question_set(view: str) -> tuple[dict, str, str]:
    cfg = yaml.safe_load((ROOT / "questions.yaml").read_text())
    qs = {k: NoulQ(question=v["question"].strip()) for k, v in cfg["views"][view].items()}
    blob = json.dumps({"pre": cfg["preamble"], "q": {k: q.question for k, q in qs.items()}},
                      sort_keys=True)
    return qs, cfg["preamble"].strip(), hashlib.sha1(blob.encode()).hexdigest()[:10]


def main(a):
    qs, pre, qhash = question_set(a.view)
    items = VIEWS[a.view](load_runs(tuple(a.writers.split(","))))
    # Other decision models (size ladder) get their own cache next to Clef's.
    cache = (ROOT / "runs" / f"clef_{a.view}.jsonl" if a.model_id == MODEL_ID
             else ROOT / "runs" / "narc_scaling" / f"{a.model_id}_{a.view}.jsonl")
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if cache.exists():
        for line in cache.read_text().splitlines():
            d = json.loads(line)
            if d["qhash"] == qhash:
                done.add(d["id"])
    todo = [i for i in items if i["id"] not in done][: a.limit or None]
    print(f"{a.view}: qhash {qhash}, {len(done)} cached, {len(todo)} to score", flush=True)
    model = SystemOneHTTP(a.url, a.model_id, timeout=300)
    lock, fails, t0 = threading.Lock(), [], time.perf_counter()
    f = cache.open("a")

    def work(it):
        req = DecisionRequest(state=f"{pre}\n\n{it['text'][: a.max_chars]}", questions=qs)
        for attempt in range(3):
            try:
                dec = model.decide(req)
                break
            except Exception as e:  # noqa: BLE001 - logged and retried
                err = repr(e)[:200]
                time.sleep(3 * (attempt + 1))
        else:
            with lock:
                fails.append((it["id"], err))
            return
        row = {"id": it["id"], "qhash": qhash, "model_id": dec.model_id,
               "p": {q: round(ans.probabilities["yes"], 5) for q, ans in dec.answers.items()},
               "latency_ms": round(dec.latency_ms, 1), "tokens_in": dec.tokens_in,
               "truncated": len(it["text"]) > a.max_chars}
        with lock:
            f.write(json.dumps(row) + "\n")
            f.flush()
            done.add(it["id"])
            if len(done) % 50 == 0:
                print(f"  {len(done)}/{len(items)}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    for i, e in fails[:5]:
        print("FAIL", i, e)
    if fails:
        raise SystemExit(f"{len(fails)} items failed; rerun to resume (cache kept)")
    print(f"done {len(done)}/{len(items)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", choices=list(VIEWS), required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8097")
    ap.add_argument("--model-id", dest="model_id", default=MODEL_ID)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    # Generated writers (load.GEN_MODELS) are scored into the same per-model cache, by id.
    ap.add_argument("--writers", default=",".join(MODELS))
    # Slot context is 8192 tokens (serve_clef.sh: -c 16384 -np 2); the longest table is ~26k chars
    ap.add_argument("--max-chars", dest="max_chars", type=int, default=28000)
    main(ap.parse_args())
