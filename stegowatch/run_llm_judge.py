"""Score a view with a general chat LLM as a yes/no judge (self-monitoring test).

Same preamble and questions as run_clef, one request per question, raw prompt to llama.cpp's
`/completion`. P(yes) is read from the first answer token: the summed top-k probability of
"yes"-like tokens over "yes" plus "no" tokens; null when neither is in the top k.
Templates are written out here because the monitors are fixed: Qwen3 (thinking off) and
GPT-OSS (Harmony, straight to the final channel, reasoning low).

Cache: runs/narc_scaling/<model-id>_<view>.jsonl, the same row shape as run_clef, so
narc_scaling and the page read it like any other monitor.

    uv run python -m stegowatch.run_llm_judge --view player --template qwen3 \
        --url http://127.0.0.1:8100 --model-id qwen3-32b-judge
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .load import ROOT, VIEWS, load_runs
from .run_clef import question_set

TEMPLATES = {
    "qwen3": "<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "gpt-oss": ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
                "Reasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be "
                "included for every message.<|end|><|start|>user<|message|>{u}<|end|>"
                "<|start|>assistant<|channel|>final<|message|>"),
}


def p_yes(url: str, prompt: str, k: int = 20) -> tuple[float | None, int]:
    body = {"prompt": prompt, "n_predict": 1, "temperature": 0, "n_probs": k, "cache_prompt": True,
            "post_sampling_probs": False}
    req = urllib.request.Request(url.rstrip("/") + "/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    yes = no = 0.0
    for t in d["completion_probabilities"][0]["top_logprobs"]:
        w = t["token"].strip().lower()
        p = math.exp(t["logprob"])
        if w in ("yes", "y"):
            yes += p
        elif w in ("no", "n"):
            no += p
    return (yes / (yes + no) if yes + no > 0 else None), d.get("tokens_evaluated", 0)


def main(a):
    qs, pre, qhash = question_set(a.view)
    items = VIEWS[a.view](load_runs())
    cache = ROOT / "runs" / "narc_scaling" / f"{a.model_id}_{a.view}.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if cache.exists():
        for line in cache.read_text().splitlines():
            d = json.loads(line)
            if d["qhash"] == qhash:
                done.add(d["id"])
    todo = [i for i in items if i["id"] not in done][: a.limit or None]
    print(f"{a.view}: qhash {qhash}, {len(done)} cached, {len(todo)} to score", flush=True)
    tpl = TEMPLATES[a.template]
    lock, fails, t0 = threading.Lock(), [], time.perf_counter()
    f = cache.open("a")

    def work(it):
        text = it["text"][: a.max_chars]
        t1, ps, toks = time.perf_counter(), {}, 0
        try:
            # player text first, question last, so the server reuses the prefix across questions
            for qid, q in qs.items():
                u = f"{pre}\n\n{text}\n\nQuestion: {q.question}\nAnswer with Yes or No only."
                ps[qid], n = p_yes(a.url, tpl.format(u=u))
                ps[qid] = None if ps[qid] is None else round(ps[qid], 5)
                toks += n
        except Exception as e:  # noqa: BLE001 - logged; rerun resumes
            with lock:
                fails.append((it["id"], repr(e)[:200]))
            return
        row = {"id": it["id"], "qhash": qhash, "model_id": a.model_id, "p": ps,
               "latency_ms": round((time.perf_counter() - t1) * 1000, 1), "tokens_in": toks,
               "truncated": len(it["text"]) > a.max_chars, "template": a.template}
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
    ap.add_argument("--template", choices=list(TEMPLATES), required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--model-id", dest="model_id", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-chars", dest="max_chars", type=int, default=28000)
    main(ap.parse_args())
