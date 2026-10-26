"""Zero-shot size ladder: score the eval splits with GGUF decision models over ``/v1/systemone``.

Every model sees the same records and the same ``covert`` question as Stage 0/2/3
(``schema.SCHEMA["covert"]``). Standard library only, so it runs on the Jetson
without a venv; ``compile_scaling.py`` joins the caches with the dataset locally.

    python3 -m experiment.clef_unseen_attack.scaling score --url http://127.0.0.1:8099 --name d3-edge
    python3 -m experiment.clef_unseen_attack.scaling latency --url ... --name d3-edge
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .schema import SCHEMA

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "clef_unseen_attack" / "dataset.jsonl"
OUT = ROOT / "runs" / "scaling"
EVAL_SPLITS = ("val", "u1", "u2", "u3", "honest")
# MAX_EVAL_LEN is 2048 tokens for the transformers runs; ~4 chars per token.
MAX_CHARS = 8000
QUESTIONS = {"covert": SCHEMA["covert"]}


def file_state(record: dict, max_chars: int = MAX_CHARS) -> str:
    """Same layout as ``model.file_state`` (PURPOSE, separator, FILE), clipped."""
    return f"PURPOSE: {record['purpose']}\n---\n{record['content']}"[:max_chars]


def request_body(state: str) -> dict:
    return {"state": state, "questions": QUESTIONS}


def ask(url: str, state: str, timeout: float = 300.0) -> dict:
    data = json.dumps(request_body(state)).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone", data=data, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        payload = json.load(response)
    latency_ms = (time.perf_counter() - started) * 1000
    payload = payload.get("result", payload)
    return {
        "p": float(payload["answers"]["covert"]["noul"]),
        "latency_ms": round(latency_ms, 1),
        "input_tokens": int(payload.get("usage", {}).get("input_tokens", 0)),
    }


def eval_records(path: Path = DATASET, splits=EVAL_SPLITS) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [row for row in rows if row["split"] in splits]


def cached_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


def score(url: str, name: str, records: list[dict], out_dir: Path = OUT, workers: int = 4) -> Path:
    """Append one row per record to ``<out_dir>/<name>.jsonl``; skips ids already cached."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    done = cached_ids(path)
    todo = [r for r in records if r["id"] not in done]
    print(f"{name}: {len(done)} cached, {len(todo)} to score", flush=True)

    def one(record: dict) -> dict:
        try:
            result = ask(url, file_state(record))
        except Exception as exc:  # noqa: BLE001 - logged, scored as missing
            result = {"p": None, "error": repr(exc)[:200]}
        return {"id": record["id"], "model": name, **result}

    with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(workers) as pool:
        for i, row in enumerate(pool.map(one, todo), 1):
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if i % 50 == 0:
                print(f"  {name}: {i}/{len(todo)}", flush=True)
    return path


def latency(url: str, name: str, out_dir: Path = OUT, reps: int = 20) -> Path:
    """One request at a time at three fixed lengths -> p50/p95 per length."""
    words = "server listen proxy_pass upstream location root index return".split()
    rng = random.Random(0)
    result = {"model": name, "reps": reps, "by_chars": {}}
    for chars in (1000, 4000, MAX_CHARS):  # ~256 / 1k / 2k tokens
        body = " ".join(rng.choice(words) for _ in range(chars // 5))[:chars]
        state = f"PURPOSE: web server config\n---\n{body}"[:chars]
        ask(url, state)  # warm-up
        times, tokens = [], 0
        for _ in range(reps):
            row = ask(url, state)
            times.append(row["latency_ms"])
            tokens = row["input_tokens"]
        times.sort()
        result["by_chars"][str(chars)] = {
            "input_tokens": tokens,
            "p50_ms": statistics.median(times),
            "p95_ms": times[int(0.95 * (len(times) - 1))],
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.latency.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["score", "latency", "probe"])
    parser.add_argument("--url", default="http://127.0.0.1:8099")
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=None, help="first N records (dry runs)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--splits", default=",".join(EVAL_SPLITS), help="comma list, e.g. u3,honest")
    args = parser.parse_args()

    if args.mode == "probe":
        print(json.dumps(ask(args.url, "PURPOSE: nginx config\n---\nserver { listen 80; }")))
    elif args.mode == "score":
        records = eval_records(args.dataset, tuple(args.splits.split(",")))[: args.limit]
        print(score(args.url, args.name, records, args.out_dir, args.workers))
    else:
        print(latency(args.url, args.name, args.out_dir))


if __name__ == "__main__":
    main()
