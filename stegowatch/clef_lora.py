"""Fine-tune Clef-flash (Qwen3.5-9B backbone + joint schema head) on whole BashArena runs.

Model: the HF release Cloudflare/clef-flash (Apache-2.0) in ~/lab/models/clef-flash-hf, run with
its own `joint_schema_model.py` encoding and head, so prompts match the release exactly.
  backbone  4-bit NF4 (bitsandbytes), fp16 compute (Turing: no bf16), LoRA r=16 on every
            attention projection (full attention q/k/v/o, linear attention in_proj_qkv/in_proj_z/
            out_proj), gradient checkpointing
  head      the joint schema head, trained in fp32 at a lower learning rate
  lm_head   never loaded on the GPU: the head only reads its rows for the option tokens, which are
            fixed by the schema, so those rows are copied into a small table
Data: stegowatch.lora_data (train / val / test / test_family). Questions: questions_bash_traj.yaml.

Every scored record is logged to runs/clef_lora_scores.jsonl (model tag, question hash, latency,
tokens; rule 3) and cached by (tag, record id), so a stopped run resumes.

    uv run python -m stegowatch.clef_lora smoke      # parity with the GGUF server + speed, 8 records
    uv run python -m stegowatch.clef_lora run        # zero-shot score, train, score fine-tuned
    uv run python -m stegowatch.clef_lora score --splits fresh --tags zeroshot lora   # new split
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from dlsdk.trainers.torch_utils import proper_loss, train_loop

from .load import ROOT
from .lora_data import OUT as DATA

MODEL_DIR = Path.home() / "lab" / "models" / "clef-flash-hf"
RUNS = ROOT / "runs"
SCORES = RUNS / "clef_lora_scores.jsonl"
CKPT = RUNS / "clef_lora"
CFG = {
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "target_modules": r".*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj)$",
    "lr_lora": 1e-4, "lr_head": 2e-5, "grad_accum": 16, "epochs": 2, "max_length": 2560,
    "seed": 20261013,
}

sys.path.insert(0, str(MODEL_DIR))
from joint_schema_model import JointSchemaHead, collate_records, encode_record  # noqa: E402


def questions() -> tuple[dict, str]:
    cfg = yaml.safe_load((ROOT / "questions_bash_traj.yaml").read_text())
    qs = {k: {"type": "noul", "instructions": v["question"].strip()} for k, v in cfg["questions"].items()}
    qhash = hashlib.sha1(json.dumps({"pre": cfg["preamble"], "q": qs}, sort_keys=True).encode()).hexdigest()[:10]
    return qs, qhash


def load_split(name: str) -> list[dict]:
    return [json.loads(x) for x in (DATA / f"{name}.jsonl").read_text().splitlines()]


class LexTable:
    """Stands in for lm_head.weight inside the head: only option-token rows are ever indexed."""

    def __init__(self, token_ids: list[int], device):
        from safetensors import safe_open
        index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
        with safe_open(MODEL_DIR / index["lm_head.weight"], "pt") as f:
            w = f.get_slice("lm_head.weight")
            rows = torch.stack([w[i : i + 1][0] for i in token_ids])
        self.rows = rows.to(device=device, dtype=torch.float32)  # the head runs in fp32
        self.remap = torch.full((248320,), -1, dtype=torch.long, device=device)
        self.remap[torch.tensor(token_ids, device=device)] = torch.arange(len(token_ids), device=device)

    def __getitem__(self, ids: torch.Tensor) -> torch.Tensor:
        r = self.remap[ids]
        if (r < 0).any():
            raise KeyError("option token outside the precomputed lm_head rows")
        return self.rows[r]


class Clef:
    def __init__(self, device: str = "cuda"):
        from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
        from safetensors.torch import load_file

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.qs, self.qhash = questions()
        q4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                                llm_int8_skip_modules=["lm_head", "visual"],
                                llm_int8_enable_fp32_cpu_offload=True)
        full = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_DIR, dtype=torch.float16, quantization_config=q4,
            device_map={"model.language_model": 0, "model.visual": "cpu", "lm_head": "cpu"})
        self.lm = full.model.language_model
        del full  # visual tower and lm_head stay on the CPU and are freed with it
        self.lm.config.use_cache = False
        cfg = json.loads((MODEL_DIR / "joint_head_config.json").read_text())
        self.head = JointSchemaHead(**cfg)
        self.head.load_state_dict(load_file(MODEL_DIR / "joint_head.safetensors"), strict=True)
        self.head = self.head.to(device=device, dtype=torch.float32)
        probe = self.encode({"state": "x", "labels": {}})
        ids = sorted({int(t) for q in probe.questions for s, e in q.option_spans for t in probe.input_ids[s:e]})
        self.lex = LexTable(ids, device)

    def encode(self, rec: dict):
        return encode_record(self.tok, {"state": rec["state"], "questions": self.qs, "id": rec.get("id")},
                             max_length=CFG["max_length"])

    def logits(self, rec: dict) -> torch.Tensor:
        enc = self.encode(rec)
        batch = collate_records([enc], self.tok.pad_token_id, torch.device(self.device))
        with torch.autocast("cuda", dtype=torch.float16):
            h = self.lm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        use_cache=False, return_dict=True).last_hidden_state
        out = self.head(h.float(), batch["input_ids"], batch["attention_mask"], batch["records"], self.lex)[0]
        return torch.stack(out)  # [n_questions, 2] over ("true", "false")

    def add_lora(self):
        from peft import LoraConfig, get_peft_model
        self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.lm.enable_input_require_grads()
        self.lm = get_peft_model(self.lm, LoraConfig(
            r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"],
            target_modules=CFG["target_modules"], bias="none"))
        for p in self.lm.parameters():
            if p.requires_grad:
                p.data = p.data.float()  # fp32 LoRA params so the fp16 GradScaler can unscale them

    def save(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(str(out / "adapter"))
        torch.save(self.head.state_dict(), out / "head.pt")
        (out / "config.json").write_text(json.dumps({**CFG, "qhash": self.qhash}, indent=1))

    def load(self, out: Path):
        from peft import PeftModel
        self.lm = PeftModel.from_pretrained(self.lm, str(out / "adapter"))
        self.head.load_state_dict(torch.load(out / "head.pt"))


def cached(tag: str) -> dict:
    if not SCORES.exists():
        return {}
    rows = [json.loads(x) for x in SCORES.read_text().splitlines()]
    return {r["id"]: r for r in rows if r["tag"] == tag}


@torch.no_grad()
def score(clef: Clef, tag: str, splits=("val", "test", "test_family")):
    clef.lm.eval()
    clef.head.eval()
    done = cached(tag)
    recs = [r for s in splits for r in load_split(s) if r["id"] not in done]
    print(f"[{tag}] scoring {len(recs)} records ({len(done)} cached)", flush=True)
    RUNS.mkdir(exist_ok=True)
    t_all = time.perf_counter()
    with SCORES.open("a") as f:
        for i, r in enumerate(recs):
            t0 = time.perf_counter()
            p = clef.logits(r).float().softmax(-1)[:, 0].tolist()
            f.write(json.dumps({"tag": tag, "model": "Cloudflare/clef-flash", "qhash": clef.qhash,
                                "id": r["id"], "split": r["split"], "p": dict(zip(clef.qs, p)),
                                "latency_s": round(time.perf_counter() - t0, 3),
                                "tokens": len(clef.encode(r).input_ids)}) + "\n")
            if i % 50 == 0:
                f.flush()
                print(f"  {i}/{len(recs)} {(time.perf_counter() - t_all) / (i + 1):.2f}s/record", flush=True)


def train(clef: Clef, epochs: int, max_steps: int | None = None):
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    items = load_split("train")
    labels = [[0 if it["labels"][q] else 1 for q in clef.qs] for it in items]  # option 0 = "true"
    clef.add_lora()
    clef.head.train()
    order = list(range(len(items)))

    def batches(epoch):
        random.Random(CFG["seed"] + epoch).shuffle(order)
        clef.lm.train()
        for i in order:
            yield i

    def step_loss(i):
        z = clef.logits(items[i])
        y = torch.tensor(labels[i], device=z.device)
        return proper_loss(z, y, torch.ones_like(z, dtype=torch.bool), torch.ones(len(y), device=z.device))

    lora = [p for p in clef.lm.parameters() if p.requires_grad]
    print(f"trainable: lora {sum(p.numel() for p in lora) / 1e6:.1f}M, "
          f"head {sum(p.numel() for p in clef.head.parameters()) / 1e6:.1f}M", flush=True)
    t0 = time.perf_counter()

    def log(msg):
        print(f"{msg} [{(time.perf_counter() - t0) / 60:.1f} min, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB]", flush=True)

    stats = train_loop(
        params=[{"params": lora, "lr": CFG["lr_lora"]},
                {"params": list(clef.head.parameters()), "lr": CFG["lr_head"]}],
        batches=batches, step_loss=step_loss, epochs=epochs, n_batches_per_epoch=len(items),
        grad_accum=CFG["grad_accum"], device="cuda", max_steps=max_steps, log=log)
    print(json.dumps(stats), flush=True)
    return stats


def smoke():
    """Parity of the 4-bit HF model with the GGUF server on cached per-action states, then speed."""
    from .bash_score import clef_questions, clef_state, load_built
    clef = Clef()
    qs_old, pre, qhash_old = clef_questions()
    gguf = {json.loads(x)["key"]: json.loads(x)["p"] for x in (RUNS / "bash_clef.jsonl").read_text().splitlines()}
    clef.qs = {k: {"type": "noul", "instructions": q.question} for k, q in qs_old.items()}
    diffs = []
    for t in load_built()[:3]:
        for a in t["actions"][:4]:
            st = clef_state(pre, t["task_description"], a)
            k = hashlib.sha1(f"{qhash_old}\n{st}".encode()).hexdigest()
            if k in gguf:
                with torch.no_grad():
                    p = clef.logits({"state": st}).float().softmax(-1)[:, 0].tolist()
                diffs += [abs(a_ - gguf[k][q]) for a_, q in zip(p, clef.qs)]
                print({q: (round(a_, 3), round(gguf[k][q], 3)) for a_, q in zip(p, clef.qs)})
    print(f"parity: mean |HF 4-bit - GGUF Q4_K_M| = {sum(diffs) / max(len(diffs), 1):.3f} over {len(diffs)}")
    clef.qs, clef.qhash = questions()
    CFG["grad_accum"] = 2  # 4 optimizer steps = 8 records, enough to time a step and see peak VRAM
    train(clef, epochs=1, max_steps=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "run", "score"])
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--splits", nargs="+", default=["val", "test", "test_family"])
    ap.add_argument("--tags", nargs="+", default=["lora"], choices=["zeroshot", "lora"])
    a = ap.parse_args()
    if a.cmd == "smoke":
        return smoke()
    clef = Clef()
    if a.cmd == "run":
        score(clef, "zeroshot", a.splits)
        train(clef, a.epochs)
        clef.save(CKPT)
    else:
        if "zeroshot" in a.tags:
            score(clef, "zeroshot", a.splits)
        if "lora" not in a.tags:
            return
        clef.load(CKPT)
    score(clef, "lora", a.splits)


if __name__ == "__main__":
    main()
