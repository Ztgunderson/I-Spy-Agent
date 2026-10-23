"""Stage 2 (CLEF QLoRA) and Stage 3 (Qwen base + fresh head) training.

Both stages train only the ``covert`` question with the same file-level data,
splits, LoRA recipe, optimizer budget, and evaluation protocol. Stage 2 starts
from the released CLEF joint head; Stage 3 starts from a freshly initialised
head on the base backbone.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional

from . import config
from .data import FileRecord, read_dataset
from .model import load_file_model
from .schema import TRAIN_SCHEMA


@dataclass(frozen=True)
class StageConfig:
    name: str
    backbone: str
    model_id: str
    pretrained_head: bool
    epochs: int = 2
    lr_lora: float = 1e-4
    lr_head: float = 2e-5
    grad_accum: int = 16
    max_length: int = config.MAX_TRAIN_LEN
    seed: int = config.SEED


STAGES = {
    "stage2": StageConfig(
        name="stage2",
        backbone="clef",
        model_id=config.MODEL_CLEF,
        pretrained_head=True,
    ),
    "stage3": StageConfig(
        name="stage3",
        backbone="base",
        model_id=config.MODEL_BASE,
        pretrained_head=False,
    ),
}


def _targets(records: list[FileRecord]) -> list[float]:
    """BCE target for the covert logit: 1 = covert, 0 = clean."""
    return [float(record.label) for record in records]


def sample_train(
    records: list[FileRecord], size: int, seed: int = config.SEED
) -> list[FileRecord]:
    """Take whole groups until at least ``size`` training records are selected.

    Synthetic clean/covert pairs share a ``group_id``; real files share a
    trajectory id. Sampling by group prevents a pair or trajectory from being
    split across the train subset and its leftovers.
    """
    train = [record for record in records if record.split == "train"]
    groups: dict[str, list[FileRecord]] = {}
    for record in train:
        groups.setdefault(record.group_id, []).append(record)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    selected: list[FileRecord] = []
    for key in keys:
        selected.extend(groups[key])
        if len(selected) >= size:
            break
    return selected


def _optimizer(model, cfg: StageConfig):
    lora = [p for p in model.lm.parameters() if p.requires_grad]
    head = list(model.head.parameters())
    groups = [
        {"params": lora, "lr": cfg.lr_lora, "name": "lora"},
        {"params": head, "lr": cfg.lr_head, "name": "head"},
    ]
    if torch.cuda.is_available():
        try:
            from bitsandbytes.optim import Adam8bit

            return Adam8bit(groups)
        except Exception:  # noqa: BLE001 - optional dependency path
            pass
    return torch.optim.AdamW(groups)


def train_one(
    cfg: StageConfig,
    records: list[FileRecord],
    out_dir: Path | str,
    max_steps: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Train one data-efficiency point and save the adapter/head."""
    out_dir = Path(out_dir)
    items = list(records)
    if dry_run:
        targets = _targets(items)
        result = {
            "stage": cfg.name,
            "dry_run": True,
            "n_train": len(items),
            "n_positive": int(sum(targets)),
            "n_negative": int(len(targets) - sum(targets)),
            "groups": len({record.group_id for record in items}),
        }
        print(json.dumps(result, indent=2))
        return result

    config.ensure_dirs()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = load_file_model(
        cfg.backbone,
        cfg.model_id,
        pretrained_head=cfg.pretrained_head,
        schema=TRAIN_SCHEMA,
    )
    model.add_lora()
    model.head.train()
    if hasattr(model.lm, "train"):
        model.lm.train()

    optimizer = _optimizer(model, cfg)
    loss_fn = functional.binary_cross_entropy_with_logits
    accum = max(1, cfg.grad_accum)
    steps = 0
    started = time.time()
    losses: list[float] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        order = list(range(len(items)))
        random.Random(cfg.seed + epoch).shuffle(order)
        for position, index in enumerate(order):
            record = items[index]
            logits = model.logits(
                record, schema=TRAIN_SCHEMA, max_length=cfg.max_length
            )
            target = torch.tensor(
                [[_targets([record])[0]]], device=logits.device, dtype=logits.dtype
            )
            loss = loss_fn(logits[:, 0].view(1, 1), target) / accum
            loss.backward()
            losses.append(float(loss.detach().cpu()) * accum)

            if (position + 1) % accum == 0 or position + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                if steps % 10 == 0:
                    elapsed = time.time() - started
                    peak = (
                        torch.cuda.max_memory_allocated() / 2**30
                        if torch.cuda.is_available()
                        else float("nan")
                    )
                    print(
                        f"[{cfg.name}] step {steps} loss {sum(losses[-10:]) / min(10, len(losses)):.4f} "
                        f"{elapsed / 60:.1f} min peak {peak:.1f} GiB",
                        flush=True,
                    )
                if max_steps and steps >= max_steps:
                    break
        if max_steps and steps >= max_steps:
            break

    model.save(out_dir)
    result = {
        **asdict(cfg),
        "n_train": len(items),
        "n_positive": int(sum(_targets(items))),
        "n_negative": int(len(items) - sum(_targets(items))),
        "groups": len({record.group_id for record in items}),
        "steps": steps,
        "final_loss": float(sum(losses[-20:]) / min(20, len(losses))) if losses else None,
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available()
            else None
        ),
        "elapsed_s": time.time() - started,
        "checkpoint": str(out_dir),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def run_sweep(
    stage: str,
    sizes: tuple[int, ...] | list[int],
    records: list[FileRecord],
    out_root: Path | str = config.RUNS,
    max_steps: int | None = None,
    dry_run: bool = False,
) -> list[dict]:
    cfg = STAGES[stage]
    results = []
    for size in sizes:
        subset = sample_train(records, size=size, seed=cfg.seed)
        result = train_one(
            cfg,
            subset,
            Path(out_root) / stage / f"size_{size}",
            max_steps=max_steps,
            dry_run=dry_run,
        )
        results.append(result)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / f"{stage}_sweep.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["stage2", "stage3", "smoke"])
    parser.add_argument("--sizes", nargs="+", type=int, default=list(config.TRAIN_SIZES))
    parser.add_argument("--stage", choices=["stage2", "stage3"], default="stage2")
    parser.add_argument("--dataset", type=Path, default=config.DATA / "dataset.jsonl")
    parser.add_argument("--out-root", type=Path, default=config.RUNS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="select and print the training subset without loading a model",
    )
    args = parser.parse_args()

    records = read_dataset(args.dataset)
    if args.command == "smoke":
        cfg = STAGES[args.stage]
        subset = sample_train(records, size=min(8, len(records)), seed=cfg.seed)
        train_one(
            cfg,
            subset,
            args.out_root / f"{args.stage}_smoke",
            max_steps=args.max_steps or 2,
            dry_run=args.dry_run,
        )
        return
    run_sweep(
        args.command,
        args.sizes,
        records,
        out_root=args.out_root,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
