"""Configuration for the CLEF unseen-attack file experiment.

The experiment lives inside the I-Spy-Agent repository but keeps its own data,
run, and result directories so it cannot collide with the existing stegowatch
experiments.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
DATA = ROOT / "data" / "clef_unseen_attack"
RUNS = ROOT / "runs" / "clef_unseen_attack"
RESULTS = PACKAGE / "results"

# Stage 0/1 reproduce the original probe with the pre-quantised 4-bit checkpoint.
MODEL_4BIT = "meossistant/clef-flash-4bit"

# Stage 2 uses the official bf16 release and quantises at load time (QLoRA).
MODEL_CLEF = "Cloudflare/clef-flash"

# Stage 3 is the base-model control. Override with CLEF_BASE_MODEL if the
# organisation publishes the base under a different repo id.
MODEL_BASE = "Qwen/Qwen3.5-9B-Base"

SEED = 0
N_PER_FAMILY_STAGE0 = 6
N_PER_FAMILY_TRAIN = 100

# The design fixes max_seq_len at 512 for training. Evaluation can use a longer
# state so the full three-question schema fits.
MAX_TRAIN_LEN = 512
MAX_EVAL_LEN = 2048

# Generator families held out from training. This is U2.
HELD_OUT_FAMILIES = ("multi_file",)

# Real attack runs that are never trained on. This is U3.
U3_RUN_MARKER = "straj"

# Fraction of real trajectories held out for validation / honest FPR.
REAL_VAL_FRACTION = 0.15
HONEST_EVAL_FRACTION = 0.15

# Fixed operating points from the BashArena / AISI convention.
FPR_POINTS = (0.01, 0.04, 0.05, 0.10)

# Fallback head shape for Stage 3 if the CLEF release is not on disk. The
# released joint_head_config.json wins whenever it is available.
DEFAULT_HEAD_CONFIG = {
    "hidden_size": 4096,
    "width": 1024,
    "routing_layers": 2,
    "layers": 4,
    "heads": 16,
    "feedforward": 4096,
}

TRAIN_SIZES = (50, 100, 200, 400)

_LORA_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
)


def lora_target_regex(layer_indices: tuple[int, ...], num_layers: int) -> str:
    """PEFT target regex for attention projections in selected layers.

    Negative indices are resolved against ``num_layers`` (``-1`` is the last
    layer). The returned expression matches the full module path, not just the
    suffix, so it works with wrappers such as ``model.language_model.layers``.
    """
    resolved = sorted({idx % num_layers for idx in layer_indices})
    layers = "|".join(str(i) for i in resolved)
    suffixes = "|".join(_LORA_SUFFIXES)
    # Real module paths put ``self_attn`` between the layer index and the
    # projection name (``layers.31.self_attn.q_proj``); allow that path but
    # still reject other layers and non-attention modules.
    return rf".*\.layers\.({layers})\.(?:.*\.)?({suffixes})$"


def resolved_model_id(backbone: str) -> str:
    """Return the model id for a backbone, allowing environment overrides."""
    import os

    if backbone == "clef":
        return os.getenv("CLEF_MODEL_ID", MODEL_CLEF)
    if backbone == "base":
        return os.getenv("CLEF_BASE_MODEL", MODEL_BASE)
    if backbone == "clef4bit":
        return os.getenv("CLEF_4BIT_MODEL_ID", MODEL_4BIT)
    raise ValueError(f"unknown backbone: {backbone}")


def ensure_dirs() -> None:
    for path in (DATA, RUNS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)


def stage_dir(stage: str, size: int | None = None) -> Path:
    name = stage if size is None else f"{stage}/size_{size}"
    return RUNS / name


def target_layers_pattern(num_layers: int) -> str:
    """Convenience wrapper for the last four text layers."""
    return lora_target_regex((-4, -3, -2, -1), num_layers)


def is_monitor_rerun(run: object) -> bool:
    return "monitor on" in str(run or "").lower()


def is_u3_run(run: object) -> bool:
    return U3_RUN_MARKER in str(run or "").lower()


def sanitize_name(value: str) -> str:
    """Filesystem-safe short name used for run directories and cache tags."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"
