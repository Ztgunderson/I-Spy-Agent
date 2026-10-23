"""Shared file-level model wrapper for Stages 2 and 3.

Stage 2 starts from the released ``Cloudflare/clef-flash`` backbone and its
joint schema head. Stage 3 starts from ``Qwen/Qwen3.5-9B-Base`` with a **fresh**
joint head and the same LoRA recipe. Both share the encoder, the option
lexicon, the quantisation path, and the scoring interface.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

from . import config
from .schema import SCHEMA


def schema_hash(schema: dict) -> str:
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def file_state(record: Any) -> str:
    """Compose the state exactly as Stage 0 did: PURPOSE, separator, FILE."""
    if isinstance(record, dict) and "state" in record:
        return str(record["state"])
    purpose = getattr(record, "purpose", "")
    content = getattr(record, "content", "")
    return f"PURPOSE: {purpose}\n---\n{content}"


def record_id(record: Any) -> str:
    if isinstance(record, dict):
        return str(record.get("id", "unknown"))
    return str(getattr(record, "id", "unknown"))


def resolve_model_path(model_id_or_path: str | Path) -> Path:
    """Resolve a local model directory or download a Hub snapshot."""
    path = Path(model_id_or_path).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(repo_id=str(model_id_or_path), token=os.getenv("HF_TOKEN"))
    )


def import_joint_schema(model_dir: Path):
    """Import the release's own encoder/head code so prompts match exactly."""
    sys.path.insert(0, str(model_dir))
    import joint_schema_model  # type: ignore

    return joint_schema_model


def _text_backbone(full: Any) -> Any:
    """Return the text transformer from a CausalLM or ConditionalGeneration."""
    base = full.get_base_model() if hasattr(full, "get_base_model") else full
    inner = getattr(base, "model", None)
    if inner is not None:
        language_model = getattr(inner, "language_model", None)
        if language_model is not None:
            return language_model
        return inner
    language_model = getattr(base, "language_model", None)
    if language_model is not None:
        return language_model
    return base


class OptionLexicon:
    """Read-only table of the ``lm_head`` rows used by schema option tokens.

    The joint head only indexes rows for option-token ids. Keeping those rows
    alone avoids putting the full 248k x 4096 output embedding on the GPU.
    """

    def __init__(self, token_ids: list[int], rows: torch.Tensor, vocab_size: int):
        self.rows = rows
        self.token_ids = list(token_ids)
        self.remap = torch.full((vocab_size,), -1, dtype=torch.long, device=rows.device)
        self.remap[torch.tensor(self.token_ids, device=rows.device)] = torch.arange(
            len(self.token_ids), device=rows.device
        )

    def __getitem__(self, ids: torch.Tensor) -> torch.Tensor:
        remapped = self.remap[ids]
        if (remapped < 0).any():
            raise KeyError("option token outside the precomputed lm_head rows")
        return self.rows[remapped]

    def to(self, device: str | torch.device) -> "OptionLexicon":
        self.rows = self.rows.to(device=device)
        self.remap = self.remap.to(device=device)
        return self

    @classmethod
    def from_weight(cls, weight: torch.Tensor, token_ids: list[int]) -> "OptionLexicon":
        rows = torch.stack([weight[i] for i in token_ids]).to(dtype=torch.float32)
        return cls(token_ids, rows, weight.shape[0])

    @classmethod
    def from_safetensors(
        cls, model_dir: Path, token_ids: list[int]
    ) -> "OptionLexicon":
        from safetensors import safe_open

        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())["weight_map"]
            shard = index["lm_head.weight"]
        else:
            shard = "model.safetensors"
        with safe_open(model_dir / shard, "pt") as handle:
            weight = handle.get_slice("lm_head.weight")
            rows = torch.stack([weight[i : i + 1][0] for i in token_ids])
        return cls(token_ids, rows.to(dtype=torch.float32), weight.get_shape()[0])


def _option_token_ids(encoded: Any) -> list[int]:
    return sorted(
        {
            int(token)
            for question in encoded.questions
            for start, end in question.option_spans
            for token in encoded.input_ids[start:end]
        }
    )


class FileModel:
    """One file-level classifier: text backbone + joint schema head."""

    def __init__(
        self,
        *,
        model_dir: Path,
        tokenizer: Any,
        joint_schema_model: Any,
        language_model: Any,
        head: Any,
        lexicon: OptionLexicon,
        backbone: str,
        device: str,
        schema: dict | None = None,
    ):
        self.model_dir = model_dir
        self.tokenizer = tokenizer
        self.joint = joint_schema_model
        self.lm = language_model
        self.head = head
        self.lex = lexicon
        self.backbone = backbone
        self.device = device
        self.schema = schema or SCHEMA
        self.qhash = schema_hash(self.schema)
        self._peft_attached = False

    def encode(self, record: Any, schema: dict | None = None, max_length: int | None = None):
        schema = schema or self.schema
        max_length = max_length or config.MAX_EVAL_LEN
        return self.joint.encode_record(
            self.tokenizer,
            {
                "state": file_state(record),
                "questions": schema,
                "id": record_id(record),
            },
            max_length=max_length,
        )

    def logits_encoded(self, encoded: Any) -> torch.Tensor:
        batch = self.joint.collate_records(
            [encoded], self.tokenizer.pad_token_id, torch.device(self.device)
        )
        autocast_device = "cuda" if self.device.startswith("cuda") else "cpu"
        with torch.autocast(
            autocast_device,
            dtype=torch.float16,
            enabled=self.device.startswith("cuda"),
        ):
            hidden = self.lm(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        outputs = self.head(
            hidden.float(),
            batch["input_ids"],
            batch["attention_mask"],
            batch["records"],
            self.lex,
        )[0]
        return torch.stack(outputs)

    def logits(
        self, record: Any, schema: dict | None = None, max_length: int | None = None
    ) -> torch.Tensor:
        return self.logits_encoded(
            self.encode(record, schema=schema, max_length=max_length)
        )

    def score(
        self, record: Any, schema: dict | None = None, max_length: int | None = None
    ) -> dict[str, dict[str, float]]:
        encoded = self.encode(record, schema=schema, max_length=max_length)
        logits = self.logits_encoded(encoded)
        probabilities = logits.float().softmax(-1).cpu()
        out: dict[str, dict[str, float]] = {}
        for question, row in zip(encoded.questions, probabilities):
            out[question.question_id] = {
                option_id: float(value)
                for option_id, value in zip(question.option_ids, row.tolist())
            }
        return out

    def add_lora(self, last_layers: int = 4) -> None:
        from peft import LoraConfig, get_peft_model

        self.lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.lm.enable_input_require_grads()
        num_layers = int(self.lm.config.num_hidden_layers)
        self.lm = get_peft_model(
            self.lm,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=config.lora_target_regex(
                    tuple(range(-last_layers, 0)), num_layers
                ),
                bias="none",
            ),
        )
        for parameter in self.lm.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self._peft_attached = True

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.lm.parameters() if p.requires_grad] + list(
            self.head.parameters()
        )

    def save(self, out_dir: Path | str) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self._peft_attached:
            self.lm.save_pretrained(str(out_dir / "adapter"))
        torch.save(self.head.state_dict(), out_dir / "head.pt")
        (out_dir / "config.json").write_text(
            json.dumps(
                {
                    "backbone": self.backbone,
                    "model_dir": str(self.model_dir),
                    "qhash": self.qhash,
                    "schema": self.schema,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out_dir

    def load(self, out_dir: Path | str) -> None:
        out_dir = Path(out_dir)
        if (out_dir / "adapter").exists():
            from peft import PeftModel

            self.lm = PeftModel.from_pretrained(self.lm, str(out_dir / "adapter"))
            self._peft_attached = True
        state = torch.load(out_dir / "head.pt", map_location=self.device)
        self.head.load_state_dict(state)


def _load_quantized(model_dir: Path, *, conditional_generation: bool, device: str):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if device.startswith("cuda"):
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32
    quant_config = None
    if device.startswith("cuda"):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head", "visual"],
            llm_int8_enable_fp32_cpu_offload=True,
        )

    kwargs = {
        "dtype": compute_dtype,
        "device_map": {"": device},
    }
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config

    if conditional_generation:
        from transformers import Qwen3_5ForConditionalGeneration

        full = Qwen3_5ForConditionalGeneration.from_pretrained(str(model_dir), **kwargs)
    else:
        try:
            full = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
        except (ValueError, KeyError):
            from transformers import Qwen3_5ForConditionalGeneration

            full = Qwen3_5ForConditionalGeneration.from_pretrained(
                str(model_dir), **kwargs
            )
    return full


def load_file_model(
    backbone: str,
    model_id: str | Path | None = None,
    *,
    device: str | None = None,
    quantized: bool = True,
    pretrained_head: bool | None = None,
    schema: dict | None = None,
) -> FileModel:
    """Load Stage 2 (``clef``) or Stage 3 (``base``) model.

    ``pretrained_head`` defaults to ``True`` only for the CLEF backbone. Stage 3
    must pass ``False`` (or use ``backbone="base"``) so the control starts from a
    fresh head.
    """
    if backbone not in {"clef", "base"}:
        raise ValueError("backbone must be 'clef' or 'base'")
    if pretrained_head is None:
        pretrained_head = backbone == "clef"
    if backbone == "base" and pretrained_head:
        raise ValueError("Stage 3 base control must use a fresh head")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = resolve_model_path(
        model_id or config.resolved_model_id(backbone)
    )
    # Stage 3 uses the base model's tokenizer and weights, but the schema
    # encoder and head definition come from the CLEF release.
    joint_dir = model_dir
    if backbone == "base":
        joint_dir = resolve_model_path(config.MODEL_CLEF)
    joint = import_joint_schema(joint_dir)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    full = _load_quantized(
        model_dir, conditional_generation=backbone == "clef", device=device
    )
    lm = _text_backbone(full)
    lm.config.use_cache = False

    head_path = joint_dir / "joint_head_config.json"
    head_config = (
        json.loads(head_path.read_text())
        if head_path.exists()
        else dict(config.DEFAULT_HEAD_CONFIG)
    )
    head = joint.JointSchemaHead(**head_config)
    if pretrained_head:
        from safetensors.torch import load_file

        head.load_state_dict(
            load_file(joint_dir / "joint_head.safetensors"), strict=True
        )
    head = head.to(device=device, dtype=torch.float32)

    probe = joint.encode_record(
        tokenizer,
        {"state": "x", "questions": schema or SCHEMA, "id": "probe"},
        max_length=config.MAX_EVAL_LEN,
    )
    token_ids = _option_token_ids(probe)
    if backbone == "clef":
        lexicon = OptionLexicon.from_safetensors(model_dir, token_ids)
    else:
        weight = full.get_output_embeddings().weight
        lexicon = OptionLexicon.from_weight(weight, token_ids)
    lexicon.to(device)

    return FileModel(
        model_dir=model_dir,
        tokenizer=tokenizer,
        joint_schema_model=joint,
        language_model=lm,
        head=head,
        lexicon=lexicon,
        backbone=backbone,
        device=device,
        schema=schema,
    )
