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
from dataclasses import dataclass
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
        # No autograd graph at scoring time: the head's parameters require grad,
        # and keeping activations for a 2048-token file OOMs a 12 GiB card.
        with torch.inference_mode():
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


def import_d3(package_dir: Path):
    """Import the d3 release's own prompt format and encoder so prompts match exactly."""
    sys.path.insert(0, str(package_dir))
    import d3_format  # type: ignore
    import d3_runtime  # type: ignore

    return d3_format, d3_runtime


class NoncausalEncoder:
    """d3's ``PromptEncoder`` for one unpadded prompt.

    Softmax-attention layers get a key mask (every position reads every other);
    the Gated DeltaNet layers keep their causal scan. With no padding, d3 passes
    no recurrent mask, so this matches its runtime on the installed transformers.
    """

    def __init__(self, text_model: Any):
        self.text = text_model
        depth = text_model.config.num_hidden_layers
        self.stack = [
            (layer, kind == "full_attention")
            for layer, kind in zip(text_model.layers[:depth], text_model.config.layer_types[:depth])
        ]

    def __call__(self, embeds: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        rows, width = present.shape
        ramp = torch.arange(width, device=embeds.device).view(1, 1, -1).expand(4, rows, -1)
        positions, mrope = ramp[0], ramp[1:]
        rope = self.text.rotary_emb(embeds, mrope)
        keys = present.bool()[:, None, None, :]
        hidden = embeds
        for layer, softmax in self.stack:
            hidden = layer(
                hidden,
                position_embeddings=rope,
                attention_mask=keys if softmax else None,
                position_ids=positions,
                past_key_values=None,
                use_cache=False,
            )
        return self.text.norm(hidden)


class D3Head(torch.nn.Module):
    """The readout rows of the two noul answer codes (false, true); trainable."""

    def __init__(self, rows: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(rows.float())

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return pooled.float() @ self.weight.T


@dataclass
class D3Encoded:
    input_ids: list[int]  # the first question's prompt (token count for the score cache)
    questions: list[tuple[str, list[int]]]


class D3FileModel:
    """d3 code-readout model behind the FileModel interface (noul questions only).

    The prompt comes from the release's ``d3_format.render``. The backbone runs
    with d3's noncausal ``PromptEncoder``, and the last position is scored
    against the readout rows of the answer codes for false and true.
    ``logits`` returns one log-odds (true - false) per question, which is what
    the trainer's BCE and ``score`` expect.
    """

    def __init__(self, *, model_dir, package_dir, tokenizer, fmt, runtime, text_model,
                 head, codes, backbone, device, schema=None):
        self.model_dir = model_dir
        self.package_dir = package_dir
        self.tokenizer = tokenizer
        self.fmt = fmt
        self.text = text_model
        self.lm = text_model
        self.encoder = NoncausalEncoder(text_model)
        self.head = head
        self.codes = codes
        self.backbone = backbone
        self.device = device
        self.schema = schema or SCHEMA
        self.qhash = schema_hash(self.schema)
        self._peft_attached = False

    def _ids(self, state: str, question: dict) -> list[int]:
        text = self.fmt.render(self.tokenizer, state, question, self.codes)
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode(self, record: Any, schema: dict | None = None, max_length: int | None = None):
        schema = schema or self.schema
        max_length = max_length or config.MAX_EVAL_LEN
        state = file_state(record)
        questions = []
        for qid, question in schema.items():
            if question.get("type") != "noul":
                raise ValueError("the d3 file model answers noul questions only")
            ids = self._ids(state, question)
            if len(ids) > max_length:
                # d3 itself never truncates; cut the state so the prompt fits.
                state_ids = self.tokenizer(state, add_special_tokens=False)["input_ids"]
                keep = max(1, len(state_ids) - (len(ids) - max_length))
                ids = self._ids(self.tokenizer.decode(state_ids[:keep]), question)
            questions.append((qid, ids))
        return D3Encoded(input_ids=questions[0][1], questions=questions)

    def logits_encoded(self, encoded: D3Encoded) -> torch.Tensor:
        rows = []
        autocast_device = "cuda" if self.device.startswith("cuda") else "cpu"
        for _, ids in encoded.questions:
            tokens = torch.tensor([ids], device=self.device)
            present = torch.ones_like(tokens)
            with torch.autocast(autocast_device, dtype=torch.float16,
                                enabled=self.device.startswith("cuda")):
                embeds = self.text.get_input_embeddings()(tokens)
                pooled = self.encoder(embeds, present)[:, -1]
            pair = self.head(pooled)[0]
            rows.append((pair[1] - pair[0]).view(1))
        return torch.stack(rows)

    def logits(self, record: Any, schema: dict | None = None, max_length: int | None = None):
        return self.logits_encoded(self.encode(record, schema=schema, max_length=max_length))

    def score(self, record: Any, schema: dict | None = None, max_length: int | None = None):
        encoded = self.encode(record, schema=schema, max_length=max_length)
        with torch.inference_mode():
            log_odds = self.logits_encoded(encoded).float().cpu()
        out: dict[str, dict[str, float]] = {}
        for (qid, _), value in zip(encoded.questions, log_odds[:, 0].tolist()):
            p = float(torch.sigmoid(torch.tensor(value)))
            out[qid] = {"false": 1.0 - p, "true": p}
        return out

    def add_lora(self, last_layers: int = 4) -> None:
        from peft import LoraConfig, get_peft_model

        num_layers = int(self.text.config.num_hidden_layers)
        # LoRA is injected into self.text in place, so the encoder keeps using it.
        self.lm = get_peft_model(
            self.text,
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
                    "package_dir": str(self.package_dir),
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

            # Wraps self.text in place; the encoder keeps calling the same layers.
            self.lm = PeftModel.from_pretrained(self.text, str(out_dir / "adapter"))
            self._peft_attached = True
        self.head.load_state_dict(torch.load(out_dir / "head.pt", map_location=self.device))


def load_d3_file_model(
    backbone: str,
    model_id: str | Path | None = None,
    *,
    device: str | None = None,
    quantized: bool = True,
    pretrained_head: bool | None = None,
    schema: dict | None = None,
) -> D3FileModel:
    """``d3``: the d3-lite release with its readout. ``d3base``: the plain
    Qwen3.5-0.8B-Base backbone with a fresh readout, prompted the d3 way."""
    if pretrained_head is None:
        pretrained_head = backbone == "d3"
    if backbone == "d3base" and pretrained_head:
        raise ValueError("the d3base control must use a fresh head")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    package_dir = resolve_model_path(config.resolved_model_id("d3"))
    model_dir = resolve_model_path(model_id or config.resolved_model_id(backbone))
    fmt, runtime = import_d3(package_dir)
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5Model

    tokenizer = AutoTokenizer.from_pretrained(str(package_dir))
    decision = json.loads((package_dir / "decision_config.json").read_text())
    codes = decision["codes"]

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    kwargs: dict[str, Any] = {"dtype": dtype, "device_map": {"": device}, "attn_implementation": "sdpa"}
    if quantized and device.startswith("cuda"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head", "visual"],
        )
    full = Qwen3_5Model.from_pretrained(str(model_dir), **kwargs)
    text_model = full.language_model
    text_model.config.use_cache = False
    hidden = int(text_model.config.hidden_size)

    if pretrained_head:
        from safetensors.torch import load_file

        rows = load_file(str(package_dir / "readout.safetensors"))["weight"][:2]
    else:
        generator = torch.Generator().manual_seed(config.SEED)
        rows = torch.randn(2, hidden, generator=generator) * 0.02
    head = D3Head(rows).to(device)
    return D3FileModel(
        model_dir=model_dir,
        package_dir=package_dir,
        tokenizer=tokenizer,
        fmt=fmt,
        runtime=runtime,
        text_model=text_model,
        head=head,
        codes=codes,
        backbone=backbone,
        device=device,
        schema=schema,
    )


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
    if backbone in {"d3", "d3base"}:
        return load_d3_file_model(
            backbone, model_id, device=device, quantized=quantized,
            pretrained_head=pretrained_head, schema=schema,
        )
    if backbone not in {"clef", "base"}:
        raise ValueError("backbone must be 'clef', 'base', 'd3' or 'd3base'")
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
        joint_dir = resolve_model_path(config.resolved_model_id("clef"))
    joint = import_joint_schema(joint_dir)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # Qwen3.5-9B-Base ships the same Qwen3_5ForConditionalGeneration layout as
    # CLEF. Load both with that class so Stage 2 vs 3 differ only in weights and
    # head init (AutoModelForCausalLM can resolve to a class with other key names).
    full = _load_quantized(model_dir, conditional_generation=True, device=device)
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
