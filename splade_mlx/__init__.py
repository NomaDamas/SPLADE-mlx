"""splade_mlx: SPLADE and V-SPLADE sparse retrieval models on Apple Silicon (MLX).

- Text SPLADE (BERT/DistilBERT): :func:`load`, :func:`load_pair`
- V-SPLADE visual document retrieval (ModernVBERT, arXiv:2605.30917):
  :func:`splade_mlx.convert_vsplade.load_vsplade`
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .convert import DEFAULT_OUT_DIR, convert
from .models.bert import BertConfig, SpladeModel

__all__ = ["load", "load_pair", "to_numpy", "SpladeModel", "SpladePair"]

# Real tokenizer artifacts. config.json / tokenizer_config.json alone are not
# enough: AutoTokenizer.from_pretrained() will then instantiate an empty BERT
# tokenizer (vocab_size == 5) instead of raising.
_TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "vocab.txt",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
)
_MIN_TOKENIZER_VOCAB = 1000



def _resolve(model: str, dtype: str | None, out_dir: Path) -> Path:
    """Return a local dir containing MLX-format weights.safetensors + config.json.

    Accepts: (a) a local pre-converted dir, (b) an HF repo already in MLX
    format (e.g. NomaDamas/*-mlx; stored dtype is used as-is), or (c) an
    original HF checkpoint id, converted to `dtype` on first use and cached.
    """
    p = Path(model)
    if p.is_dir() and (p / "config.json").exists():
        return p

    from huggingface_hub import hf_hub_download, snapshot_download

    try:
        cfg = json.loads(Path(hf_hub_download(model, "config.json")).read_text())
    except Exception:
        cfg = {}
    if "bert" in cfg:  # already MLX format on the hub
        return Path(snapshot_download(model))

    conversion_dtype = dtype or "float32"
    model_name = model.split("/")[-1]
    dest = out_dir / f"{model_name}-{conversion_dtype}"
    if not (dest / "weights.safetensors").exists():
        convert(model, out_dir=dest.parent, dtype=conversion_dtype)
        # convert() writes to out_dir/model_name; move to dtype-suffixed dir
        plain = dest.parent / model_name
        if plain != dest:
            plain.rename(dest)
    return dest


def _has_tokenizer_files(path: Path) -> bool:
    return any((path / name).is_file() for name in _TOKENIZER_ARTIFACTS)


def _tokenizer_source(dest: Path, hf_id: str) -> str | Path:
    """Prefer a self-contained tokenizer in `dest`; otherwise the upstream id."""
    if _has_tokenizer_files(dest):
        return dest
    if not hf_id:
        raise ValueError(f"{dest} has no tokenizer files and no upstream hf_id")
    return hf_id


def _load_tokenizer(dest: Path, hf_id: str):
    from transformers import AutoTokenizer

    source = _tokenizer_source(dest, hf_id)
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.vocab_size < _MIN_TOKENIZER_VOCAB and source != hf_id:
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
    return tokenizer


def _stored_dtype(dest: Path, requested_dtype: str | None) -> str:
    config = json.loads((dest / "config.json").read_text())
    stored_dtype = config.get("dtype")
    if stored_dtype is None:
        raise ValueError(f"{dest} does not declare its stored weight dtype")
    if requested_dtype is not None and requested_dtype != stored_dtype:
        raise ValueError(
            f"{dest} stores {stored_dtype} weights; requested dtype={requested_dtype}. "
            "Load without dtype to use the stored weights, or load the original "
            "upstream checkpoint to create a conversion at the requested dtype."
        )
    return stored_dtype


def to_numpy(array: mx.array, dtype: mx.Dtype = mx.float32) -> np.ndarray:
    """Evaluate an MLX result and export it as a NumPy-compatible dtype.

    NumPy cannot consume MLX bfloat16 buffers directly, so sparse vectors are
    cast to float32 by default before transfer.
    """
    converted = array.astype(dtype)
    mx.eval(converted)
    return np.array(converted)


def load(
    hf_id: str,
    dtype: str | None = None,
    quantize_bits: int | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
):
    """Load a SPLADE model as MLX, converting from HF on first use.

    Returns (model, tokenizer). Tokenizer is the HF fast tokenizer so that
    tokenization is bit-identical to the PyTorch reference path.
    """
    dest = _resolve(hf_id, dtype, out_dir)
    config = json.loads((dest / "config.json").read_text())
    _stored_dtype(dest, dtype)
    model = SpladeModel(BertConfig.from_hf(config["bert"]))
    model.load_weights(str(dest / "weights.safetensors"))
    if quantize_bits is not None:
        nn.quantize(model, group_size=64, bits=quantize_bits)
    mx.eval(model.parameters())
    tokenizer = _load_tokenizer(dest, config["hf_id"])
    return model, tokenizer


class SpladePair:
    """Asymmetric pair of SPLADE text encoders (separate query and document models)."""

    def __init__(self, query_model, query_tokenizer, doc_model, doc_tokenizer):
        self.query_model = query_model
        self.query_tokenizer = query_tokenizer
        self.doc_model = doc_model
        self.doc_tokenizer = doc_tokenizer

    def _encode(self, model, tokenizer, texts: list[str], max_length: int) -> mx.array:
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="np"
        )
        kwargs = {}
        if "token_type_ids" in enc:
            kwargs["token_type_ids"] = mx.array(enc["token_type_ids"])
        out = model.encode(
            mx.array(enc["input_ids"]), mx.array(enc["attention_mask"]), **kwargs
        )
        mx.eval(out)
        return out

    def encode_query(self, texts: list[str], max_length: int = 64) -> mx.array:
        return self._encode(self.query_model, self.query_tokenizer, texts, max_length)

    def encode_doc(self, texts: list[str], max_length: int = 256) -> mx.array:
        return self._encode(self.doc_model, self.doc_tokenizer, texts, max_length)

    def encode_query_numpy(self, texts: list[str], max_length: int = 64) -> np.ndarray:
        return to_numpy(self.encode_query(texts, max_length))

    def encode_doc_numpy(self, texts: list[str], max_length: int = 256) -> np.ndarray:
        return to_numpy(self.encode_doc(texts, max_length))


def load_pair(
    query_hf_id: str,
    doc_hf_id: str,
    dtype: str | None = None,
    quantize_bits: int | None = None,
) -> SpladePair:
    qm, qt = load(query_hf_id, dtype=dtype, quantize_bits=quantize_bits)
    dm, dt = load(doc_hf_id, dtype=dtype, quantize_bits=quantize_bits)
    return SpladePair(qm, qt, dm, dt)
