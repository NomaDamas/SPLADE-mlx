"""splade_mlx: SPLADE / V-SPLADE sparse retrieval models on Apple Silicon (MLX)."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .convert import DEFAULT_OUT_DIR, convert
from .models.bert import BertConfig, SpladeModel

__all__ = ["load", "load_pair", "SpladeModel", "SpladePair"]


def load(
    hf_id: str,
    dtype: str = "float32",
    quantize_bits: int | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
):
    """Load a SPLADE model as MLX, converting from HF on first use.

    Returns (model, tokenizer). Tokenizer is the HF fast tokenizer so that
    tokenization is bit-identical to the PyTorch reference path.
    """
    from transformers import AutoTokenizer

    model_name = hf_id.split("/")[-1]
    dest = out_dir / f"{model_name}-{dtype}"
    if not (dest / "weights.safetensors").exists():
        convert(hf_id, out_dir=dest.parent, dtype=dtype)
        # convert() writes to out_dir/model_name; move to dtype-suffixed dir
        plain = dest.parent / model_name
        if plain != dest:
            plain.rename(dest)

    config = json.loads((dest / "config.json").read_text())
    model = SpladeModel(BertConfig.from_hf(config["bert"]))
    model.load_weights(str(dest / "weights.safetensors"))
    if quantize_bits is not None:
        nn.quantize(model, group_size=64, bits=quantize_bits)
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(config["hf_id"])
    return model, tokenizer


class SpladePair:
    """Asymmetric V-SPLADE pair: separate query and document encoders."""

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


def load_pair(
    query_hf_id: str = "naver/efficient-splade-V-large-query",
    doc_hf_id: str = "naver/efficient-splade-V-large-doc",
    dtype: str = "float32",
    quantize_bits: int | None = None,
) -> SpladePair:
    qm, qt = load(query_hf_id, dtype=dtype, quantize_bits=quantize_bits)
    dm, dt = load(doc_hf_id, dtype=dtype, quantize_bits=quantize_bits)
    return SpladePair(qm, qt, dm, dt)
