from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from splade_mlx import (
    _load_tokenizer,
    _stored_dtype,
    _tokenizer_source,
    to_numpy,
)


def test_to_numpy_casts_bfloat16_to_float32():
    array = mx.array([[1.0, 2.0]], dtype=mx.bfloat16)

    result = to_numpy(array)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [[1.0, 2.0]])


def test_stored_dtype_accepts_matching_request(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"dtype": "bfloat16"}))

    assert _stored_dtype(tmp_path, "bfloat16") == "bfloat16"


def test_stored_dtype_rejects_mismatched_request(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"dtype": "bfloat16"}))

    with pytest.raises(ValueError, match="stores bfloat16 weights"):
        _stored_dtype(tmp_path, "float32")


def test_tokenizer_source_skips_weight_only_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights.safetensors").write_bytes(b"x")

    assert _tokenizer_source(tmp_path, "prithivida/Splade_PP_en_v1") == (
        "prithivida/Splade_PP_en_v1"
    )


def test_tokenizer_source_uses_shipped_tokenizer(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}")

    assert _tokenizer_source(tmp_path, "prithivida/Splade_PP_en_v1") == tmp_path


def test_load_tokenizer_falls_back_from_tiny_vocab(tmp_path, monkeypatch):
    (tmp_path / "tokenizer.json").write_text("{}")
    calls: list[str] = []

    class DummyTok:
        def __init__(self, source: str, vocab_size: int):
            self.name_or_path = source
            self.vocab_size = vocab_size

        @classmethod
        def from_pretrained(cls, source, *args, **kwargs):
            calls.append(str(source))
            vocab = 5 if Path(source) == tmp_path else 30522
            return cls(str(source), vocab)

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", DummyTok)

    tok = _load_tokenizer(tmp_path, "prithivida/Splade_PP_en_v1")

    assert tok.vocab_size == 30522
    assert calls == [str(tmp_path), "prithivida/Splade_PP_en_v1"]

