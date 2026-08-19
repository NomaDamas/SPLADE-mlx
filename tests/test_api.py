from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from splade_mlx import _stored_dtype, to_numpy


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

