"""Torch <-> MLX numerical parity gates.

Prerequisite: `uv run python -m bench.save_reference` must have been run
(writes data/reference/{model_key}.npz from fp32 CPU torch).

Gates (from PLAN.md):
  1. MLM logits (fp32): max |delta| < 1e-3 on the saved logits subset
  2. Sparse vectors: per-row top-k term index overlap >= 99%, cosine >= 0.9999
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.workloads import P0_MODELS, PARITY_LOGITS_COUNT, REFERENCE_DIR

MODEL_KEYS = list(P0_MODELS)


@pytest.fixture(scope="module", params=MODEL_KEYS)
def model_case(request):
    import mlx.core as mx

    from splade_mlx import load

    model_key = request.param
    ref_path = REFERENCE_DIR / f"{model_key}.npz"
    if not ref_path.exists():
        pytest.skip(f"missing reference {ref_path}; run `uv run python -m bench.save_reference`")
    ref = np.load(ref_path)
    model, _ = load(P0_MODELS[model_key]["hf_id"], dtype="float32")
    ids = mx.array(ref["input_ids"])
    mask = mx.array(ref["attention_mask"])
    logits = np.array(model(ids, mask).astype(mx.float32))
    sparse = np.array(model.encode(ids, mask).astype(mx.float32))
    return model_key, ref, logits, sparse


def test_logits_parity(model_case):
    model_key, ref, logits, _ = model_case
    delta = np.abs(logits[:PARITY_LOGITS_COUNT] - ref["logits_subset"]).max()
    assert delta < 1e-3, f"{model_key}: max logits delta {delta:.2e} >= 1e-3"


def test_sparse_cosine_parity(model_case):
    model_key, ref, _, sparse = model_case
    a, b = sparse, ref["sparse"]
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = num / np.maximum(den, 1e-12)
    assert cos.min() >= 0.9999, f"{model_key}: min cosine {cos.min():.6f} < 0.9999"


def test_sparse_topk_overlap(model_case):
    model_key, ref, _, sparse = model_case
    overlaps = []
    for row_mlx, row_ref in zip(sparse, ref["sparse"]):
        active = int((row_ref > 0).sum())
        k = min(64, active)
        if k == 0:
            continue
        top_mlx = set(np.argsort(-row_mlx)[:k].tolist())
        top_ref = set(np.argsort(-row_ref)[:k].tolist())
        overlaps.append(len(top_mlx & top_ref) / k)
    mean_overlap = float(np.mean(overlaps))
    assert mean_overlap >= 0.99, f"{model_key}: top-k overlap {mean_overlap:.4f} < 0.99"
