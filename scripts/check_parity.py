"""Ad-hoc fp32 parity check (torch vs MLX) for any supported SPLADE checkpoint.

Gates (same as tests/test_parity.py):
  max |MLM logit delta| < 1e-3, sparse cosine >= 0.9999, top-64 overlap >= 99%

Usage:
    uv run python scripts/check_parity.py naver/splade-v3
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEXTS = [
    "what causes vitamin d deficiency",
    "The role of vitamin D in calcium absorption and bone health has been studied extensively.",
    "apple silicon unified memory architecture",
    "SPLADE is a sparse lexical expansion model for first-stage retrieval.",
]


def main() -> None:
    import mlx.core as mx
    import numpy as np
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from bench.bench_torch import splade_pool
    from splade_mlx import load

    src = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(src)
    tm = AutoModelForMaskedLM.from_pretrained(src, dtype=torch.float32).eval()
    enc = tok(TEXTS, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
    with torch.inference_mode():
        tl = tm(**inputs).logits
        ts = splade_pool(tl, enc["attention_mask"]).numpy()

    mm, _ = load(src, dtype="float32")
    ids, mask = mx.array(enc["input_ids"].numpy()), mx.array(enc["attention_mask"].numpy())
    ml = np.array(mm(ids, mask))
    ms = np.array(mm.encode(ids, mask))

    dl = np.abs(ml - tl.numpy()).max()
    cos = (ms * ts).sum(1) / np.maximum(
        np.linalg.norm(ms, axis=1) * np.linalg.norm(ts, axis=1), 1e-12
    )
    ov = []
    for a, b in zip(ms, ts):
        k = min(64, int((b > 0).sum()))
        if k:
            ov.append(len(set(np.argsort(-a)[:k].tolist()) & set(np.argsort(-b)[:k].tolist())) / k)
    print(f"{src}: max|logit delta|={dl:.2e}  min cosine={cos.min():.6f}  topk overlap={np.mean(ov):.4f}")
    assert dl < 1e-3 and cos.min() >= 0.9999 and np.mean(ov) >= 0.99
    print(f"PARITY_OK {src}")


if __name__ == "__main__":
    main()
