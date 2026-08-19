"""V-SPLADE fp32 parity check: torch (trust_remote_code) vs MLX.

Gates: doc-side max |logit delta| < 1e-3, sparse cosine >= 0.9999,
top-64 term overlap >= 99%; query-side lookup table must match the shipped
Sentence Transformers static embedding exactly.

Usage:
    uv run python scripts/check_parity_vsplade.py [naver/v-splade-efficient]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw


def make_doc_image(seed: int, size=(700, 900)) -> Image.Image:
    """Deterministic synthetic document page (text lines + table + figure)."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    d.text((40, 20), f"Quarterly Report {2020 + seed}", fill="black")
    for i in range(24):  # text lines
        y = 70 + i * 22
        d.rectangle([40, y, 40 + int(rng.integers(300, 600)), y + 8], fill=(30, 30, 30))
    for r in range(5):  # table
        for c in range(4):
            x, y = 60 + c * 140, 620 + r * 40
            d.rectangle([x, y, x + 130, y + 34], outline="black")
            d.text((x + 8, y + 8), f"{rng.integers(0, 999)}", fill="black")
    return img


def main() -> None:
    import torch
    from transformers import AutoModelForMaskedLM, AutoProcessor

    hf_id = sys.argv[1] if len(sys.argv) > 1 else "naver/v-splade-efficient"

    processor = AutoProcessor.from_pretrained(hf_id)
    images = [[make_doc_image(0)], [make_doc_image(1, size=(900, 640))]]
    texts = ["<image>", "<image>"]
    enc = processor(text=texts, images=images, return_tensors="pt", padding=True)
    print("input_ids:", tuple(enc["input_ids"].shape), "pixel_values:", tuple(enc["pixel_values"].shape))

    # --- torch reference ---
    tm = AutoModelForMaskedLM.from_pretrained(hf_id, trust_remote_code=True, dtype=torch.float32)
    tm.eval()
    with torch.inference_mode():
        tl = tm(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            pixel_values=enc["pixel_values"],
        ).logits
        mask = enc["attention_mask"].unsqueeze(-1).to(tl.dtype)
        ts = (torch.log1p(torch.relu(tl)) * mask).max(dim=1).values.numpy()
    tl = tl.numpy()

    # --- MLX ---
    import mlx.core as mx

    from splade_mlx.convert_vsplade import load_vsplade

    model, query_encoder, _ = load_vsplade(hf_id, dtype="float32")
    ids = mx.array(enc["input_ids"].numpy())
    am = mx.array(enc["attention_mask"].numpy())
    pv = enc["pixel_values"].numpy()
    ml = np.array(model(ids, am, pv))
    ms = np.array(model.encode(ids, am, pv))

    dl = np.abs(ml - tl).max()
    cos = (ms * ts).sum(1) / np.maximum(np.linalg.norm(ms, axis=1) * np.linalg.norm(ts, axis=1), 1e-12)
    ov = []
    for a, b in zip(ms, ts):
        k = min(64, int((b > 0).sum()))
        if k:
            ov.append(len(set(np.argsort(-a)[:k].tolist()) & set(np.argsort(-b)[:k].tolist())) / k)
    print(f"doc:   max|logit delta|={dl:.2e}  min cosine={cos.min():.6f}  topk overlap={np.mean(ov):.4f}")

    # --- query-side: exact match vs the shipped static embedding ---
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    ref_path = hf_hub_download(hf_id, "query_0_VSPLADEStaticEmbedding/model.safetensors")
    with safe_open(ref_path, framework="np") as f:
        ref_weight = f.get_tensor(list(f.keys())[0]).astype(np.float32)
    qd = np.abs(query_encoder.weight - ref_weight).max()
    print(f"query: max|lookup delta| vs shipped table = {qd:.2e}")

    # scatter-add behaviour on a repeated-token query (raw ids: BPE makes
    # "what what" tokenize to two different ids, so test with explicit ids)
    some_id = int(np.argmax(query_encoder.weight))  # a token with nonzero weight
    q_ids = np.array([[some_id, some_id, 50283]])  # repeated + [PAD]
    q_mask = np.array([[1, 1, 0]])
    qv = query_encoder.encode(q_ids, q_mask)
    assert abs(qv[0, some_id] - 2 * query_encoder.weight[some_id]) < 1e-6, "scatter-add broken"
    assert qv[0].sum() - qv[0, some_id] < 1e-6, "padding leaked into query vector"

    assert dl < 1e-3 and cos.min() >= 0.9999 and np.mean(ov) >= 0.99 and qd < 1e-5
    print(f"VSPLADE_PARITY_OK {hf_id}")


if __name__ == "__main__":
    main()
