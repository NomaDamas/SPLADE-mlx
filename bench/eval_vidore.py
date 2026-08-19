"""ViDoRe (visual document retrieval) nDCG@5 parity: torch vs MLX for V-SPLADE.

Task: vidore/docvqa_test_subsampled (500 page images, text queries; the
relevant document for each query is its own page — standard ViDoRe protocol,
binary relevance, nDCG@5).

Queries use the inference-free lookup (identical numpy math for every
backend), so any ranking difference comes from the document encoders.
Gate: |nDCG@5(mlx-fp32) - nDCG@5(torch-mps-fp32)| <= 0.002.

Usage:
    uv run python -m bench.eval_vidore [--variant efficient|quality] [--batch 4]
Writes/updates results/quality_vidore.json
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from bench.workloads import RESULTS_DIR

DOC_PROMPT = "User:<image><end_of_utterance>\nAssistant:"
DATASET = "vidore/docvqa_test_subsampled"
CONFIGS = ["torch-mps-fp32", "mlx-float32", "mlx-bfloat16"]


def load_dataset():
    import pyarrow.parquet as pq

    path = hf_hub_download(DATASET, "data/test-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(path)
    rows = table.to_pylist()
    corpus: dict[str, Image.Image] = {}
    qrels: list[tuple[str, str]] = []  # (query, relevant_doc_id)
    for row in rows:
        doc_id = row["image_filename"]
        if doc_id not in corpus:
            corpus[doc_id] = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        if row.get("query"):
            qrels.append((row["query"], doc_id))
    return corpus, qrels


class TorchDocEncoder:
    def __init__(self, hf_id: str, processor, device: str = "mps"):
        import torch
        from transformers import AutoModelForMaskedLM

        self.torch = torch
        self.device = device
        self.processor = processor
        self.model = AutoModelForMaskedLM.from_pretrained(
            hf_id, trust_remote_code=True, dtype=torch.float32
        )
        self.model.eval().to(device)

    def encode_batch(self, images: list[Image.Image]) -> np.ndarray:
        enc = self.processor(
            text=[DOC_PROMPT] * len(images),
            images=[[im] for im in images],
            return_tensors="pt",
            padding=True,
        )
        t = self.torch
        with t.inference_mode():
            out = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
                pixel_values=enc["pixel_values"].to(self.device),
            ).logits
            mask = enc["attention_mask"].to(self.device).unsqueeze(-1).to(out.dtype)
            sparse = (t.log1p(t.relu(out)) * mask).max(dim=1).values
        return sparse.float().cpu().numpy()


class MlxDocEncoder:
    def __init__(self, hf_id: str, processor, dtype: str):
        from splade_mlx.convert_vsplade import load_vsplade

        self.processor = processor
        self.model, self.query_encoder, _ = load_vsplade(hf_id, dtype=dtype)

    def encode_batch(self, images: list[Image.Image]) -> np.ndarray:
        import mlx.core as mx

        enc = self.processor(
            text=[DOC_PROMPT] * len(images),
            images=[[im] for im in images],
            return_tensors="np",
            padding=True,
        )
        sparse = self.model.encode(
            mx.array(enc["input_ids"]),
            mx.array(enc["attention_mask"]),
            enc["pixel_values"].astype(np.float32),
        )
        return np.array(sparse.astype(mx.float32))


def ndcg_at_5(scores: np.ndarray, doc_ids: list[str], qrels: list[tuple[str, str]]) -> float:
    vals = []
    for qi, (_, rel_doc) in enumerate(qrels):
        top = np.argsort(-scores[qi])[:5]
        gain = 0.0
        for rank, di in enumerate(top):
            if doc_ids[di] == rel_doc:
                gain = 1.0 / math.log2(rank + 2)
                break
        vals.append(gain)  # IDCG = 1 (single relevant doc)
    return float(np.mean(vals))


def main() -> None:
    from transformers import AutoProcessor

    from splade_mlx.convert_vsplade import load_vsplade

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["efficient", "quality"], default="efficient")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    hf_id = f"naver/v-splade-{args.variant}"

    corpus, qrels = load_dataset()
    doc_ids = list(corpus)
    images = [corpus[d] for d in doc_ids]
    print(f"### {DATASET}: {len(doc_ids)} pages, {len(qrels)} queries [{hf_id}]", flush=True)

    # queries: inference-free lookup, computed once (identical for all backends)
    _, query_encoder, processor = load_vsplade(hf_id, dtype="float32")
    q_enc = processor.tokenizer(
        [q for q, _ in qrels], return_tensors="np", padding=True
    )
    Q = query_encoder.encode(q_enc["input_ids"], q_enc["attention_mask"])

    out_path = RESULTS_DIR / "quality_vidore.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results.setdefault(hf_id, {})

    for config in CONFIGS:
        t0 = time.time()
        if config == "torch-mps-fp32":
            encoder = TorchDocEncoder(hf_id, processor)
        else:
            encoder = MlxDocEncoder(hf_id, processor, config.replace("mlx-", ""))
        D = np.concatenate(
            [
                encoder.encode_batch(images[i : i + args.batch])
                for i in range(0, len(images), args.batch)
            ]
        )
        ndcg = ndcg_at_5(Q @ D.T, doc_ids, qrels)
        results[hf_id][config] = ndcg
        ref = results[hf_id].get("torch-mps-fp32")
        delta = f" (delta {ndcg - ref:+.4f})" if ref is not None and config != "torch-mps-fp32" else ""
        print(f"  {config:16s} nDCG@5 {ndcg:.4f}{delta}  [{time.time() - t0:.0f}s]", flush=True)
        del encoder, D

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print("EVAL_VIDORE_DONE", flush=True)


if __name__ == "__main__":
    main()
