"""BEIR retrieval-quality parity: torch fp32 reference vs MLX dtypes/quantization.

Encodes the full corpus + test queries of NFCorpus and SciFact with each
backend config, ranks by sparse dot product, and reports nDCG@10
(trec_eval-style linear gain). Gate from PLAN.md: MLX float32 must be within
+-0.002 of the torch fp32 reference.

Usage:
    uv run python -m bench.eval_beir
Writes results/quality_ndcg.json
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
from huggingface_hub import hf_hub_download

from bench.workloads import RESULTS_DIR, _load_parquet_texts

DATASETS = ["nfcorpus", "scifact"]

MODEL_FAMILIES = {
    "splade-cocondenser-ensembledistil": {
        "query": "naver/splade-cocondenser-ensembledistil",
        "doc": "naver/splade-cocondenser-ensembledistil",
    },
}

CONFIGS = ["torch-mps-fp32", "mlx-float32", "mlx-bfloat16", "mlx-q8", "mlx-q4"]

QUERY_MAX_LEN = 64
DOC_MAX_LEN = 256
BATCH_SIZE = 32


def load_beir(dataset: str):
    docs = _load_parquet_texts(
        f"BeIR/{dataset}", "corpus/corpus-00000-of-00001.parquet", "_id", ["title", "text"]
    )
    queries = _load_parquet_texts(
        f"BeIR/{dataset}", "queries/queries-00000-of-00001.parquet", "_id", ["text"]
    )
    qrels_path = hf_hub_download(
        repo_id=f"BeIR/{dataset}-qrels", filename="test.tsv", repo_type="dataset"
    )
    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_path) as f:
        next(f)  # header
        for line in f:
            qid, did, score = line.strip().split("\t")
            qrels.setdefault(qid, {})[did] = int(score)
    queries = [(qid, text) for qid, text in queries if qid in qrels]
    return docs, queries, qrels


class TorchEncoder:
    def __init__(self, hf_id: str, device: str = "mps"):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForMaskedLM.from_pretrained(hf_id, dtype=torch.float32)
        self.model.eval().to(device)

    def encode(self, texts: list[str], max_length: int) -> np.ndarray:
        from bench.bench_torch import splade_pool

        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), BATCH_SIZE):
                enc = self.tokenizer(
                    texts[i : i + BATCH_SIZE],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                batch = {
                    k: v.to(self.device)
                    for k, v in enc.items()
                    if k in ("input_ids", "attention_mask", "token_type_ids")
                }
                sparse = splade_pool(self.model(**batch).logits, batch["attention_mask"])
                out.append(sparse.float().cpu().numpy())
        return np.concatenate(out)


class MlxEncoder:
    def __init__(self, hf_id: str, dtype: str, quantize_bits: int | None):
        from splade_mlx import load

        self.model, self.tokenizer = load(hf_id, dtype=dtype, quantize_bits=quantize_bits)

    def encode(self, texts: list[str], max_length: int) -> np.ndarray:
        import mlx.core as mx

        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            enc = self.tokenizer(
                texts[i : i + BATCH_SIZE],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
            kwargs = {}
            if "token_type_ids" in enc:
                kwargs["token_type_ids"] = mx.array(enc["token_type_ids"])
            sparse = self.model.encode(
                mx.array(enc["input_ids"]), mx.array(enc["attention_mask"]), **kwargs
            )
            out.append(np.array(sparse.astype(mx.float32)))
        return np.concatenate(out)


def make_encoder(config: str, hf_id: str):
    if config == "torch-mps-fp32":
        return TorchEncoder(hf_id)
    if config == "mlx-float32":
        return MlxEncoder(hf_id, "float32", None)
    if config == "mlx-bfloat16":
        return MlxEncoder(hf_id, "bfloat16", None)
    if config == "mlx-q8":
        return MlxEncoder(hf_id, "float32", 8)
    if config == "mlx-q4":
        return MlxEncoder(hf_id, "float32", 4)
    raise ValueError(config)


def ndcg_at_k(ranked_doc_ids: list[str], rels: dict[str, int], k: int = 10) -> float:
    dcg = 0.0
    for i, did in enumerate(ranked_doc_ids[:k]):
        rel = rels.get(did, 0)
        dcg += rel / math.log2(i + 2)
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main() -> None:
    results: dict = {}
    for dataset in DATASETS:
        docs, queries, qrels = load_beir(dataset)
        doc_ids = [d for d, _ in docs]
        doc_texts = [t for _, t in docs]
        query_ids = [q for q, _ in queries]
        query_texts = [t for _, t in queries]
        print(f"### {dataset}: {len(docs)} docs, {len(queries)} test queries", flush=True)
        results[dataset] = {}

        for family, spec in MODEL_FAMILIES.items():
            results[dataset][family] = {}
            for config in CONFIGS:
                t0 = time.time()
                q_encoder = make_encoder(config, spec["query"])
                d_encoder = (
                    q_encoder
                    if spec["doc"] == spec["query"]
                    else make_encoder(config, spec["doc"])
                )
                Q = q_encoder.encode(query_texts, QUERY_MAX_LEN)
                D = d_encoder.encode(doc_texts, DOC_MAX_LEN)
                scores = Q @ D.T
                ndcgs = []
                for qi, qid in enumerate(query_ids):
                    order = np.argsort(-scores[qi])[:10]
                    ranked = [doc_ids[j] for j in order]
                    ndcgs.append(ndcg_at_k(ranked, qrels[qid]))
                ndcg = float(np.mean(ndcgs))
                results[dataset][family][config] = ndcg
                ref = results[dataset][family].get("torch-mps-fp32")
                delta = f" (delta {ndcg - ref:+.4f})" if ref is not None else ""
                print(
                    f"  {family:36s} {config:16s} nDCG@10 {ndcg:.4f}{delta}"
                    f"  [{time.time() - t0:.0f}s]",
                    flush=True,
                )
                del q_encoder, d_encoder, Q, D, scores

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "quality_ndcg.json").write_text(json.dumps(results, indent=2))
    print("wrote results/quality_ndcg.json", flush=True)
    print("EVAL_BEIR_DONE", flush=True)


if __name__ == "__main__":
    main()
