"""V-SPLADE document/query encoding latency on Apple Silicon.

Measures end-to-end page encoding (vision + text + SPLADE head; image
preprocessing measured separately) for torch-MPS and MLX at matched precision,
plus the inference-free query lookup cost.

Usage:
    uv run python -m bench.bench_vsplade --backend torch --dtype fp32
    uv run python -m bench.bench_vsplade --backend mlx --dtype float32
    uv run python -m bench.bench_vsplade --backend mlx --dtype bfloat16
Writes results/vsplade_{backend}_{dtype}.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.workloads import MAX_ITERS, MIN_ITERS, RESULTS_DIR, TARGET_SECONDS, WARMUP_ITERS

HF_ID = "naver/v-splade-efficient"
DOC_PROMPT = "User:<image><end_of_utterance>\nAssistant:"
BATCHES = [1, 4, 8]


def make_page(seed: int):
    from scripts.check_parity_vsplade import make_doc_image

    return make_doc_image(seed)


def timed(fn) -> dict:
    times = []
    for _ in range(WARMUP_ITERS):
        fn()
    start = time.perf_counter()
    while len(times) < MAX_ITERS and (
        len(times) < MIN_ITERS or time.perf_counter() - start < TARGET_SECONDS
    ):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    n = len(times)
    return {
        "iters": n,
        "mean_ms": sum(times) / n * 1e3,
        "p50_ms": statistics.median(times) * 1e3,
        "p95_ms": times[min(n - 1, int(round(0.95 * (n - 1))))] * 1e3,
    }


def main() -> None:
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "mlx"], required=True)
    parser.add_argument("--dtype", default=None)
    args = parser.parse_args()
    dtype = args.dtype or ("fp32" if args.backend == "torch" else "float32")

    processor = AutoProcessor.from_pretrained(HF_ID)
    pages = [make_page(i) for i in range(max(BATCHES))]

    results = {
        "backend": args.backend,
        "dtype": dtype,
        "machine": platform.platform(),
        "chip": "Apple M4 Max 16c/64GB",
        "model": HF_ID,
        "workloads": {},
    }

    # image preprocessing cost (CPU, identical for both backends)
    t0 = time.perf_counter()
    for _ in range(5):
        enc1 = processor(text=[DOC_PROMPT], images=[[pages[0]]], return_tensors="np", padding=True)
    results["preprocess_1page_ms"] = (time.perf_counter() - t0) / 5 * 1e3
    results["tiles_per_page"] = int(enc1["pixel_values"].shape[1])
    results["seq_len_1page"] = int(enc1["input_ids"].shape[1])

    if args.backend == "torch":
        import torch
        from transformers import AutoModelForMaskedLM

        model = AutoModelForMaskedLM.from_pretrained(HF_ID, trust_remote_code=True, dtype=torch.float32)
        model.eval().to("mps")

        def make_fn(enc):
            ids = enc["input_ids"].to("mps")
            am = enc["attention_mask"].to("mps")
            pv = enc["pixel_values"].to("mps")

            def fn():
                with torch.inference_mode():
                    out = model(input_ids=ids, attention_mask=am, pixel_values=pv).logits
                    sparse = (torch.log1p(torch.relu(out)) * am.unsqueeze(-1)).max(dim=1).values
                torch.mps.synchronize()
                return sparse

            return fn

        def encode_inputs(batch):
            return processor(
                text=[DOC_PROMPT] * batch, images=[[p] for p in pages[:batch]],
                return_tensors="pt", padding=True,
            )

    else:
        import mlx.core as mx

        from splade_mlx.convert_vsplade import load_vsplade

        model, query_encoder, _ = load_vsplade(HF_ID, dtype=dtype)

        def make_fn(enc):
            ids = mx.array(enc["input_ids"])
            am = mx.array(enc["attention_mask"])
            pv = enc["pixel_values"].astype(np.float32)

            def fn():
                mx.eval(model.encode(ids, am, pv))

            return fn

        def encode_inputs(batch):
            return processor(
                text=[DOC_PROMPT] * batch, images=[[p] for p in pages[:batch]],
                return_tensors="np", padding=True,
            )

    for batch in BATCHES:
        enc = encode_inputs(batch)
        stats = timed(make_fn(enc))
        stats["pages_per_s"] = batch / (stats["mean_ms"] / 1e3)
        results["workloads"][f"pages-B{batch}"] = stats
        print(
            f"  pages-B{batch}  mean {stats['mean_ms']:8.2f} ms  "
            f"p50 {stats['p50_ms']:8.2f}  {stats['pages_per_s']:6.2f} pages/s",
            flush=True,
        )

    # inference-free query lookup (numpy, backend-independent; measured once)
    from splade_mlx.convert_vsplade import load_vsplade as _lv

    if args.backend == "mlx":
        q_enc = processor.tokenizer(["what is the total revenue in 2023"], return_tensors="np")
        _, query_encoder, _ = _lv(HF_ID, dtype="float32")

        def qfn():
            query_encoder.encode(q_enc["input_ids"], q_enc["attention_mask"])

        qstats = timed(qfn)
        results["query_lookup_ms"] = qstats["mean_ms"]
        print(f"  query lookup  {qstats['mean_ms'] * 1e3:.1f} us/query", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"vsplade_{args.backend}_{dtype}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
