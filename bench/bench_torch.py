"""PyTorch baseline benchmark for SPLADE models on macOS (CPU / MPS).

Usage:
    uv run python -m bench.bench_torch --backend cpu --dtype fp32
    uv run python -m bench.bench_torch --backend mps --dtype fp32
    uv run python -m bench.bench_torch --backend mps --dtype fp16
    uv run python -m bench.bench_torch --quick   # smoke test

Writes results/baseline_{backend}_{dtype}.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import psutil
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from bench.workloads import (
    MAX_ITERS,
    MIN_ITERS,
    P0_MODELS,
    QUICK_PROTOCOL,
    RESULTS_DIR,
    TARGET_SECONDS,
    WARMUP_ITERS,
    WORKLOADS,
    Workload,
    texts_for,
)


def splade_pool(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """log(1 + relu(logits)) masked, max-pooled over the sequence axis."""
    scores = torch.log1p(torch.relu(logits))
    scores = scores * attention_mask.unsqueeze(-1).to(scores.dtype)
    return scores.max(dim=1).values


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def bench_one(model, batch: dict, device: str) -> dict:
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            out = splade_pool(model(**batch).logits, batch["attention_mask"])
            sync(device)
        start_all = time.perf_counter()
        while len(times) < MAX_ITERS and (
            len(times) < MIN_ITERS or time.perf_counter() - start_all < TARGET_SECONDS
        ):
            t0 = time.perf_counter()
            out = splade_pool(model(**batch).logits, batch["attention_mask"])
            sync(device)
            times.append(time.perf_counter() - t0)
    del out
    times.sort()
    n = len(times)
    mean = sum(times) / n
    return {
        "iters": n,
        "mean_ms": mean * 1e3,
        "p50_ms": statistics.median(times) * 1e3,
        "p95_ms": times[min(n - 1, int(round(0.95 * (n - 1))))] * 1e3,
        "min_ms": times[0] * 1e3,
    }


def make_batch(tokenizer, texts: list[str], wl: Workload, device: str, dtype) -> dict:
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=wl.seq_len,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--quick", action="store_true", help="smoke test: 1 workload, few iters")
    args = parser.parse_args()

    global MIN_ITERS, MAX_ITERS, TARGET_SECONDS
    if args.quick:
        MIN_ITERS = QUICK_PROTOCOL["min_iters"]
        MAX_ITERS = QUICK_PROTOCOL["max_iters"]
        TARGET_SECONDS = QUICK_PROTOCOL["target_seconds"]

    device = args.backend
    torch_dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    if device == "cpu" and args.dtype == "fp16":
        raise SystemExit("fp16 on CPU is not a meaningful baseline; skipping by design")

    proc = psutil.Process()
    results: dict = {
        "backend": device,
        "dtype": args.dtype,
        "framework": f"torch-{torch.__version__}",
        "machine": platform.platform(),
        "chip": "Apple M4 Max 16c/64GB",
        "torch_num_threads": torch.get_num_threads(),
        "protocol": {
            "warmup": WARMUP_ITERS,
            "min_iters": MIN_ITERS,
            "max_iters": MAX_ITERS,
            "target_seconds": TARGET_SECONDS,
            "padding": "max_length",
            "note": "latency excludes tokenization; tokenize_ms reported separately",
        },
        "models": {},
    }

    workloads = WORKLOADS[:1] if args.quick else WORKLOADS

    for model_key, spec in P0_MODELS.items():
        print(f"=== {model_key} [{device}/{args.dtype}] ===", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"])
        model = AutoModelForMaskedLM.from_pretrained(spec["hf_id"], dtype=torch_dtype)
        model.eval().to(device)
        rss_after_load = proc.memory_info().rss

        entry: dict = {
            "hf_id": spec["hf_id"],
            "rss_after_load_mb": rss_after_load / 1e6,
            "workloads": {},
        }

        for wl in workloads:
            if wl.kind not in spec["roles"]:
                continue
            texts = texts_for(wl.kind, wl.batch_size)
            # tokenization cost, measured separately on the same texts
            t0 = time.perf_counter()
            for _ in range(5):
                tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=wl.seq_len,
                    return_tensors="pt",
                )
            tokenize_ms = (time.perf_counter() - t0) / 5 * 1e3

            batch = make_batch(tokenizer, texts, wl, device, torch_dtype)
            stats = bench_one(model, batch, device)
            stats["tokenize_ms"] = tokenize_ms
            stats["throughput_seq_s"] = wl.batch_size / (stats["mean_ms"] / 1e3)
            stats["throughput_tok_s"] = wl.batch_size * wl.seq_len / (stats["mean_ms"] / 1e3)
            stats["peak_rss_mb"] = proc.memory_info().rss / 1e6
            if device == "mps":
                stats["mps_driver_alloc_mb"] = torch.mps.driver_allocated_memory() / 1e6
            entry["workloads"][wl.name] = stats
            print(
                f"  {wl.name:16s} mean {stats['mean_ms']:8.2f} ms  "
                f"p50 {stats['p50_ms']:8.2f}  p95 {stats['p95_ms']:8.2f}  "
                f"{stats['throughput_seq_s']:8.1f} seq/s",
                flush=True,
            )

        results["models"][model_key] = entry
        del model
        if device == "mps":
            torch.mps.empty_cache()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"baseline_{device}_{args.dtype}{'_quick' if args.quick else ''}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
