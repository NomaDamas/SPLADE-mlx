"""MLX benchmark for SPLADE models, mirroring bench_torch's protocol exactly.

Usage:
    uv run python -m bench.bench_mlx --dtype float32
    uv run python -m bench.bench_mlx --dtype bfloat16
    uv run python -m bench.bench_mlx --dtype bfloat16 --compile
    uv run python -m bench.bench_mlx --dtype float32 --quantize-bits 8
    uv run python -m bench.bench_mlx --quick

Writes results/mlx_{dtype}[_q{bits}][_compile].json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx
import psutil

from bench import workloads as W
from bench.workloads import (
    BENCHMARK_SUITE,
    P0_MODELS,
    RESULTS_DIR,
    WORKLOADS,
    Workload,
    hardware_metadata,
    texts_for,
)
from splade_mlx import load


def make_batch(tokenizer, texts: list[str], wl: Workload) -> dict:
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=wl.seq_len,
        return_tensors="np",
    )
    batch = {
        "input_ids": mx.array(enc["input_ids"]),
        "attention_mask": mx.array(enc["attention_mask"]),
    }
    if "token_type_ids" in enc:
        batch["token_type_ids"] = mx.array(enc["token_type_ids"])
    mx.eval(*batch.values())
    return batch


def bench_one(encode_fn, batch: dict, protocol: dict) -> dict:
    times: list[float] = []
    for _ in range(W.WARMUP_ITERS):
        mx.eval(encode_fn(**batch))
    start_all = time.perf_counter()
    while len(times) < protocol["max_iters"] and (
        len(times) < protocol["min_iters"]
        or time.perf_counter() - start_all < protocol["target_seconds"]
    ):
        t0 = time.perf_counter()
        mx.eval(encode_fn(**batch))
        times.append(time.perf_counter() - t0)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="float32"
    )
    parser.add_argument("--quantize-bits", type=int, choices=[4, 8], default=None)
    parser.add_argument("--compile", action="store_true", dest="use_compile")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    protocol = {
        "min_iters": W.MIN_ITERS,
        "max_iters": W.MAX_ITERS,
        "target_seconds": W.TARGET_SECONDS,
    }
    if args.quick:
        protocol = dict(W.QUICK_PROTOCOL)

    proc = psutil.Process()
    tag = args.dtype
    if args.quantize_bits:
        tag += f"_q{args.quantize_bits}"
    if args.use_compile:
        tag += "_compile"

    results: dict = {
        "benchmark_suite": BENCHMARK_SUITE,
        "backend": "mlx",
        "dtype": args.dtype,
        "quantize_bits": args.quantize_bits,
        "compile": args.use_compile,
        "framework": f"mlx-{mx.__version__}",
        **hardware_metadata(),
        "protocol": {
            "warmup": W.WARMUP_ITERS,
            **protocol,
            "padding": "max_length",
            "note": "latency excludes tokenization; tokenize_ms reported separately",
        },
        "models": {},
    }

    workloads = WORKLOADS[:1] if args.quick else WORKLOADS

    for model_key, spec in P0_MODELS.items():
        print(f"=== {model_key} [mlx/{tag}] ===", flush=True)
        mx.reset_peak_memory()
        model, tokenizer = load(
            spec["hf_id"], dtype=args.dtype, quantize_bits=args.quantize_bits
        )
        encode_fn = model.encode
        if args.use_compile:
            encode_fn = mx.compile(model.encode)

        entry: dict = {
            "hf_id": spec["hf_id"],
            "rss_after_load_mb": proc.memory_info().rss / 1e6,
            "mlx_active_after_load_mb": mx.get_active_memory() / 1e6,
            "workloads": {},
        }

        for wl in workloads:
            if wl.kind not in spec["roles"]:
                continue
            texts = texts_for(wl.kind, wl.batch_size)
            t0 = time.perf_counter()
            for _ in range(5):
                tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=wl.seq_len,
                    return_tensors="np",
                )
            tokenize_ms = (time.perf_counter() - t0) / 5 * 1e3

            batch = make_batch(tokenizer, texts, wl)
            stats = bench_one(encode_fn, batch, protocol)
            stats["tokenize_ms"] = tokenize_ms
            stats["throughput_seq_s"] = wl.batch_size / (stats["mean_ms"] / 1e3)
            stats["throughput_tok_s"] = wl.batch_size * wl.seq_len / (stats["mean_ms"] / 1e3)
            stats["peak_rss_mb"] = proc.memory_info().rss / 1e6
            stats["mlx_peak_mb"] = mx.get_peak_memory() / 1e6
            entry["workloads"][wl.name] = stats
            print(
                f"  {wl.name:16s} mean {stats['mean_ms']:8.2f} ms  "
                f"p50 {stats['p50_ms']:8.2f}  p95 {stats['p95_ms']:8.2f}  "
                f"{stats['throughput_seq_s']:8.1f} seq/s",
                flush=True,
            )

        results["models"][model_key] = entry
        del model, encode_fn
        mx.clear_cache()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"mlx_{tag}{'_quick' if args.quick else ''}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
