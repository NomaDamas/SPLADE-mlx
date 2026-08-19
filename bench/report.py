"""Aggregate results/*.json into markdown comparison tables.

Usage:
    uv run python -m bench.report            # prints markdown to stdout
"""

from __future__ import annotations

import json

from bench.workloads import BENCHMARK_SUITE, P0_MODELS, RESULTS_DIR, WORKLOADS

# display order; anything else found is appended alphabetically
PREFERRED_ORDER = [
    "baseline_cpu_fp32",
    "baseline_mps_fp32",
    "baseline_mps_fp16",
    "mlx_float32",
    "mlx_bfloat16",
    "mlx_float16",
    "mlx_float32_q8",
    "mlx_bfloat16_q8",
    "mlx_bfloat16_q4",
    "mlx_bfloat16_compile",
]

TORCH_BEST_KEY = "baseline_mps_fp16"  # speedup denominator (best torch config)


def validate_run(path, data: dict) -> None:
    suite = data.get("benchmark_suite")
    if suite != BENCHMARK_SUITE:
        raise ValueError(
            f"{path}: benchmark suite {suite!r} does not match {BENCHMARK_SUITE!r}"
        )
    expected_models = set(P0_MODELS)
    actual_models = set(data["models"])
    if actual_models != expected_models:
        raise ValueError(
            f"{path}: model set {sorted(actual_models)} does not match "
            f"{sorted(expected_models)}"
        )
    for model_key, spec in P0_MODELS.items():
        expected_workloads = {
            workload.name for workload in WORKLOADS if workload.kind in spec["roles"]
        }
        actual_workloads = set(data["models"][model_key].get("workloads", {}))
        if actual_workloads != expected_workloads:
            raise ValueError(
                f"{path}: workload set for {model_key} does not match current suite"
            )


def load_runs() -> dict[str, dict]:
    runs = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.stem.endswith("_quick"):
            continue
        data = json.loads(path.read_text())
        if "models" not in data:  # e.g. quality_ndcg.json
            continue
        validate_run(path, data)
        runs[path.stem] = data
    ordered = {k: runs[k] for k in PREFERRED_ORDER if k in runs}
    for k in sorted(runs):
        ordered.setdefault(k, runs[k])
    return ordered


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit(f"no result JSONs in {RESULTS_DIR}")

    model_keys: list[str] = []
    for run in runs.values():
        for mk in run["models"]:
            if mk not in model_keys:
                model_keys.append(mk)

    print("# SPLADE on Apple Silicon: PyTorch vs MLX\n")
    first = next(iter(runs.values()))
    print(f"Machine: {first.get('chip', '?')} — {first.get('machine', '?')}\n")
    print("Latency = encoder forward + SPLADE pooling, tokenization excluded")
    print("(tokenize_ms reported separately in results/*.json). mean over")
    print("adaptive iterations after warmup; see protocol block in each JSON.\n")

    for mk in model_keys:
        print(f"\n## {mk}\n")
        wl_names: list[str] = []
        for run in runs.values():
            for wl in run["models"].get(mk, {}).get("workloads", {}):
                if wl not in wl_names:
                    wl_names.append(wl)

        header = ["workload"] + [
            k.replace("baseline_", "torch-").replace("mlx_", "mlx-") for k in runs
        ]
        if TORCH_BEST_KEY in runs:
            header.append(f"best-mlx speedup vs {TORCH_BEST_KEY.replace('baseline_', 'torch-')}")
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))

        for wl in wl_names:
            cells = [wl]
            torch_best = None
            best_mlx = None
            for run_key, run in runs.items():
                stats = run["models"].get(mk, {}).get("workloads", {}).get(wl)
                if stats is None:
                    cells.append("—")
                    continue
                cells.append(f"{stats['mean_ms']:.2f} ms")
                if run_key == TORCH_BEST_KEY:
                    torch_best = stats["mean_ms"]
                if run_key.startswith("mlx_"):
                    if best_mlx is None or stats["mean_ms"] < best_mlx:
                        best_mlx = stats["mean_ms"]
            if TORCH_BEST_KEY in runs:
                if torch_best and best_mlx:
                    cells.append(f"{torch_best / best_mlx:.2f}x")
                else:
                    cells.append("—")
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
