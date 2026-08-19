"""Generate V-SPLADE (visual) README charts from measured results/*.json.

Outputs:
    assets/vsplade_latency.png  - doc-page encoding latency torch vs MLX (matched precision)
    assets/vsplade_quality.png  - ViDoRe nDCG@5 parity, torch fp32 vs MLX fp32/bf16

Usage:
    uv run python scripts/make_charts_vsplade.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"

TORCH_C = "#ee6c4d"
MLX_C = "#0a84ff"


def latency_chart() -> None:
    runs = {
        k: json.loads((RESULTS / f"vsplade_{k}.json").read_text())
        for k in ("torch_fp32", "mlx_float32", "mlx_bfloat16")
    }
    batches = ["pages-B1", "pages-B4", "pages-B8"]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    w = 0.27
    series = [
        ("torch_fp32", "PyTorch MPS fp32", TORCH_C),
        ("mlx_float32", "MLX fp32", MLX_C),
        ("mlx_bfloat16", "MLX bf16", "#30b0c7"),
    ]
    for ci, (key, label, color) in enumerate(series):
        xs = [i + (ci - 1) * w for i in range(len(batches))]
        ys = [runs[key]["workloads"][b]["mean_ms"] for b in batches]
        bars = ax.bar(xs, ys, w, label=label, color=color)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    # speedup annotations (mlx bf16 vs torch fp32)
    for i, b in enumerate(batches):
        t = runs["torch_fp32"]["workloads"][b]["mean_ms"]
        m = runs["mlx_bfloat16"]["workloads"][b]["mean_ms"]
        ax.annotate(
            f"{t / m:.2f}x",
            (i + w, m),
            textcoords="offset points",
            xytext=(0, 16),
            ha="center",
            fontsize=9,
            color="#30b0c7",
            fontweight="bold",
        )
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels([b.replace("pages-", "batch ") for b in batches])
    ax.set_ylabel("latency (ms) — lower is better")
    ax.set_title(
        "V-SPLADE document-page encoding on Apple M4 Max (measured)\n"
        "ModernVBERT backbone, 13 tiles/page, seq 871 — inference-free queries add only ~6 µs",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "vsplade_latency.png", dpi=180)
    print("wrote assets/vsplade_latency.png")


def quality_chart() -> None:
    q = json.loads((RESULTS / "quality_vidore.json").read_text())
    variants = ["v-splade-efficient", "v-splade-quality"]
    configs = [("mlx-float32", "MLX fp32"), ("mlx-bfloat16", "MLX bf16")]
    colors = [MLX_C, "#30b0c7"]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.axhspan(-0.002, 0.002, color="#34c759", alpha=0.12, label="parity gate (±0.002)")
    ax.axhline(0, color="#333", lw=0.8)
    w = 0.3
    for ci, ((cfg, label), color) in enumerate(zip(configs, colors)):
        xs, ys = [], []
        for i, v in enumerate(variants):
            ref = q[f"naver/{v}"]["torch-mps-fp32"]
            xs.append(i + (ci - 0.5) * w)
            ys.append(q[f"naver/{v}"][cfg] - ref)
        bars = ax.bar(xs, ys, w, label=label, color=color)
        ax.bar_label(bars, fmt="%+.4f", fontsize=9, padding=2)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.replace("v-splade-", "V-SPLADE ") for v in variants], fontsize=10)
    ax.set_ylabel("nDCG@5 delta vs PyTorch fp32")
    ax.set_ylim(-0.006, 0.005)
    ax.set_title(
        "ViDoRe docvqa_test_subsampled (500 pages) — MLX fp32 is bit-exact (Δ = 0.0000)",
        fontsize=11,
    )
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "vsplade_quality.png", dpi=180)
    print("wrote assets/vsplade_quality.png")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    latency_chart()
    quality_chart()
