"""Generate README charts from measured results/*.json.

Outputs:
    assets/latency_m4max.png   - PyTorch vs MLX inference latency (measured, M4 Max)
    assets/quality_parity.png  - nDCG@10 delta vs PyTorch fp32, with the +-0.002 gate band

Usage:
    uv run python scripts/make_charts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"

TORCH_CPU = "#9aa0a6"
TORCH_MPS = "#ee6c4d"
MLX = "#0a84ff"


def _load(stem: str) -> dict:
    return json.loads((RESULTS / f"{stem}.json").read_text())


def _mean_ms(run: dict, model: str, workload: str) -> float:
    return run["models"][model]["workloads"][workload]["mean_ms"]


def latency_chart() -> None:
    """Matched-precision comparison: torch and MLX at the SAME precision per group,
    so the speedup shown is attributable to MLX itself, not quantization."""
    cpu32 = _load("baseline_cpu_fp32")
    mps32 = _load("baseline_mps_fp32")
    mps16 = _load("baseline_mps_fp16")
    mlx32 = _load("mlx_float32")
    mlx16 = _load("mlx_bfloat16")
    mlxq8 = _load("mlx_float32_q8")

    panels = [
        ("V-SPLADE query encoder — query (seq 32, batch 1)", "efficient-splade-V-large-query", "query-L32-B1"),
        ("SPLADE++ (BERT-base) — documents (seq 256, batch 32)", "splade-cocondenser-ensembledistil", "doc-L256-B32"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, (title, model, workload) in zip(axes, panels):
        v = {
            "cpu32": _mean_ms(cpu32, model, workload),
            "mps32": _mean_ms(mps32, model, workload),
            "mlx32": _mean_ms(mlx32, model, workload),
            "mps16": _mean_ms(mps16, model, workload),
            "mlx16": _mean_ms(mlx16, model, workload),
            "q8": _mean_ms(mlxq8, model, workload),
        }
        # precision-matched groups: fp32 | 16-bit | 8-bit (torch n/a)
        bars = [
            (0.0, v["cpu32"], "PyTorch CPU", TORCH_CPU, None),
            (0.8, v["mps32"], "PyTorch MPS", TORCH_MPS, None),
            (1.6, v["mlx32"], "MLX", MLX, v["mps32"]),
            (2.9, v["mps16"], "PyTorch MPS", TORCH_MPS, None),
            (3.7, v["mlx16"], "MLX", MLX, v["mps16"]),
            (5.0, v["q8"], "MLX", MLX, None),
        ]
        for x, val, _, color, ref in bars:
            hatch = "//" if x == 5.0 else None
            b = ax.bar([x], [val], 0.7, color=color, hatch=hatch, edgecolor="white")
            ax.bar_label(b, fmt="%.1f", fontsize=8, padding=2)
            if ref is not None:
                ax.annotate(
                    f"{ref / val:.1f}x faster\nsame precision",
                    (x, val),
                    textcoords="offset points",
                    xytext=(0, 16),
                    ha="center",
                    fontsize=8,
                    color=MLX,
                    fontweight="bold",
                )
        ax.set_xticks([x for x, *_ in bars])
        ax.set_xticklabels(
            ["CPU\nfp32", "MPS\nfp32", "MLX\nfp32", "MPS\nfp16", "MLX\nbf16", "MLX\n8-bit*"],
            fontsize=8.5,
        )
        for xc, lbl in ((0.8, "fp32"), (3.3, "16-bit"), (5.0, "8-bit")):
            ax.annotate(
                lbl, (xc, 1.0), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=9, color="#555",
            )
        ax.axvline(2.25, color="#ddd", lw=1)
        ax.axvline(4.35, color="#ddd", lw=1)
        ax.set_ylabel("latency (ms) — lower is better")
        ax.set_title(title, fontsize=10, pad=22)
        ax.set_ylim(0, v["cpu32"] * 1.28)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "SPLADE inference on Apple M4 Max — matched precision per group "
        "(*8-bit is an optional MLX extra; PyTorch not measured quantized)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "latency_m4max.png", dpi=180)
    print("wrote assets/latency_m4max.png")


def quality_chart() -> None:
    q = json.loads((RESULTS / "quality_ndcg.json").read_text())
    cells = [
        (ds, fam, short)
        for ds in ("nfcorpus", "scifact")
        for fam, short in (
            ("splade-cocondenser-ensembledistil", "SPLADE++"),
            ("efficient-splade-V-large", "V-SPLADE"),
        )
    ]
    configs = [("mlx-float32", "fp32"), ("mlx-bfloat16", "bf16"), ("mlx-q8", "8-bit"), ("mlx-q4", "4-bit")]
    colors = ["#0a84ff", "#30b0c7", "#64d2ff", "#a5b4fc"]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axhspan(-0.002, 0.002, color="#34c759", alpha=0.12, label="parity gate (±0.002)")
    ax.axhline(0, color="#333", lw=0.8)
    w = 0.19
    for ci, ((cfg, cfg_label), color) in enumerate(zip(configs, colors)):
        xs, ys = [], []
        for i, (ds, fam, _) in enumerate(cells):
            ref = q[ds][fam]["torch-mps-fp32"]
            xs.append(i + (ci - 1.5) * w)
            ys.append(q[ds][fam][cfg] - ref)
        bars = ax.bar(xs, ys, w, label=f"MLX {cfg_label}", color=color)
        ax.bar_label(bars, fmt="%+.4f", fontsize=7, padding=2, rotation=90)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(
        [f"{short}\n{ds.upper() if ds == 'nfcorpus' else 'SciFact'}" for ds, _, short in cells],
        fontsize=9,
    )
    ax.set_ylabel("nDCG@10 delta vs PyTorch fp32")
    ax.set_ylim(-0.006, 0.008)
    ax.set_title(
        "Retrieval quality is preserved: BEIR nDCG@10 delta vs PyTorch (MLX fp32 is bit-exact: Δ = 0.0000)",
        fontsize=11,
    )
    ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "quality_parity.png", dpi=180)
    print("wrote assets/quality_parity.png")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    latency_chart()
    quality_chart()
