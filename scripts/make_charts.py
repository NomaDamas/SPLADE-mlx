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


def _best_mlx(model: str, workload: str) -> tuple[float, str]:
    best, tag = None, ""
    for path in RESULTS.glob("mlx_*.json"):
        if path.stem.endswith("_quick"):
            continue
        run = json.loads(path.read_text())
        wl = run["models"].get(model, {}).get("workloads", {}).get(workload)
        if wl and (best is None or wl["mean_ms"] < best):
            best = wl["mean_ms"]
            tag = path.stem.replace("mlx_", "").replace("float32_", "").replace("bfloat16", "bf16")
    return best, tag


def latency_chart() -> None:
    cpu, mps = _load("baseline_cpu_fp32"), _load("baseline_mps_fp16")
    panels = [
        (
            "Query encoding (seq 32, batch 1)",
            "query-L32-B1",
            [
                ("V-SPLADE query\n(DistilBERT)", "efficient-splade-V-large-query"),
                ("SPLADE++\n(BERT-base)", "splade-cocondenser-ensembledistil"),
            ],
        ),
        (
            "Document encoding (seq 256, batch 32)",
            "doc-L256-B32",
            [
                ("V-SPLADE doc\n(DistilBERT)", "efficient-splade-V-large-doc"),
                ("SPLADE++\n(BERT-base)", "splade-cocondenser-ensembledistil"),
            ],
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (title, workload, models) in zip(axes, panels):
        labels = [lbl for lbl, _ in models]
        cpu_v = [_mean_ms(cpu, m, workload) for _, m in models]
        mps_v = [_mean_ms(mps, m, workload) for _, m in models]
        mlx_v, mlx_tag = zip(*[_best_mlx(m, workload) for _, m in models])
        x = range(len(models))
        w = 0.26
        b1 = ax.bar([i - w for i in x], cpu_v, w, label="PyTorch CPU fp32", color=TORCH_CPU)
        b2 = ax.bar(x, mps_v, w, label="PyTorch MPS fp16", color=TORCH_MPS)
        b3 = ax.bar([i + w for i in x], mlx_v, w, label="MLX (best)", color=MLX)
        for bars in (b1, b2, b3):
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        for i, (m_ms, t_ms, tag) in enumerate(zip(mlx_v, mps_v, mlx_tag)):
            ax.annotate(
                f"{t_ms / m_ms:.1f}x faster\n({tag})",
                (i + w, m_ms),
                textcoords="offset points",
                xytext=(0, 16),
                ha="center",
                fontsize=8,
                color=MLX,
                fontweight="bold",
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("latency (ms) — lower is better")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(cpu_v) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("SPLADE inference: PyTorch vs MLX on Apple M4 Max (measured)", fontsize=12)
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
