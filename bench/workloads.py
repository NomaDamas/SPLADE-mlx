"""Shared workload definitions.

Both the PyTorch baseline (bench_torch) and the MLX benchmark (bench_mlx) import
from this module so that model lists, input texts, sequence lengths, and batch
sizes are guaranteed identical across backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
REFERENCE_DIR = DATA_DIR / "reference"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# roles: which workload kinds this checkpoint is used for.
# splade-cocondenser is symmetric (one encoder for queries and docs).
# efficient-splade-V ("V-SPLADE") is an asymmetric query/doc pair.
P0_MODELS: dict[str, dict] = {
    "splade-cocondenser-ensembledistil": {
        "hf_id": "naver/splade-cocondenser-ensembledistil",
        "roles": ("query", "doc"),
    },
    "efficient-splade-V-large-query": {
        "hf_id": "naver/efficient-splade-V-large-query",
        "roles": ("query",),
    },
    "efficient-splade-V-large-doc": {
        "hf_id": "naver/efficient-splade-V-large-doc",
        "roles": ("doc",),
    },
}

# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Workload:
    kind: str  # "query" | "doc"
    seq_len: int  # fixed padded length (padding="max_length")
    batch_size: int

    @property
    def name(self) -> str:
        return f"{self.kind}-L{self.seq_len}-B{self.batch_size}"


WORKLOADS: list[Workload] = [
    Workload("query", 32, 1),
    Workload("query", 32, 8),
    Workload("query", 32, 32),
    Workload("doc", 128, 1),
    Workload("doc", 128, 8),
    Workload("doc", 128, 32),
    Workload("doc", 256, 1),
    Workload("doc", 256, 8),
    Workload("doc", 256, 32),
    Workload("doc", 256, 64),
]

# ---------------------------------------------------------------------------
# Text sources (BEIR NFCorpus via HF hub parquet files)
# ---------------------------------------------------------------------------


def _load_parquet_texts(repo_id: str, filename: str, id_col: str, text_cols: list[str]) -> list[tuple[str, str]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    table = pq.read_table(path)
    rows = table.to_pylist()
    out = []
    for row in rows:
        text = " ".join(str(row[c]).strip() for c in text_cols if row.get(c))
        out.append((str(row[id_col]), text))
    out.sort(key=lambda x: x[0])  # deterministic order
    return out


@lru_cache(maxsize=1)
def nfcorpus_queries() -> list[tuple[str, str]]:
    return _load_parquet_texts(
        "BeIR/nfcorpus", "queries/queries-00000-of-00001.parquet", "_id", ["text"]
    )


@lru_cache(maxsize=1)
def nfcorpus_docs() -> list[tuple[str, str]]:
    return _load_parquet_texts(
        "BeIR/nfcorpus", "corpus/corpus-00000-of-00001.parquet", "_id", ["title", "text"]
    )


def texts_for(kind: str, n: int) -> list[str]:
    """Deterministic slice of real texts for benchmarking."""
    source = nfcorpus_queries() if kind == "query" else nfcorpus_docs()
    texts = [t for _, t in source[:n]]
    # cycle if the request exceeds the corpus (never happens for NFCorpus sizes)
    while len(texts) < n:
        texts.append(texts[len(texts) % max(1, len(source))])
    return texts


def parity_texts() -> list[str]:
    """Fixed 32 inputs (16 queries + 16 docs) used for torch<->mlx parity."""
    q = [t for _, t in nfcorpus_queries()[:16]]
    d = [t for _, t in nfcorpus_docs()[:16]]
    return q + d


PARITY_MAX_LEN = 256  # tokenizer truncation length for parity inputs
PARITY_LOGITS_COUNT = 4  # save full logits for only this many inputs (size)

# ---------------------------------------------------------------------------
# Shared measurement protocol (used by bench_torch and bench_mlx)
# ---------------------------------------------------------------------------
WARMUP_ITERS = 3
MIN_ITERS = 12
MAX_ITERS = 50
TARGET_SECONDS = 8.0
QUICK_PROTOCOL = {"min_iters": 3, "max_iters": 3, "target_seconds": 1.0}
