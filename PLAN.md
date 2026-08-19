# SPLADE-mlx Project Plan (phase 1: text SPLADE)

> **Correction note**: this plan's original "V-SPLADE" target was misidentified as the
> SIGIR'22 `efficient-splade-V-large` pair (efficiency Level V, text-only). The actual
> **V-SPLADE** ([arXiv:2605.30917](https://arxiv.org/abs/2605.30917),
> `naver/v-splade-{efficient,quality}`) is a multimodal inference-free sparse retriever
> for visual document retrieval (ModernVBERT backbone, Apache-2.0). Its MLX port is a
> separate follow-up phase; references to "Efficient-SPLADE" below are the SIGIR'22
> text models.

Problem: there is no SPLADE / Efficient-SPLADE inference implementation specialized for Apple
Silicon (MLX). Overall flow: **measure the existing (PyTorch) models on a Mac → port to
MLX → measure the ported models and report speed/quality.**

- Work machine: Apple M4 Max (12P+4E, 64 GB), macOS, mlx 0.32.x, uv
- Reference project: [Blaizzy/mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) —
  supports dense embedding models (BERT/XLM-R/ModernBERT, ...) but **no MLM head +
  SPLADE sparse head**, which is exactly the gap this project fills. Its code layout
  (`models/bert.py`, `convert`, `load` API) is used as a design benchmark.

---

## 1. Background

### What SPLADE is

A model that builds a sparse vector on top of the **MLM head (30,522-dim vocab logits)**
of a BERT-family encoder:

```
w_j = max_{i in tokens} log(1 + ReLU(logit_{ij}))   (j = vocab index)
```

- Attention-masked max-pooling over the sequence axis → each query/document becomes a
  30,522-dim sparse vector
- Retrieval score = dot product of sparse vectors (inverted-index compatible)

### What Efficient-SPLADE is

The **efficiency Level V** configuration from *An Efficiency Study for SPLADE Models*
(Lassance & Clinchant, SIGIR'22). Its key trait is **separate (asymmetric) query and
document encoders**:

- `naver/efficient-splade-V-large-query` (query encoder)
- `naver/efficient-splade-V-large-doc` (document encoder)
- MS MARCO dev 38.8 MRR@10, 45.3 ms inference latency (per the paper)
- Level VI-BT uses a BERT-tiny query encoder (0.7 ms query encoding) — a good stretch goal

### What the port actually is

= implementing `BertForMaskedLM` (encoder + MLM prediction head, tied embeddings) in MLX
+ SPLADE activation/pooling + HF safetensors → MLX weight conversion + tokenizer
integration. mlx-embeddings' `bert.py` stops at the encoder, so the MLM head is new work.

---

## 2. Target model matrix

| Priority | Model | Architecture | Notes |
|---|---|---|---|
| **P0** | `naver/splade-cocondenser-ensembledistil` | BERT-base + MLM | Flagship SPLADE++ model, symmetric (query = doc), ungated |
| **P0** | `naver/efficient-splade-V-large-{query,doc}` | **DistilBERT** ×2 | **Efficient-SPLADE**, asymmetric pair. (Confirmed during implementation: DistilBERT, not BERT) |
| P1 | `naver/splade-v3-distilbert` | DistilBERT + MLM | DistilBERT support already done in P0. `splade-v3` itself is HF-gated (terms acknowledgement) |
| P1 | `naver/efficient-splade-VI-BT-large-{query,doc}` | BERT-tiny (query) + BERT-base (doc) | Ultra-fast query-encoding demo |
| Alt | `prithivida/Splade_PP_en_v1` | BERT-base + MLM | Apache-2.0. Fallback if the naver CC BY-NC-SA (non-commercial) license is an issue |

> **License note**: the naver SPLADE family is CC BY-NC-SA 4.0 (non-commercial).
> Research/benchmarking is fine; redistribution of converted weights requires the same
> license + attribution, and commercial use requires switching to the prithivida line.

---

## 3. Phase plan

### Phase 0 — Scaffolding (0.5 d)

- `uv init` + **pin Python 3.12** (system 3.14 risks torch/transformers wheel
  compatibility → 3.12 in the venv)
- Dependencies: `mlx`, `torch`, `transformers`, `tokenizers`, `safetensors`, `numpy`,
  `pyarrow` (BEIR parquet), `pytest`
- Repository layout:

```
SPLADE-mlx/
├── pyproject.toml
├── splade_mlx/
│   ├── __init__.py           # public load() API
│   ├── models/
│   │   └── bert.py           # BertModel + MLM head + SpladeHead (BERT & DistilBERT)
│   └── convert.py            # HF safetensors → MLX weights (key sanitize, dtype cast)
├── bench/
│   ├── workloads.py          # shared workload definitions (identical torch/mlx inputs)
│   ├── bench_torch.py        # Phase 1
│   ├── bench_mlx.py          # Phase 4
│   └── report.py             # result JSON → tables/report
├── tests/
│   └── test_parity.py        # torch ↔ mlx numeric parity tests
├── results/                  # benchmark result JSONs (committed)
└── data/                     # dataset cache (gitignored)
```

**Done when**: `uv run python -c "import mlx.core, torch, transformers"` succeeds.

### Phase 1 — Baseline measurement of existing models on the Mac (1 d)

Backends measured: **PyTorch CPU / PyTorch MPS** (both are needed to state clearly what
MLX actually beat).

**Workloads** (bitwise-identical inputs for torch/mlx, fixed seed):
- Query encoding: MS MARCO/BEIR query samples, seq_len ~32, batch {1, 8, 32}
- Document encoding: BEIR NFCorpus documents, seq_len {128, 256}, batch {1, 8, 32, 64}

**Metrics**:
- latency p50/p95 (warm-up then adaptive repetitions; `torch.mps.synchronize()` on MPS)
- throughput (docs/s, tokens/s)
- peak memory (RSS + MPS allocated)
- tokenization time measured separately (encoder-only and end-to-end)

**Extra deliverable — parity references**: for 32 fixed inputs, save torch fp32 logits
and sparse vectors as `.npz` → the answer key for Phase 3 tests.

**Done when**: `results/baseline_{cpu,mps}.json` and the reference `.npz` files exist.

### Phase 2 — MLX port (2–3 d, core)

1. **BERT encoder + MLM head** (`models/bert.py`)
   - embeddings (word/position/token_type + LayerNorm) → encoder layers → MLM transform
     (dense + gelu + LayerNorm) → decoder (tied word embedding + bias)
   - `SpladeHead`: `log1p(relu(logits))` → apply attention mask → max-pool
     (`mx.log1p`, `mx.maximum`)
2. **Weight conversion** (`convert.py`)
   - download safetensors from the HF hub → key mapping/sanitizing → `mx.save_safetensors`
   - dtype: fp32 (for parity validation) / **bf16 (default)** save options
   - Efficient-SPLADE loads the query/doc checkpoints as one asymmetric pair API:
     `SpladePair.encode_query() / .encode_doc()`
3. **Tokenizer**: HF `tokenizers` as-is (100% identical tokens to the torch path →
   removes a parity variable)
4. **Public API** (mlx-embeddings style):
   ```python
   from splade_mlx import load
   model, tokenizer = load("naver/splade-cocondenser-ensembledistil")
   sparse = model.encode(["hello world"])   # (B, 30522) or top-k (indices, values)
   ```
5. **Quantization (later)**: `mx.quantize` 8-bit/4-bit options. The MLM decoder is
   embedding-tied, so quantization can hurt quality → only ship what passes the Phase 3
   quality gate
6. P1: add DistilBERT (v3-distilbert) and BERT-tiny (VI-BT query) architectures

**Done when**: both P0 models load and encode in MLX, with sane output shapes and
sparsity (number of active terms).

### Phase 3 — Parity validation (1 d)

The premise of any speed report is proof that it is "the same model". Three gates:

1. **Logit parity**: 32 fixed inputs, MLX fp32 vs the torch fp32 reference →
   `max |Δ| < 1e-3`
2. **Sparse-vector parity**: top-k (k=64) term-index overlap ≥ 99%, active-term weight
   cosine ≥ 0.9999
3. **Retrieval-quality parity**: full encoding of BEIR **NFCorpus + SciFact** (small
   enough for local runs) → brute-force dot-product ranking → **nDCG@10 within ±0.002
   of the torch result**
   - bf16/quantized variants pass through the same gate to quantify degradation →
     dtype-by-dtype quality table in the report

**Done when**: `pytest tests/` passes and the per-dtype nDCG table exists.

### Phase 4 — MLX benchmark & performance report (1 d)

- Measure MLX with the **same harness and workloads** as Phase 1 (bf16 default; fp32 /
  4-bit / 8-bit additional)
- MLX-specific care: force lazy evaluation with `mx.eval()` before timing, separate
  kernel-compilation cost via warm-up, compare with/without `mx.compile`
- Write **REPORT.md**:
  - latency/throughput tables for model × backend (CPU/MPS/MLX) × batch × seq_len,
    with speedup multiples
  - memory comparison, per-dtype quality (nDCG@10) vs speed trade-off
  - highlight the asymmetric Efficient-SPLADE query-encoding latency (on-device search angle)
  - one-line reproduction commands

**Done when**: REPORT.md and `results/*.json` are complete, every number measured.

### Phase 5 — Stretch (optional)

- Upstream PR of the SPLADE architectures to mlx-embeddings (note: GPL v3 project)
- Upload converted weights to HF (`mlx-community` style) — **after license review**
  (done: NomaDamas org, CC BY-NC-SA tag + attribution for naver, Apache for prithivida)
- A "sub-millisecond query encoding on M4" demo with VI-BT (BERT-tiny query encoder)
- Local end-to-end search demo: top-k sparse output + a simple inverted index

---

## 4. Risks & upfront decisions

| Risk | Mitigation |
|---|---|
| torch wheel issues on Python 3.14 | pin 3.12 via uv |
| `naver/splade-v3` is gated | requires HF terms acknowledgement → deferred to P1; P0 uses ungated models only |
| naver license is CC BY-NC-SA | benchmarking/research OK; redistribution handled with same-license + attribution; Apache alternative: prithivida |
| bf16 numeric drift in `log1p(relu(·))` | pass fp32 parity first → judge bf16 by the quality gate (nDCG) |
| "unfair MPS baseline" claims | strict synchronization, identical warm-up, tokenization measured separately, methodology documented in the report |
| MLM-decoder quantization quality collapse | quantization is optional; if the gate fails, use mixed quantization excluding embedding/decoder |

## 5. Timeline summary

| Phase | Content | Estimate |
|---|---|---|
| 0 | Scaffolding | 0.5 d |
| 1 | PyTorch baseline measurement + reference outputs | 1 d |
| 2 | MLX port (BERT+MLM+SPLADE, convert, API) | 2–3 d |
| 3 | Parity validation (logits/vectors/nDCG) | 1 d |
| 4 | MLX benchmark + REPORT.md | 1 d |
| 5 | Stretch (upstream PR, HF upload, demos) | optional |

**Total 5.5–6.5 d (excluding stretch).**

---

## References

- Efficient-SPLADE (Efficient SPLADE Level V): https://huggingface.co/naver/efficient-splade-V-large-doc , https://huggingface.co/naver/efficient-splade-V-large-query
- Paper: *An Efficiency Study for SPLADE Models* — https://reneuir.org/assets/slides/ReNeuIR2022-efficient-splade.pdf
- Original SPLADE repository: https://github.com/naver/splade
- SPLADE v2 paper: https://arxiv.org/abs/2109.10086
- mlx-embeddings: https://github.com/Blaizzy/mlx-embeddings
