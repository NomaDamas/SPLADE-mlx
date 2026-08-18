# SPLADE-mlx

**SPLADE / V-SPLADE sparse retrieval models, natively on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).**

- **1.3–4.0x faster** than the best PyTorch config (MPS fp16) on an M4 Max — single-query encoding in **1.6–2.1 ms**
- **Bit-exact quality**: MLX fp32 matches PyTorch fp32 nDCG@10 to the last float digit on BEIR NFCorpus & SciFact
- BERT and DistilBERT MLM backbones, asymmetric V-SPLADE query/doc pair API, bf16 / 8-bit / 4-bit quantization

Full benchmark methodology and tables: [REPORT.md](REPORT.md) · Project plan: [PLAN.md](PLAN.md)

## Install

```bash
uv sync   # or: pip install -e .
```

## Usage

```python
from splade_mlx import load, load_pair

# symmetric SPLADE++ (one encoder for queries and docs)
model, tok = load("naver/splade-cocondenser-ensembledistil", dtype="bfloat16")
enc = tok(["what causes vitamin d deficiency"], return_tensors="np", padding=True)
import mlx.core as mx
sparse = model.encode(mx.array(enc["input_ids"]), mx.array(enc["attention_mask"]))  # (1, 30522)

# asymmetric V-SPLADE (efficient-splade-V): separate query / doc encoders
pair = load_pair(dtype="bfloat16")
q = pair.encode_query(["what causes vitamin d deficiency"])
d = pair.encode_doc(["Vitamin D deficiency is commonly caused by ..."])
score = (q @ d.T)  # sparse dot product
```

`load()` accepts either an original HF checkpoint id (converted to MLX on first
use and cached) or a pre-converted MLX repo (e.g. `NomaDamas/*-mlx`).

Quantization: `load(..., quantize_bits=8)` (see REPORT.md §2 for the quality
impact; bf16 and q8 are within ±0.0014 nDCG@10 of fp32).

## Benchmarks (Apple M4 Max, vs PyTorch MPS fp16)

| model | workload | torch mps fp16 | best MLX | speedup |
|---|---|---|---|---|
| efficient-splade-V-query | query B1 | 4.65 ms | 1.57 ms (q8) | **2.96x** |
| efficient-splade-V-query | query B32 | 49.41 ms | 12.22 ms (bf16+compile) | **4.04x** |
| efficient-splade-V-doc | doc L256-B64 | 273.49 ms | 179.66 ms (bf16+compile) | **1.52x** |
| splade-cocondenser | query B1 | 6.77 ms | 2.51 ms (q8) | **2.70x** |
| splade-cocondenser | doc L256-B64 | 393.32 ms | 294.41 ms (bf16) | **1.34x** |

Reproduce: see [REPORT.md §6](REPORT.md). Parity gates: `uv run pytest tests/`.

## License

Code: Apache-2.0.

Model weights keep their upstream licenses:
- `naver/*` SPLADE checkpoints: **CC BY-NC-SA 4.0 (non-commercial)** — NAVER Corp
- `prithivida/Splade_PP_en_v1`: Apache-2.0
