# SPLADE-mlx

**SPLADE / V-SPLADE sparse retrieval models, natively on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).**

![Inference latency: PyTorch vs MLX on Apple M4 Max](assets/latency_m4max.png)

![Retrieval quality is preserved](assets/quality_parity.png)

*All numbers above are measured on an Apple M4 Max (12P+4E, 64 GB). MLX fp32 reproduces
the PyTorch fp32 nDCG@10 to the last floating-point digit on BEIR NFCorpus and SciFact —
the ranking is identical, not just close. bf16 and 8-bit stay within ±0.0014.*

## Highlights

- **1.3–4.0x faster** than the best PyTorch configuration (MPS fp16); up to **8.9x** vs PyTorch CPU
- **Single-query encoding in 1.6 ms** (V-SPLADE query encoder, 8-bit) — fast enough for on-device, per-keystroke retrieval
- **Bit-exact quality** in fp32, verified with three parity gates plus full BEIR evaluations
- BERT and DistilBERT MLM backbones, asymmetric V-SPLADE query/doc pair API, bf16 / 8-bit / 4-bit quantization
- Pre-converted weights ready on the Hugging Face Hub (table below)

Full methodology and tables: [REPORT.md](REPORT.md) · Project plan: [PLAN.md](PLAN.md)

## Supported models

| MLX weights (Hugging Face) | Upstream checkpoint | Backbone | fp32 parity (max logit Δ) | License |
|---|---|---|---|---|
| [NomaDamas/splade-cocondenser-ensembledistil-mlx](https://huggingface.co/NomaDamas/splade-cocondenser-ensembledistil-mlx) | [naver/splade-cocondenser-ensembledistil](https://huggingface.co/naver/splade-cocondenser-ensembledistil) | BERT-base | 6.1e-05 | CC BY-NC-SA 4.0 |
| [NomaDamas/efficient-splade-V-large-query-mlx](https://huggingface.co/NomaDamas/efficient-splade-V-large-query-mlx) | [naver/efficient-splade-V-large-query](https://huggingface.co/naver/efficient-splade-V-large-query) | DistilBERT | 5.2e-05 | CC BY-NC-SA 4.0 |
| [NomaDamas/efficient-splade-V-large-doc-mlx](https://huggingface.co/NomaDamas/efficient-splade-V-large-doc-mlx) | [naver/efficient-splade-V-large-doc](https://huggingface.co/naver/efficient-splade-V-large-doc) | DistilBERT | 4.1e-05 | CC BY-NC-SA 4.0 |
| [NomaDamas/splade-v3-mlx](https://huggingface.co/NomaDamas/splade-v3-mlx) | [naver/splade-v3](https://huggingface.co/naver/splade-v3) (gated) | BERT-base | 7.6e-05 | CC BY-NC-SA 4.0 |
| [NomaDamas/splade-v3-doc-mlx](https://huggingface.co/NomaDamas/splade-v3-doc-mlx) | [naver/splade-v3-doc](https://huggingface.co/naver/splade-v3-doc) (gated) | BERT-base | 4.8e-05 | CC BY-NC-SA 4.0 |
| [NomaDamas/splade-v3-lexical-mlx](https://huggingface.co/NomaDamas/splade-v3-lexical-mlx) | [naver/splade-v3-lexical](https://huggingface.co/naver/splade-v3-lexical) (gated) | BERT-base | 3.1e-05 | CC BY-NC-SA 4.0 |
| [NomaDamas/splade-v3-distilbert-mlx](https://huggingface.co/NomaDamas/splade-v3-distilbert-mlx) | [naver/splade-v3-distilbert](https://huggingface.co/naver/splade-v3-distilbert) | DistilBERT | 5.8e-05 | CC BY-NC-SA 4.0 |
| [NomaDamas/Splade_PP_en_v1-mlx](https://huggingface.co/NomaDamas/Splade_PP_en_v1-mlx) | [prithivida/Splade_PP_en_v1](https://huggingface.co/prithivida/Splade_PP_en_v1) | BERT-base | 5.5e-05 | **Apache-2.0** |

Every row passed the same fp32 parity gates before upload: max MLM-logit delta < 1e-3,
sparse-vector cosine ≥ 0.9999, and top-64 expansion-term overlap ≥ 99% vs the PyTorch
reference. Any other BERT / DistilBERT `*ForMaskedLM` SPLADE checkpoint can be converted
with `splade_mlx.convert` as well.

> The `naver/*` weights are **non-commercial** (CC BY-NC-SA 4.0). For commercial use,
> pick `Splade_PP_en_v1-mlx` (Apache-2.0).

## Install

```bash
uv sync   # or: pip install -e .
```

## Usage

```python
import mlx.core as mx
from splade_mlx import load, load_pair

# Symmetric SPLADE (one encoder for queries and documents).
# Accepts a pre-converted MLX repo, or an original HF checkpoint id
# (converted to MLX on first use and cached).
model, tok = load("NomaDamas/splade-v3-mlx")
enc = tok(["what causes vitamin d deficiency"], return_tensors="np", padding=True)
sparse = model.encode(mx.array(enc["input_ids"]), mx.array(enc["attention_mask"]))  # (1, 30522)

# Asymmetric V-SPLADE: separate query / document encoders.
pair = load_pair(
    "NomaDamas/efficient-splade-V-large-query-mlx",
    "NomaDamas/efficient-splade-V-large-doc-mlx",
)
q = pair.encode_query(["what causes vitamin d deficiency"])
d = pair.encode_doc(["Vitamin D deficiency is commonly caused by ..."])
score = q @ d.T  # sparse dot product

# Quantization (see the quality chart above for the impact).
model, tok = load("NomaDamas/splade-v3-mlx", quantize_bits=8)
```

## Benchmarks

Measured with an identical harness on both frameworks: same tokenizer, same real
NFCorpus texts, fixed-length padding, warm-up plus adaptive repetitions, and explicit
synchronization (`torch.mps.synchronize()` / `mx.eval()`). Tokenization time is
excluded and reported separately. Selected results (full grid in
[REPORT.md](REPORT.md) and `results/*.json`):

| model | workload | PyTorch MPS fp16 | best MLX | speedup |
|---|---|---|---|---|
| efficient-splade-V-query | query, batch 1 | 4.65 ms | 1.57 ms (8-bit) | **2.96x** |
| efficient-splade-V-query | query, batch 32 | 49.41 ms | 12.22 ms (bf16+compile) | **4.04x** |
| efficient-splade-V-doc | doc L256, batch 64 | 273.49 ms | 179.66 ms (bf16+compile) | **1.52x** |
| splade-cocondenser | query, batch 1 | 6.77 ms | 2.51 ms (8-bit) | **2.70x** |
| splade-cocondenser | doc L256, batch 64 | 393.32 ms | 294.41 ms (bf16) | **1.34x** |

Quality was evaluated on two public BEIR benchmarks — **NFCorpus** (3.6k docs, 323 test
queries) and **SciFact** (5.2k docs, 300 test queries) — by encoding the full corpus and
ranking with the sparse dot product (nDCG@10, trec_eval-style linear gain). Absolute
scores match published SPLADE results, which independently validates the pipeline.

### Reproduce

```bash
uv run python -m bench.save_reference                      # torch fp32 parity references
uv run python -m bench.bench_torch --backend cpu --dtype fp32
uv run python -m bench.bench_torch --backend mps --dtype fp32
uv run python -m bench.bench_torch --backend mps --dtype fp16
uv run pytest tests/                                       # numeric parity gates
uv run python -m bench.eval_beir                           # BEIR nDCG@10 parity
uv run python -m bench.bench_mlx --dtype bfloat16          # (--compile / --quantize-bits 8|4)
uv run python -m bench.report                              # markdown comparison tables
uv run python scripts/make_charts.py                       # README charts
```

## License

Code: [Apache-2.0](LICENSE).

Model weights keep their upstream licenses (see the table above):
`naver/*` checkpoints are CC BY-NC-SA 4.0 (© NAVER Corp, non-commercial, redistributed
as adapted material under the same license); `prithivida/Splade_PP_en_v1` is Apache-2.0.
This project is not affiliated with or endorsed by NAVER.
