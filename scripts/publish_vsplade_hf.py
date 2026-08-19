"""Package and upload MLX-converted V-SPLADE (visual) weights to NomaDamas.

Repos are created PRIVATE; flip to public at launch time. Upstream weights are
Apache-2.0 (naver/v-splade-{efficient,quality}), so redistribution is
unrestricted with attribution.

Usage:
    uv run python scripts/publish_vsplade_hf.py [--only efficient|quality]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from splade_mlx.convert import DEFAULT_OUT_DIR  # noqa: E402
from splade_mlx.convert_vsplade import convert_vsplade  # noqa: E402

STAGING_ROOT = DEFAULT_OUT_DIR.parent / "hf-upload"
DTYPE = "bfloat16"

VARIANTS = ["efficient", "quality"]

CARD = """---
license: apache-2.0
base_model: naver/v-splade-{variant}
tags:
- mlx
- v-splade
- splade
- visual-document-retrieval
- sparse-retrieval
- multimodal
pipeline_tag: visual-document-retrieval
---

# v-splade-{variant}-mlx

MLX ({dtype}) conversion of [`naver/v-splade-{variant}`](https://huggingface.co/naver/v-splade-{variant})
(V-SPLADE, [arXiv:2605.30917](https://arxiv.org/abs/2605.30917)) for Apple Silicon,
produced by [NomaDamas/SPLADE-mlx](https://github.com/NomaDamas/SPLADE-mlx).

V-SPLADE is an inference-free sparse retriever for visual document retrieval:
document pages (rendered PDFs, slides, scans) are encoded by a ModernVBERT
backbone (SigLIP vision tower + pixel-shuffle connector + ModernBERT text
encoder) with a SPLADE MLM head into a 50,368-dim vocabulary-space sparse
vector, while queries are resolved by a learned Bag-of-Words lookup with no
neural encoding at all.

**Contents**: `weights.safetensors` (document encoder, {dtype}),
`query_lookup.npy` (inference-free query table, fp32), `config.json`,
plus tokenizer/processor configs for self-contained loading.

**Changes from upstream**: PyTorch checkpoint converted to MLX safetensors
(parameter re-mapping, conv weight transposed to NHWC, cast to {dtype}); the
query lookup table `softplus(embedding @ projection + bias)` is precomputed
with special tokens zeroed. No training or fine-tuning was performed.

**Quality** (see repo REPORT.md for methodology):
- fp32 parity vs the PyTorch reference: max |logit delta| {logit_delta} on
  real document-page inputs, sparse-vector cosine 1.000000, top-64 term
  overlap 100%; the query table matches the shipped Sentence Transformers
  static embedding to 1.2e-07.
- ViDoRe `docvqa_test_subsampled` nDCG@5 (fp32): {torch_ndcg} (torch) ->
  {mlx_ndcg} (MLX), delta {ndcg_delta:+.4f} (gate: ±0.002).

## Usage

```python
from splade_mlx.convert_vsplade import load_vsplade
import mlx.core as mx
from PIL import Image

model, query_encoder, processor = load_vsplade("NomaDamas/v-splade-{variant}-mlx")

# documents (page images)
enc = processor(text=["User:<image><end_of_utterance>\\nAssistant:"],
                images=[[Image.open("page.png")]], return_tensors="np")
d = model.encode(mx.array(enc["input_ids"]), mx.array(enc["attention_mask"]),
                 enc["pixel_values"])   # (1, 50368)

# queries: inference-free lookup, no neural network
q = processor.tokenizer(["total revenue 2023"], return_tensors="np")
qw = query_encoder.encode(q["input_ids"], q["attention_mask"])  # (1, 50368)

score = d @ qw.T
```

## License

Apache-2.0, same as the upstream checkpoint (© NAVER Corp).
This repository is not affiliated with or endorsed by NAVER.
"""


def main() -> None:
    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=VARIANTS, default=None)
    args = parser.parse_args()

    stats = {
        "efficient": {"logit_delta": "1.5e-04"},
        "quality": {"logit_delta": "6.4e-04"},
    }
    vidore_path = Path(__file__).resolve().parent.parent / "results" / "quality_vidore.json"
    if vidore_path.exists():
        vidore = json.loads(vidore_path.read_text())
        for variant in VARIANTS:
            cell = vidore.get(f"naver/v-splade-{variant}", {})
            if "torch-mps-fp32" in cell and "mlx-float32" in cell:
                stats[variant]["torch_ndcg"] = f"{cell['torch-mps-fp32']:.4f}"
                stats[variant]["mlx_ndcg"] = f"{cell['mlx-float32']:.4f}"
                stats[variant]["ndcg_delta"] = cell["mlx-float32"] - cell["torch-mps-fp32"]

    api = HfApi()
    for variant in VARIANTS:
        if args.only and args.only != variant:
            continue
        src = f"naver/v-splade-{variant}"
        dst = f"NomaDamas/v-splade-{variant}-mlx"
        print(f"=== {dst}", flush=True)

        dest = convert_vsplade(src, out_dir=DEFAULT_OUT_DIR, dtype=DTYPE)
        staging = STAGING_ROOT / f"v-splade-{variant}-mlx"
        staging.mkdir(parents=True, exist_ok=True)
        for name in ("weights.safetensors", "query_lookup.npy", "config.json"):
            shutil.copy(dest / name, staging / name)
        # tokenizer + processor configs for self-contained loading
        src_files = snapshot_download(
            src,
            allow_patterns=[
                "tokenizer*", "special_tokens_map.json", "preprocessor_config.json",
                "processor_config.json", "chat_template.jinja",
            ],
        )
        for f in Path(src_files).iterdir():
            shutil.copy(f, staging / f.name)
        s = stats[variant]
        (staging / "README.md").write_text(
            CARD.format(
                variant=variant,
                dtype=DTYPE,
                logit_delta=s["logit_delta"],
                torch_ndcg=s.get("torch_ndcg", "n/a"),
                mlx_ndcg=s.get("mlx_ndcg", "n/a"),
                ndcg_delta=s.get("ndcg_delta", float("nan")),
            )
        )
        api.create_repo(dst, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=dst, folder_path=str(staging),
            commit_message=f"MLX {DTYPE} conversion of {src}",
        )
        print(f"  uploaded -> https://huggingface.co/{dst}", flush=True)
    print("PUBLISH_VSPLADE_DONE", flush=True)


if __name__ == "__main__":
    import json

    main()
