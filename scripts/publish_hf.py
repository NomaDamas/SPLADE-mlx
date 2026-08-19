"""Package and upload MLX-converted SPLADE weights to the NomaDamas HF org.

Repos are created PRIVATE; flip to public at launch time.
License compliance:
  - naver/* checkpoints are CC BY-NC-SA 4.0 (c) NAVER Corp. Redistribution of
    the converted (adapted) weights is permitted under the same license with
    attribution and a statement of changes (ShareAlike). Uploaded with the
    cc-by-nc-sa-4.0 tag + non-commercial notice.
  - prithivida/Splade_PP_en_v1 is Apache-2.0.

Usage:
    uv run python scripts/publish_hf.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from splade_mlx import _resolve  # noqa: E402
from splade_mlx.convert import DEFAULT_OUT_DIR  # noqa: E402

STAGING_ROOT = DEFAULT_OUT_DIR.parent / "hf-upload"
DTYPE = "bfloat16"

MODELS = [
    {
        "src": "naver/splade-cocondenser-ensembledistil",
        "dst": "NomaDamas/splade-cocondenser-ensembledistil-mlx",
        "license": "cc-by-nc-sa-4.0",
        "nc": True,
        "desc": "SPLADE++ (CoCondenser-EnsembleDistil): symmetric sparse encoder for queries and documents (BERT-base).",
        "quality": "BEIR quality parity vs PyTorch fp32: nDCG@10 delta +0.0001 (NFCorpus) / +0.0004 (SciFact) in bfloat16.",
    },
    {
        "src": "naver/splade-v3-distilbert",
        "dst": "NomaDamas/splade-v3-distilbert-mlx",
        "license": "cc-by-nc-sa-4.0",
        "nc": True,
        "desc": "SPLADE-v3 DistilBERT variant: symmetric sparse encoder for queries and documents.",
        "quality": "fp32 parity vs PyTorch: max |logit delta| 5.8e-05, sparse cosine 1.000000, top-64 term overlap 100%.",
    },
    {
        "src": "naver/splade-v3",
        "dst": "NomaDamas/splade-v3-mlx",
        "license": "cc-by-nc-sa-4.0",
        "nc": True,
        "desc": "SPLADE-v3 (BERT-base): symmetric sparse encoder for queries and documents; the main SPLADE-v3 family model. Upstream is gated on the HF hub (click-through, CC BY-NC-SA acknowledgement).",
        "quality": "fp32 parity vs PyTorch: max |logit delta| 7.6e-05, sparse cosine 1.000000, top-64 term overlap 100%.",
    },
    {
        "src": "naver/splade-v3-doc",
        "dst": "NomaDamas/splade-v3-doc-mlx",
        "license": "cc-by-nc-sa-4.0",
        "nc": True,
        "desc": "SPLADE-v3-Doc (BERT-base): document-side encoder; the query side is inference-free (bag-of-words) in the SPLADE-v3-Doc setup.",
        "quality": "fp32 parity vs PyTorch: max |logit delta| 4.8e-05, sparse cosine 1.000000, top-64 term overlap 100%.",
    },
    {
        "src": "naver/splade-v3-lexical",
        "dst": "NomaDamas/splade-v3-lexical-mlx",
        "license": "cc-by-nc-sa-4.0",
        "nc": True,
        "desc": "SPLADE-v3-Lexical (BERT-base): lexical-matching-focused SPLADE-v3 variant (see the SPLADE-v3 paper, arXiv:2403.06789).",
        "quality": "fp32 parity vs PyTorch: max |logit delta| 3.1e-05, sparse cosine 1.000000, top-64 term overlap 100%.",
    },
    {
        "src": "prithivida/Splade_PP_en_v1",
        "dst": "NomaDamas/Splade_PP_en_v1-mlx",
        "license": "apache-2.0",
        "nc": False,
        "desc": "Independent Apache-2.0 SPLADE++ reproduction (BERT-base), freely usable commercially.",
        "quality": "fp32 parity vs PyTorch: max |logit delta| 5.5e-05, sparse cosine 1.000000, top-64 term overlap 100%.",
    },
]

CARD = """---
license: {license}
base_model: {src}
tags:
- mlx
- splade
- sparse-retrieval
- retrieval
pipeline_tag: feature-extraction
---

# {name}

MLX ({dtype}) conversion of [`{src}`](https://huggingface.co/{src}) for Apple Silicon,
produced by [NomaDamas/SPLADE-mlx](https://github.com/NomaDamas/SPLADE-mlx).

{desc}

**Changes from upstream**: PyTorch checkpoint converted to MLX `safetensors`
(parameter re-mapping, cast to {dtype}). No training or fine-tuning was performed.

**Quality**: {quality}
Numeric parity gates (fp32): MLM logits max |delta| < 1e-3, sparse-vector cosine >= 0.9999,
top-64 term overlap >= 99% vs the PyTorch reference. Full methodology: see the SPLADE-mlx report.

## Usage

```python
from splade_mlx import load
import mlx.core as mx

model, tok = load("{dst}")
enc = tok(["what causes vitamin d deficiency"], return_tensors="np", padding=True)
sparse = model.encode(mx.array(enc["input_ids"]), mx.array(enc["attention_mask"]))  # (1, 30522)
```

## License

{license_body}
"""

NC_BODY = (
    "**CC BY-NC-SA 4.0** — the original weights are Copyright (c) NAVER Corp. "
    "(NAVER LABS Europe) and are licensed for **non-commercial use only**. "
    "This conversion is Adapted Material redistributed under the same "
    "CC BY-NC-SA 4.0 license (ShareAlike), with attribution and the changes stated above. "
    "This repository is not affiliated with or endorsed by NAVER. "
    "For commercial use, consider `NomaDamas/Splade_PP_en_v1-mlx` (Apache-2.0)."
)
APACHE_BODY = "Apache-2.0, same as the upstream checkpoint."


def main() -> None:
    import argparse

    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="substring filter on src/dst; publish only matching models")
    args = parser.parse_args()

    api = HfApi()
    for spec in MODELS:
        if args.only and args.only not in spec["src"] and args.only not in spec["dst"]:
            continue
        name = spec["dst"].split("/")[-1]
        print(f"=== {spec['dst']}", flush=True)
        src_dir = _resolve(spec["src"], DTYPE, DEFAULT_OUT_DIR)

        staging = STAGING_ROOT / name
        staging.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_dir / "weights.safetensors", staging / "weights.safetensors")
        shutil.copy(src_dir / "config.json", staging / "config.json")
        AutoTokenizer.from_pretrained(spec["src"]).save_pretrained(staging)
        (staging / "README.md").write_text(
            CARD.format(
                name=name,
                dtype=DTYPE,
                license=spec["license"],
                src=spec["src"],
                dst=spec["dst"],
                desc=spec["desc"],
                quality=spec["quality"],
                license_body=NC_BODY if spec["nc"] else APACHE_BODY,
            )
        )

        api.create_repo(spec["dst"], repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=spec["dst"],
            folder_path=str(staging),
            commit_message=f"MLX {DTYPE} conversion of {spec['src']}",
        )
        print(f"  uploaded -> https://huggingface.co/{spec['dst']}", flush=True)
    print("PUBLISH_HF_DONE", flush=True)


if __name__ == "__main__":
    main()
