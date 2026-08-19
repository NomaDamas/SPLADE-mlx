"""Convert naver/v-splade-{efficient,quality} checkpoints to MLX format.

Writes {out_dir}/{model_name}-{dtype}/:
    weights.safetensors   - document encoder (vision + text + MLM head)
    query_lookup.npy      - inference-free query lookup table (vocab,), fp32
    config.json           - unified config with model_type "vsplade"

Usage:
    uv run python -m splade_mlx.convert_vsplade --hf-id naver/v-splade-efficient
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import shutil

import mlx.core as mx
import numpy as np

from .convert import DEFAULT_OUT_DIR
from .models.modernvbert import SPECIAL_TOKEN_IDS

_PREFIX = "encoder.encoder.model."

_VISION_LAYER_RE = re.compile(r"^vision_model\.encoder\.(layers\.\d+\..+)$")


def _map_key(key: str) -> str | None:
    if key.startswith("query_encoder."):
        return None  # handled separately
    if key.startswith("encoder.mlm_head."):
        return key.replace("encoder.mlm_head.", "mlm_head.")
    if not key.startswith(_PREFIX):
        raise KeyError(f"unmapped v-splade key: {key}")
    k = key[len(_PREFIX) :]
    if k.startswith("vision_model.head."):
        return None  # SigLIP pooling head, unused by V-SPLADE
    m = _VISION_LAYER_RE.match(k)
    if m:
        return f"vision_model.{m.group(1)}"
    if k == "connector.modality_projection.proj.weight":
        return "connector.proj.weight"
    if k.startswith("text_model.embeddings.tok_embeddings.additional_embedding."):
        return k.replace(
            "text_model.embeddings.tok_embeddings.additional_embedding.",
            "text_model.embeddings_tok.additional_embedding.",
        )
    if k.startswith("text_model.embeddings.tok_embeddings."):
        return k.replace(
            "text_model.embeddings.tok_embeddings.", "text_model.embeddings_tok.tok."
        )
    if k.startswith("text_model.embeddings.norm."):
        return k.replace("text_model.embeddings.norm.", "text_model.embeddings_norm.")
    if k == "text_model.layers.0.attn_norm.weight":
        return None  # layer 0 uses Identity in the reference implementation
    if k.startswith(("vision_model.", "text_model.")):
        return k
    raise KeyError(f"unmapped v-splade key: {key}")


def convert_vsplade(hf_id: str, out_dir: Path = DEFAULT_OUT_DIR, dtype: str = "float32") -> Path:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(hf_id, "model.safetensors")
    hf_config = json.loads(Path(hf_hub_download(hf_id, "config.json")).read_text())

    mx_dtype = getattr(mx, dtype)
    weights: dict[str, mx.array] = {}
    query_raw: dict[str, np.ndarray] = {}
    # the checkpoint is stored in bfloat16, which numpy cannot represent ->
    # read through torch and upcast to fp32 (same as the torch fp32 reference)
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key).to_dense().float().numpy()
            if key.startswith("query_encoder."):
                query_raw[key] = tensor
                continue
            mlx_key = _map_key(key)
            if mlx_key is None:
                continue
            if mlx_key == "vision_model.embeddings.patch_embedding.weight":
                tensor = tensor.transpose(0, 2, 3, 1)  # OIHW -> OHWI for mlx Conv2d
            weights[mlx_key] = mx.array(tensor).astype(mx_dtype)

    # Inference-free query lookup: softplus(projection(embedding)), specials zeroed.
    emb = query_raw["query_encoder.embeddings.weight"].astype(np.float64)
    w = query_raw["query_encoder.projection.weight"].astype(np.float64)  # (1, hidden)
    b = query_raw["query_encoder.projection.bias"].astype(np.float64)  # (1,)
    z = emb @ w[0] + b[0]
    lookup = np.logaddexp(0.0, z).astype(np.float32)  # softplus, numerically stable
    lookup[list(SPECIAL_TOKEN_IDS)] = 0.0

    model_name = hf_id.split("/")[-1]
    dest = out_dir / f"{model_name}-{dtype}"
    dest.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dest / "weights.safetensors"), weights)
    np.save(dest / "query_lookup.npy", lookup)
    (dest / "config.json").write_text(
        json.dumps(
            {
                "hf_id": hf_id,
                "dtype": dtype,
                "model_type": "vsplade",
                "vsplade": {
                    "text_config": hf_config["text_config"],
                    "vision_config": hf_config["vision_config"],
                    "pixel_shuffle_factor": hf_config["pixel_shuffle_factor"],
                    "additional_vocab_size": hf_config["additional_vocab_size"],
                    "image_token_id": hf_config["image_token_id"],
                },
            },
            indent=2,
        )
    )
    return dest


def load_vsplade(
    hf_id_or_dir: str,
    dtype: str | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
):
    """Returns (doc_model, query_encoder, processor).

    Accepts a local converted dir, a pre-converted MLX hub repo (e.g.
    NomaDamas/*-mlx; loaded directly), or an original HF checkpoint id
    (converted to `dtype` on first use and cached).
    """
    from huggingface_hub import hf_hub_download
    from transformers import AutoProcessor

    from .models.modernvbert import VSpladeConfig, VSpladeModel, VSpladeQueryEncoder

    p = Path(hf_id_or_dir)
    if p.is_dir() and (p / "config.json").exists():
        dest = p
    else:
        try:  # pre-converted MLX repo on the hub?
            cfg = json.loads(
                Path(hf_hub_download(hf_id_or_dir, "config.json")).read_text()
            )
        except Exception:
            cfg = {}
        if "vsplade" in cfg:
            dest = out_dir / f"hub-{hf_id_or_dir.split('/')[-1]}"
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("weights.safetensors", "query_lookup.npy"):
                if not (dest / name).exists():
                    shutil.copy(hf_hub_download(hf_id_or_dir, name), dest / name)
            (dest / "config.json").write_text(json.dumps(cfg))
        else:  # original checkpoint -> convert
            conversion_dtype = dtype or "float32"
            dest = out_dir / f"{hf_id_or_dir.split('/')[-1]}-{conversion_dtype}"
            if not (dest / "weights.safetensors").exists():
                dest = convert_vsplade(
                    hf_id_or_dir, out_dir=out_dir, dtype=conversion_dtype
                )

    config = json.loads((dest / "config.json").read_text())
    stored_dtype = config.get("dtype")
    if stored_dtype is None:
        raise ValueError(f"{dest} does not declare its stored weight dtype")
    if dtype is not None and dtype != stored_dtype:
        raise ValueError(
            f"{dest} stores {stored_dtype} weights; requested dtype={dtype}. "
            "Load without dtype to use the stored weights, or load the original "
            "upstream checkpoint to create a conversion at the requested dtype."
        )
    model = VSpladeModel(VSpladeConfig.from_hf(config["vsplade"]))
    model.load_weights(str(dest / "weights.safetensors"))
    mx.eval(model.parameters())
    query = VSpladeQueryEncoder(np.load(dest / "query_lookup.npy"))
    try:  # self-contained MLX repos ship processor configs
        processor = AutoProcessor.from_pretrained(hf_id_or_dir)
    except Exception:
        processor = AutoProcessor.from_pretrained(config["hf_id"])
    return model, query, processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-id", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    args = parser.parse_args()
    dest = convert_vsplade(args.hf_id, args.out_dir, args.dtype)
    print(f"converted {args.hf_id} -> {dest}")


if __name__ == "__main__":
    main()
