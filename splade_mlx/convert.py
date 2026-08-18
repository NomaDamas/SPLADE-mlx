"""Convert HF BertForMaskedLM checkpoints (SPLADE family) to MLX weights.

Usage:
    uv run python -m splade_mlx.convert --hf-id naver/splade-cocondenser-ensembledistil
    uv run python -m splade_mlx.convert --hf-id ... --dtype bfloat16

Writes {out_dir}/{model_name}/{weights.safetensors, config.json} where the
default out_dir is data/mlx-models. Tokenizer files are left on the HF hub;
load() pulls them with AutoTokenizer using the recorded hf_id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mlx-models"

# HF BertForMaskedLM parameter name -> MLX SpladeModel parameter name
_STATIC_MAP = {
    "bert.embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
    "bert.embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
    "bert.embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
    "bert.embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
    "bert.embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",
    "cls.predictions.transform.dense.weight": "mlm_head.dense.weight",
    "cls.predictions.transform.dense.bias": "mlm_head.dense.bias",
    "cls.predictions.transform.LayerNorm.weight": "mlm_head.layer_norm.weight",
    "cls.predictions.transform.LayerNorm.bias": "mlm_head.layer_norm.bias",
    "cls.predictions.decoder.weight": "mlm_head.decoder.weight",
    "cls.predictions.bias": "mlm_head.decoder.bias",
}

_LAYER_MAP = {
    "attention.self.query.weight": "attention.query.weight",
    "attention.self.query.bias": "attention.query.bias",
    "attention.self.key.weight": "attention.key.weight",
    "attention.self.key.bias": "attention.key.bias",
    "attention.self.value.weight": "attention.value.weight",
    "attention.self.value.bias": "attention.value.bias",
    "attention.output.dense.weight": "attention.out.weight",
    "attention.output.dense.bias": "attention.out.bias",
    "attention.output.LayerNorm.weight": "attention_norm.weight",
    "attention.output.LayerNorm.bias": "attention_norm.bias",
    "intermediate.dense.weight": "mlp_up.weight",
    "intermediate.dense.bias": "mlp_up.bias",
    "output.dense.weight": "mlp_down.weight",
    "output.dense.bias": "mlp_down.bias",
    "output.LayerNorm.weight": "mlp_norm.weight",
    "output.LayerNorm.bias": "mlp_norm.bias",
}

# HF DistilBertForMaskedLM (used by efficient-splade-V) -> same MLX tree
_DISTIL_STATIC_MAP = {
    "distilbert.embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
    "distilbert.embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
    "distilbert.embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
    "distilbert.embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",
    "vocab_transform.weight": "mlm_head.dense.weight",
    "vocab_transform.bias": "mlm_head.dense.bias",
    "vocab_layer_norm.weight": "mlm_head.layer_norm.weight",
    "vocab_layer_norm.bias": "mlm_head.layer_norm.bias",
    "vocab_projector.weight": "mlm_head.decoder.weight",
    "vocab_projector.bias": "mlm_head.decoder.bias",
}

_DISTIL_LAYER_MAP = {
    "attention.q_lin.weight": "attention.query.weight",
    "attention.q_lin.bias": "attention.query.bias",
    "attention.k_lin.weight": "attention.key.weight",
    "attention.k_lin.bias": "attention.key.bias",
    "attention.v_lin.weight": "attention.value.weight",
    "attention.v_lin.bias": "attention.value.bias",
    "attention.out_lin.weight": "attention.out.weight",
    "attention.out_lin.bias": "attention.out.bias",
    "sa_layer_norm.weight": "attention_norm.weight",
    "sa_layer_norm.bias": "attention_norm.bias",
    "ffn.lin1.weight": "mlp_up.weight",
    "ffn.lin1.bias": "mlp_up.bias",
    "ffn.lin2.weight": "mlp_down.weight",
    "ffn.lin2.bias": "mlp_down.bias",
    "output_layer_norm.weight": "mlp_norm.weight",
    "output_layer_norm.bias": "mlp_norm.bias",
}

_SKIP_SUBSTRINGS = ("position_ids", "cls.seq_relationship", "cls.predictions.decoder.bias")


def map_key(hf_key: str, model_type: str) -> str | None:
    if any(s in hf_key for s in _SKIP_SUBSTRINGS):
        return None
    if model_type == "bert":
        if hf_key in _STATIC_MAP:
            return _STATIC_MAP[hf_key]
        if hf_key.startswith("bert.encoder.layer."):
            rest = hf_key[len("bert.encoder.layer.") :]
            idx, _, sub = rest.partition(".")
            if sub in _LAYER_MAP:
                return f"encoder.layers.{idx}.{_LAYER_MAP[sub]}"
    elif model_type == "distilbert":
        if hf_key in _DISTIL_STATIC_MAP:
            return _DISTIL_STATIC_MAP[hf_key]
        if hf_key.startswith("distilbert.transformer.layer."):
            rest = hf_key[len("distilbert.transformer.layer.") :]
            idx, _, sub = rest.partition(".")
            if sub in _DISTIL_LAYER_MAP:
                return f"encoder.layers.{idx}.{_DISTIL_LAYER_MAP[sub]}"
    raise KeyError(f"unmapped HF key ({model_type}): {hf_key}")


def convert(hf_id: str, out_dir: Path = DEFAULT_OUT_DIR, dtype: str = "float32") -> Path:
    import torch
    from transformers import AutoConfig, AutoModelForMaskedLM

    hf_config = AutoConfig.from_pretrained(hf_id)
    if hf_config.model_type not in ("bert", "distilbert"):
        raise ValueError(f"unsupported model_type: {hf_config.model_type}")

    model = AutoModelForMaskedLM.from_pretrained(hf_id, dtype=torch.float32)
    state = model.state_dict()

    # tied decoder weight may be absent from the serialized checkpoint but is
    # always present in state_dict(); decoder bias is tied to cls.predictions.bias
    weights: dict[str, mx.array] = {}
    mx_dtype = getattr(mx, dtype)
    for hf_key, tensor in state.items():
        mlx_key = map_key(hf_key, hf_config.model_type)
        if mlx_key is None:
            continue
        weights[mlx_key] = mx.array(tensor.numpy()).astype(mx_dtype)

    model_name = hf_id.split("/")[-1]
    dest = out_dir / model_name
    dest.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dest / "weights.safetensors"), weights)
    (dest / "config.json").write_text(
        json.dumps(
            {
                "hf_id": hf_id,
                "dtype": dtype,
                "model_type": hf_config.model_type,
                "bert": _unified_config(hf_config),
            },
            indent=2,
        )
    )
    return dest


def _unified_config(hf_config) -> dict:
    if hf_config.model_type == "bert":
        return {
            "vocab_size": hf_config.vocab_size,
            "hidden_size": hf_config.hidden_size,
            "num_hidden_layers": hf_config.num_hidden_layers,
            "num_attention_heads": hf_config.num_attention_heads,
            "intermediate_size": hf_config.intermediate_size,
            "max_position_embeddings": hf_config.max_position_embeddings,
            "type_vocab_size": hf_config.type_vocab_size,
            "layer_norm_eps": hf_config.layer_norm_eps,
            "use_token_type": True,
        }
    # distilbert
    return {
        "vocab_size": hf_config.vocab_size,
        "hidden_size": hf_config.dim,
        "num_hidden_layers": hf_config.n_layers,
        "num_attention_heads": hf_config.n_heads,
        "intermediate_size": hf_config.hidden_dim,
        "max_position_embeddings": hf_config.max_position_embeddings,
        "type_vocab_size": 0,
        "layer_norm_eps": 1e-12,
        "use_token_type": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-id", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="float32"
    )
    args = parser.parse_args()
    dest = convert(args.hf_id, args.out_dir, args.dtype)
    print(f"converted {args.hf_id} -> {dest}")


if __name__ == "__main__":
    main()
