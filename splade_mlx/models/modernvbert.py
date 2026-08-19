"""V-SPLADE (visual document retrieval) in MLX.

Port of NAVER's V-SPLADE (arXiv:2605.30917, `naver/v-splade-{efficient,quality}`):
a ModernVBERT backbone (SigLIP vision tower -> pixel-shuffle connector ->
ModernBERT text encoder) with a SPLADE MLM head on the document side, and an
inference-free Li-LSR static lookup on the query side.

Reference implementations (Apache-2.0):
  - transformers.models.modernvbert / modernbert / siglip
  - the checkpoint's modeling_vsplade.py / modeling_st_vsplade.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# [UNK], [CLS], [SEP], [PAD], [MASK] - zeroed out of the sparse representation
SPECIAL_TOKEN_IDS = (50280, 50281, 50282, 50283, 50284)


@dataclass
class VSpladeConfig:
    # vision (SigLIP)
    vision_hidden_size: int = 768
    vision_layers: int = 12
    vision_heads: int = 12
    vision_intermediate: int = 3072
    image_size: int = 512
    patch_size: int = 16
    vision_layer_norm_eps: float = 1e-6
    # connector
    pixel_shuffle_factor: int = 4
    # text (ModernBERT)
    vocab_size: int = 50368  # base MLM vocabulary
    additional_vocab_size: int = 40
    hidden_size: int = 768
    text_layers: int = 22
    text_heads: int = 12
    text_intermediate: int = 1152
    norm_eps: float = 1e-5
    global_attn_every_n_layers: int = 3
    local_attention: int = 128  # total window; half-window = 64
    rope_theta: float = 160000.0
    image_token_id: int = 50407

    layer_types: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = [
                "full_attention" if i % self.global_attn_every_n_layers == 0 else "sliding_attention"
                for i in range(self.text_layers)
            ]

    @classmethod
    def from_hf(cls, config: dict) -> "VSpladeConfig":
        tc, vc = config["text_config"], config["vision_config"]
        return cls(
            vision_hidden_size=vc["hidden_size"],
            vision_layers=vc["num_hidden_layers"],
            vision_heads=vc["num_attention_heads"],
            vision_intermediate=vc["intermediate_size"],
            image_size=vc["image_size"],
            patch_size=vc["patch_size"],
            vision_layer_norm_eps=vc.get("layer_norm_eps", 1e-6),
            pixel_shuffle_factor=config["pixel_shuffle_factor"],
            vocab_size=tc["vocab_size"] - config["additional_vocab_size"],
            additional_vocab_size=config["additional_vocab_size"],
            hidden_size=tc["hidden_size"],
            text_layers=tc["num_hidden_layers"],
            text_heads=tc["num_attention_heads"],
            text_intermediate=tc["intermediate_size"],
            norm_eps=tc.get("norm_eps", 1e-5),
            global_attn_every_n_layers=tc["global_attn_every_n_layers"],
            local_attention=tc["local_attention"],
            rope_theta=tc.get("global_rope_theta", 160000.0),
            image_token_id=config["image_token_id"],
            layer_types=tc.get("layer_types", []),
        )


# ---------------------------------------------------------------------------
# SigLIP vision tower (pooling head intentionally omitted - unused by V-SPLADE)
# ---------------------------------------------------------------------------


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: VSpladeConfig):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            3, config.vision_hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )
        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches, config.vision_hidden_size)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        # pixel_values: (B, H, W, C) in NHWC
        x = self.patch_embedding(pixel_values)  # (B, H/16, W/16, D)
        B = x.shape[0]
        x = x.reshape(B, -1, x.shape[-1])
        return x + self.position_embedding(mx.arange(x.shape[1])[None, :])


class SiglipAttention(nn.Module):
    def __init__(self, config: VSpladeConfig):
        super().__init__()
        d = config.vision_hidden_size
        self.num_heads = config.vision_heads
        self.head_dim = d // config.vision_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, L, D))


class SiglipMLP(nn.Module):
    def __init__(self, config: VSpladeConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.vision_hidden_size, config.vision_intermediate)
        self.fc2 = nn.Linear(config.vision_intermediate, config.vision_hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))  # gelu_pytorch_tanh


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: VSpladeConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.vision_hidden_size, eps=config.vision_layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.vision_hidden_size, eps=config.vision_layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class SiglipVisionModel(nn.Module):
    def __init__(self, config: VSpladeConfig):
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings(config)
        self.layers = [SiglipEncoderLayer(config) for _ in range(config.vision_layers)]
        self.post_layernorm = nn.LayerNorm(config.vision_hidden_size, eps=config.vision_layer_norm_eps)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        x = self.embeddings(pixel_values)
        for layer in self.layers:
            x = layer(x)
        return self.post_layernorm(x)


# ---------------------------------------------------------------------------
# Connector: pixel shuffle + linear projection (mirrors ModernVBertConnector)
# ---------------------------------------------------------------------------


class VSpladeConnector(nn.Module):
    def __init__(self, config: VSpladeConfig):
        super().__init__()
        self.factor = config.pixel_shuffle_factor
        self.proj = nn.Linear(
            config.vision_hidden_size * config.pixel_shuffle_factor**2,
            config.hidden_size,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        f = self.factor
        h = w = int(L**0.5)
        x = x.reshape(B, h, w, D)
        x = x.reshape(B, h, w // f, D * f)
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(B, w // f, h // f, D * f * f)
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(B, L // (f * f), D * f * f)
        return self.proj(x)


# ---------------------------------------------------------------------------
# ModernBERT text encoder with decoupled (base + additional) embeddings
# ---------------------------------------------------------------------------


class DecoupledEmbedding(nn.Module):
    """Base MLM vocabulary + separately-stored additional (vision chat) tokens."""

    def __init__(self, config: VSpladeConfig):
        super().__init__()
        self.num_base = config.vocab_size
        self.tok = nn.Embedding(config.vocab_size, config.hidden_size)
        self.additional_embedding = nn.Embedding(config.additional_vocab_size, config.hidden_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        is_extra = input_ids >= self.num_base
        base = self.tok(mx.where(is_extra, 0, input_ids))
        extra = self.additional_embedding(mx.where(is_extra, input_ids - self.num_base, 0))
        return mx.where(is_extra[..., None], extra, base)


class ModernBertAttention(nn.Module):
    def __init__(self, config: VSpladeConfig, layer_idx: int):
        super().__init__()
        d = config.hidden_size
        self.num_heads = config.text_heads
        self.head_dim = d // config.text_heads
        self.scale = self.head_dim**-0.5
        self.rope_theta = config.rope_theta
        self.Wqkv = nn.Linear(d, 3 * d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        B, L, D = x.shape
        qkv = self.Wqkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=0)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.Wo(out.transpose(0, 2, 1, 3).reshape(B, L, D))


class ModernBertMLP(nn.Module):
    def __init__(self, config: VSpladeConfig):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, 2 * config.text_intermediate, bias=False)
        self.Wo = nn.Linear(config.text_intermediate, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate_in = self.Wi(x)
        inp, gate = mx.split(gate_in, 2, axis=-1)
        return self.Wo(nn.gelu(inp) * gate)


class ModernBertLayer(nn.Module):
    def __init__(self, config: VSpladeConfig, layer_idx: int):
        super().__init__()
        self.is_first = layer_idx == 0
        if not self.is_first:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        self.attn = ModernBertAttention(config, layer_idx)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        self.mlp = ModernBertMLP(config)
        self.attention_type = config.layer_types[layer_idx]

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        h = x if self.is_first else self.attn_norm(x)
        x = x + self.attn(h, mask)
        return x + self.mlp(self.mlp_norm(x))


class ModernBertTextModel(nn.Module):
    def __init__(self, config: VSpladeConfig):
        super().__init__()
        self.config = config
        self.embeddings_tok = DecoupledEmbedding(config)
        self.embeddings_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        self.layers = [ModernBertLayer(config, i) for i in range(config.text_layers)]
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)

    def _masks(self, attention_mask: mx.array, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        """(global_mask, sliding_mask), both additive (B, 1, L, L)."""
        L = attention_mask.shape[1]
        neg = mx.finfo(dtype).min
        pad = (1.0 - attention_mask[:, None, None, :].astype(dtype)) * neg
        idx = mx.arange(L)
        dist_ok = mx.abs(idx[:, None] - idx[None, :]) <= self.config.local_attention // 2
        sliding = pad + mx.where(dist_ok, 0.0, neg).astype(dtype)[None, None]
        return pad, sliding

    def __call__(self, inputs_embeds: mx.array, attention_mask: mx.array) -> mx.array:
        x = self.embeddings_norm(inputs_embeds)
        global_mask, sliding_mask = self._masks(attention_mask, x.dtype)
        for layer in self.layers:
            mask = global_mask if layer.attention_type == "full_attention" else sliding_mask
            x = layer(x, mask)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# V-SPLADE document encoder
# ---------------------------------------------------------------------------


class VSpladeMLMHead(nn.Module):
    def __init__(self, config: VSpladeConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)  # default eps 1e-5, with bias
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.decoder(self.norm(nn.gelu(self.dense(x))))


class VSpladeModel(nn.Module):
    """Document-side V-SPLADE encoder (vision + text -> vocab-space sparse logits)."""

    def __init__(self, config: VSpladeConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionModel(config)
        self.connector = VSpladeConnector(config)
        self.text_model = ModernBertTextModel(config)
        self.mlm_head = VSpladeMLMHead(config)
        self.logit_scale = config.hidden_size**-0.25
        special = np.ones(config.vocab_size, dtype=np.float32)
        special[list(SPECIAL_TOKEN_IDS)] = 0.0
        self._special_mask = mx.array(special)

    def _image_features(self, pixel_values: np.ndarray) -> tuple[mx.array, np.ndarray]:
        """pixel_values: (B, T, 3, H, W) float np array (torch processor layout).

        Returns (features (real_tiles * image_seq_len, hidden), real-tile mask).
        All-zero padding tiles are dropped, mirroring the reference."""
        B, T = pixel_values.shape[:2]
        flat = pixel_values.reshape(B * T, *pixel_values.shape[2:])
        real = ~(np.abs(flat).sum(axis=(1, 2, 3)) == 0)
        if not real.any():
            real[0] = True
        tiles = flat[real]  # (N, 3, H, W)
        x = mx.array(tiles.transpose(0, 2, 3, 1))  # NHWC
        feats = self.connector(self.vision_model(x))  # (N, 64, hidden)
        return feats.reshape(-1, feats.shape[-1]), real

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        pixel_values: np.ndarray | None = None,
    ) -> mx.array:
        """Returns SPLADE term logits (B, L, vocab): scaled, specials zeroed."""
        embeds = self.text_model.embeddings_tok(input_ids)
        if pixel_values is not None:
            image_flat, _ = self._image_features(pixel_values)
            ids = np.asarray(input_ids)
            pos = np.nonzero(ids.reshape(-1) == self.config.image_token_id)[0]
            if pos.size:
                B, L, D = embeds.shape
                flat = embeds.reshape(B * L, D)
                flat[mx.array(pos)] = image_flat[: pos.size].astype(flat.dtype)
                embeds = flat.reshape(B, L, D)
        x = self.text_model(embeds, attention_mask)
        logits = self.mlm_head(x) * self.logit_scale
        # zero the special tokens so they never activate (log1p(relu(0)) == 0)
        return logits * self._special_mask.astype(logits.dtype)

    def encode(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        pixel_values: np.ndarray | None = None,
    ) -> mx.array:
        """Sparse document vectors (B, vocab)."""
        logits = self(input_ids, attention_mask, pixel_values)
        scores = mx.log1p(mx.maximum(logits, 0.0))
        scores = scores * attention_mask[:, :, None].astype(scores.dtype)
        return scores.max(axis=1)


class VSpladeQueryEncoder:
    """Inference-free Li-LSR query encoder: a static per-token weight lookup.

    weight[token_id] = softplus(projection(embedding))[token_id], specials zeroed.
    Repeated query tokens accumulate (scatter-add); ids outside the base
    vocabulary (added vision tokens) contribute nothing.
    """

    def __init__(self, weight: np.ndarray):
        self.weight = weight.astype(np.float32)  # (vocab,)
        self.num_dimensions = weight.shape[0]

    def encode(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        input_ids = np.asarray(input_ids)
        attention_mask = np.asarray(attention_mask)
        valid = (input_ids < self.num_dimensions) & (attention_mask > 0)
        safe = np.clip(input_ids, 0, self.num_dimensions - 1)
        scores = self.weight[safe] * valid
        out = np.zeros((input_ids.shape[0], self.num_dimensions), dtype=np.float32)
        for b in range(input_ids.shape[0]):
            np.add.at(out[b], safe[b], scores[b])
        return out
