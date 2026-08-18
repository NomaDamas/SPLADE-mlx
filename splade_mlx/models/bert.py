"""BERT/DistilBERT encoder + MLM head + SPLADE pooling in MLX.

Faithful port of HF `BertForMaskedLM` / `DistilBertForMaskedLM` inference
(post-LN, exact GELU), with a SPLADE sparse head:
log(1 + relu(logits)) masked max-pool. DistilBERT (used by the
efficient-splade-V pair) shares the exact block structure minus
token-type embeddings, so both map onto the same module tree.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    use_token_type: bool = True  # False for DistilBERT

    @classmethod
    def from_hf(cls, config: dict) -> "BertConfig":
        return cls(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_position_embeddings"],
            type_vocab_size=config.get("type_vocab_size", 2),
            layer_norm_eps=config.get("layer_norm_eps", 1e-12),
            use_token_type=config.get("use_token_type", True),
        )


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.use_token_type = config.use_token_type
        if config.use_token_type:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size
            )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array | None = None) -> mx.array:
        seq_len = input_ids.shape[1]
        positions = mx.arange(seq_len)[None, :]
        x = self.word_embeddings(input_ids) + self.position_embeddings(positions)
        if self.use_token_type:
            if token_type_ids is None:
                token_type_ids = mx.zeros_like(input_ids)
            x = x + self.token_type_embeddings(token_type_ids)
        return self.layer_norm(x)


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        B, L, D = x.shape
        q = self.query(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        attn = attn.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out(attn)


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp_up = nn.Linear(config.hidden_size, config.intermediate_size)
        self.mlp_down = nn.Linear(config.intermediate_size, config.hidden_size)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        x = self.attention_norm(x + self.attention(x, mask))
        return self.mlp_norm(x + self.mlp_down(nn.gelu(self.mlp_up(x))))


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layers = [BertLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class MLMHead(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layer_norm(nn.gelu(self.dense(x)))
        return self.decoder(x)


class SpladeModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.mlm_head = MLMHead(config)

    @staticmethod
    def _additive_mask(attention_mask: mx.array, dtype: mx.Dtype) -> mx.array:
        # (B, L) 1/0 -> (B, 1, 1, L) additive: 0 where attended, large negative elsewhere
        mask = (1.0 - attention_mask[:, None, None, :].astype(dtype)) * mx.finfo(dtype).min
        return mask.astype(dtype)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        token_type_ids: mx.array | None = None,
    ) -> mx.array:
        """Returns MLM logits (B, L, vocab)."""
        x = self.embeddings(input_ids, token_type_ids)
        mask = None
        if attention_mask is not None:
            mask = self._additive_mask(attention_mask, x.dtype)
        x = self.encoder(x, mask)
        return self.mlm_head(x)

    def encode(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        token_type_ids: mx.array | None = None,
    ) -> mx.array:
        """SPLADE sparse vectors (B, vocab): log1p(relu(logits)) masked max-pool."""
        logits = self(input_ids, attention_mask, token_type_ids)
        scores = mx.log1p(mx.maximum(logits, 0.0))
        scores = scores * attention_mask[:, :, None].astype(scores.dtype)
        return scores.max(axis=1)
