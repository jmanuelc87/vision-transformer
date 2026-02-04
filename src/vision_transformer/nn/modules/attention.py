from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_transformer.nn.modules.config import ViTConfig

from typing import Any, Mapping


log = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "Hidden size is not a multiple of the number of attention heads"
            )

        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.num_attention_heads * self.attention_head_size
        self.dropout_p = config.attention_dropout_prob
        self.n_heads = config.num_attention_heads
        self.scaling = self.attention_head_size**-0.5
        self.config = config

        self.query = nn.Linear(
            config.hidden_size,
            self.all_head_size,
            bias=config.qkv_bias,
        )

        self.key = nn.Linear(
            config.hidden_size,
            self.all_head_size,
            bias=config.qkv_bias,
        )

        self.value = nn.Linear(
            config.hidden_size,
            self.all_head_size,
            bias=config.qkv_bias,
        )

        self.output = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.qkv_bias,
        )

        self.dropout_a = nn.Dropout(p=config.hidden_dropout_p)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def scaled_dot_product_attention(self, q, k, v):
        # calculate the attention scores
        attn = q @ k.transpose(-2, -1) * (self.scaling)

        attn = F.softmax(attn, dim=-1)

        attn = self.dropout_a(attn)

        out = attn @ v

        return out

    def forward(self, hidden_states: torch.Tensor):
        b, t, e = hidden_states.size()

        k, q, v = (
            self.key(hidden_states),
            self.query(hidden_states),
            self.value(hidden_states),
        )

        # reshape
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)

        if self.flash:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.config.attention_dropout_prob if self.training else 0.0,
                is_causal=False,
            )
        else:
            x = self.scaled_dot_product_attention(q, k, v)

        x = self.combine(x)

        x = self.output(x)

        return x

    def split(self, tensor: torch.Tensor):
        # (batch, length, d_model) --> (batch, length, num_heads, d_k) --> (batch, num_heads, length, d_k)
        batch, length, _ = tensor.shape
        return tensor.view(
            batch, length, self.n_heads, self.attention_head_size
        ).transpose(1, 2)

    def combine(self, tensor: torch.Tensor):
        # (batch, num_heads, length, d_k) --> (batch, length, num_heads, d_k) --> (batch, length, dim_model)
        batch, num_heads, length, d_k = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch, length, num_heads * d_k)
