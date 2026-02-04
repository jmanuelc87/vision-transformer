from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_transformer.nn.base import BaseModel
from vision_transformer.nn.modules.attention import MultiHeadAttention
from vision_transformer.nn.modules.embeddings import Embeddings

from vision_transformer.nn.modules.config import ViTConfig


log = logging.getLogger(__name__)


class ViTLayer(nn.Module):

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(config)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.attention(input_tensor)
        return x


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [ViTLayer(config) for _ in range(config.num_attention_layers)]
        )


class ViT(BaseModel):

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.embeddings = Embeddings(config)
        self.encoder = ViTEncoder(config)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_tensor)
        x = self.encoder(x)

        return x

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        for key in list(state_dict.keys()):
            if "classifier" not in key:
                new_key = key.lstrip("vit.")
                state_dict[new_key] = state_dict.pop(key)
        for key in list(state_dict.keys()):
            if "attention.attention" in key:
                new_key = key.replace("attention.attention", "attention")
                state_dict[new_key] = state_dict.pop(key)
        for key in list(state_dict.keys()):
            if "dense" in key:
                new_key = key.replace("dense.", "")
                state_dict[new_key] = state_dict.pop(key)
