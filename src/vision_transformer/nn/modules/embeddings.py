from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from vision_transformer.nn.modules.config import ViTConfig


class Embeddings(torch.nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.patch_embeddings = PatchEmbeddings(config)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, config.num_patches + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_p)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = input_tensor.shape
        embeddings = self.patch_embeddings(input_tensor)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_token, embeddings], dim=1)

        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class PatchEmbeddings(torch.nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()

        self.im_size = (config.im_size, config.im_size)
        self.p_size = (config.patch_size, config.patch_size)

        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        _, num_channels, height, width = input_tensor.shape

        if num_channels != self.num_channels:
            raise ValueError(
                "input tensor don't match with the expected channel dimension"
            )

        if height != self.im_size[0] or width != self.im_size[1]:
            raise ValueError("Input image spatial dimension doesn't match model")

        embedding = self.projection(input_tensor)  # (B, C, W, H)
        return torch.flatten(embedding, start_dim=2).transpose(1, 2)  # (B, W*H, C)
