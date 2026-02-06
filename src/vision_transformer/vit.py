from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers.modeling_outputs import ImageClassifierOutput
from transformers import PretrainedConfig, PreTrainedModel


class ViTEmbeddings(torch.nn.Module):
    """
    Vision Transformer Embeddings Module.
    This module combines patch embeddings, class token, and position embeddings
    for the Vision Transformer architecture. It processes input images by:
    1. Converting image patches into embeddings
    2. Prepending a learnable class token
    3. Adding position embeddings
    4. Applying dropout regularization
    Attributes:
        cls_token (nn.Parameter): Learnable class token of shape (1, 1, hidden_size).
        patch_embeddings (ViTPatchEmbeddings): Module that converts image patches to embeddings.
        position_embeddings (nn.Parameter): Learnable position embeddings of shape
            (1, num_patches + 1, hidden_size). The +1 accounts for the class token.
        dropout (nn.Dropout): Dropout layer for regularization with dropout probability
            specified by config.hidden_dropout_p.
    Args:
        config (ViTConfig): Configuration class with the following attributes:
            - hidden_size (int): Dimensionality of the embeddings.
            - num_patches (int): Number of image patches.
            - hidden_dropout_p (float): Dropout probability for the dropout layer.
    Returns:
        torch.Tensor: Embedded tensor of shape (batch_size, num_patches + 1, hidden_size)
            containing patch embeddings with prepended class token and position information.
    Example:
        >>> config = ViTConfig(hidden_size=768, num_patches=196, hidden_dropout_p=0.1)
        >>> embeddings = ViTEmbeddings(config)
        >>> input_tensor = torch.randn(2, 3, 224, 224)  # batch_size=2, 3 channels, 224x224 image
        >>> output = embeddings(input_tensor)
        >>> output.shape
        torch.Size([2, 197, 768])
    """

    def __init__(
        self,
        config: ViTConfig,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.patch_embeddings = ViTPatchEmbeddings(config)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, config.num_patches + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_p)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, _ = input_tensor.shape
        embeddings = self.patch_embeddings(input_tensor)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_token, embeddings], dim=1)

        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class ViTPatchEmbeddings(torch.nn.Module):
    """
    ViTPatchEmbeddings is a PyTorch module that converts input images into patch embeddings
    for the Vision Transformer architecture. It utilizes a convolutional layer to project
    image patches into a higher-dimensional space, enabling the model to learn meaningful
    representations from the input images.
    Attributes:
        image_size (tuple): The size of the input images as (height, width).
        p_size (tuple): The size of the patches as (height, width).
        num_channels (int): The number of channels in the input images.
        projection (nn.Conv2d): A convolutional layer that projects the input image patches
            into the hidden size specified in the configuration.
    Args:
        config (ViTConfig): Configuration object containing the following attributes:
            - image_size (int): The height and width of the input images.
            - patch_size (int): The height and width of the patches.
            - num_channels (int): The number of channels in the input images.
            - hidden_size (int): The dimensionality of the output embeddings.
    Methods:
        forward(input_tensor: torch.Tensor) -> torch.Tensor:
            Processes the input tensor through the projection layer and returns the
            flattened patch embeddings. The output shape is (batch_size, num_patches, hidden_size).
    Raises:
        ValueError: If the number of channels in the input tensor does not match the expected
        number of channels or if the spatial dimensions of the input tensor do not match
        the expected image size.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()

        self.image_size = (config.image_size, config.image_size)
        self.p_size = (config.patch_size, config.patch_size)
        self.num_channels = config.num_channels

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

        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError("Input image spatial dimension doesn't match model")

        embedding = self.projection(input_tensor)  # (B, C, H/P, W/P)
        return torch.flatten(embedding, start_dim=2).transpose(
            1, 2
        )  # (B, (H/P)*(W/P), C)


class ViTSelfAttention(nn.Module):
    """
    ViTSelfAttention Module
    A multi-head self-attention module for Vision Transformer (ViT) architecture.
    This module implements scaled dot-product attention with support for PyTorch's
    optimized flash attention when available.
    The module transforms input hidden states into query, key, and value projections,
    splits them across multiple attention heads, computes attention weights, and
    combines the results back into the original embedding dimension.
    Attributes:
        attention_head_size (int): Dimensionality of each attention head, calculated as
            hidden_size / num_attention_heads.
        all_head_size (int): Total size of all attention heads combined, equal to hidden_size.
        dropout_p (float): Dropout probability for attention weights.
        n_heads (int): Number of attention heads.
        scaling (float): Scaling factor for attention scores, calculated as
            (attention_head_size)^-0.5 for numerical stability.
        config (ViTConfig): Configuration object containing model hyperparameters.
        query (nn.Linear): Linear projection layer for query computation.
        key (nn.Linear): Linear projection layer for key computation.
        value (nn.Linear): Linear projection layer for value computation.
        output (nn.Linear): Output linear projection layer to combine attention heads.
        dropout (nn.Dropout): Dropout layer applied to attention weights.
        flash (bool): Flag indicating whether PyTorch's scaled_dot_product_attention
            is available for optimized computation.
    Args:
        config (ViTConfig): Configuration class with the following required attributes:
            - hidden_size (int): Dimensionality of input/output embeddings.
            - num_attention_heads (int): Number of attention heads.
            - attention_dropout_prob (float): Dropout probability for attention weights.
            - hidden_dropout_p (float): Dropout probability for the output projection.
            - qkv_bias (bool): Whether to use bias in query, key, value, and output projections.
    Raises:
        ValueError: If hidden_size is not divisible by num_attention_heads.
    Methods:
        forward(hidden_states: torch.Tensor) -> torch.Tensor:
            Applies multi-head self-attention to the input hidden states.
            Args:
                hidden_states: Input tensor of shape (batch_size, sequence_length, hidden_size).
            Returns:
                Attention output tensor of shape (batch_size, sequence_length, hidden_size).
        scaled_dot_product_attention(q, k, v):
            Computes scaled dot-product attention manually when flash attention is unavailable.
            Args:
                q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim).
                k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim).
                v: Value tensor of shape (batch_size, num_heads, seq_len, head_dim).
            Returns:
                Attention output tensor of shape (batch_size, num_heads, seq_len, head_dim).
        split(tensor: torch.Tensor) -> torch.Tensor:
            Reshapes and rearranges tensor for multi-head attention computation.
            Transforms from (batch, length, hidden_size) to (batch, num_heads, length, head_dim).
            Args:
                tensor: Input tensor of shape (batch_size, sequence_length, hidden_size).
            Returns:
                Reshaped tensor of shape (batch_size, num_heads, sequence_length, head_dim).
        combine(tensor: torch.Tensor) -> torch.Tensor:
            Combines attention heads back into a single embedding dimension.
            Transforms from (batch, num_heads, length, head_dim) to (batch, length, hidden_size).
            Args:
                tensor: Input tensor of shape (batch_size, num_heads, sequence_length, head_dim).
            Returns:
                Combined tensor of shape (batch_size, sequence_length, hidden_size).
    Example:
        >>> config = ViTConfig(
        ...     image_size=224,
        ...     patch_size=16,
        ...     num_channels=3,
        ...     hidden_size=768,
        ...     num_attention_heads=12,
        ...     attention_dropout_prob=0.1,
        ...     hidden_dropout_p=0.1
        ... )
        >>> attention = ViTSelfAttention(config)
        >>> hidden_states = torch.randn(2, 197, 768)  # batch_size=2, seq_len=197
        >>> output = attention(hidden_states)
        >>> output.shape
        torch.Size([2, 197, 768])
    Note:
        - The module automatically uses PyTorch's optimized scaled_dot_product_attention
          (flash attention) if available, falling back to manual implementation otherwise.
        - During training, attention dropout is applied; during evaluation, no dropout is applied.
    """

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

        self.dropout = nn.Dropout(p=config.hidden_dropout_p)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def scaled_dot_product_attention(self, q, k, v):
        # calculate the attention scores
        attn = q @ k.transpose(-2, -1) * (self.scaling)

        attn = F.softmax(attn, dim=-1)

        attn = self.dropout(attn)

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


class ViTConfig(PretrainedConfig):
    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_size: int = 768,
        qkv_bias: bool = True,
        hidden_dropout_p: float = 0.0,
        attention_dropout_prob: float = 0.0,
        num_attention_heads: int = 12,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        dtype: str | torch.dtype | None = None,
        chunk_size_feed_forward: int = 0,
        is_encoder_decoder: bool = False,
        architectures: list[str] | None = None,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        num_labels: int | None = None,
        problem_type: str | None = None,
        **kwargs,
    ):
        super().__init__(
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            dtype=dtype,
            chunk_size_feed_forward=chunk_size_feed_forward,
            is_encoder_decoder=is_encoder_decoder,
            architectures=architectures,
            id2label=id2label,
            label2id=label2id,
            num_labels=num_labels,
            problem_type=problem_type,
            **kwargs,
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.qkv_bias = qkv_bias
        self.hidden_dropout_p = hidden_dropout_p
        self.attention_dropout_prob = attention_dropout_prob
        self.num_attention_heads = num_attention_heads

        self.num_patches = (self.image_size // self.patch_size) ** 2


class ViT(nn.Module):

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.embeddings = ViTEmbeddings(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixel_values)

        return x


class ViTModel(PreTrainedModel):

    config_class = ViTConfig

    all_tied_weights_keys = {}

    def __init__(self, config: ViTConfig):
        super().__init__(config)
        self.vit = ViT(config)

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None):
        logits = self.vit(pixel_values)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassifierOutput(loss=loss, logits=logits)  # type: ignore
