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
            specified by config.hidden_dropout_prob.
    Args:
        config (ViTConfig): Configuration class with the following attributes:
            - hidden_size (int): Dimensionality of the embeddings.
            - num_patches (int): Number of image patches.
            - hidden_dropout_prob (float): Dropout probability for the dropout layer.
    Returns:
        torch.Tensor: Embedded tensor of shape (batch_size, num_patches + 1, hidden_size)
            containing patch embeddings with prepended class token and position information.
    Example:
        >>> config = ViTConfig(hidden_size=768, num_patches=196, hidden_dropout_prob=0.1)
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
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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
            - hidden_dropout_prob (float): Dropout probability for the output projection.
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
        ...     hidden_dropout_prob=0.1
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

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

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


class ViTSelfOutput(nn.Module):
    """
    ViTSelfOutput Module
    A post-processing layer for Vision Transformer self-attention output. This module applies
    a dense linear transformation followed by dropout regularization to the attention output.
    This layer is typically used after the ViTSelfAttention module to project the attention
    output and apply regularization during training.
    Attributes:
        dense (nn.Linear): Linear projection layer that maps from hidden_size to hidden_size,
            preserving the embedding dimension.
        dropout (nn.Dropout): Dropout layer for regularization with dropout probability
            specified by config.hidden_dropout_prob.
    Args:
        config (ViTConfig): Configuration class with the following attributes:
            - hidden_size (int): Dimensionality of input and output embeddings.
            - hidden_dropout_prob (float): Dropout probability for the dropout layer.
    Methods:
        forward(hidden_states: torch.Tensor) -> torch.Tensor:
            Applies linear transformation and dropout to the input hidden states.
            Args:
                hidden_states: Input tensor of shape (batch_size, sequence_length, hidden_size).
            Returns:
                Output tensor of shape (batch_size, sequence_length, hidden_size).
    Example:
        >>> config = ViTConfig(hidden_size=768, hidden_dropout_prob=0.1)
        >>> self_output = ViTSelfOutput(config)
        >>> hidden_states = torch.randn(2, 197, 768)  # batch_size=2, seq_len=197
        >>> output = self_output(hidden_states)
        >>> output.shape
        torch.Size([2, 197, 768])
    Note:
        - During training, dropout is applied; during evaluation, no dropout is applied.
        - This module is typically used as part of the ViTAttention block.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ViTAttention(nn.Module):
    """
    ViTAttention Module
    A composite attention module for Vision Transformer (ViT) architecture that combines
    self-attention and output projection layers. This module applies multi-head self-attention
    to the input hidden states, followed by a dense transformation and dropout regularization.
    The ViTAttention module is a key building block in the ViT encoder stack, typically used
    within ViTLayer blocks to enable the model to attend to different parts of the input
    sequence.
    Attributes:
        attention (ViTSelfAttention): Multi-head self-attention layer that computes attention
            over input hidden states.
        output (ViTSelfOutput): Output projection layer that applies a dense transformation
            and dropout to the attention output.
    Args:
        config (ViTConfig): Configuration class with the following attributes:
            - hidden_size (int): Dimensionality of input/output embeddings.
            - num_attention_heads (int): Number of attention heads.
            - attention_dropout_prob (float): Dropout probability for attention weights.
            - hidden_dropout_prob (float): Dropout probability for the output projection.
            - qkv_bias (bool): Whether to use bias in query, key, value, and output projections.
    Methods:
        forward(hidden_states: torch.Tensor) -> torch.Tensor:
            Applies self-attention followed by output projection to the input hidden states.
            Args:
                hidden_states: Input tensor of shape (batch_size, sequence_length, hidden_size).
            Returns:
                Attention output tensor of shape (batch_size, sequence_length, hidden_size).
    Example:
        >>> config = ViTConfig(
        ...     hidden_size=768,
        ...     num_attention_heads=12,
        ...     attention_dropout_prob=0.1,
        ...     hidden_dropout_prob=0.1
        ... )
        >>> attention_module = ViTAttention(config)
        >>> hidden_states = torch.randn(2, 197, 768)  # batch_size=2, seq_len=197
        >>> output = attention_module(hidden_states)
        >>> output.shape
        torch.Size([2, 197, 768])
    Note:
        - This module is typically used as part of the ViTLayer block in the encoder.
        - During training, dropout is applied; during evaluation, no dropout is applied.
        - The module automatically uses PyTorch's optimized scaled_dot_product_attention
          (flash attention) in ViTSelfAttention if available.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self_attn_output = self.attention(hidden_states)
        output = self.output(self_attn_output)
        return output


class ViTIntermediate(nn.Module):
    """
    ViTIntermediate Module
    An intermediate feed-forward layer for Vision Transformer (ViT) architecture. This module
    applies a dense linear transformation followed by a GELU activation function to expand
    the hidden dimensionality, which is a key component of the feed-forward network in
    transformer blocks.
    The ViTIntermediate module expands the hidden states from hidden_size to intermediate_size,
    allowing the model to learn more complex non-linear transformations. This is typically
    followed by a ViTOutput module that projects back to the original hidden_size dimension.
    Attributes:
        dense (nn.Linear): Linear projection layer that expands from hidden_size to
            intermediate_size, enabling increased representational capacity.
        intermediate_act_fn (nn.GELU): GELU (Gaussian Error Linear Unit) activation function
            that introduces non-linearity after the dense transformation.
    Args:
        config (ViTConfig): Configuration class with the following attributes:
            - hidden_size (int): Dimensionality of input embeddings.
            - intermediate_size (int): Dimensionality of the intermediate representation,
              typically 4x the hidden_size in standard transformer architectures.
    Methods:
        forward(hidden_states: torch.Tensor) -> torch.Tensor:
            Applies the intermediate feed-forward transformation to the input hidden states.
            Args:
                hidden_states: Input tensor of shape (batch_size, sequence_length, hidden_size).
            Returns:
                Expanded tensor of shape (batch_size, sequence_length, intermediate_size).
    Example:
        >>> config = ViTConfig(hidden_size=768, intermediate_size=3072)
        >>> intermediate = ViTIntermediate(config)
        >>> hidden_states = torch.randn(2, 197, 768)  # batch_size=2, seq_len=197
        >>> output = intermediate(hidden_states)
        >>> output.shape
        torch.Size([2, 197, 3072])
    Note:
        - This module is typically used as part of the ViTLayer block between self-attention
          and output projection layers.
        - The intermediate_size is usually set to 4 times the hidden_size, following the
          original Transformer architecture design.
        - GELU activation provides smooth non-linearity compared to ReLU.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ViTOutput(nn.Module):
    """
    ViTOutput Module
    A feed-forward output projection layer for Vision Transformer (ViT) architecture. This module
    applies a dense linear transformation followed by dropout regularization and residual connection
    to project the intermediate feed-forward representations back to the original hidden dimensionality.
    The ViTOutput module is typically used after the ViTIntermediate module to complete the feed-forward
    network component of transformer blocks. It implements a residual connection by adding the output
    to the original input tensor, which aids in gradient flow and model training stability.
    Attributes:
        dense (nn.Linear): Linear projection layer that maps from intermediate_size back to
            hidden_size, reducing the dimensionality after the intermediate expansion.
        dropout (nn.Dropout): Dropout layer for regularization with dropout probability
            specified by config.hidden_dropout_prob.
    Args:
        config (ViTConfig): Configuration class with the following attributes:
            - intermediate_size (int): Dimensionality of the intermediate representation,
              typically 4x the hidden_size.
            - hidden_size (int): Dimensionality of the output embeddings.
            - hidden_dropout_prob (float): Dropout probability for the dropout layer.
    Methods:
        forward(hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
            Applies the output projection and residual connection to the intermediate hidden states.
            Args:
                hidden_states: Intermediate tensor of shape (batch_size, sequence_length, intermediate_size).
                input_tensor: Original hidden states tensor of shape (batch_size, sequence_length, hidden_size)
                    used for the residual connection.
            Returns:
                Output tensor of shape (batch_size, sequence_length, hidden_size).
    Example:
        >>> config = ViTConfig(intermediate_size=3072, hidden_size=768, hidden_dropout_prob=0.1)
        >>> output_layer = ViTOutput(config)
        >>> hidden_states = torch.randn(2, 197, 3072)  # batch_size=2, seq_len=197
        >>> input_tensor = torch.randn(2, 197, 768)
        >>> output = output_layer(hidden_states, input_tensor)
        >>> output.shape
        torch.Size([2, 197, 768])
    Note:
        - This module is typically used as part of the ViTLayer block following ViTIntermediate.
        - The residual connection (hidden_states + input_tensor) helps with gradient flow and
          model training stability.
        - During training, dropout is applied; during evaluation, no dropout is applied.
        - The input_tensor parameter should have the same hidden_size dimension as the output
          for the residual connection to be valid.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class ViTLayer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_norm = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states_norm)

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        return layer_output


class ViTEncoder(nn.Module):
    """
    ViTEncoder Module
    The encoder component of the Vision Transformer (ViT) architecture that stacks multiple
    transformer layers to progressively refine token representations. This module applies a series
    of ViTLayer blocks sequentially to the input hidden states, enabling the model to capture
    multi-level hierarchical features through self-attention and feed-forward transformations.
    The ViTEncoder is a core building block of the Vision Transformer, responsible for transforming
    the embedded patch representations (with class token and position embeddings) into contextualized
    token representations that capture global dependencies across the entire image.
    Attributes:
        config (ViTConfig): Configuration object containing model hyperparameters and architectural
            settings used to construct the encoder layers.
        layer (nn.ModuleList): A list of ViTLayer modules, each implementing a complete transformer
            block with self-attention, layer normalization, and feed-forward networks.
            The number of layers is determined by config.num_hidden_layers.
        gradient_checkpointing (bool): Flag to enable/disable gradient checkpointing for memory
            efficiency during training. When enabled, intermediate activations are not stored
            during the forward pass and are recomputed during backpropagation.
    Args:
        config (ViTConfig): Configuration class with the following attributes:
            - num_hidden_layers (int): Number of transformer layers in the encoder.
            - hidden_size (int): Dimensionality of hidden states.
            - intermediate_size (int): Dimensionality of intermediate representations in feed-forward.
            - num_attention_heads (int): Number of attention heads in self-attention layers.
            - attention_dropout_prob (float): Dropout probability for attention weights.
            - hidden_dropout_prob (float): Dropout probability for hidden states.
            - layer_norm_eps (float): Epsilon for layer normalization numerical stability.
            - qkv_bias (bool): Whether to use bias in query, key, value projections.
            - chunk_size_feed_forward (int): Chunk size for feed-forward layer computation.
    Methods:
        forward(hidden_states: torch.Tensor) -> torch.Tensor:
            Sequentially applies all transformer layers to the input hidden states.
            Args:
                hidden_states: Input tensor of shape (batch_size, sequence_length, hidden_size)
                    containing embedded patch representations with class token.
            Returns:
                Output tensor of shape (batch_size, sequence_length, hidden_size) containing
                contextualized token representations after processing through all encoder layers.
    Example:
        >>> config = ViTConfig(
        ...     hidden_size=768,
        ...     num_hidden_layers=12,
        ...     num_attention_heads=12,
        ...     intermediate_size=3072,
        ...     attention_dropout_prob=0.1,
        ...     hidden_dropout_prob=0.1
        ... )
        >>> encoder = ViTEncoder(config)
        >>> hidden_states = torch.randn(2, 197, 768)  # batch_size=2, seq_len=197 (196 patches + 1 class token)
        >>> output = encoder(hidden_states)
        >>> output.shape
        torch.Size([2, 197, 768])
    Note:
        - The encoder processes tokens sequentially through each layer, with each layer adding
          attention and feed-forward transformations.
        - Gradient checkpointing can be enabled by setting gradient_checkpointing=True to reduce
          memory consumption during training at the cost of slower backpropagation.
        - This module is typically used after ViTEmbeddings and before the final layer normalization
          in the complete Vision Transformer architecture.
        - The module preserves the sequence length throughout the forward pass, enabling
          token-level output analysis.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [ViTLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)

        return hidden_states


class ViT(nn.Module):

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embedding_output)

        sequence_output = self.layernorm(encoder_outputs)

        return sequence_output


class ViTConfig(PretrainedConfig):
    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        qkv_bias: bool = True,
        layer_norm_eps: float = 1e-12,
        num_hidden_layers: int = 12,
        hidden_dropout_prob: float = 0.0,
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
        self.intermediate_size = intermediate_size
        self.qkv_bias = qkv_bias
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.num_attention_heads = num_attention_heads

        self.num_patches = (self.image_size // self.patch_size) ** 2


class ViTForImageClassification(PreTrainedModel):
    """
    ViTForImageClassification Module
    A Vision Transformer (ViT) model for image classification tasks. This module combines the core
    Vision Transformer architecture with a classification head to perform end-to-end image classification.
    The ViTForImageClassification model processes input images through the ViT encoder to generate
    contextualized token representations, extracts the class token representation, and passes it through
    a linear classifier to produce class logits. It supports both inference and training with optional
    loss computation.
    The model follows the HuggingFace PreTrainedModel interface, enabling seamless integration with
    the Transformers library ecosystem, including model saving/loading, configuration management, and
    standardized training workflows.
    Attributes:
        config_class (type): Configuration class for this model, set to ViTConfig.
        all_tied_weights_keys (dict): Dictionary of tied weight keys (empty for this model).
        vit (ViT): The core Vision Transformer encoder that processes image patches and generates
            contextualized token representations.
        classifier (nn.Linear): Linear classification head that projects the class token representation
            from hidden_size to num_labels dimensions for multi-class or multi-label classification.
        loss_fn (nn.BCEWithLogitsLoss): Binary Cross Entropy loss function with logits for computing
            classification loss during training. Suitable for multi-label classification tasks.
    Args:
        config (ViTConfig): Model configuration class with the following attributes:
            - hidden_size (int): Dimensionality of hidden states throughout the model.
            - num_labels (int): Number of classification labels/classes.
            - image_size (int): Height and width of input images.
            - patch_size (int): Height and width of image patches.
            - num_channels (int): Number of input image channels.
            - num_hidden_layers (int): Number of transformer layers in the encoder.
            - num_attention_heads (int): Number of attention heads in self-attention layers.
            - intermediate_size (int): Dimensionality of intermediate representations in feed-forward.
            - attention_dropout_prob (float): Dropout probability for attention weights.
            - hidden_dropout_prob (float): Dropout probability for hidden states.
            - layer_norm_eps (float): Epsilon for layer normalization numerical stability.
            - qkv_bias (bool): Whether to use bias in query, key, value projections.
    Methods:
        forward(pixel_values: torch.Tensor, labels: torch.Tensor | None = None) -> ImageClassifierOutput:
            Processes input images through the model and optionally computes classification loss.
            Args:
                pixel_values: Input image tensor of shape (batch_size, num_channels, image_height, image_width).
                    Expected to be normalized RGB images with shape (batch_size, 3, 224, 224) by default.
                labels: Optional target labels tensor of shape (batch_size, num_labels) for multi-label
                    classification or (batch_size,) for single-label classification. If provided, loss is computed.
            Returns:
                ImageClassifierOutput: Output object containing:
                    - loss: Classification loss (only computed if labels are provided).
                    - logits: Classification logits of shape (batch_size, num_labels).
    Example:
        >>> from vision_transformer.vit import ViTConfig, ViTForImageClassification
        >>> import torch
        >>>
        >>> config = ViTConfig(
        ...     image_size=224,
        ...     patch_size=16,
        ...     num_channels=3,
        ...     hidden_size=768,
        ...     num_labels=10,
        ...     num_hidden_layers=12,
        ...     num_attention_heads=12,
        ...     intermediate_size=3072
        ... )
        >>> model = ViTForImageClassification(config)
        >>>
        >>> # Inference
        >>> pixel_values = torch.randn(4, 3, 224, 224)  # batch_size=4
        >>> outputs = model(pixel_values)
        >>> logits = outputs.logits  # shape: (4, 10)
        >>>
        >>> # Training with loss computation
        >>> labels = torch.randint(0, 2, (4, 10))  # multi-label targets
        >>> outputs = model(pixel_values, labels=labels)
        >>> loss = outputs.loss
    Note:
        - The model uses BCEWithLogitsLoss, which is suitable for multi-label classification. For
          single-label multi-class classification, consider using CrossEntropyLoss instead.
        - The class token (first token in the sequence) is extracted and used for classification,
          following the standard ViT design pattern.
        - Input images should be pre-processed and normalized before being passed to the model.
        - The model can be easily extended by replacing the loss function or classifier head for
          different classification scenarios.
        - This model is compatible with HuggingFace training utilities and model hubs.
    """

    config_class = ViTConfig

    all_tied_weights_keys = {}

    def __init__(self, config: ViTConfig):
        super().__init__(config)
        self.vit = ViT(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_criterion()

    def init_criterion(self):
        """Initialize loss function based on the number of labels."""
        if self.config.num_labels > 1:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None):
        sequence_output = self.vit(pixel_values)
        pooled_output = sequence_output[:, 0, :]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None and self.loss_fn is not None:
            loss = self.loss_fn(logits, labels)

        return ImageClassifierOutput(loss=loss, logits=logits)
