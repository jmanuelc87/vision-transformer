import pytest
import torch

from vision_transformer.vit import (
    ViTPatchEmbeddings,
    ViTEmbeddings,
    ViTSelfAttention,
    ViTConfig,
    ViTModel,
)

im_size = 224
patch_size = 16

config = ViTConfig(
    im_size=im_size,
    patch_size=patch_size,
    num_channels=3,
    hidden_size=768,
)


def test_vit_patch_embeddings():
    layer = ViTPatchEmbeddings(config)

    inputs = torch.rand(size=(1, 3, 224, 224))
    output = layer(inputs)

    assert (
        output.size(0) == 1
        and output.size(1) == (im_size // patch_size) * (im_size // patch_size)
        and output.size(2) == 768
    )


def test_vit_embeddings():
    layer = ViTEmbeddings(config)

    inputs = torch.rand(size=(1, 3, 224, 224))
    output = layer(inputs)

    assert (
        output.size(0) == 1
        and output.size(1) == (im_size // patch_size) ** 2 + 1
        and output.size(2) == 768
    )


def test_vit_self_attention():
    layer = ViTSelfAttention(config)

    inputs = torch.rand(size=(1, 197, 768))
    output = layer(inputs)

    assert output.size(0) == 1 and output.size(1) == 197 and output.size(2) == 768


def test_load_vit():
    model = ViTModel.from_pretrained("google/vit-base-patch16-224")

    assert model is not None
