import torch
import torchvision
import logging

from vision_transformer.vit import (
    ViTPatchEmbeddings,
    ViTEmbeddings,
    ViTSelfAttention,
    ViTSelfOutput,
    ViTAttention,
    ViTIntermediate,
    ViTOutput,
    ViTLayer,
    ViT,
    ViTConfig,
    ViTForImageClassification,
)

im_size = 224
patch_size = 16

config = ViTConfig(
    im_size=im_size,
    patch_size=patch_size,
)

log = logging.getLogger(__file__)


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


def test_vit_self_output():
    layer = ViTSelfOutput(config)

    inputs = torch.rand(size=(1, 197, 768))
    output = layer(inputs)

    assert output.size(0) == 1 and output.size(1) == 197 and output.size(2) == 768


def test_vit_attention():
    layer = ViTAttention(config)

    inputs = torch.rand(size=(1, 197, 768))
    output = layer(inputs)

    assert output.size(0) == 1 and output.size(1) == 197 and output.size(2) == 768


def test_vit_intermediate():
    layer = ViTIntermediate(config)

    inputs = torch.rand(size=(1, 197, 768))
    output = layer(inputs)

    assert output.size(0) == 1 and output.size(1) == 197 and output.size(2) == 3072


def test_vit_output():
    layer = ViTOutput(config)

    hidden_states = torch.rand(size=(1, 197, 3072))
    input_tensor = torch.rand(size=(1, 197, 768))

    output = layer(hidden_states, input_tensor)

    assert output.size(0) == 1 and output.size(1) == 197 and output.size(2) == 768


def test_vit_layer():
    layer = ViTLayer(config)

    inputs = torch.rand(size=(1, 197, 768))
    output = layer(inputs)

    assert output.size(0) == 1 and output.size(1) == 197 and output.size(2) == 768


def test_vit():
    layer = ViT(config)

    inputs = torch.rand(size=(1, 3, 224, 224))
    output = layer(inputs)

    assert output.size(0) == 1 and output.size(1) == 197 and output.size(2) == 768


def test_vit_for_image_classification():
    layer = ViTForImageClassification(config)
    inputs = torch.rand(size=(1, 3, 224, 224))
    output = layer(inputs)
    assert output.logits.size(0) == 1 and output.logits.size(1) == config.num_labels


def test_vit_for_image_classification_with_labels():
    layer = ViTForImageClassification(config)
    inputs = torch.rand(size=(1, 3, 224, 224))
    labels = torch.tensor([0])
    output = layer(inputs, labels=labels)
    assert output.loss is not None
    assert output.logits.size(0) == 1 and output.logits.size(1) == config.num_labels


def test_vit_for_image_classification_batch():
    layer = ViTForImageClassification(config)
    batch_size = 4
    inputs = torch.rand(size=(batch_size, 3, 224, 224))
    output = layer(inputs)
    assert (
        output.logits.size(0) == batch_size
        and output.logits.size(1) == config.num_labels
    )


def test_vit_for_image_classification_batch_with_labels():
    layer = ViTForImageClassification(config)
    batch_size = 4
    inputs = torch.rand(size=(batch_size, 3, 224, 224))
    labels = torch.randint(0, config.num_labels, (batch_size,))
    output = layer(inputs, labels=labels)
    assert output.loss is not None
    assert (
        output.logits.size(0) == batch_size
        and output.logits.size(1) == config.num_labels
    )


def test_inference_vit():
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    inputs = torch.rand(size=(1, 3, 224, 224))
    output = model(inputs)

    assert output is not None

    log.info(output.logits.shape)


def test_inference_vit_with_image():

    test_image = torchvision.io.read_image("tests/tench.jpg")
    test_image = torchvision.transforms.Resize((224, 224))(test_image)
    test_image = test_image / 255.0
    test_image = torch.unsqueeze(test_image, dim=0)

    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.eval()

    with torch.no_grad():
        output = model(test_image)

    assert output.logits is not None
    assert output.logits.shape[0] == 1
    assert output.logits.shape[1] == model.config.num_labels

    # Get predicted class
    predicted_class = output.logits.argmax(dim=-1).item()

    assert predicted_class == 0

    log.info("Predicted Class: %s", predicted_class)

    assert isinstance(predicted_class, int)
    assert 0 <= predicted_class < model.config.num_labels
