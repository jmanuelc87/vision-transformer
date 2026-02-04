import torch

from typing import Any
from vision_transformer.nn.base import BaseModel
from vision_transformer.models.vit import ViT, ViTConfig


def test_load_base_model():
    class MyModel(BaseModel):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.classifier = torch.nn.Linear(768, 1000)

        def forward(self, x):
            x = self.classifier(x)

            return x

    model = MyModel.load("google/vit-base-patch16-224", None)


def test_load_vit_model():
    config = ViTConfig()
    model = ViT.load("google/vit-base-patch16-224", config)
