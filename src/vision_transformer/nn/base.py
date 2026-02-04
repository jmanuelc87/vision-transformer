import torch
import logging

from typing import Any
from safetensors.torch import load_file
from vision_transformer.nn.utils import download_model_weights


log = logging.getLogger(__file__)


class BaseModel(torch.nn.Module):

    def forward(self, x, *args, **kwargs):

        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, *args, **kwargs) -> Any:
        return self(x)

    def loss(self, batch, preds=None) -> Any:
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)

    def init_criterion(self):
        raise NotImplementedError("Subclasses must implement init_criterion method")

    @classmethod
    def load(cls, repo_id: str, config, cache_dir: str = "./cache"):
        """Load model weights from a repository.

        Args:
            repo_id: Repository identifier
            cache_dir: Directory to cache downloaded weights

        Returns:
            Model instance with loaded weights
        """
        model = cls(config)
        paths = download_model_weights(repo_id, cache_dir=cache_dir)

        for path in paths:
            state_dict = load_file(path)
            model.load_state_dict(state_dict)

        return model
