from __future__ import annotations

import logging

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable

from safetensors.torch import load_file

from huggingface_hub import hf_hub_download


log = logging.getLogger(__file__)


class VisionTransformerBase(nn.Module):

    config_class: Callable

    def __init__(self, config):
        super().__init__()
        self.config = config

    @classmethod
    def _from_config(cls, repo_id: str, config: str, cache_dir: str = "./.cache"):

        config_file = hf_hub_download(
            repo_id=repo_id,
            filename=config,
            cache_dir=cache_dir,
        )

        try:
            with open(config_file, "r") as f:
                if cls.config_class is not None:
                    config_dict = json.load(f)

                    return cls.config_class(**config_dict)
        except FileNotFoundError as e:
            raise ValueError(f"Error: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error: {e}")

        raise ValueError("Error: config_class is None")

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: str = "model.safetensors",
        config: str = "config.json",
        weights: dict | None = None,
        cache_dir: str = "./.cache",
    ):
        """Load pretrained weights from a repo_id or dict of weights."""

        config_instance = cls._from_config(repo_id, config)

        model = cls(config_instance)  # type: ignore

        if weights is not None:
            # Use provided weights dict
            state_dict = weights
        elif isinstance(repo_id, str):
            # Load from huggingface hub
            weights_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
            )
            state_dict = load_file(weights_file, device="cpu")
        else:
            raise ValueError("Can't load the weights")

        model.load_state_dict(state_dict, strict=False)
        return model
