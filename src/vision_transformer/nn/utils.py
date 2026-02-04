import os
import logging
import pathlib
import huggingface_hub as hub


log = logging.getLogger(__file__)


def download_model_weights(repo_id: str, cache_dir: str):
    files = hub.list_repo_tree(repo_id=repo_id)
    to_download = [file.path for file in files if file.path.endswith("safetensors")]
    paths = []
    for file in to_download:
        path = hub.hf_hub_download(repo_id, filename=file, cache_dir=cache_dir)
        paths.append(path)

    return paths


