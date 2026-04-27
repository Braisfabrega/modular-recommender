from __future__ import annotations

import os
import pickle
from logging import Logger
from typing import Any, Optional


def dataset_cache_path(project_root: str, dataset_key: str) -> str:
    return os.path.join(project_root, "dataset", f".{dataset_key}_cache.pkl")


def load_cached_dataset(cache_path: str, logger: Logger) -> Optional[Any]:
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "rb") as cache_file:
            dataset = pickle.load(cache_file)
        logger.info("Cache carregada: %s", cache_path)
        return dataset
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("No s'ha pogut carregar la cache %s (%s).", cache_path, exc)
        return None


def save_dataset_cache(dataset: Any, cache_path: str, logger: Logger) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as cache_file:
        pickle.dump(dataset, cache_file)
    logger.info("Cache desada: %s", cache_path)
