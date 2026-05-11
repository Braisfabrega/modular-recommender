from __future__ import annotations

import os
import pickle
from logging import Logger
from typing import Any, Optional


def dataset_cache_path(project_root: str, dataset_key: str) -> str:
    """Return the file path for a cached dataset object.

    Parameters
    ----------
    project_root : str
        Absolute path to the project root directory.
    dataset_key : str
        Short dataset identifier (e.g. ``"movies"``).

    Returns
    -------
    str
        Path to the ``.pkl`` cache file inside ``dataset/``.
    """
    return os.path.join(project_root, "dataset", f".{dataset_key}_cache.pkl")


def load_cached_dataset(cache_path: str, logger: Logger) -> Optional[Any]:
    """Load a previously pickled dataset from *cache_path*.

    Parameters
    ----------
    cache_path : str
        Full path to the pickle file.
    logger : logging.Logger
        Logger instance for info / warning messages.

    Returns
    -------
    Any or None
        Deserialised dataset object, or ``None`` when the file does not exist
        or cannot be deserialised.
    """
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
    """Serialise *dataset* to *cache_path* with pickle.

    Parameters
    ----------
    dataset : Any
        Dataset object to serialise.
    cache_path : str
        Destination file path.
    logger : logging.Logger
        Logger instance for info messages.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as cache_file:
        pickle.dump(dataset, cache_file)
    logger.info("Cache desada: %s", cache_path)


def recommender_cache_path(project_root: str, dataset_key: str, method_key: str) -> str:
    """Return the file path for a cached recommender object.

    Parameters
    ----------
    project_root : str
        Absolute path to the project root directory.
    dataset_key : str
        Short dataset identifier (e.g. ``"movies"``).
    method_key : str
        Short method identifier (e.g. ``"content"``).

    Returns
    -------
    str
        Path of the form ``recommender_<dataset>_<method>.dat``.
    """
    return os.path.join(project_root, f"recommender_{dataset_key}_{method_key}.dat")


def load_cached_recommender(cache_path: str, logger: Logger) -> Optional[Any]:
    """Load a previously pickled recommender from *cache_path*.

    Parameters
    ----------
    cache_path : str
        Full path to the ``.dat`` pickle file.
    logger : logging.Logger
        Logger instance for info / warning messages.

    Returns
    -------
    Any or None
        Deserialised recommender object, or ``None`` when the file does not
        exist or cannot be deserialised.
    """
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "rb") as f:
            recommender = pickle.load(f)
        logger.info("Recomanador carregat des de cache: %s", cache_path)
        return recommender
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("No s'ha pogut carregar el recomanador %s (%s).", cache_path, exc)
        return None


def save_recommender_cache(recommender: Any, cache_path: str, logger: Logger) -> None:
    """Serialise *recommender* to *cache_path* with pickle.

    Parameters
    ----------
    recommender : Any
        Trained recommender object to serialise.
    cache_path : str
        Destination file path (typically ``*.dat``).
    logger : logging.Logger
        Logger instance for info messages.
    """
    with open(cache_path, "wb") as f:
        pickle.dump(recommender, f)
    logger.info("Recomanador desat a cache: %s", cache_path)
