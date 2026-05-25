from __future__ import annotations

import logging

from datasets import BooksDataset, Dataset, MovieLensDataset
from recommenders import (
    CollaborativeRecommender,
    ContentBasedRecommender,
    Recommender,
    SimpleRecommender,
)

logger = logging.getLogger("recommender_system.factories")


def build_dataset(dataset_key: str, project_root: str) -> Dataset:
    """Instantiate, load, and return the dataset for *dataset_key*.

    Parameters
    ----------
    dataset_key : str
        One of ``"movies"`` or ``"books"``.
    project_root : str
        Absolute path to the project root directory.

    Returns
    -------
    Dataset
        Fully loaded dataset instance.

    Raises
    ------
    ValueError
        When *dataset_key* is not recognised.
    """
    if dataset_key == "movies":
        dataset = MovieLensDataset(project_root)
    elif dataset_key == "books":
        dataset = BooksDataset(project_root, max_books=10_000)
    else:
        logger.error("Error: el dataset '%s' no és suportat. Opcions vàlides: movies, books.", dataset_key)
        raise ValueError(f"Dataset no suportat: {dataset_key}")

    dataset.load()
    return dataset


def build_recommender(method_key: str, dataset: Dataset) -> Recommender:
    """Instantiate, prepare, and return the recommender for *method_key*.

    Parameters
    ----------
    method_key : str
        One of ``"simple"``, ``"collaborative"``, or ``"content"``.
    dataset : Dataset
        Loaded dataset to bind to the recommender.

    Returns
    -------
    Recommender
        Prepared recommender instance.

    Raises
    ------
    ValueError
        When *method_key* is not recognised.
    """
    if method_key == "simple":
        recommender = SimpleRecommender(dataset, min_votes=10)
    elif method_key == "collaborative":
        recommender = CollaborativeRecommender(dataset, k=5)
    elif method_key == "content":
        recommender = ContentBasedRecommender(dataset)
    else:
        logger.error("Error: el mètode '%s' no és suportat. Opcions vàlides: simple, collaborative, content.", method_key)
        raise ValueError(f"Metode no suportat: {method_key}")

    recommender.prepare()
    return recommender
