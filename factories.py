from __future__ import annotations

from datasets import BooksDataset, Dataset, MovieLensDataset
from recommenders import CollaborativeRecommender, Recommender, SimpleRecommender


def build_dataset(dataset_key: str, project_root: str) -> Dataset:
    if dataset_key == "movies":
        dataset = MovieLensDataset(project_root)
    elif dataset_key == "books":
        dataset = BooksDataset(project_root)
    else:
        raise ValueError(f"Dataset no suportat: {dataset_key}")

    dataset.load()
    return dataset


def build_recommender(method_key: str, dataset: Dataset) -> Recommender:
    if method_key == "simple":
        recommender = SimpleRecommender(dataset, min_votes=10)
    elif method_key == "collaborative":
        recommender = CollaborativeRecommender(dataset, k=5)
    else:
        raise ValueError(f"Metode no suportat: {method_key}")

    recommender.prepare()
    return recommender
