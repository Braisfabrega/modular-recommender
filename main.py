from __future__ import annotations

import os
import sys
from typing import List, Tuple

from cache_utils import (
    dataset_cache_path,
    load_cached_dataset,
    load_cached_recommender,
    recommender_cache_path,
    save_dataset_cache,
    save_recommender_cache,
)
from datasets import Dataset
from evaluation import evaluate_user
from factories import build_dataset, build_recommender
from logging_utils import build_logger
from recommenders import Recommender


VALID_DATASETS = {"movies", "books"}
VALID_METHODS = {"simple", "collaborative", "content"}


def print_usage() -> None:
    print("Us: python main.py <dataset> <metode>")
    print("  <dataset>: movies | books")
    print("  <metode>: simple | collaborative | content")


def read_action() -> str:
    print("\nAccions disponibles:")
    print("  1) Recomanar")
    print("  2) Avaluar")
    print("  3) Sortir")

    action = input("Escull una opcio (1/2/3): ").strip().lower()
    if action in {"1", "recomanar", "recommend"}:
        return "recommend"
    if action in {"2", "avaluar", "evaluate"}:
        return "evaluate"
    if action in {"3", "sortir", "exit"}:
        return "exit"
    return "invalid"


def show_recommendations(dataset: Dataset, recommendations: List[Tuple[str, float]]) -> None:
    """Print the top-N recommendations for a user.

    Parameters
    ----------
    dataset : Dataset
        Dataset used to format item display strings.
    recommendations : list of (str, float)
        Ordered list of ``(item_id, score)`` pairs.
    """
    if not recommendations:
        print("No hi ha recomanacions disponibles per aquest usuari.")
        return

    print("\nTop 5 recomanacions:")
    for idx, (item_id, score) in enumerate(recommendations, start=1):
        item_info = dataset.format_item_for_display(item_id)
        print(f"{idx}. {item_info} (id={item_id}, score={score:.3f})")


def show_evaluation(dataset: Dataset, recommender: Recommender, user_id: str) -> None:
    """Compute and display MAE / RMSE for a user.

    Parameters
    ----------
    dataset : Dataset
        Dataset containing the ground-truth ratings.
    recommender : Recommender
        Trained recommender to evaluate.
    user_id : str
        Target user identifier.
    """
    print("\nCalculant metriques d'avaluacio (pot tardar uns segons)...")
    mae_score, rmse_score = evaluate_user(recommender, dataset, user_id)

    if mae_score is None or rmse_score is None:
        print("No s'han pogut calcular les metriques per aquest usuari.")
        return

    print(f"\nResultats per l'usuari {user_id}:")
    print(f"  MAE  = {mae_score:.4f}")
    print(f"  RMSE = {rmse_score:.4f}")


def load_dataset(project_root: str, dataset_key: str, logger) -> Dataset:
    """Load a dataset from cache or from CSV files.

    Parameters
    ----------
    project_root : str
        Absolute path to the project root directory.
    dataset_key : str
        Short dataset identifier (e.g. ``"movies"``).
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    Dataset
        Fully loaded dataset instance.
    """
    cache_path = dataset_cache_path(project_root, dataset_key)
    dataset = load_cached_dataset(cache_path, logger)
    if dataset is not None:
        return dataset

    logger.info("No hi ha cache per '%s'. Carregant CSV...", dataset_key)
    dataset = build_dataset(dataset_key, project_root)
    save_dataset_cache(dataset, cache_path, logger)
    return dataset


def load_recommender(
    project_root: str,
    dataset_key: str,
    method_key: str,
    dataset: Dataset,
    logger,
) -> Recommender:
    """Load a prepared recommender from a ``.dat`` file, or build and save one.

    Parameters
    ----------
    project_root : str
        Absolute path to the project root directory.
    dataset_key : str
        Short dataset identifier (e.g. ``"movies"``).
    method_key : str
        Short method identifier (e.g. ``"content"``).
    dataset : Dataset
        Loaded dataset to bind when building from scratch.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    Recommender
        Prepared recommender instance.
    """
    cache_path = recommender_cache_path(project_root, dataset_key, method_key)
    recommender = load_cached_recommender(cache_path, logger)
    if recommender is not None:
        return recommender

    logger.info(
        "No hi ha cache de recomanador per '%s/%s'. Inicialitzant...",
        dataset_key,
        method_key,
    )
    recommender = build_recommender(method_key, dataset)
    save_recommender_cache(recommender, cache_path, logger)
    return recommender


def run_interactive_loop(dataset: Dataset, recommender: Recommender, logger) -> int:
    """Run the main interactive loop.

    Parameters
    ----------
    dataset : Dataset
        Loaded dataset.
    recommender : Recommender
        Prepared recommender.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    int
        Exit code (``0`` for normal exit, ``1`` for errors).
    """
    while True:
        user_id = input("\nIntrodueix l'ID d'usuari (buit per sortir): ").strip()
        if user_id == "":
            logger.info("Sortida de l'aplicacio per entrada buida de user_id.")
            print("Fins aviat.")
            return 0

        if not dataset.has_user(user_id):
            logger.warning("Usuari no trobat: %s", user_id)
            print("Aquest usuari no existeix al dataset seleccionat.")
            continue

        while True:
            action = read_action()
            if action == "invalid":
                print("Opcio no valida. Torna-ho a provar.")
                continue

            if action == "exit":
                logger.info("Sortida des del menu d'accions.")
                print("Fins aviat.")
                return 0

            if action == "evaluate":
                logger.info("Accio Avaluar per usuari %s.", user_id)
                show_evaluation(dataset, recommender, user_id)
                break

            if action == "recommend":
                logger.info("Accio Recomanar per usuari %s.", user_id)
                recommendations = recommender.recommend(user_id=user_id, top_n=5)
                show_recommendations(dataset, recommendations)
                break


def main() -> int:
    if len(sys.argv) != 3:
        print_usage()
        return 1

    dataset_key = sys.argv[1].strip().lower()
    method_key = sys.argv[2].strip().lower()

    if dataset_key not in VALID_DATASETS or method_key not in VALID_METHODS:
        print_usage()
        return 1

    project_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = project_root
    logger = build_logger(log_dir)
    logger.info("Inici aplicacio. dataset=%s, metode=%s", dataset_key, method_key)

    try:
        dataset = load_dataset(project_root, dataset_key, logger)
        recommender = load_recommender(project_root, dataset_key, method_key, dataset, logger)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Error inicialitzant el sistema.")
        print(f"Error inicialitzant el sistema: {exc}")
        return 1

    return run_interactive_loop(dataset, recommender, logger)


if __name__ == "__main__":
    sys.exit(main())
