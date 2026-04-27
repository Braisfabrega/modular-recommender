from __future__ import annotations

import os
import sys
from typing import List, Tuple

from cache_utils import dataset_cache_path, load_cached_dataset, save_dataset_cache
from datasets import Dataset
from factories import build_dataset, build_recommender
from logging_utils import build_logger
from recommenders import Recommender


VALID_DATASETS = {"movies", "books"}
VALID_METHODS = {"simple", "collaborative"}


def print_usage() -> None:
    print("Us: python main.py <dataset> <metode>")
    print("  <dataset>: movies | books")
    print("  <metode>: simple | collaborative")


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
    if not recommendations:
        print("No hi ha recomanacions disponibles per aquest usuari.")
        return

    print("\nTop 5 recomanacions:")
    for idx, (item_id, score) in enumerate(recommendations, start=1):
        item_info = dataset.format_item_for_display(item_id)
        print(f"{idx}. {item_info} (id={item_id}, score={score:.3f})")


def load_dataset(project_root: str, dataset_key: str, logger) -> Dataset:
    cache_path = dataset_cache_path(project_root, dataset_key)
    dataset = load_cached_dataset(cache_path, logger)
    if dataset is not None:
        return dataset

    logger.info("No hi ha cache per '%s'. Carregant CSV...", dataset_key)
    dataset = build_dataset(dataset_key, project_root)
    save_dataset_cache(dataset, cache_path, logger)
    return dataset


def run_interactive_loop(dataset: Dataset, recommender: Recommender, logger) -> int:
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
                print("Avaluacio MAE/RMSE pendent per Fase 2.")
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
    logger = build_logger(os.path.join(project_root, "log.txt"))
    logger.info("Inici aplicacio. dataset=%s, metode=%s", dataset_key, method_key)

    try:
        dataset = load_dataset(project_root, dataset_key, logger)
        recommender = build_recommender(method_key, dataset)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Error inicialitzant el sistema.")
        print(f"Error inicialitzant el sistema: {exc}")
        return 1

    return run_interactive_loop(dataset, recommender, logger)


if __name__ == "__main__":
    sys.exit(main())
