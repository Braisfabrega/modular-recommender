from __future__ import annotations

import os
import sys
from typing import List, Tuple
from datasets import Dataset
from evaluation import evaluate_user, plot_evaluation
from factories import build_dataset, build_recommender
from logging_utils import build_logger
from recommenders import Recommender


VALID_DATASETS = {"movies", "books"}
VALID_METHODS = {"simple", "collaborative", "content"}


ALL_METHODS = ["simple", "collaborative", "content"]
METHOD_LABELS = {
    "simple": "Simple",
    "collaborative": "Colaboratiu",
    "content": "Contingut",
}


def print_usage() -> None:
    print("Us: python main.py <dataset> <metode>")
    print("  <dataset>: movies | books")
    print("  <metode>: simple | collaborative | content")


def read_action() -> str:
    print("\nAccions disponibles:")
    print("  1) Recomanar")
    print("  2) Avaluar")
    print("  3) Comparar tots els metodes")
    print("  4) Sortir")

    action = input("Escull una opcio (1/2/3/4): ").strip().lower()
    if action in {"1", "recomanar", "recommend"}:
        return "recommend"
    if action in {"2", "avaluar", "evaluate"}:
        return "evaluate"
    if action in {"3", "comparar", "compare"}:
        return "compare"
    if action in {"4", "sortir", "exit"}:
        return "exit"
    return "invalid"


def show_recommendations(
    dataset: Dataset,
    recommendations: List[Tuple[str, float]],
    N: int = 5,
) -> None:
    """Print the top-N recommendations for a user.

    Parameters
    ----------
    dataset : Dataset
        Dataset used to format item display strings.
    recommendations : list of (str, float)
        Ordered list of ``(item_id, score)`` pairs.
    N : int, optional
        Maximum number of recommendations to display.  Defaults to ``5``.
    """
    if not recommendations:
        print("No hi ha recomanacions disponibles per aquest usuari.")
        return

    recommendations = recommendations[:N]
    print(f"\nTop {N} recomanacions:")
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


def show_comparison(dataset: Dataset, user_id: str, logger) -> None:
    """Build all three recommenders and compare their MAE/RMSE for *user_id*.

    For each method (Simple, Collaborative, Content) the recommender is
    instantiated via :func:`~factories.build_recommender` (cached builds are
    reused), evaluated with :func:`~evaluation.evaluate_user`, and the results
    are printed as a table and displayed as a bar chart.

    Parameters
    ----------
    dataset : Dataset
        Loaded dataset containing ground-truth ratings.
    user_id : str
        Target user identifier.
    logger : logging.Logger
        Logger instance.
    """
    print("\nComparant tots els metodes (pot tardar uns minuts)...")

    mae_dict: dict = {}
    rmse_dict: dict = {}

    for method_key in ALL_METHODS:
        label = METHOD_LABELS[method_key]
        print(f"  [{label}] Preparant...", end=" ", flush=True)
        try:
            rec = build_recommender(method_key, dataset)
            mae_score, rmse_score = evaluate_user(rec, dataset, user_id)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Error avaluant %s: %s", method_key, exc)
            print(f"error: {exc}")
            continue

        if mae_score is None or rmse_score is None:
            print("no hi ha prou dades per avaluar.")
            continue

        mae_dict[label] = mae_score
        rmse_dict[label] = rmse_score
        print(f"MAE={mae_score:.4f}  RMSE={rmse_score:.4f}")

    if not mae_dict:
        print("No s'han pogut calcular metriques per cap metode.")
        return

    col_w = 15
    print(f"\nResultats de comparacio per l'usuari {user_id}:")
    print(f"  {'Metode':<{col_w}} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-' * col_w} {'-' * 8} {'-' * 8}")
    for label in mae_dict:
        print(f"  {label:<{col_w}} {mae_dict[label]:>8.4f} {rmse_dict[label]:>8.4f}")

    plot_evaluation(mae_dict, rmse_dict)


def load_dataset(project_root: str, dataset_key: str, logger) -> Dataset:
    """Load a dataset from CSV files.

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
    logger.info("Carregant dataset '%s' des de CSV...", dataset_key)
    dataset = build_dataset(dataset_key, project_root)
    return dataset


def load_recommender(method_key: str, dataset: Dataset, logger) -> Recommender:
    """Build (or restore from cache) a recommender for the given method and dataset.

    Parameters
    ----------
    method_key : str
        Short method identifier (e.g. ``"content"``).
    dataset : Dataset
        Loaded dataset to bind to the recommender.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    Recommender
        Prepared recommender instance.
    """
    logger.info("Inicialitzant recomanador '%s'...", method_key)
    recommender = build_recommender(method_key, dataset)
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
            logger.error("Error: l'ID d'usuari introduït no existeix al dataset: %s", user_id)
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

            if action == "compare":
                logger.info("Accio Comparar tots els metodes per usuari %s.", user_id)
                show_comparison(dataset, user_id, logger)
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
    log_dir = project_root+"/logs"
    logger = build_logger(log_dir)
    logger.info("Inici aplicacio. dataset=%s, metode=%s", dataset_key, method_key)

    try:
        dataset = load_dataset(project_root, dataset_key, logger)
        recommender = load_recommender(method_key, dataset, logger)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Error inicialitzant el sistema.")
        print(f"Error inicialitzant el sistema: {exc}")
        return 1

    return run_interactive_loop(dataset, recommender, logger)


if __name__ == "__main__":
    sys.exit(main())
