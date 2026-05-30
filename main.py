from __future__ import annotations

import os
import sys
from typing import List, Tuple
from datasets import Dataset
from evaluation import evaluate_user
from controller import Controller
from logging_utils import build_logger
from recommenders import Recommender


#: Conjunt de claus de dataset acceptades com a primer argument de la CLI.
VALID_DATASETS = {"movies", "books"}
#: Conjunt de claus de mètode acceptades com a segon argument de la CLI.
VALID_METHODS = {"simple", "collaborative", "content"}

#: Ordre d'iteració dels mètodes en la comparació múltiple (show_comparison).
ALL_METHODS = ["simple", "collaborative", "content"]
#: Etiquetes llegibles per a cada mètode, usades a la taula comparativa.
METHOD_LABELS = {
    "simple": "Simple",
    "collaborative": "Colaboratiu",
    "content": "Contingut",
}


def print_usage() -> None:
    """Mostra per consola les instruccions del format d'ús correcte de l'script."""
    print("Us: python main.py <dataset> <metode>")
    print("  <dataset>: movies | books")
    print("  <metode>: simple | collaborative | content")


def read_action() -> str:
    """Mostra el menú d'opcions interactiu i en llegeix la decisió de l'usuari.

    Returns
    -------
    str
        Cadena normalitzada de l'acció a realitzar (``"recommend"``, ``"evaluate"``, 
        ``"compare"``, ``"exit"`` o ``"invalid"``).
    """
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
    """Mostra per consola de manera formatejada el Top-N ítems recomanats per a un usuari.

    Parameters
    ----------
    dataset : Dataset
        Conjunt de dades utilitzat per traduir i formatejar les metadades de l'ítem.
    recommendations : list of tuple of (str, float)
        Llista ordenada de parelles on cada element conté ``(item_id, puntuacio_score)``.
    N : int, opcional
        Nombre màxim de recomanacions a llistar per pantalla. Per defecte és ``5``.
    """
    if not recommendations:
        print("No hi ha recomanacions disponibles per aquest usuari.")
        return

    recommendations = recommendations[:N]
    print(f"\nTop {N} recomanacions:")
    for idx, (item_id, score) in enumerate(recommendations, start=1):
        item_info = dataset.format_item_for_display(item_id)
        print(f"{idx}. {item_info} (id={item_id}, score={score:.3f})")


def show_evaluation(dataset: Dataset, recommender: Recommender, user_id: str, top_n: int = 5) -> None:
    """Mostra les top-N prediccions, les valoracions reals i les mètriques MAE/RMSE per a un usuari.

    Parameters
    ----------
    dataset : Dataset
        Conjunt de dades que conté les valoracions reals (ground-truth).
    recommender : Recommender
        Instància del recomanador ja entrenat al qual es demanen les prediccions.
    user_id : str
        Identificador de l'usuari que es vol avaluar.
    top_n : int, opcional
        Nombre màxim de prediccions a mostrar. Per defecte és ``5``.
    """
    print("\nCalculant metriques d'avaluacio (pot tardar uns segons)...")

    recommendations = recommender.recommend(user_id=user_id, top_n=top_n)
    print(f"\nTop {top_n} prediccions per l'usuari {user_id}:")
    if recommendations:
        for idx, (item_id, score) in enumerate(recommendations, start=1):
            item_info = dataset.format_item_for_display(item_id)
            print(f"  {idx}. {item_info} (id={item_id}, prediccio={score:.4f})")
    else:
        print("  No hi ha prediccions disponibles.")

    actual_ratings = dataset.get_user_ratings(user_id)
    print(f"\nValoracions de l'usuari {user_id}:")
    if actual_ratings:
        for item_id, rating in actual_ratings.items():
            item_info = dataset.format_item_for_display(item_id)
            print(f"  {item_info} (id={item_id}, valoracio={rating:.1f})")
    else:
        print("  No hi ha valoracions disponibles.")

    mae_score, rmse_score = evaluate_user(recommender, dataset, user_id)
    if mae_score is None or rmse_score is None:
        print("\nNo s'han pogut calcular les metriques per aquest usuari.")
        return

    print(f"\nMesures de comparacio per l'usuari {user_id}:")
    print(f"  MAE  = {mae_score:.4f}")
    print(f"  RMSE = {rmse_score:.4f}")


def show_comparison(dataset: Dataset, user_id: str, logger, controller: Controller) -> None:
    """Construeix els tres mètodes de recomanació i en compara els errors MAE/RMSE per consola.

    Cada tècnica (Simple, Col·laboratiu, Basat en contingut) s'instancia mitjançant el controlador
    (reutilitzant construccions memoritzades de memòria cau si s'escau), s'avalua i es tabulen els
    resultats comparatius per pantalla.

    Parameters
    ----------
    dataset : Dataset
        Conjunt de dades carregat amb el contingut de les interaccions reals.
    user_id : str
        Identificador de l'usuari objectiu per a la comparació multimetòdica.
    logger : logging.Logger
        Instància del logger configurat pel seguiment de l'execució.
    controller : Controller
        Instància de la fàbrica de control encarregada d'instanciar els models.
    """
    print("\nComparant tots els metodes (pot tardar uns minuts)...")

    mae_dict: dict = {}
    rmse_dict: dict = {}

    for method_key in ALL_METHODS:
        label = METHOD_LABELS[method_key]
        print(f"  [{label}] Preparant...", end=" ", flush=True)
        try:
            rec = controller.build_recommender(method_key)
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



def load_dataset(project_root: str, dataset_key: str, logger, controller: Controller) -> Dataset:
    """Carrega completament les metadades i valoracions d'un dataset mitjançant el controlador.

    Parameters
    ----------
    project_root : str
        Ruta absoluta al directori arrel on s'ubiquen els binaris i CSVs de dades.
    dataset_key : str
        Clau identificadora alfanumèrica curta (p. ex. ``"movies"`` o ``"books"``).
    logger : logging.Logger
        Instància del logger de traces.
    controller : Controller
        Fàbrica de control encarregada de delegar i carregar la instància.

    Returns
    -------
    Dataset
        Instància del dataset completament inicialitzada i processada en memòria.
    """
    logger.info("Carregant dataset '%s' des de CSV...", dataset_key)
    dataset = controller.build_dataset(dataset_key, project_root)
    return dataset


def load_recommender(method_key: str, logger, controller: Controller) -> Recommender:
    """Inicialitza i entrena el motor de recomanació associat al mètode sol·licitat.

    Parameters
    ----------
    method_key : str
        Identificador de la tècnica algorísmica (p. ex. ``"content"``, ``"collaborative"``).
    logger : logging.Logger
        Instància del logger de traces.
    controller : Controller
        Instància d'un controlador actiu que conté el dataset prèviament vinculat.

    Returns
    -------
    Recommender
        Instància del motor de recomanació ja preparada per predir o recomanar.
    """
    logger.info("Inicialitzant recomanador '%s'...", method_key)
    recommender = controller.build_recommender(method_key)
    return recommender


def run_interactive_loop(
    dataset: Dataset,
    recommender: Recommender,
    logger,
    controller: Controller,
) -> int:
    """Executa el bucle d'interacció principal de la interfície de text per consola.

    Sol·licita contínuament IDs d'usuari i accions associades a realitzar sobre el
    recomanador escollit fins que rep una instrucció expressa o línia buida de sortida.

    Parameters
    ----------
    dataset : Dataset
        Conjunt de dades objectiu amb el qual opera l'aplicació.
    recommender : Recommender
        Motor de recomanació triat durant el llançament inicial pel terminal.
    logger : logging.Logger
        Instància activa del logger de traces de l'aplicació.
    controller : Controller
        Instància del controlador de dades per a l'orquestració de consultes dinàmiques.

    Returns
    -------
    int
        Codi d'estat de sortida del procés de l'aplicació (``0`` per a una sortida normal).
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
                show_comparison(dataset, user_id, logger, controller)
                break


def main() -> int:
    """Punt d'entrada principal (main entrypoint) de l'aplicació en línia de comandes.

    S'encarrega d'analitzar els paràmetres de la línia de comandes, instanciar el logger,
    carregar les dades inicials i llançar el bucle interactiu de la consola.

    Returns
    -------
    int
        Codi de retorn de sortida del sistema (``0`` si és correcte, ``1`` si conté errors).
    """
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
        controller = Controller()
        dataset = load_dataset(project_root, dataset_key, logger, controller)
        recommender = load_recommender(method_key, logger, controller)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Error inicialitzant el sistema.")
        print(f"Error inicialitzant el sistema: {exc}")
        return 1

    return run_interactive_loop(dataset, recommender, logger, controller)


if __name__ == "__main__":
    sys.exit(main())
