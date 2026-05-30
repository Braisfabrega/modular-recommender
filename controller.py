from __future__ import annotations

import logging
from typing import Callable, Dict

from datasets import BooksDataset, Dataset, MovieLensDataset
from recommenders import (
    CollaborativeRecommender,
    ContentBasedRecommender,
    Recommender,
    SimpleRecommender,
)

logger = logging.getLogger("recommender_system.factories")

_DATASET_BUILDERS: Dict[str, Callable[[str], Dataset]] = {
    "movies": lambda root: MovieLensDataset(root),
    "books": lambda root: BooksDataset(root, max_books=10_000),
}

_RECOMMENDER_BUILDERS: Dict[str, Callable[[Dataset], Recommender]] = {
    "simple": lambda ds: SimpleRecommender(ds, min_votes=10),
    "collaborative": lambda ds: CollaborativeRecommender(ds, k=5),
    "content": lambda ds: ContentBasedRecommender(ds),
}


class Controller:
    """Controlador i fàbrica (Factory) per a la creació de datasets i recomanadors.

    Attributes
    ----------
    _logger : logging.Logger
        Instància del logger utilitzada per registrar traces de l'execució.
    _dataset : Dataset or None
        Instància de l'últim conjunt de dades instanciat i carregat.
    _recommender : Recommender or None
        Instància de l'últim sistema de recomanació preparat.
    """

    def __init__(self, logger_instance: logging.Logger | None = None) -> None:
        """Inicialitza el controlador i en configura el sistema de logging de traces.

        Parameters
        ----------
        logger_instance : logging.Logger, opcional
            Instància personalitzada de logging a utilitzar. Si és ``None``,
            s'utilitzarà el logger genèric del mòdul per defecte.
        """
        self._logger = logger_instance or logger
        self._dataset: Dataset | None = None
        self._recommender: Recommender | None = None

    def build_dataset(self, dataset_key: str, project_root: str) -> Dataset:
        """Instancia i retorna el dataset corresponent a *dataset_key*.

        Parameters
        ----------
        dataset_key : str
            Identificador del dataset. Ha de ser ``"movies"`` o ``"books"``.
        project_root : str
            Ruta absoluta al directori arrel del projecte on s'ubiquen els fitxers.

        Returns
        -------
        Dataset
            Instància del conjunt de dades completament carregada i a punt per fer servir.

        Raises
        ------
        ValueError
            Si el valor de *dataset_key* proporcionat no és reconegut pel sistema.
        """
        builder = _DATASET_BUILDERS.get(dataset_key)
        if builder is None:
            self._logger.error(
                "Error: el dataset '%s' no és suportat. Opcions vàlides: %s.",
                dataset_key,
                ", ".join(_DATASET_BUILDERS),
            )
            raise ValueError(f"Dataset no suportat: {dataset_key}")

        dataset = builder(project_root)
        self._dataset = dataset
        return dataset

    def build_recommender(self, method_key: str) -> Recommender:
        """Instancia i retorna el sistema de recomanació per a *method_key*.

        Parameters
        ----------
        method_key : str
            Identificador de la tècnica. Ha de ser ``"simple"``, ``"collaborative"`` o ``"content"``.

        Returns
        -------
        Recommender
            Instància del recomanador completament inicialitzada i preparada.

        Raises
        ------
        ValueError
            Si no s'ha carregat cap dataset prèviament o si el valor de *method_key*
            no és cap de les opcions suportades.
        """
        if self._dataset is None:
            self._logger.error("No hi ha cap dataset carregat. Cal cridar build_dataset primer.")
            raise ValueError("Dataset no carregat")

        builder = _RECOMMENDER_BUILDERS.get(method_key)
        if builder is None:
            self._logger.error(
                "Error: el mètode '%s' no és suportat. Opcions vàlides: %s.",
                method_key,
                ", ".join(_RECOMMENDER_BUILDERS),
            )
            raise ValueError(f"Metode no suportat: {method_key}")

        recommender = builder(self._dataset)
        self._recommender = recommender
        return recommender
