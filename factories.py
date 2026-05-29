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
        """Instancia, carrega des de disc i retorna el dataset corresponent a *dataset_key*.

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
        if dataset_key == "movies":
            dataset = MovieLensDataset(project_root)
        elif dataset_key == "books":
            dataset = BooksDataset(project_root, max_books=10_000)
        else:
            self._logger.error(
                "Error: el dataset '%s' no és suportat. Opcions vàlides: movies, books.",
                dataset_key,
            )
            raise ValueError(f"Dataset no suportat: {dataset_key}")

        dataset.load()
        self._dataset = dataset
        return dataset

    def build_recommender(self, method_key: str) -> Recommender:
        """Instancia, entrena/prepara i retorna el sistema de recomanació per a *method_key*.

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

        dataset = self._dataset
        if method_key == "simple":
            recommender = SimpleRecommender(dataset, min_votes=10)
        elif method_key == "collaborative":
            recommender = CollaborativeRecommender(dataset, k=5)
        elif method_key == "content":
            recommender = ContentBasedRecommender(dataset)
        else:
            self._logger.error(
                "Error: el mètode '%s' no és suportat. Opcions vàlides: simple, collaborative, content.",
                method_key,
            )
            raise ValueError(f"Metode no suportat: {method_key}")

        recommender.prepare()
        self._recommender = recommender
        return recommender
