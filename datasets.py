from __future__ import annotations

import csv
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple


class Dataset(ABC):
    """Classe base abstracta per als conjunts de dades de recomanació.

    Attributes
    ----------
    _project_root : str
        Ruta absoluta al directori arrel del projecte.
    _dataset_name : str
        Identificador curt d'aquest conjunt de dades (p. ex. ``"movies"``).
    _items : dict
        Diccionari que mapeja item_id amb un diccionari de camps de metadades.
    _known_users : set
        Conjunt de tots els IDs d'usuari presents en el dataset.
    _user_ratings : dict
        Diccionari que mapeja user_id amb ``{item_id: rating}``.
    _item_ratings : dict
        Diccionari que mapeja item_id amb ``{user_id: rating}``.
    """

    def __init__(self, project_root: str, dataset_name: str) -> None:
        """Inicialitza els atributs base del dataset i el sistema de logging.

        Parameters
        ----------
        project_root : str
            Ruta absoluta al directori arrel del projecte.
        dataset_name : str
            Identificador curt d'aquest conjunt de dades.
        """
        self._project_root = project_root
        self._dataset_name = dataset_name
        self._items: Dict[str, Dict[str, str]] = {}
        self._known_users: Set[str] = set()
        self._user_ratings: Dict[str, Dict[str, float]] = {}
        self._item_ratings: Dict[str, Dict[str, float]] = {}
        self._min_rating: Optional[float] = None
        self._max_rating: Optional[float] = None
        self._logger = logging.getLogger(f"recommender_system.{self.__class__.__name__}")

    @abstractmethod
    def get_cache_key(self) -> str:
        """Retorna una cadena única que identifica aquest dataset i la seva configuració.

        S'utilitza per construir els noms de fitxer de la memòria cau (pickle) de manera
        que les caches de diferents configuracions (p. ex. diferents valors de ``max_books``)
        mai col·lideixin.

        Returns
        -------
        str
            Una clau alfanumèrica curta, p. ex. ``"movies"`` o ``"books_10000"``.
        """
        raise NotImplementedError

    @abstractmethod
    def _load_items(self) -> None:
        """Mètode abstracte per carregar les metadades dels ítems des del disc a les estructures internes.

        Raises
        ------
        NotImplementedError
            Si la subclasse no implementa aquest mètode.
        """
        raise NotImplementedError

    @abstractmethod
    def _load_ratings(self) -> None:
        """Mètode abstracte per carregar les valoracions des del disc a les estructures internes.

        Raises
        ------
        NotImplementedError
            Si la subclasse no implementa aquest mètode.
        """
        raise NotImplementedError

    def _load_users(self) -> None:
        """Ganxo (hook) opcional per carregar usuaris explícitament des d'un fitxer dedicat.

        No fa res per defecte, ja que els usuaris es poden inferir a partir de les valoracions.
        Les subclasses poden redefinir-lo si existeix un fitxer específic d'usuaris.
        """
        return

    @abstractmethod
    def format_item_for_display(self, item_id: str) -> str:
        """Retorna una cadena de text formatejada i llegible per a humans per a un ``item_id``.

        Parameters
        ----------
        item_id : str
            Identificador de l'ítem a formatejar.

        Returns
        -------
        str
            Cadena de text de visualització formatejada.
        """
        raise NotImplementedError

    @abstractmethod
    def get_item_content_text(self, item_id: str) -> str:
        """Retorna el contingut de text utilitzat per construir els vectors TF-IDF d'un ítem.

        Parameters
        ----------
        item_id : str
            Identificador de l'ítem.

        Returns
        -------
        str
            Representació en text lliure de les característiques del contingut de l'ítem.
            Retorna una cadena buida si no hi ha contingut disponible.
        """
        raise NotImplementedError

    def load(self) -> None:
        """Carrega els ítems, usuaris i valoracions des del disc, i calcula els límits de puntuació."""
        self._load_items()
        self._load_users()
        self._load_ratings()
        self._compute_rating_bounds()

    def _compute_rating_bounds(self) -> None:
        """Calcula i emmagatzema els valors màxim i mínim globals de les valoracions carregades.

        Si no hi ha cap valoració registrada, s'estableixen els límits per defecte en un rang de 0.0 a 5.0.
        """
        all_ratings: List[float] = []
        for ratings in self._user_ratings.values():
            all_ratings.extend(ratings.values())

        if not all_ratings:
            self._min_rating = 0.0
            self._max_rating = 5.0
            return

        self._min_rating = min(all_ratings)
        self._max_rating = max(all_ratings)

    def _register_rating(self, user_id: str, item_id: str, rating: float) -> None:
        """Registra una valoració d'un usuari dins dels mapes d'interacció bidireccionals.

        Parameters
        ----------
        user_id : str
            Identificador de l'usuari que fa la valoració.
        item_id : str
            Identificador de l'ítem valorat.
        rating : float
            Valor numèric de la puntuació atorgada.
        """
        self._known_users.add(user_id)

        if user_id not in self._user_ratings:
            self._user_ratings[user_id] = {}
        self._user_ratings[user_id][item_id] = rating

        if item_id not in self._item_ratings:
            self._item_ratings[item_id] = {}
        self._item_ratings[item_id][user_id] = rating

    def get_name(self) -> str:
        """Obté el nom identificador d'aquest conjunt de dades.

        Returns
        -------
        str
            El nom curt assignat a la instància del dataset.
        """
        return self._dataset_name

    def get_project_root(self) -> str:
        """Obté la ruta absoluta al directori arrel del projecte.

        Returns
        -------
        str
            Ruta absoluta del directori arrel.
        """
        return self._project_root

    def get_user_ids(self) -> List[str]:
        """Obté una llista de tots els identificadors únics d'usuari coneguts ordenats naturalment.

        Returns
        -------
        List[str]
            Llista ordenada dels IDs d'usuari.
        """
        return sorted(self._known_users, key=self._sort_user_id)

    def has_user(self, user_id: str) -> bool:
        """Comprova si un identificador d'usuari existeix en el conjunt de dades.

        Parameters
        ----------
        user_id : str
            Identificador de l'usuari objectiu.

        Returns
        -------
        bool
            True si l'usuari és present al dataset, False en cas contrari.
        """
        return user_id in self._known_users

    def get_item_ids(self) -> List[str]:
        """Obté una llista ordenada de tots els identificadors d'ítem carregats.

        Returns
        -------
        List[str]
            Llista ordenada dels IDs d'ítem vàlids.
        """
        return sorted(self._items.keys())

    def get_user_ratings(self, user_id: str) -> Dict[str, float]:
        """Obté l'historial complet de valoracions realitzades per un únic usuari.

        Parameters
        ----------
        user_id : str
            Identificador de l'usuari.

        Returns
        -------
        Dict[str, float]
            Una còpia del diccionari que mapeja IDs d'ítem amb les valoracions de l'usuari.
        """
        return dict(self._user_ratings.get(user_id, {}))

    def get_user_rated_items(self, user_id: str) -> Set[str]:
        """Obté el conjunt de tots els identificadors d'ítem avaluats per un usuari concret.

        Parameters
        ----------
        user_id : str
            Identificador de l'usuari.

        Returns
        -------
        Set[str]
            Conjunt que conté els IDs dels ítems valorats per l'usuari.
        """
        return set(self._user_ratings.get(user_id, {}).keys())

    def get_unrated_items(self, user_id: str) -> List[str]:
        """Identifica tots els ítems del sistema que un usuari encara no ha valorat.

        Parameters
        ----------
        user_id : str
            Identificador de l'usuari objectiu.

        Returns
        -------
        List[str]
            Llista d'IDs d'ítem que no apareixen a l'historial d'interaccions de l'usuari.
        """
        rated_items = self.get_user_rated_items(user_id)
        return [item_id for item_id in self._items if item_id not in rated_items]

    def get_item_metadata(self, item_id: str) -> Dict[str, str]:
        """Recupera els atributs descriptius de metadades d'un ítem específic.

        Parameters
        ----------
        item_id : str
            Identificador de l'ítem.

        Returns
        -------
        Dict[str, str]
            Una còpia del diccionari amb les característiques de l'ítem (títol, autor, gèneres, etc.).
        """
        return dict(self._items.get(item_id, {}))

    def get_item_user_ratings(self, item_id: str) -> Dict[str, float]:
        """Recupera totes les valoracions rebudes per un ítem específic de part dels usuaris.

        Parameters
        ----------
        item_id : str
            Identificador de l'ítem.

        Returns
        -------
        Dict[str, float]
            Una còpia del diccionari que mapeja els IDs d'usuari amb les seves valoracions per a aquest ítem.
        """
        return dict(self._item_ratings.get(item_id, {}))

    def get_item_average(self, item_id: str) -> Optional[float]:
        """Calcula la puntuació mitjana rebuda per un ítem específic.

        Parameters
        ----------
        item_id : str
            Identificador de l'ítem.

        Returns
        -------
        Optional[float]
            La mitjana aritmètica de les valoracions de l'ítem, o None si no té cap valoració.
        """
        ratings = self._item_ratings.get(item_id, {})
        if not ratings:
            return None
        return sum(ratings.values()) / len(ratings)

    def get_user_average(self, user_id: str) -> Optional[float]:
        """Calcula la puntuació mitjana atorgada per un usuari al llarg del seu historial.

        Parameters
        ----------
        user_id : str
            Identificador de l'usuari.

        Returns
        -------
        Optional[float]
            La mitjana aritmètica de les valoracions de l'usuari, o None si no ha valorat res.
        """
        ratings = self._user_ratings.get(user_id, {})
        if not ratings:
            return None
        return sum(ratings.values()) / len(ratings)

    def get_rating(self, user_id: str, item_id: str) -> Optional[float]:
        """Obté la puntuació exacta donada per un usuari a un ítem concret.

        Parameters
        ----------
        user_id : str
            Identificador de l'usuari.
        item_id : str
            Identificador de l'ítem.

        Returns
        -------
        Optional[float]
            El valor numèric de la puntuació si existeix, o None si no hi ha registre d'interació.
        """
        return self._user_ratings.get(user_id, {}).get(item_id)

    def get_rating_bounds(self) -> Tuple[float, float]:
        """Obté els límits de rang de les valoracions possibles en aquest conjunt de dades.

        Returns
        -------
        Tuple[float, float]
            Una tupla que conté (valoracio_minima, valoracio_maxima).
        """
        min_rating = 0.0 if self._min_rating is None else self._min_rating
        max_rating = 5.0 if self._max_rating is None else self._max_rating
        return min_rating, max_rating

    @staticmethod
    def _sort_user_id(user_id: str) -> Tuple[int, str]:
        """Funció clau auxiliar per ordenar naturalment els IDs d'usuari (númerics abans que alfabètics).

        Parameters
        ----------
        user_id : str
            La cadena de l'identificador d'usuari per a la qual es generarà la clau.

        Returns
        -------
        Tuple[int, str]
            Una tupla de comparació on els dígits s'ordenen amb farcit i el text en brut.
        """
        if user_id.isdigit():
            return (0, f"{int(user_id):020d}")
        return (1, user_id)

    @staticmethod
    def _ensure_file(path: str) -> None:
        """Valida si un fitxer destí existeix realment al disc.

        Parameters
        ----------
        path : str
            La ruta del fitxer del sistema a comprovar.

        Raises
        ------
        FileNotFoundError
            Si el fitxer requerit no es troba a la ruta especificada.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No s'ha trobat el fitxer requerit: {path}")


class MovieLensDataset(Dataset):
    """Carregador del conjunt de dades MovieLens 100k.

    Parameters
    ----------
    project_root : str
        Ruta absoluta al directori arrel del projecte.
    """

    def __init__(self, project_root: str) -> None:
        """Inicialitza les rutes d'instància de MovieLens i les seves propietats."""
        super().__init__(project_root, "movies")
        self._dataset_dir = os.path.join(project_root, "dataset", "MovieLens100k")
        self.load()

    def get_cache_key(self) -> str:
        """Retorna una cadena única que identifica aquest dataset i la seva configuració.

        S'utilitza per construir els noms de fitxer de la memòria cau (pickle) de manera
        que les caches de diferents configuracions (p. ex. diferents valors de ``max_books``)
        mai col·lideixin.

        Returns
        -------
        str
            Una clau alfanumèrica curta, p. ex. ``"movies"`` o ``"books_10000"``.
        """
        return "movies"

    def _load_items(self) -> None:
        """Carrega les entrades de metadades de pel·lícules des del fitxer 'movies.csv'.

        Processa l'ID de la pel·lícula, el títol i els gèneres, registrant-los al diccionari intern.
        """
        movies_path = os.path.join(self._dataset_dir, "movies.csv")
        self._ensure_file(movies_path)

        with open(movies_path, "r", encoding="utf-8-sig", newline="") as csv_file:
            csvreader = csv.reader(csv_file)
            _ = next(csvreader, None)
            for row in csvreader:
                if len(row) < 3:
                    continue
                item_id = row[0].strip()
                title = row[1].strip()
                genres = row[2].strip()
                if not item_id:
                    continue
                self._items[item_id] = {
                    "title": title,
                    "genres": genres,
                }
        self._logger.info("CSV de pel·lícules carregat correctament: %d items des de '%s'.", len(self._items), movies_path)

    def _load_ratings(self) -> None:
        """Carrega els registres d'interacció de valoracions des del fitxer 'ratings.csv'.

        Filtra els registres associats a pel·lícules no vàlides o no indexades, i en registra els vincles.
        """
        ratings_path = os.path.join(self._dataset_dir, "ratings.csv")
        self._ensure_file(ratings_path)

        with open(ratings_path, "r", encoding="utf-8-sig", newline="") as csv_file:
            csvreader = csv.reader(csv_file)
            _ = next(csvreader, None)
            for row in csvreader:
                if len(row) < 3:
                    continue
                user_id = row[0].strip()
                item_id = row[1].strip()
                if not user_id or not item_id:
                    continue

                try:
                    rating = float(row[2].strip())
                except ValueError:
                    continue

                if rating <= 0:
                    continue
                if item_id not in self._items:
                    continue

                self._register_rating(user_id, item_id, rating)

        total_ratings = sum(len(v) for v in self._user_ratings.values())
        self._logger.info(
            "CSV de valoracions carregat correctament: %d valoracions, %d usuaris únics des de '%s'.",
            total_ratings, len(self._known_users), ratings_path,
        )

    def format_item_for_display(self, item_id: str) -> str:
        """Retorna una cadena de text formatejada i llegible per a humans per a un ``item_id``.

        Parameters
        ----------
        item_id : str
            Identificador de l'ítem a formatejar.

        Returns
        -------
        str
            Cadena de text de visualització formatejada.
        """
        metadata = self.get_item_metadata(item_id)
        title = metadata.get("title", "Sense titol")
        genres = metadata.get("genres", "Sense generes")
        return f"{title} | {genres}"

    def get_item_content_text(self, item_id: str) -> str:
        """Retorna el contingut de text d'un gènere per a un ``item_id`` substituint els caràcters ``|`` per espais.

        Parameters
        ----------
        item_id : str
            Identificador de la pel·lícula.

        Returns
        -------
        str
            Tokens de gènere separats per espais (p. ex. ``"Action Comedy Drama"``).
        """
        genres = self._items.get(item_id, {}).get("genres", "")
        return genres.replace("|", " ")


class BooksDataset(Dataset):
    """Carregador del conjunt de dades Book-Crossing.

    Parameters
    ----------
    project_root : str
        Ruta absoluta al directori arrel del projecte.
    max_books : int, optional
        Nombre màxim de llibres a carregar de Books.csv. Un valor de ``0`` significa sense límit.
        Per defecte és ``10_000``.
    """

    def __init__(self, project_root: str, max_books: int = 10_000) -> None:
        """Inicialitza les variables d'instència de Books i l'estructura de directoris d'emmagatzematge."""
        super().__init__(project_root, "books")
        self._dataset_dir = os.path.join(project_root, "dataset", "Books")
        self._max_books = max_books
        self.load()

    def get_cache_key(self) -> str:
        """Retorna una cadena única que identifica aquest dataset i la seva configuració.

        S'utilitza per construir els noms de fitxer de la memòria cau (pickle) de manera
        que les caches de diferents configuracions (p. ex. diferents valors de ``max_books``)
        mai col·lideixin.

        Returns
        -------
        str
            Una clau alfanumèrica curta, p. ex. ``"movies"`` o ``"books_10000"``.
        """
        suffix = "all" if self._max_books == 0 else str(self._max_books)
        return f"books_{suffix}"

    def _load_items(self) -> None:
        """Carrega metadades de llibres des de 'Books.csv' analitzant dinàmicament la capçalera.

        Verifica la presència obligatòria de columnes estructurals i controla el llindar màxim de `max_books`.

        Raises
        ------
        ValueError
            Si el fitxer CSV té una capçalera buida o si li falta alguna columna requerida de manera estàndard.
        """
        books_path = os.path.join(self._dataset_dir, "Books.csv")
        self._ensure_file(books_path)

        with open(books_path, "r", encoding="utf-8-sig", newline="") as csv_file:
            csvreader = csv.reader(csv_file)
            header = next(csvreader, None)
            if not header:
                raise ValueError("Books.csv no conte cap capcalera.")

            header_map = {name.strip(): index for index, name in enumerate(header)}
            required_columns = ["ISBN", "Book-Title", "Book-Author"]
            for column in required_columns:
                if column not in header_map:
                    raise ValueError(f"Books.csv no conte la columna requerida: {column}")

            isbn_idx = header_map["ISBN"]
            title_idx = header_map["Book-Title"]
            author_idx = header_map["Book-Author"]

            for row in csvreader:
                if not row:
                    continue

                max_idx = max(isbn_idx, title_idx, author_idx)
                if len(row) <= max_idx:
                    continue

                item_id = row[isbn_idx].strip()
                if not item_id:
                    continue

                self._items[item_id] = {
                    "title": row[title_idx].strip(),
                    "author": row[author_idx].strip(),
                }

                if self._max_books > 0 and len(self._items) >= self._max_books:
                    break

        self._logger.info("CSV de llibres carregat correctament: %d items des de '%s'.", len(self._items), books_path)

    def _load_users(self) -> None:
        """Carrega els IDs d'usuari vàlids de 'Users.csv' cap al conjunt global d'usuaris coneguts."""
        users_path = os.path.join(self._dataset_dir, "Users.csv")
        self._ensure_file(users_path)

        with open(users_path, "r", encoding="utf-8-sig", newline="") as csv_file:
            csvreader = csv.reader(csv_file)
            header = next(csvreader, None)
            if not header:
                return

            header_map = {name.strip(): index for index, name in enumerate(header)}
            if "User-ID" not in header_map:
                return

            user_id_idx = header_map["User-ID"]
            for row in csvreader:
                if len(row) <= user_id_idx:
                    continue
                user_id = row[user_id_idx].strip()
                if user_id:
                    self._known_users.add(user_id)

    def _load_ratings(self) -> None:
        """Carrega els perfils d'interaccions i valoracions des de 'Ratings.csv'.

        Fa una validació creuada comparant les entrades amb els llibres ja carregats per descartar
        valoracions d'ítems inexistents.

        Raises
        ------
        ValueError
            Si el fitxer de valoracions té estructures buides o no conté les capçaleres obligatòries.
        """
        ratings_path = os.path.join(self._dataset_dir, "Ratings.csv")
        self._ensure_file(ratings_path)

        with open(ratings_path, "r", encoding="utf-8-sig", newline="") as csv_file:
            csvreader = csv.reader(csv_file)
            header = next(csvreader, None)
            if not header:
                raise ValueError("Ratings.csv no conte cap capcalera.")

            header_map = {name.strip(): index for index, name in enumerate(header)}
            required_columns = ["User-ID", "ISBN", "Book-Rating"]
            for column in required_columns:
                if column not in header_map:
                    raise ValueError(f"Ratings.csv no conte la columna requerida: {column}")

            user_id_idx = header_map["User-ID"]
            item_id_idx = header_map["ISBN"]
            rating_idx = header_map["Book-Rating"]

            for row in csvreader:
                max_idx = max(user_id_idx, item_id_idx, rating_idx)
                if len(row) <= max_idx:
                    continue

                user_id = row[user_id_idx].strip()
                item_id = row[item_id_idx].strip()
                if not user_id or not item_id:
                    continue

                self._known_users.add(user_id)

                try:
                    rating = float(row[rating_idx].strip())
                except ValueError:
                    continue

                if rating <= 0:
                    continue
                if item_id not in self._items:
                    continue

                self._register_rating(user_id, item_id, rating)

        total_ratings = sum(len(v) for v in self._user_ratings.values())
        self._logger.info(
            "CSV de valoracions de llibres carregat correctament: %d valoracions, %d usuaris únics des de '%s'.",
            total_ratings, len(self._known_users), ratings_path,
        )

    def format_item_for_display(self, item_id: str) -> str:
        """Retorna una cadena de text formatejada i llegible per a humans per a un ``item_id``.

        Parameters
        ----------
        item_id : str
            Identificador de l'ítem a formatejar.

        Returns
        -------
        str
            Cadena de text de visualització formatejada.
        """
        metadata = self.get_item_metadata(item_id)
        title = metadata.get("title", "Sense titol")
        author = metadata.get("author", "Autor desconegut")
        return f"{title} | {author}"

    def get_item_content_text(self, item_id: str) -> str:
        """Retorna el nom de l'autor com a text de contingut per a un ``item_id``.

        Parameters
        ----------
        item_id : str
            Identificador ISBN del llibre.

        Returns
        -------
        str
            Nom de l'autor utilitzat com a característica de contingut per a TF-IDF.
        """
        return self._items.get(item_id, {}).get("author", "")
