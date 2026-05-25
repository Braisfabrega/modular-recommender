from __future__ import annotations

import csv
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple


class Dataset(ABC):
    """Abstract base class for recommendation datasets.

    Attributes
    ----------
    _project_root : str
        Absolute path to the project root directory.
    _dataset_name : str
        Short identifier for this dataset (e.g. ``"movies"``).
    _items : dict
        Mapping from item_id to a dict of metadata fields.
    _known_users : set
        Set of all user IDs present in the dataset.
    _user_ratings : dict
        Mapping from user_id to ``{item_id: rating}``.
    _item_ratings : dict
        Mapping from item_id to ``{user_id: rating}``.
    """

    def __init__(self, project_root: str, dataset_name: str) -> None:
        self._project_root = project_root
        self._dataset_name = dataset_name
        self._items: Dict[str, Dict[str, str]] = {}
        self._known_users: Set[str] = set()
        self._user_ratings: Dict[str, Dict[str, float]] = {}
        self._item_ratings: Dict[str, Dict[str, float]] = {}
        self._min_rating: Optional[float] = None
        self._max_rating: Optional[float] = None
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def _load_items(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _load_ratings(self) -> None:
        raise NotImplementedError

    def _load_users(self) -> None:
        return

    @abstractmethod
    def format_item_for_display(self, item_id: str) -> str:
        """Return a human-readable string for ``item_id``.

        Parameters
        ----------
        item_id : str
            Identifier of the item to format.

        Returns
        -------
        str
            Formatted display string.
        """
        raise NotImplementedError

    @abstractmethod
    def get_item_content_text(self, item_id: str) -> str:
        """Return the text content used to build TF-IDF vectors for an item.

        Parameters
        ----------
        item_id : str
            Identifier of the item.

        Returns
        -------
        str
            Free-text representation of the item's content features.
            Returns an empty string when no content is available.
        """
        raise NotImplementedError

    def load(self) -> None:
        """Load items, users, and ratings from disk, then compute rating bounds."""
        self._load_items()
        self._load_users()
        self._load_ratings()
        self._compute_rating_bounds()

    def _compute_rating_bounds(self) -> None:
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
        self._known_users.add(user_id)

        if user_id not in self._user_ratings:
            self._user_ratings[user_id] = {}
        self._user_ratings[user_id][item_id] = rating

        if item_id not in self._item_ratings:
            self._item_ratings[item_id] = {}
        self._item_ratings[item_id][user_id] = rating

    def get_name(self) -> str:
        return self._dataset_name

    def get_project_root(self) -> str:
        return self._project_root

    def get_user_ids(self) -> List[str]:
        return sorted(self._known_users, key=self._sort_user_id)

    def has_user(self, user_id: str) -> bool:
        return user_id in self._known_users

    def get_item_ids(self) -> List[str]:
        return sorted(self._items.keys())

    def get_user_ratings(self, user_id: str) -> Dict[str, float]:
        return dict(self._user_ratings.get(user_id, {}))

    def get_user_rated_items(self, user_id: str) -> Set[str]:
        return set(self._user_ratings.get(user_id, {}).keys())

    def get_unrated_items(self, user_id: str) -> List[str]:
        rated_items = self.get_user_rated_items(user_id)
        return [item_id for item_id in self._items if item_id not in rated_items]

    def get_item_metadata(self, item_id: str) -> Dict[str, str]:
        return dict(self._items.get(item_id, {}))

    def get_item_user_ratings(self, item_id: str) -> Dict[str, float]:
        return dict(self._item_ratings.get(item_id, {}))

    def get_item_average(self, item_id: str) -> Optional[float]:
        ratings = self._item_ratings.get(item_id, {})
        if not ratings:
            return None
        return sum(ratings.values()) / len(ratings)

    def get_user_average(self, user_id: str) -> Optional[float]:
        ratings = self._user_ratings.get(user_id, {})
        if not ratings:
            return None
        return sum(ratings.values()) / len(ratings)

    def get_rating(self, user_id: str, item_id: str) -> Optional[float]:
        return self._user_ratings.get(user_id, {}).get(item_id)

    def get_rating_bounds(self) -> Tuple[float, float]:
        min_rating = 0.0 if self._min_rating is None else self._min_rating
        max_rating = 5.0 if self._max_rating is None else self._max_rating
        return min_rating, max_rating

    @staticmethod
    def _sort_user_id(user_id: str) -> Tuple[int, str]:
        if user_id.isdigit():
            return (0, f"{int(user_id):020d}")
        return (1, user_id)

    @staticmethod
    def _ensure_file(path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No s'ha trobat el fitxer requerit: {path}")


class MovieLensDataset(Dataset):
    """MovieLens 100k dataset loader.

    Parameters
    ----------
    project_root : str
        Absolute path to the project root directory.
    """

    def __init__(self, project_root: str) -> None:
        super().__init__(project_root, "movies")
        self._dataset_dir = os.path.join(project_root, "dataset", "MovieLens100k")

    def _load_items(self) -> None:
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
        metadata = self.get_item_metadata(item_id)
        title = metadata.get("title", "Sense titol")
        genres = metadata.get("genres", "Sense generes")
        return f"{title} | {genres}"

    def get_item_content_text(self, item_id: str) -> str:
        """Return genre string for ``item_id``, with ``|`` replaced by spaces.

        Parameters
        ----------
        item_id : str
            Movie identifier.

        Returns
        -------
        str
            Space-separated genre tokens (e.g. ``"Action Comedy Drama"``).
        """
        genres = self._items.get(item_id, {}).get("genres", "")
        return genres.replace("|", " ")


class BooksDataset(Dataset):
    """Book-Crossing dataset loader.

    Parameters
    ----------
    project_root : str
        Absolute path to the project root directory.
    max_books : int, optional
        Maximum number of books to load from Books.csv. ``0`` means no limit.
        Defaults to ``10_000``.
    """

    def __init__(self, project_root: str, max_books: int = 10_000) -> None:
        super().__init__(project_root, "books")
        self._dataset_dir = os.path.join(project_root, "dataset", "Books")
        self._max_books = max_books

    def _load_items(self) -> None:
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
        metadata = self.get_item_metadata(item_id)
        title = metadata.get("title", "Sense titol")
        author = metadata.get("author", "Autor desconegut")
        return f"{title} | {author}"

    def get_item_content_text(self, item_id: str) -> str:
        """Return the author name as content text for ``item_id``.

        Parameters
        ----------
        item_id : str
            Book ISBN identifier.

        Returns
        -------
        str
            Author name used as the content feature for TF-IDF.
        """
        return self._items.get(item_id, {}).get("author", "")
