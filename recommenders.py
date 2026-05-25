from __future__ import annotations

import logging
import math
import os
import pickle
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from datasets import Dataset


Recommendation = Tuple[str, float]


def _load_pickle_cache(cache_path: str, logger: logging.Logger) -> Optional[Any]:
    if not os.path.exists(cache_path):
        logger.info("Caché MISS — fitxer no trobat: %s", cache_path)
        return None

    try:
        with open(cache_path, "rb") as cache_file:
            payload = pickle.load(cache_file)
        logger.info("Caché HIT — carregat des de: %s", cache_path)
        return payload
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Caché CORRUPTE — error carregant %s (%s). Es recalcularà des de zero.",
            cache_path,
            exc,
        )
        return None


def _save_pickle_cache(cache_path: str, payload: Any, logger: logging.Logger) -> None:
    cache_dir = os.path.dirname(cache_path)
    tmp_path: Optional[str] = None
    try:
        os.makedirs(cache_dir, exist_ok=True)
        # Escriptura atòmica: escrivim a un fitxer temporal i fem rename
        # per evitar fitxers de caché corromputs si el procés s'interromp.
        with tempfile.NamedTemporaryFile("wb", dir=cache_dir, delete=False) as tmp:
            tmp_path = tmp.name
            pickle.dump(payload, tmp)
        os.replace(tmp_path, cache_path)
        logger.info("Caché desada correctament: %s", cache_path)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Error desant caché %s: %s", cache_path, exc)
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class Recommender(ABC):
    """Abstract base class for all recommendation engines.

    Parameters
    ----------
    dataset : Dataset
        The dataset this recommender operates on.

    Attributes
    ----------
    _dataset : Dataset
        Bound dataset instance.
    _is_prepared : bool
        ``True`` after :meth:`prepare` has completed successfully.
    """

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._is_prepared = False
        self._logger = logging.getLogger(f"recommender_system.{self.__class__.__name__}")

    def _cache_path(self, filename: str) -> str:
        cache_dir = os.path.join(self._dataset.get_project_root(), "dataset", "cache")
        return os.path.join(cache_dir, filename)

    @abstractmethod
    def prepare(self) -> None:
        """Pre-compute any internal state needed for recommendations."""
        raise NotImplementedError

    @abstractmethod
    def recommend(self, user_id: str, top_n: int = 5) -> List[Recommendation]:
        """Return the top-N recommended items for *user_id*.

        Parameters
        ----------
        user_id : str
            Target user identifier.
        top_n : int, optional
            Maximum number of recommendations to return.  Defaults to ``5``.

        Returns
        -------
        list of (str, float)
            List of ``(item_id, score)`` pairs sorted by descending score.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_rating(self, user_id: str, item_id: str) -> Optional[float]:
        """Predict the rating *user_id* would give *item_id*.

        Used by the evaluation module to compute MAE / RMSE metrics.

        Parameters
        ----------
        user_id : str
            Target user identifier.
        item_id : str
            Item identifier to predict.

        Returns
        -------
        float or None
            Predicted rating, or ``None`` when the recommender cannot produce
            a meaningful estimate.
        """
        raise NotImplementedError

    def _fallback_by_item_average(self, top_n: int, excluded_items: Set[str]) -> List[Recommendation]:
        candidates: List[Recommendation] = []
        for item_id in self._dataset.get_item_ids():
            if item_id in excluded_items:
                continue
            avg_item = self._dataset.get_item_average(item_id)
            if avg_item is None:
                continue
            candidates.append((item_id, avg_item))

        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates[:top_n]


class SimpleRecommender(Recommender):
    """Bayesian-average popularity recommender.

    Scores each item using a weighted combination of the item's own average
    rating and the global average, controlled by a minimum-votes threshold.

    Parameters
    ----------
    dataset : Dataset
        Dataset to recommend from.
    min_votes : int, optional
        Minimum number of ratings an item must have to be eligible.
        Also used as the regularisation weight.  Defaults to ``10``.

    Attributes
    ----------
    _min_votes : int
        Minimum-vote threshold / regularisation weight.
    _item_scores : dict
        Mapping from item_id to its Bayesian-average score.
    """

    def __init__(self, dataset: Dataset, min_votes: int = 10) -> None:
        super().__init__(dataset)
        self._min_votes = min_votes
        self._item_scores: Dict[str, float] = {}

    def prepare(self) -> None:
        """Compute Bayesian-average scores for all eligible items, amb caché pickle."""
        cache_path = self._cache_path(
            f"simple_{self._dataset.get_cache_key()}_minvotes{self._min_votes}.pkl"
        )
        self._logger.info(
            "Preparant SimpleRecommender per al dataset '%s' (min_votes=%d)...",
            self._dataset.get_cache_key(),
            self._min_votes,
        )

        cached = _load_pickle_cache(cache_path, self._logger)
        if isinstance(cached, dict) and "item_scores" in cached:
            self._item_scores = cached["item_scores"]
            self._is_prepared = True
            self._logger.info(
                "SimpleRecommender carregat des de caché (%d ítems puntuats).",
                len(self._item_scores),
            )
            return

        self._logger.info(
            "SimpleRecommender: computant puntuacions des de zero..."
        )
        eligible_items: List[Tuple[str, int, float, float]] = []
        total_rating_sum = 0.0
        total_rating_count = 0

        for item_id in self._dataset.get_item_ids():
            item_ratings = self._dataset.get_item_user_ratings(item_id)
            num_votes = len(item_ratings)
            if num_votes < self._min_votes:
                continue

            item_sum = sum(item_ratings.values())
            avg_item = item_sum / num_votes
            eligible_items.append((item_id, num_votes, avg_item, item_sum))
            total_rating_sum += item_sum
            total_rating_count += num_votes

        self._item_scores = {}
        if total_rating_count == 0:
            self._is_prepared = True
            self._logger.info("SimpleRecommender: cap ítem elegible, caché no es desa.")
            return

        avg_global = total_rating_sum / total_rating_count
        for item_id, num_votes, avg_item, _ in eligible_items:
            denominator = num_votes + self._min_votes
            score = ((num_votes / denominator) * avg_item) + ((self._min_votes / denominator) * avg_global)
            self._item_scores[item_id] = score

        _save_pickle_cache(cache_path, {"item_scores": self._item_scores}, self._logger)
        self._is_prepared = True
        self._logger.info(
            "SimpleRecommender preparat des de zero (%d ítems puntuats).",
            len(self._item_scores),
        )

    def recommend(self, user_id: str, top_n: int = 5) -> List[Recommendation]:
        if not self._is_prepared:
            self.prepare()

        rated_items = self._dataset.get_user_rated_items(user_id)
        candidates = [
            (item_id, score)
            for item_id, score in self._item_scores.items()
            if item_id not in rated_items
        ]

        if not candidates:
            return self._fallback_by_item_average(top_n, rated_items)

        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates[:top_n]

    def predict_rating(self, user_id: str, item_id: str) -> Optional[float]:
        """Return the Bayesian-average score for *item_id* as a predicted rating.

        Parameters
        ----------
        user_id : str
            Ignored; simple recommender scores are user-independent.
        item_id : str
            Target item identifier.

        Returns
        -------
        float or None
            Pre-computed Bayesian score, or ``None`` when the item has fewer
            votes than ``min_votes``.
        """
        if not self._is_prepared:
            self.prepare()
        return self._item_scores.get(item_id)


class CollaborativeRecommender(Recommender):
    """User-based collaborative filtering recommender.

    Uses Pearson-like cosine similarity computed over mean-centred ratings.

    Parameters
    ----------
    dataset : Dataset
        Dataset to recommend from.
    k : int, optional
        Number of nearest neighbours to consider.  Defaults to ``5``.

    Attributes
    ----------
    _k : int
        Neighbourhood size.
    _user_means : dict
        Mapping from user_id to that user's average rating.
    _neighbors_cache : dict or None
        Mapping from user_id to ``[(neighbor_id, similarity), ...]``.
    """

    def __init__(self, dataset: Dataset, k: int = 5) -> None:
        super().__init__(dataset)
        self._k = k
        self._user_means: Dict[str, float] = {}
        self._neighbors_cache: Optional[Dict[str, List[Tuple[str, float]]]] = None

    def prepare(self) -> None:
        """Compute per-user mean ratings i veïns, amb caché pickle."""
        cache_path = self._cache_path(
            f"collaborative_{self._dataset.get_cache_key()}_k{self._k}.pkl"
        )
        self._logger.info(
            "Preparant CollaborativeRecommender per al dataset '%s' (k=%d)...",
            self._dataset.get_cache_key(),
            self._k,
        )

        cached = _load_pickle_cache(cache_path, self._logger)
        if isinstance(cached, dict) and "user_means" in cached and "neighbors" in cached:
            self._user_means = cached["user_means"]
            self._neighbors_cache = cached["neighbors"]
            self._is_prepared = True
            self._logger.info(
                "CollaborativeRecommender carregat des de caché (%d usuaris, %d amb veïns).",
                len(self._user_means),
                len(self._neighbors_cache),
            )
            return

        self._logger.info(
            "CollaborativeRecommender: computant mitjanes i veïns des de zero..."
        )
        self._user_means = {}
        for user_id in self._dataset.get_user_ids():
            avg_user = self._dataset.get_user_average(user_id)
            if avg_user is not None:
                self._user_means[user_id] = avg_user
        self._neighbors_cache = self._build_neighbors_cache()
        _save_pickle_cache(
            cache_path,
            {"user_means": self._user_means, "neighbors": self._neighbors_cache},
            self._logger,
        )
        self._is_prepared = True
        self._logger.info(
            "CollaborativeRecommender preparat des de zero (%d usuaris, %d amb veïns).",
            len(self._user_means),
            len(self._neighbors_cache),
        )

    def _build_neighbors_cache(self) -> Dict[str, List[Tuple[str, float]]]:
        stats: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(lambda: [0.0, 0.0, 0.0])
        )

        for item_id in self._dataset.get_item_ids():
            item_ratings = self._dataset.get_item_user_ratings(item_id)
            if len(item_ratings) < 2:
                continue

            users = list(item_ratings.items())
            for idx_u in range(len(users)):
                user_u, rating_u = users[idx_u]
                for idx_v in range(idx_u + 1, len(users)):
                    user_v, rating_v = users[idx_v]
                    stats[user_u][user_v][0] += rating_u * rating_v
                    stats[user_u][user_v][1] += rating_u * rating_u
                    stats[user_u][user_v][2] += rating_v * rating_v
                    stats[user_v][user_u][0] += rating_u * rating_v
                    stats[user_v][user_u][1] += rating_v * rating_v
                    stats[user_v][user_u][2] += rating_u * rating_u

        neighbors_cache: Dict[str, List[Tuple[str, float]]] = {}
        for user_id, neighbor_stats in stats.items():
            neighbors: List[Tuple[str, float]] = []
            for other_user, (dot, norm_u, norm_v) in neighbor_stats.items():
                denominator = math.sqrt(norm_u) * math.sqrt(norm_v)
                if denominator == 0:
                    continue
                similarity = dot / denominator
                if similarity <= 0:
                    continue
                neighbors.append((other_user, similarity))
            neighbors.sort(key=lambda item: (-item[1], item[0]))
            neighbors_cache[user_id] = neighbors[: self._k]
            self._logger.debug(
                "Calculant similitud per a l'usuari %s: %d veïns seleccionats.",
                user_id, len(neighbors_cache[user_id]),
            )

        return neighbors_cache

    def _top_k_neighbors(self, user_id: str) -> List[Tuple[str, float]]:
        if self._neighbors_cache is not None:
            return self._neighbors_cache.get(user_id, [])

        target_ratings = self._dataset.get_user_ratings(user_id)
        if not target_ratings:
            return []

        dot_products = defaultdict(float)
        norm_u = defaultdict(float)
        norm_v = defaultdict(float)

        for item_id, rating_u in target_ratings.items():
            for other_user, rating_v in self._dataset.get_item_user_ratings(item_id).items():
                if other_user == user_id:
                    continue

                dot_products[other_user] += rating_u * rating_v
                norm_u[other_user] += rating_u * rating_u
                norm_v[other_user] += rating_v * rating_v

        similarities: List[Tuple[str, float]] = []
        for other_user, dot in dot_products.items():
            denominator = math.sqrt(norm_u[other_user]) * math.sqrt(norm_v[other_user])
            if denominator == 0:
                continue
            similarity = dot / denominator
            if similarity <= 0:
                continue
            similarities.append((other_user, similarity))

        similarities.sort(key=lambda item: (-item[1], item[0]))
        return similarities[: self._k]

    def _predict_for_user(
        self,
        user_id: str,
        neighbors: Sequence[Tuple[str, float]],
        top_n: int,
    ) -> List[Recommendation]:
        rated_items = self._dataset.get_user_rated_items(user_id)
        avg_u = self._user_means.get(user_id)
        if avg_u is None:
            return self._fallback_by_item_average(top_n, rated_items)

        min_rating, max_rating = self._dataset.get_rating_bounds()
        predictions: List[Recommendation] = []

        for item_id in self._dataset.get_unrated_items(user_id):
            numerator = 0.0
            denominator = 0.0

            for neighbor_id, similarity in neighbors:
                rating_vi = self._dataset.get_rating(neighbor_id, item_id)
                if rating_vi is None:
                    continue

                avg_v = self._user_means.get(neighbor_id)
                if avg_v is None:
                    continue

                numerator += similarity * (rating_vi - avg_v)
                denominator += abs(similarity)

            if denominator == 0:
                continue

            score = avg_u + (numerator / denominator)
            if score < min_rating:
                score = min_rating
            if score > max_rating:
                score = max_rating
            predictions.append((item_id, score))

        if not predictions:
            return self._fallback_by_item_average(top_n, rated_items)

        predictions.sort(key=lambda item: (-item[1], item[0]))
        return predictions[:top_n]

    def recommend(self, user_id: str, top_n: int = 5) -> List[Recommendation]:
        if not self._is_prepared:
            self.prepare()

        if user_id not in self._user_means:
            return self._fallback_by_item_average(top_n, self._dataset.get_user_rated_items(user_id))

        neighbors = self._top_k_neighbors(user_id)
        if not neighbors:
            return self._fallback_by_item_average(top_n, self._dataset.get_user_rated_items(user_id))

        return self._predict_for_user(user_id, neighbors, top_n)

    def predict_rating(self, user_id: str, item_id: str) -> Optional[float]:
        """Predict the rating *user_id* would give *item_id* using CF.

        Parameters
        ----------
        user_id : str
            Target user identifier.
        item_id : str
            Item identifier to predict.

        Returns
        -------
        float or None
            CF prediction clamped to the dataset rating bounds, or ``None``
            when there are no neighbours with a rating for *item_id*.
        """
        if not self._is_prepared:
            self.prepare()

        avg_u = self._user_means.get(user_id)
        if avg_u is None:
            return None

        neighbors = self._top_k_neighbors(user_id)
        if not neighbors:
            return None

        min_rating, max_rating = self._dataset.get_rating_bounds()
        numerator = 0.0
        denominator = 0.0

        for neighbor_id, similarity in neighbors:
            rating_vi = self._dataset.get_rating(neighbor_id, item_id)
            if rating_vi is None:
                continue
            avg_v = self._user_means.get(neighbor_id)
            if avg_v is None:
                continue
            numerator += similarity * (rating_vi - avg_v)
            denominator += abs(similarity)

        if denominator == 0:
            return None

        score = avg_u + (numerator / denominator)
        return max(min_rating, min(max_rating, score))


class ContentBasedRecommender(Recommender):
    """Content-based recommender using TF-IDF item vectors.

    Builds a TF-IDF matrix from each item's textual content (e.g. genres for
    MovieLens, author for Books).  A user profile is computed as the
    rating-weighted average of the TF-IDF vectors of items the user has rated.
    Candidates are ranked by dot-product similarity (S_u = M · Q_u^T) between
    the user profile and each unrated item vector, scaled to the dataset's
    maximum rating.

    Parameters
    ----------
    dataset : Dataset
        Dataset to recommend from.

    Attributes
    ----------
    _tfidf_matrix : scipy.sparse matrix or None
        TF-IDF matrix of shape ``(n_items, n_features)``.
    _vectorizer : TfidfVectorizer or None
        Fitted scikit-learn vectorizer.
    _item_index : dict
        Mapping from item_id to row index in ``_tfidf_matrix``.
    _index_item : list
        Mapping from row index to item_id.
    """

    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset)
        self._tfidf_matrix = None
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._item_index: Dict[str, int] = {}
        self._index_item: List[str] = []

    def prepare(self) -> None:
        """Fit TF-IDF vectorizer on all item content texts, amb caché pickle."""
        cache_path = self._cache_path(f"content_{self._dataset.get_cache_key()}.pkl")
        self._logger.info(
            "Preparant ContentBasedRecommender per al dataset '%s'...",
            self._dataset.get_cache_key(),
        )

        cached = _load_pickle_cache(cache_path, self._logger)
        if isinstance(cached, dict) and {
            "vectorizer",
            "tfidf_matrix",
            "item_index",
            "index_item",
        }.issubset(cached):
            self._vectorizer = cached["vectorizer"]
            self._tfidf_matrix = cached["tfidf_matrix"]
            self._item_index = cached["item_index"]
            self._index_item = cached["index_item"]
            self._is_prepared = True
            self._logger.info(
                "ContentBasedRecommender carregat des de caché "
                "(%d ítems indexats, %d característiques TF-IDF).",
                self._tfidf_matrix.shape[0],
                self._tfidf_matrix.shape[1],
            )
            return

        self._logger.info(
            "ContentBasedRecommender: construint matriu TF-IDF des de zero..."
        )
        texts: List[str] = []
        valid_ids: List[str] = []

        for item_id in self._dataset.get_item_ids():
            text = self._dataset.get_item_content_text(item_id)
            if text.strip():
                texts.append(text)
                valid_ids.append(item_id)

        if not texts:
            self._is_prepared = True
            self._logger.info("ContentBasedRecommender: cap text de contingut disponible, caché no es desa.")
            return

        self._vectorizer = TfidfVectorizer()
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)
        self._index_item = valid_ids
        self._item_index = {item_id: idx for idx, item_id in enumerate(valid_ids)}
        self._logger.debug(
            "Matriu TF-IDF generada amb forma %s (%d items, %d característiques).",
            str(self._tfidf_matrix.shape), self._tfidf_matrix.shape[0], self._tfidf_matrix.shape[1],
        )
        _save_pickle_cache(
            cache_path,
            {
                "vectorizer": self._vectorizer,
                "tfidf_matrix": self._tfidf_matrix,
                "item_index": self._item_index,
                "index_item": self._index_item,
            },
            self._logger,
        )
        self._is_prepared = True
        self._logger.info(
            "ContentBasedRecommender preparat des de zero "
            "(%d ítems indexats, %d característiques TF-IDF).",
            self._tfidf_matrix.shape[0],
            self._tfidf_matrix.shape[1],
        )

    def _compute_similarity(
        self,
        user_profile: np.ndarray,
        item_vector_or_matrix,
    ) -> np.ndarray:
        """Compute raw dot-product similarity following S_u = M · Q_u^T.

        No normalisation is applied — this is a pure inner-product as required
        by the project formula.  The method handles both the full-matrix case
        (all items at once) and the single-item case transparently via the
        ``@`` operator.

        Parameters
        ----------
        user_profile : np.ndarray
            Dense profile vector of shape ``(n_features,)``.
        item_vector_or_matrix : np.ndarray or scipy sparse matrix
            Either a 1-D item vector ``(n_features,)`` for a single item, or
            the full TF-IDF matrix of shape ``(n_items, n_features)`` for all
            items at once.

        Returns
        -------
        np.ndarray
            1-D array of similarity scores — shape ``(n_items,)`` for the
            matrix case, or a scalar for the single-vector case.
        """
        return item_vector_or_matrix @ user_profile

    def _compute_user_profile(self, user_id: str) -> Optional[np.ndarray]:
        """Return the rating-weighted mean TF-IDF vector for *user_id*.

        Parameters
        ----------
        user_id : str
            Target user identifier.

        Returns
        -------
        numpy.ndarray or None
            Dense profile vector of shape ``(n_features,)``, or ``None`` when
            the user has no ratings on items present in the TF-IDF index.
        """
        ratings = self._dataset.get_user_ratings(user_id)
        if not ratings or self._tfidf_matrix is None:
            return None

        n_features = self._tfidf_matrix.shape[1]
        weighted_sum = np.zeros(n_features)
        total_weight = 0.0

        for item_id, rating in ratings.items():
            idx = self._item_index.get(item_id)
            if idx is None:
                continue
            item_vec = self._tfidf_matrix[idx].toarray().flatten()
            weighted_sum += rating * item_vec
            total_weight += rating

        if total_weight == 0:
            return None

        return weighted_sum / total_weight

    def recommend(self, user_id: str, top_n: int = 5) -> List[Recommendation]:
        if not self._is_prepared:
            self.prepare()

        rated_items = self._dataset.get_user_rated_items(user_id)

        if self._tfidf_matrix is None:
            return self._fallback_by_item_average(top_n, rated_items)

        user_profile = self._compute_user_profile(user_id)
        if user_profile is None:
            return self._fallback_by_item_average(top_n, rated_items)

        _, max_rating = self._dataset.get_rating_bounds()

        sims = self._compute_similarity(user_profile, self._tfidf_matrix)

        candidates: List[Recommendation] = []
        for idx, item_id in enumerate(self._index_item):
            if item_id in rated_items:
                continue
            candidates.append((item_id, float(sims[idx]) * max_rating))

        if not candidates:
            return self._fallback_by_item_average(top_n, rated_items)

        candidates.sort(key=lambda x: (-x[1], x[0]))
        return candidates[:top_n]

    def predict_rating(self, user_id: str, item_id: str) -> Optional[float]:
        """Predict the rating *user_id* would give *item_id* via content similarity.

        The prediction is ``dot(user_profile, item_vector) * max_rating``.

        Parameters
        ----------
        user_id : str
            Target user identifier.
        item_id : str
            Item identifier to predict.

        Returns
        -------
        float or None
            Predicted rating, or ``None`` when the item has no content vector
            or the user has no ratable history.
        """
        if not self._is_prepared:
            self.prepare()

        if self._tfidf_matrix is None:
            return None

        idx = self._item_index.get(item_id)
        if idx is None:
            return None

        user_profile = self._compute_user_profile(user_id)
        if user_profile is None:
            return None

        _, max_rating = self._dataset.get_rating_bounds()
        item_vec = self._tfidf_matrix[idx].toarray().flatten()
        sim = float(self._compute_similarity(user_profile, item_vec))
        return sim * max_rating
