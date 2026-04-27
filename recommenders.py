from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

from datasets import Dataset


Recommendation = Tuple[str, float]


class Recommender(ABC):
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._is_prepared = False

    @abstractmethod
    def prepare(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def recommend(self, user_id: str, top_n: int = 5) -> List[Recommendation]:
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
    def __init__(self, dataset: Dataset, min_votes: int = 10) -> None:
        super().__init__(dataset)
        self._min_votes = min_votes
        self._item_scores: Dict[str, float] = {}

    def prepare(self) -> None:
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
            return

        avg_global = total_rating_sum / total_rating_count
        for item_id, num_votes, avg_item, _ in eligible_items:
            denominator = num_votes + self._min_votes
            score = ((num_votes / denominator) * avg_item) + ((self._min_votes / denominator) * avg_global)
            self._item_scores[item_id] = score

        self._is_prepared = True

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


class CollaborativeRecommender(Recommender):
    def __init__(self, dataset: Dataset, k: int = 5) -> None:
        super().__init__(dataset)
        self._k = k
        self._user_means: Dict[str, float] = {}

    def prepare(self) -> None:
        self._user_means = {}
        for user_id in self._dataset.get_user_ids():
            avg_user = self._dataset.get_user_average(user_id)
            if avg_user is not None:
                self._user_means[user_id] = avg_user
        self._is_prepared = True

    def _top_k_neighbors(self, user_id: str) -> List[Tuple[str, float]]:
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
