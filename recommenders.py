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


def _sort_by_score(items: List[Tuple[str, float]]) -> None:
    """Ordena in-place la llista de parelles (id, score) per score descendent i id ascendent."""
    items.sort(key=lambda x: (-x[1], x[0]))


def _load_pickle_cache(cache_path: str, logger: logging.Logger) -> Optional[Any]:
    """Carrega un fitxer d'emmagatzematge serialitzat (pickle cache) des del disc.

    Parameters
    ----------
    cache_path : str
        Ruta absoluta o relativa al fitxer .pkl que es vol llegir.
    logger : logging.Logger
        Instància de logging activa per registrar l'estat de la càrrega.

    Returns
    -------
    Any or None
        L'objecte deserialitzat si la lectura és correcta; ``None`` si el fitxer
        no existeix o es detecta corrupció en les dades.
    """
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
    """Desa un objecte en disc en format serialitzat de manera atòmica i segura.

    Utilitza un fitxer temporal i l'operació atòmica de substitució del sistema de 
    fitxers per garantir que no es generin binaris corromputs si la seva escriptura
    s'interromp abruptament.

    Parameters
    ----------
    cache_path : str
        Ruta objectiu on es vol emmagatzemar de manera definitiva el fitxer de memòria cau.
    payload : Any
        Estructura o objecte de Python que es desitja serialitzar.
    logger : logging.Logger
        Instància de logging activa per registrar l'estat de l'operació.
    """
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
    """Classe base abstracta reguladora de tots els motors de recomanació del sistema.

    Parameters
    ----------
    dataset : Dataset
        El conjunt de dades actiu sobre el qual operarà aquest recomanador.

    Attributes
    ----------
    _dataset : Dataset
        Instància del conjunt de dades vinculada internament.
    _logger : logging.Logger
        Instància del registre de traces personalitzat amb el nom de la subclasse.
    """

    def __init__(self, dataset: Dataset) -> None:
        """Inicialitza l'esquelet base del recomanador abstracte."""
        self._dataset = dataset
        self._logger = logging.getLogger(f"recommender_system.{self.__class__.__name__}")

    def _cache_path(self, filename: str) -> str:
        """Calcula de manera interna la ruta del directori de memòria cau per a un fitxer.

        Parameters
        ----------
        filename : str
            Nom del fitxer que es pretén ubicar.

        Returns
        -------
        str
            Ruta completa normalitzada apuntant cap a la subcarpeta de memòria cau.
        """
        cache_dir = os.path.join(self._dataset.get_project_root(), "dataset", "cache")
        return os.path.join(cache_dir, filename)

    @abstractmethod
    def prepare(self) -> None:
        """Precomputa o entrena l'estat intern i les estructures requerides pels models."""
        raise NotImplementedError

    @abstractmethod
    def recommend(self, user_id: str, top_n: int = 5) -> List[Recommendation]:
        """Calcula i obté els Top-N ítems millor recomanats per a un usuari concret.

        Parameters
        ----------
        user_id : str
            Identificador únic de l'usuari objectiu.
        top_n : int, opcional
            Nombre màxim de recomanacions de sortida sol·licitades. Per defecte és ``5``.

        Returns
        -------
        list of tuple of (str, float)
            Llista ordenada descendentment segons la puntuació de parelles ``(item_id, score)``.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_rating(self, user_id: str, item_id: str) -> Optional[float]:
        """Estima numèricament la valoració predita que un usuari donaria a un ítem concret.

        Invocat directament pel mòdul d'avaluació externa per extreure les mètriques 
        comparatives globals MAE i RMSE.

        Parameters
        ----------
        user_id : str
            Identificador de l'usuari d'estudi.
        item_id : str
            Identificador de l'ítem del qual es desitja la predicció.

        Returns
        -------
        float or None
            Puntuació numèrica estimada clampada, o ``None`` quan el recomanador no disposa 
            de suficients dades contextuals com per fer un càlcul coherent.
        """
        raise NotImplementedError

    def _fallback_by_item_average(self, top_n: int, excluded_items: Set[str]) -> List[Recommendation]:
        """Mètode de contingència que recomana els ítems més ben valorats en mitjana global.

        S'utilitza per evitar trencar el flux del programa en situacions on no hi ha prou 
        veïns (en col·laboratiu) o vectors de paraules (en contingut) per a l'usuari objectiu.

        Parameters
        ----------
        top_n : int
            Nombre màxim d'ítems de contingència a llistar.
        excluded_items : set of str
            Conjunt d'identificadors d'ítems que han de ser omesos (p. ex. ja consumits).

        Returns
        -------
        list of tuple of (str, float)
            Llista d'ítems candidats ordenats descendentment per la seva puntuació mitjana.
        """
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
    """Sistema de recomanació simple basat en la Popularitat Bayesiana (Bayesian Average).

    Calcula la puntuació dels ítems aplicant una ponderació equilibrada que combina 
    la mitjana de valoracions d'un propi ítem amb la mitjana de valoracions global de tot el 
    dataset, penalitzant o regularitzant aquells ítems amb molt poques interaccions.

    Parameters
    ----------
    dataset : Dataset
        Conjunt de dades d'on s'extreuen les interaccions.
    min_votes : int, opcional
        Llindar de vots mínims exigibles per considerar un ítem com a elegible i pes 
        de regularització utilitzat en la fórmula bayesiana. Per defecte és ``10``.

    Attributes
    ----------
    _min_votes : int
        Valor límit de regularització bayesiana / vots mínims exigibles.
    _item_scores : dict of (str, float)
        Diccionari que mapeja cada ``item_id`` amb la seva puntuació ponderada precomputada.
    """

    def __init__(self, dataset: Dataset, min_votes: int = 10) -> None:
        """Inicialitza i assigna els paràmetres de configuració de Popularitat Bayesiana."""
        super().__init__(dataset)
        self._min_votes = min_votes
        self._item_scores: Dict[str, float] = {}
        self.prepare()

    def prepare(self) -> None:
        """Calcula la puntuació Bayesiana de tots els ítems vàlids, recorrent a cache en disc."""
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
            self._logger.info("SimpleRecommender: cap ítem elegible, caché no es desa.")
            return

        avg_global = total_rating_sum / total_rating_count
        for item_id, num_votes, avg_item, _ in eligible_items:
            denominator = num_votes + self._min_votes
            score = ((num_votes / denominator) * avg_item) + ((self._min_votes / denominator) * avg_global)
            self._item_scores[item_id] = score

        _save_pickle_cache(cache_path, {"item_scores": self._item_scores}, self._logger)
        self._logger.info(
            "SimpleRecommender preparat des de zero (%d ítems puntuats).",
            len(self._item_scores),
        )

    def recommend(self, user_id: str, top_n: int = 5) -> List[Recommendation]:
        """Calcula i obté els Top-N ítems recomanats per a un usuari concret.

        Filtra els ítems que l'usuari ja ha valorat prèviament i ordena els 
        candidats restants de manera descendent segons la seva puntuació Bayesiana.
        Si no hi ha candidats elegibles, recorre al mètode de contingència.

        Parameters
        ----------
        user_id : str
            Identificador únic de l'usuari objectiu.
        top_n : int, opcional
            Nombre màxim de recomanacions a retornar. Per defecte és ``5``.

        Returns
        -------
        list of tuple of (str, float)
            Llista ordenada descendentment de parelles ``(item_id, score)``.
        """
        rated_items = self._dataset.get_user_rated_items(user_id)
        candidates = [
            (item_id, score)
            for item_id, score in self._item_scores.items()
            if item_id not in rated_items
        ]

        if not candidates:
            return self._fallback_by_item_average(top_n, rated_items)

        _sort_by_score(candidates)
        return candidates[:top_n]

    def predict_rating(self, user_id: str, item_id: str) -> Optional[float]:
        """Retorna la puntuació Bayesiana d'un ítem com a predicció de valoració.

        Aquesta predicció és independent de l'usuari, ja que es basa en la 
        popularitat global ponderada de l'ítem.

        Parameters
        ----------
        user_id : str
            Identificador de l'usuari (s'ignora en aquest recomanador).
        item_id : str
            Identificador únic de l'ítem del qual es vol la predicció.

        Returns
        -------
        float or None
            La puntuació Bayesiana precomputada, o ``None`` si l'ítem no té 
            prou vots per haver estat indexat a ``_item_scores``.
        """
        return self._item_scores.get(item_id)


class CollaborativeRecommender(Recommender):
    """Filtre Col·laboratiu basat en l'usuari (User-Based Collaborative Filtering).

    Estimeu el comportament futur d'un usuari mitjançant l'estudi de perfils de veïns 
    similars, utilitzant una aproximació a la correlació de Pearson mitjançant similituds 
    de cosinus aplicades directament sobre les puntuacions centrades en la mitjana.

    Parameters
    ----------
    dataset : Dataset
        Conjunt de dades de d'on s'extreuen usuaris i valoracions.
    k : int, opcional
        Mida de l'entorn. Nombre màxim de veïns propers que es consideren per a la 
        predicció de puntuacions. Per defecte és ``5``.

    Attributes
    ----------
    _k : int
        Mida de l'entorn de veïnatge.
    _user_means : dict of (str, float)
        Mapeig del perfil d'usuaris amb la seva mitjana de puntuacions personals.
    _neighbors_cache : dict or None
        Estructura de memòria cau on es desen per a cada usuari llistes ordenades 
        de parelles tipus ``[(neighbor_id, similarity_value), ...]``.
    """

    def __init__(self, dataset: Dataset, k: int = 5) -> None:
        """Inicialitza el motor col·laboratiu configurant el paràmetre de veïns K."""
        super().__init__(dataset)
        self._k = k
        self._user_means: Dict[str, float] = {}
        self._neighbors_cache: Optional[Dict[str, List[Tuple[str, float]]]] = None
        self.prepare()

    def prepare(self) -> None:
        """Calcula les mitjanes dels usuaris i construeix el mapa d'afinitat de veïns."""
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
        self._logger.info(
            "CollaborativeRecommender preparat des de zero (%d usuaris, %d amb veïns).",
            len(self._user_means),
            len(self._neighbors_cache),
        )

    @staticmethod
    def _cosine_sim(dot: float, norm_u: float, norm_v: float) -> Optional[float]:
        """Retorna la similitud de cosinus o None si el denominador és zero o el resultat ≤ 0."""
        denom = math.sqrt(norm_u) * math.sqrt(norm_v)
        if denom == 0:
            return None
        sim = dot / denom
        return sim if sim > 0 else None

    def _build_neighbors_cache(self) -> Dict[str, List[Tuple[str, float]]]:
        """Calcula eficientment de forma creuada la similitud de cosinus entre tots els usuaris.

        Returns
        -------
        dict of (str, list of tuple)
            Diccionari d'entorns de veïnatge indexat per ID d'usuari.
        """
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
                sim = self._cosine_sim(dot, norm_u, norm_v)
                if sim is not None:
                    neighbors.append((other_user, sim))
            _sort_by_score(neighbors)
            neighbors_cache[user_id] = neighbors[: self._k]
            self._logger.debug(
                "Calculant similitud per a l'usuari %s: %d veïns seleccionats.",
                user_id, len(neighbors_cache[user_id]),
            )

        return neighbors_cache

    def _top_k_neighbors(self, user_id: str) -> List[Tuple[str, float]]:
        """Recupera o calcula sota demanda els Top-K veïns d'un usuari.

        Parameters
        ----------
        user_id : str
            Identificador únic de l'usuari consultat.

        Returns
        -------
        list of tuple of (str, float)
            Llista de fins a K parelles amb ID de veí i la seva similitud associada.
        """
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
            sim = self._cosine_sim(dot, norm_u[other_user], norm_v[other_user])
            if sim is not None:
                similarities.append((other_user, sim))

        _sort_by_score(similarities)
        return similarities[: self._k]

    def _predict_for_user(
        self,
        user_id: str,
        neighbors: Sequence[Tuple[str, float]],
        top_n: int,
    ) -> List[Recommendation]:
        """Calcula de manera agrupada les puntuacions estimades d'un entorn per als ítems no valorats.

        Parameters
        ----------
        user_id : str
            Usuari objectiu de l'estudi.
        neighbors : sequence of tuple
            Llista ordenada dels veïns afins a l'usuari.
        top_n : int
            Nombre de recomanacions màximes sol·licitades.

        Returns
        -------
        list of tuple of (str, float)
            Llista de les Top-N recomanacions ordenades.
        """
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

        _sort_by_score(predictions)
        return predictions[:top_n]

    def recommend(self, user_id: str, top_n: int = 5) -> List[Recommendation]:
        """Calcula els Top-N ítems recomanats mitjançant filtre col·laboratiu.

        Identifica els veïns més propers de l'usuari objectiu i utilitza les seves 
        interaccions per predir la puntuació dels ítems que l'usuari encara no 
        ha valorat. Si l'usuari o el seu entorn no tenen dades suficients, es 
        recorre a la mitjana global per ítem.

        Parameters
        ----------
        user_id : str
            Identificador únic de l'usuari objectiu.
        top_n : int, opcional
            Nombre màxim de recomanacions a retornar. Per defecte és ``5``.

        Returns
        -------
        list of tuple of (str, float)
            Llista de les millors recomanacions ordenades de manera descendent.
        """
        if user_id not in self._user_means:
            return self._fallback_by_item_average(top_n, self._dataset.get_user_rated_items(user_id))

        neighbors = self._top_k_neighbors(user_id)
        if not neighbors:
            return self._fallback_by_item_average(top_n, self._dataset.get_user_rated_items(user_id))

        return self._predict_for_user(user_id, neighbors, top_n)

    def predict_rating(self, user_id: str, item_id: str) -> Optional[float]:
        """Prediu la valoració d'un ítem basant-se en l'entorn de veïns (K-NN).

        Aplica la fórmula col·laborativa agregant les desviacions respecte a la 
        mitjana dels veïns que sí que han puntuat l'ítem, ponderat per la seva 
        similitud. El resultat final es clampa automàticament als límits de 
        valoració del dataset.

        Parameters
        ----------
        user_id : str
            Identificador únic de l'usuari d'estudi.
        item_id : str
            Identificador de l'ítem del qual es desitja estimar la nota.

        Returns
        -------
        float or None
            Predicció numèrica estimada clampada, o ``None`` si l'usuari no és 
            vàlid, no té veïns, o cap dels seus veïns ha puntuat aquest ítem.
        """
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
    """Sistema de recomanació Basat en Contingut utilitzant perfils de matriu de text TF-IDF.

    Forma una matriu esparsa basant-se en metadades textuals de cada ítem (com ara gèneres a
    MovieLens o autors a Books). Es modela el perfil de l'usuari calculant la mitjana ponderada 
    segons les seves valoracions històriques dels vectors de text consumits. La similitud final
    es mesura mitjançant el producte escalar (S_u = M · Q_u^T) i s'escala a la cota màxima.

    Parameters
    ----------
    dataset : Dataset
        Conjunt de dades amb atributs textuals associats als ítems.

    Attributes
    ----------
    _tfidf_matrix : scipy.sparse.csr_matrix or None
        Matriu conceptual de característiques de termes amb forma ``(n_items, n_features)``.
    _vectorizer : TfidfVectorizer or None
        Mecanisme extractor d'sklearn de termes i freqüències ajustat.
    _item_index : dict of (str, int)
        Diccionari mapejador de codis d'ítem cap a posicions de fila de la matriu.
    _index_item : list of str
        Llista indexada d'identificadors d'ítems per fer correspondències inverses de fila.
    """

    def __init__(self, dataset: Dataset) -> None:
        """Inicialitza l'esquelet intern dels components de text i vectorització TF-IDF."""
        super().__init__(dataset)
        self._tfidf_matrix = None
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._item_index: Dict[str, int] = {}
        self._index_item: List[str] = []
        self.prepare()

    def prepare(self) -> None:
        """Ajusta l'extractor TfidfVectorizer analitzant totes les descripcions de text textuals."""
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
        """Calcula el producte vectorial pur de similitud basat en la directriu S_u = M · Q_u^T.

        Sense aplicar normalitzacions de cosinus afegides, seguint la descripció del 
        disseny teòric del projecte. Multiplica vectors densos o matrius esparses 
        transparentment amb l'operador ``@``.

        Parameters
        ----------
        user_profile : np.ndarray
            Vector de perfil de l'usuari multidimensional complet d'estructura ``(n_features,)``.
        item_vector_or_matrix : np.ndarray or scipy.sparse matrix
            Vector d'ítem ``(n_features,)`` o matriu TF-IDF sencera ``(n_items, n_features)``.

        Returns
        -------
        np.ndarray
            Vector numèric unidimensional o escalar que llista les afinitats resultants.
        """
        return item_vector_or_matrix @ user_profile

    def _compute_user_profile(self, user_id: str) -> Optional[np.ndarray]:
        """Desenvolupa el vector mitjà ponderat TF-IDF que caracteritza els gustos de l'usuari.

        Parameters
        ----------
        user_id : str
            Identificador únic de l'usuari consultat.

        Returns
        -------
        np.ndarray or None
            Vector profile dens ajustat de dimensions ``(n_features,)``, o ``None`` quan 
            l'usuari manqui de qualsevol interacció indexable.
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
        """Genera els Top-N ítems recomanats basant-se en perfils de text TF-IDF.

        Construeix el perfil de gustos de l'usuari fent una combinació lineal 
        de les descripcions dels ítems consumits. Posteriorment, calcula el 
        producte escalar amb els ítems no valorats, escalant el resultat final 
        amb la valoració màxima admissible.

        Parameters
        ----------
        user_id : str
            Identificador de l'usuari objectiu de les recomanacions.
        top_n : int, opcional
            Nombre de recomanacions màximes sol·licitades. Per defecte és ``5``.

        Returns
        -------
        list of tuple of (str, float)
            Llista de parelles ``(item_id, score)`` sorted descendentment.
        """
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

        _sort_by_score(candidates)
        return candidates[:top_n]

    def predict_rating(self, user_id: str, item_id: str) -> Optional[float]:
        """Estima la valoració predita mitjançant el contingut del propi ítem.

        Calcula el producte escalar directament entre el vector de paraules 
        TF-IDF de l'ítem objectiu i el perfil d'interaccions dens de l'usuari, 
        multiplicat per la nota màxima.

        Parameters
        ----------
        user_id : str
            Identificador únic de l'usuari d'estudi.
        item_id : str
            Identificador de l'ítem del qual es demana la predicció.

        Returns
        -------
        float or None
            Predicció en format float d'acord amb el contingut textual, o 
            ``None`` si l'ítem no disposa de metadades o l'usuari no té historial.
        """
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
