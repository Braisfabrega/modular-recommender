# simple_recommender.py

from recommender import Recommender
from collections import defaultdict  #nos permite crear diccionarios donde cada clave empieza con una lista vacía sin tener que inicializarla a mano.
import numpy as np

class SimpleRecommender(Recommender):
    #loader =(users,items,ratings)
    #utlitzem min_votes=5 per exemple pot ser altre valor
    def __init__(self, loader, min_votes=5):
        super().__init__(loader)
        self.min_votes = min_votes

    def recommend(self, user_id):
        # ratings = [(user, item, rating), ...]
        ratings = self.loader.ratings

        # Utilitzem defaultdict pq ens permet agrupar totes les valoracions per item sense haber de comprovar si la clau existeix
        #{"034545104X": [5, 7, 8, 10]", "0155061224": [6, 10, 9]...}
        item_scores = defaultdict(list)

        for user, item, rating in ratings:
            if rating > 0:  # en aquest dataset, 0 també és un vot, però el considerem vàlid
                item_scores[item].append(rating)

        # mean calcula la mitja de la llista de values [3, 4, 2, 1, 5, 7] --> 3.66
        #items_avgs queda una llista amb totes les mitjes [3.66, 8.0, 6,2...]
        item_avgs = [np.mean(v) for v in item_scores.values() if len(v) >= self.min_votes]
        if len(item_avgs) == 0:
            return []  # evitar divisió per zero
        global_avg = np.mean(item_avgs)

        # Ítems que l'usuari ja ha valorat
        rated_by_user = {item for (user, item, rating) in ratings if user == user_id}

        results = []

        for item, vals in item_scores.items():
            num_vots = len(vals)

            # descartar ítems amb pocs vots
            if num_vots < self.min_votes:
                continue

            avg_item = np.mean(vals)

            # Fórmula del projecte
            score = (num_vots / (num_vots + self.min_votes)) * avg_item + \
                    (self.min_votes / (num_vots + self.min_votes)) * global_avg

            # Només recomanem ítems que l'usuari no ha vist
            if item not in rated_by_user:
                results.append((item, score))

        # Ordenar per score descendent
        #key=lambda x: x[1] -->Ordena la llista utilitzant com clau el segon element de cada tupla (score) no item
        #reverse=True --> per defecte Python ordena de menor a major pero volem de major a menor
        results.sort(key=lambda x: x[1], reverse=True)

        # Retornar els 5 primers per exemple 
        return results[:5]
