# collaborative_recommender.py

from recommender import Recommender
import numpy as np
from collections import defaultdict

class CollaborativeRecommender(Recommender):
#posem k=5 per exemple pot ser un altre valor
    def __init__(self, loader, k=5):
        super().__init__(loader)
        self.k = k  # nombre de veïns més semblants

    def recommend(self, user_id):
        ratings = self.loader.ratings

        # 1) Construir matriu usuari-item com diccionari de diccionaris
        # user_item[u][item] = rating
        #user_item = user_id_1: { item_1: rating, item_2: rating, ... }, 276726: {"0155061224": 5, "0446520802": 4}}


        user_item = defaultdict(dict)
        for user, item, rating in ratings:
            if rating > 0:
                user_item[user][item] = rating
                #
        # Si l'usuari no existeix o no té valoracions → no podem fer col·laboratiu
        if user_id not in user_item:
            return []

        #  Vector del nostre usuari
        #{"034545104X": 7, "052165615X": 3} --> tots els items que l'usuari ja ha valorat
        target_ratings = user_item[user_id]

        #  Calcular semblança (correlació) amb tots els altres usuaris
        similarities = []

        #Recorrem tots els usuaris del sistema, si other_user==user_id el saltem no te sentit comparar-se amb si mateix
        #user_item.items() --> (user_id, diccionario_de_items)

        for other_user, other_ratings in user_item.items():
            if other_user == user_id:
                continue

            # Ítems que han valorat tots dos
            comuns = set(target_ratings.keys()) & set(other_ratings.keys())

            if len(comuns) < 2:
                continue  # no hi ha prou dades per calcular correlació

            # Crear vectors alineats
            #Agafem les puntuacion dels dos usuaris sobre els mateixos items
            v1 = np.array([target_ratings[i] for i in comuns])
            v2 = np.array([other_ratings[i] for i in comuns])

            
            # Similitud del coseno (la del PDF)
            #Si tenim v1=[7,6] i v2=[6,7], num fa 7*6+6*7, den fa sqrt(7^^2+6^^2)*sqrt(6^^2+7^^2)
            num = np.dot(v1, v2)
            den = np.linalg.norm(v1) * np.linalg.norm(v2)

            if den == 0:
                continue
            
            #quan mes semblant sigui sim a 1 mes similars els gustos
            sim = num / den
            similarities.append((other_user, sim))

        # Si no hi ha veïns → no podem recomanar
        if len(similarities) == 0:
            return []

        # Agafar els k veïns més semblants
        #Ordenem la sim de mes gran a mes petit
        similarities.sort(key=lambda x: x[1], reverse=True)

        #Ens quedem amb els k primers usuaris mes semblants
        #neighbors=(neighbor_user_id, similarity_value)

        neighbors = similarities[:self.k]

        # 5) Agafar ítems que els veïns han vist i l'usuari no
        #rated_by_user --> conjunt d'items que el usari ja ha valorat
        #creem candidate_score on guardem el item i la llista de ratings que li han donat els veïns (item--> [ratings de vecinos])
        rated_by_user = set(target_ratings.keys())
        candidate_scores = defaultdict(list)

        for neighbor, sim in neighbors:
            for item, rating in user_item[neighbor].items():
                if item not in rated_by_user:
                    candidate_scores[item].append(rating)

        # 6) Fer la mitja de cada ítem i ordenar
        
        results = []
        for item, vals in candidate_scores.items():
            avg = np.mean(vals)
            results.append((item, avg))

        results.sort(key=lambda x: x[1], reverse=True)

        # Retornar top-5
        return results[:5]
