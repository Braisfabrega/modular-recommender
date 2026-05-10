# recommender.py

class Recommender:
    def __init__(self, loader):
        # loader conté totes les dades carregades (usuaris, items, ratings)
        self.loader = loader

    def recommend(self, user_id):
        raise NotImplementedError("Aquest mètode s'ha d'implementar a les subclasses")
