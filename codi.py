import sys
import pandas as pd


# ============================
#   DATASETS
# ============================

class BaseDataset:
    def __init__(self, ruta):
        self.ruta = ruta
        self.items = None
        self.ratings = None

    def cargar(self):
        pass


class MovieLens(BaseDataset):
    def cargar(self):
        movies = pd.read_csv(self.ruta + "/movies.csv")
        ratings = pd.read_csv(self.ruta + "/ratings.csv")

        movies = movies.rename(columns={
            "movieId": "item_id",
            "title": "title"
        })

        ratings = ratings.rename(columns={
            "userId": "user_id",
            "movieId": "item_id",
            "rating": "rating"
        })

        self.items = movies
        self.ratings = ratings


class Books(BaseDataset):
    def cargar(self):
        books = pd.read_csv(self.ruta + "/Books.csv")
        ratings = pd.read_csv(self.ruta + "/Ratings.csv")

        books = books.rename(columns={
            "ISBN": "item_id",
            "Book-Title": "title"
        })

        ratings = ratings.rename(columns={
            "User-ID": "user_id",
            "Book-ID": "item_id",
            "Book-Rating": "rating"
        })

        self.items = books
        self.ratings = ratings


# ============================
#   RECOMMENDERS
# ============================

class BaseRecommender:
    def __init__(self, dataset):
        self.dataset = dataset

    def recomendar(self, user_id):
        pass


class SimpleRecommender(BaseRecommender):
    def recomendar(self, user_id):
        df = self.dataset.ratings
        medias = df.groupby("item_id")["rating"].mean()
        top = medias.sort_values(ascending=False).head(5).index.tolist()
        return self.dataset.items[self.dataset.items["item_id"].isin(top)]


class CollaborativeRecommender(BaseRecommender):
    def recomendar(self, user_id):
        df = self.dataset.ratings

        tabla = df.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating"
        )

        if user_id not in tabla.index:
            return SimpleRecommender(self.dataset).recomendar(user_id)

        usuario = tabla.loc[user_id]
        simil = tabla.corrwith(usuario, axis=1)
        simil = simil.drop(labels=[user_id], errors="ignore")
        simil = simil.dropna()

        if simil.empty:
            return SimpleRecommender(self.dataset).recomendar(user_id)

        vecinos = simil.sort_values(ascending=False).head(5).index
        sub = tabla.loc[vecinos]
        medias = sub.mean().sort_values(ascending=False)

        vistos = tabla.loc[user_id].dropna().index
        medias = medias.drop(labels=vistos, errors="ignore")

        top = medias.head(5).index.tolist()
        return self.dataset.items[self.dataset.items["item_id"].isin(top)]


# ============================
#   CARGA
# ============================

def cargar_dataset(nombre):
    if nombre == "movies":
        ds = MovieLens("dataset/MovieLens100k")
    elif nombre == "books":
        ds = Books("dataset/Books")
    else:
        print("Dataset no válido")
        sys.exit(1)

    ds.cargar()
    return ds


def cargar_recommender(metodo, dataset):
    if metodo == "simple":
        return SimpleRecommender(dataset)
    elif metodo == "collaborative":
        return CollaborativeRecommender(dataset)
    else:
        print("Método no válido")
        sys.exit(1)


# ============================
#   MAIN
# ============================

def main():
    if len(sys.argv) != 3:
        print("Uso: python main.py <dataset> <method>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    method_name = sys.argv[2]

    dataset = cargar_dataset(dataset_name)
    rec = cargar_recommender(method_name, dataset)

    while True:
        user = input("User ID (enter para salir): ")
        if user.strip() == "":
            break

        try:
            user = int(user)
        except:
            print("ID inválido")
            continue

        recomendaciones = rec.recomendar(user)

        print("\nRecomendaciones:")
        cols = [c for c in ["item_id", "title"] if c in recomendaciones.columns]
        print(recomendaciones[cols].head(5).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
