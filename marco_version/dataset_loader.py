import csv

class DatasetLoader:
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self.users = {}
        self.items = {}
        self.ratings = []

    def load(self):
        if self.dataset_type == "movies":
            self._load_movies()
        elif self.dataset_type == "books":
            self._load_books()
        else:
            raise ValueError("Dataset no reconegut")

    def _load_movies(self):
        with open("datasets/MovieLens100k/movies.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                movie_id = int(row[0])
                title = row[1]
                genres = row[2]
                self.items[movie_id] = {"title": title, "genres": genres}

        with open("datasets/MovieLens100k/ratings.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                user = int(row[0])
                movie = int(row[1])
                rating = float(row[2])
                self.ratings.append((user, movie, rating))

    def _load_books(self):
        with open("datasets/Books/Books.csv") as f:
            reader = csv.reader(f)
            next(reader)  # saltar capçalera
            for row in reader:
                isbn = row[0]  # ISBN és el bookId
                title = row[1]
                author = row[2]
                year = row[3]
                publisher = row[4]

                self.items[isbn] = {"title": title, "author": author, "year": year, "publisher": publisher}

        with open("datasets/Books/Users.csv", encoding="utf8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                user_id = int(row[0])
                location = row[1]

                age_raw = row[2] if len(row) > 2 else ""

                # Casos:
                # ""  → buit
                # "18.0" → float
                # "0.0" → probablement sense edat
                if age_raw == "":
                    age = None
                else:
                    try:
                        age = int(float(age_raw))
                        if age <= 0:
                            age = None
                    except:
                        age = None

                self.users[user_id] = {"location": location, "age": age}

        # Carregar valoracions
        with open("datasets/Books/Ratings.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                user = int(row[0])
                isbn = row[1]
                rating = float(row[2])

                # Afegim la tupla (usuari, llibre, puntuació)
                self.ratings.append((user, isbn, rating))