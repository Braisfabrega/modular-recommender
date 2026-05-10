from dataset_loader import DatasetLoader
from simple_recommender import SimpleRecommender
from collaborative_recommender import CollaborativeRecommender

def main():
    # Cargar dataset
    # Load s'encarrega de llegir movies.csv, ratings.csv i omplir loader.items, loader.ratings i loader.users
    loader = DatasetLoader("books")   # o "books"
    loader.load()

    print("Tipus de recomanador:")
    print("1 - Simple")
    print("2 - Col·laboratiu (cosinus)")
    opcion = input("Escull una opció (1/2): ")

    user_id = int(input("Introdueix l'user_id: "))

    if opcion == "1":
        rec = SimpleRecommender(loader)
    else:
        rec = CollaborativeRecommender(loader, k=5)

    recomendaciones = rec.recommend(user_id)

    print("\nRecomanacions per a l'usuari", user_id)
    if not recomendaciones:
        print("No hi ha recomanacions disponibles.")
    else:
        for item, score in recomendaciones:
            print(f"Ítem: {item}  |  Score: {score:.3f}")

if __name__ == "__main__":
    main()
