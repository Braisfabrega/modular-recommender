# Sistema de recomendacion (Proyecto PA)

## Descripcion general
Proyecto de recomendacion con tres enfoques (simple, colaborativo y basado en contenido) y soporte para dos datasets (MovieLens100k y Book-Crossing). Incluye carga de datos desde CSV, generacion de recomendaciones, evaluacion con MAE/RMSE y comparacion grafica entre metodos.

## Estructura del proyecto
```
projectePA/
  datasets.py
  evaluation.py
  factories.py
  fase.py
  logging_utils.py
  main.py
  recommenders.py
  dataset/
    Books/
      Books.csv
      Ratings.csv
      Users.csv
    MovieLens100k/
      links.csv
      movies.csv
      ratings.csv
      tags.csv
    cache/
      (caches .pkl generados automaticamente)
  logs/
    log_YYYYMMDD-HHMMSS.txt
```

## Archivos y contenido

### datasets.py
- `Dataset` (abstracta): capa base de datos. Gestiona items, usuarios, ratings, rangos de rating y utilidades de consulta.
- `MovieLensDataset`: carga items y ratings desde `dataset/MovieLens100k`.
- `BooksDataset`: carga items, usuarios y ratings desde `dataset/Books`, con limite de libros configurable.

### recommenders.py
- `Recommender` (abstracta): interfaz comun para preparar, recomendar y predecir ratings. Gestiona rutas de cache.
- `SimpleRecommender`: popularidad con promedio bayesiano (usa `min_votes`).
- `CollaborativeRecommender`: filtrado colaborativo usuario-usuario con similitud coseno (media centrada).
- `ContentBasedRecommender`: TF-IDF sobre contenido de items (generos o autor) y perfil de usuario ponderado por rating.
- Utilidades internas: `_load_pickle_cache`, `_save_pickle_cache` para caches atomicos.

### evaluation.py
- `mae`, `rmse`: metricas de error.
- `evaluate_user`: evalua un usuario con MAE/RMSE usando `predict_rating`.
- `plot_evaluation`: grafico comparativo con matplotlib.

### factories.py
- `Controller`: clase con metodos `build_dataset` y `build_recommender`.

### logging_utils.py
- `build_logger`: logger con salida a consola y fichero en `logs/`.

### main.py
- CLI principal. Valida argumentos, carga dataset/recommender, y ejecuta un bucle interactivo con acciones: recomendar, evaluar, comparar.

### fase.py
- Punto de entrada alternativo que ejecuta `main()`.

### dataset/
- Datos CSV de MovieLens100k y Book-Crossing.
- `cache/`: se generan caches `.pkl` para acelerar el preprocesado de los recomendadores.

### logs/
- Logs con timestamp generados en cada ejecucion.

## Diagrama de clases (completo)

```mermaid
classDiagram
  class Dataset {
    <<abstract>>
    -str _project_root
    -str _dataset_name
    -dict _items
    -set _known_users
    -dict _user_ratings
    -dict _item_ratings
    -float _min_rating
    -float _max_rating
    +get_cache_key() str
    +load() void
    +format_item_for_display(item_id) str
    +get_item_content_text(item_id) str
    +get_user_ids() List~str~
    +get_item_ids() List~str~
    +get_user_ratings(user_id) Dict~str,float~
    +get_item_user_ratings(item_id) Dict~str,float~
    +get_item_average(item_id) float
    +get_user_average(user_id) float
    +get_rating(user_id,item_id) float
    +get_rating_bounds() Tuple~float,float~
  }

  class MovieLensDataset {
    +get_cache_key() str
    +format_item_for_display(item_id) str
    +get_item_content_text(item_id) str
  }

  class BooksDataset {
    +get_cache_key() str
    +format_item_for_display(item_id) str
    +get_item_content_text(item_id) str
  }

  Dataset <|-- MovieLensDataset : extends
  Dataset <|-- BooksDataset : extends

  class Controller {
    -logging.Logger _logger
    -Dataset _dataset
    -Recommender _recommender
    +build_dataset(dataset_key, project_root) Dataset
    +build_recommender(method_key) Recommender
  }

  class Recommender {
    <<abstract>>
    -Dataset _dataset
    -bool _is_prepared
    +prepare() void
    +recommend(user_id, top_n) List~Tuple~str,float~~
    +predict_rating(user_id, item_id) float
  }

  class SimpleRecommender {
    -int _min_votes
    -dict _item_scores
    +prepare() void
    +recommend(user_id, top_n) List~Tuple~str,float~~
    +predict_rating(user_id, item_id) float
  }

  class CollaborativeRecommender {
    -int _k
    -dict _user_means
    -dict _neighbors_cache
    +prepare() void
    +recommend(user_id, top_n) List~Tuple~str,float~~
    +predict_rating(user_id, item_id) float
  }

  class ContentBasedRecommender {
    -tfidf_matrix _tfidf_matrix
    -TfidfVectorizer _vectorizer
    -dict _item_index
    -list _index_item
    +prepare() void
    +recommend(user_id, top_n) List~Tuple~str,float~~
    +predict_rating(user_id, item_id) float
  }

  Recommender <|-- SimpleRecommender : extends
  Recommender <|-- CollaborativeRecommender : extends
  Recommender <|-- ContentBasedRecommender : extends

  Recommender --> Dataset : 1
  Controller "1" --> "1" Dataset
  Controller "1" --> "1" Recommender
```

## Diagramas de secuencia (6 casos)

### Books + Simple (primero)

```mermaid
sequenceDiagram
  autonumber
  actor Usuario
  participant Main as main.py
  participant Logger as logging_utils.build_logger
  participant Controller as factories.Controller
  participant Dataset as BooksDataset
  participant Recommender as SimpleRecommender
  participant Cache as dataset/cache (pickle)

  Usuario->>Main: Ejecuta `python main.py books simple`
  Main->>Logger: build_logger(log_dir)
  Logger-->>Main: logger
  Main->>Controller: build_dataset("books", project_root)
  Controller->>Dataset: BooksDataset(...)
  Dataset-->>Controller: dataset
  Controller->>Dataset: load() = _load_items/_load_users/_load_ratings
  Dataset-->>Controller: items/usuarios/ratings cargados
  Controller-->>Main: dataset
  Main->>Controller: build_recommender("simple")
  Controller->>Recommender: SimpleRecommender(dataset)
  Recommender-->>Controller: recommender
  Controller->>Recommender: prepare()
  Recommender->>Cache: _load_pickle_cache(...)
  alt cache HIT
    Cache-->>Recommender: item_scores
  else cache MISS
    Recommender->>Dataset: get_item_ids/get_item_user_ratings
    Dataset-->>Recommender: items/ratings
    Recommender->>Cache: _save_pickle_cache(...)
    Cache-->>Recommender: ok
  end
  Recommender-->>Controller: preparado
  Controller-->>Main: recommender
  Usuario->>Main: Accion Recomendar (user_id)
  Main->>Recommender: recommend(user_id, top_n)
  Recommender->>Dataset: get_user_rated_items/get_item_average
  Dataset-->>Recommender: ratings/medias
  Recommender-->>Main: [(item_id, score)]
  Main-->>Usuario: Lista de recomendaciones
```

### Books + Collaborative

```mermaid
sequenceDiagram
  autonumber
  actor Usuario
  participant Main as main.py
  participant Logger as logging_utils.build_logger
  participant Controller as factories.Controller
  participant Dataset as BooksDataset
  participant Recommender as CollaborativeRecommender
  participant Cache as dataset/cache (pickle)

  Usuario->>Main: Ejecuta `python main.py books collaborative`
  Main->>Logger: build_logger(log_dir)
  Logger-->>Main: logger
  Main->>Controller: build_dataset("books", project_root)
  Controller->>Dataset: BooksDataset(...)
  Dataset-->>Controller: dataset
  Controller->>Dataset: load() = _load_items/_load_users/_load_ratings
  Dataset-->>Controller: items/usuarios/ratings cargados
  Controller-->>Main: dataset
  Main->>Controller: build_recommender("collaborative")
  Controller->>Recommender: CollaborativeRecommender(dataset)
  Recommender-->>Controller: recommender
  Controller->>Recommender: prepare()
  Recommender->>Cache: _load_pickle_cache(...)
  alt cache HIT
    Cache-->>Recommender: user_means, neighbors
  else cache MISS
    Recommender->>Dataset: get_user_ids/get_item_ids/get_item_user_ratings
    Dataset-->>Recommender: users/items/ratings
    Recommender->>Cache: _save_pickle_cache(...)
    Cache-->>Recommender: ok
  end
  Recommender-->>Controller: preparado
  Controller-->>Main: recommender
  Usuario->>Main: Accion Recomendar (user_id)
  Main->>Recommender: recommend(user_id, top_n)
  Recommender->>Dataset: get_user_rated_items/get_rating/get_rating_bounds
  Dataset-->>Recommender: ratings/rango
  Recommender-->>Main: [(item_id, score)]
  Main-->>Usuario: Lista de recomendaciones
```

### Books + Content

```mermaid
sequenceDiagram
  autonumber
  actor Usuario
  participant Main as main.py
  participant Logger as logging_utils.build_logger
  participant Controller as factories.Controller
  participant Dataset as BooksDataset
  participant Recommender as ContentBasedRecommender
  participant Cache as dataset/cache (pickle)

  Usuario->>Main: Ejecuta `python main.py books content`
  Main->>Logger: build_logger(log_dir)
  Logger-->>Main: logger
  Main->>Controller: build_dataset("books", project_root)
  Controller->>Dataset: BooksDataset(...)
  Dataset-->>Controller: dataset
  Controller->>Dataset: load() = _load_items/_load_users/_load_ratings
  Dataset-->>Controller: items/usuarios/ratings cargados
  Controller-->>Main: dataset
  Main->>Controller: build_recommender("content")
  Controller->>Recommender: ContentBasedRecommender(dataset)
  Recommender-->>Controller: recommender
  Controller->>Recommender: prepare()
  Recommender->>Cache: _load_pickle_cache(...)
  alt cache HIT
    Cache-->>Recommender: vectorizer, tfidf_matrix
  else cache MISS
    Recommender->>Dataset: get_item_ids/get_item_content_text
    Dataset-->>Recommender: items/textos
    Recommender->>Cache: _save_pickle_cache(...)
    Cache-->>Recommender: ok
  end
  Recommender-->>Controller: preparado
  Controller-->>Main: recommender
  Usuario->>Main: Accion Recomendar (user_id)
  Main->>Recommender: recommend(user_id, top_n)
  Recommender->>Dataset: get_user_ratings/get_rating_bounds
  Dataset-->>Recommender: ratings/rango
  Recommender-->>Main: [(item_id, score)]
  Main-->>Usuario: Lista de recomendaciones
```

### Movies + Simple

```mermaid
sequenceDiagram
  autonumber
  actor Usuario
  participant Main as main.py
  participant Logger as logging_utils.build_logger
  participant Controller as factories.Controller
  participant Dataset as MovieLensDataset
  participant Recommender as SimpleRecommender
  participant Cache as dataset/cache (pickle)

  Usuario->>Main: Ejecuta `python main.py movies simple`
  Main->>Logger: build_logger(log_dir)
  Logger-->>Main: logger
  Main->>Controller: build_dataset("movies", project_root)
  Controller->>Dataset: MovieLensDataset(...)
  Dataset-->>Controller: dataset
  Controller->>Dataset: load() = _load_items/_load_ratings
  Dataset-->>Controller: items/ratings cargados
  Controller-->>Main: dataset
  Main->>Controller: build_recommender("simple")
  Controller->>Recommender: SimpleRecommender(dataset)
  Recommender-->>Controller: recommender
  Controller->>Recommender: prepare()
  Recommender->>Cache: _load_pickle_cache(...)
  alt cache HIT
    Cache-->>Recommender: item_scores
  else cache MISS
    Recommender->>Dataset: get_item_ids/get_item_user_ratings
    Dataset-->>Recommender: items/ratings
    Recommender->>Cache: _save_pickle_cache(...)
    Cache-->>Recommender: ok
  end
  Recommender-->>Controller: preparado
  Controller-->>Main: recommender
  Usuario->>Main: Accion Recomendar (user_id)
  Main->>Recommender: recommend(user_id, top_n)
  Recommender->>Dataset: get_user_rated_items/get_item_average
  Dataset-->>Recommender: ratings/medias
  Recommender-->>Main: [(item_id, score)]
  Main-->>Usuario: Lista de recomendaciones
```

### Movies + Collaborative

```mermaid
sequenceDiagram
  autonumber
  actor Usuario
  participant Main as main.py
  participant Logger as logging_utils.build_logger
  participant Controller as factories.Controller
  participant Dataset as MovieLensDataset
  participant Recommender as CollaborativeRecommender
  participant Cache as dataset/cache (pickle)

  Usuario->>Main: Ejecuta `python main.py movies collaborative`
  Main->>Logger: build_logger(log_dir)
  Logger-->>Main: logger
  Main->>Controller: build_dataset("movies", project_root)
  Controller->>Dataset: MovieLensDataset(...)
  Dataset-->>Controller: dataset
  Controller->>Dataset: load() = _load_items/_load_ratings
  Dataset-->>Controller: items/ratings cargados
  Controller-->>Main: dataset
  Main->>Controller: build_recommender("collaborative")
  Controller->>Recommender: CollaborativeRecommender(dataset)
  Recommender-->>Controller: recommender
  Controller->>Recommender: prepare()
  Recommender->>Cache: _load_pickle_cache(...)
  alt cache HIT
    Cache-->>Recommender: user_means, neighbors
  else cache MISS
    Recommender->>Dataset: get_user_ids/get_item_ids/get_item_user_ratings
    Dataset-->>Recommender: users/items/ratings
    Recommender->>Cache: _save_pickle_cache(...)
    Cache-->>Recommender: ok
  end
  Recommender-->>Controller: preparado
  Controller-->>Main: recommender
  Usuario->>Main: Accion Recomendar (user_id)
  Main->>Recommender: recommend(user_id, top_n)
  Recommender->>Dataset: get_user_rated_items/get_rating/get_rating_bounds
  Dataset-->>Recommender: ratings/rango
  Recommender-->>Main: [(item_id, score)]
  Main-->>Usuario: Lista de recomendaciones
```

### Movies + Content

```mermaid
sequenceDiagram
  autonumber
  actor Usuario
  participant Main as main.py
  participant Logger as logging_utils.build_logger
  participant Controller as factories.Controller
  participant Dataset as MovieLensDataset
  participant Recommender as ContentBasedRecommender
  participant Cache as dataset/cache (pickle)

  Usuario->>Main: Ejecuta `python main.py movies content`
  Main->>Logger: build_logger(log_dir)
  Logger-->>Main: logger
  Main->>Controller: build_dataset("movies", project_root)
  Controller->>Dataset: MovieLensDataset(...)
  Dataset-->>Controller: dataset
  Controller->>Dataset: load() = _load_items/_load_ratings
  Dataset-->>Controller: items/ratings cargados
  Controller-->>Main: dataset
  Main->>Controller: build_recommender("content")
  Controller->>Recommender: ContentBasedRecommender(dataset)
  Recommender-->>Controller: recommender
  Controller->>Recommender: prepare()
  Recommender->>Cache: _load_pickle_cache(...)
  alt cache HIT
    Cache-->>Recommender: vectorizer, tfidf_matrix
  else cache MISS
    Recommender->>Dataset: get_item_ids/get_item_content_text
    Dataset-->>Recommender: items/textos
    Recommender->>Cache: _save_pickle_cache(...)
    Cache-->>Recommender: ok
  end
  Recommender-->>Controller: preparado
  Controller-->>Main: recommender
  Usuario->>Main: Accion Recomendar (user_id)
  Main->>Recommender: recommend(user_id, top_n)
  Recommender->>Dataset: get_user_ratings/get_rating_bounds
  Dataset-->>Recommender: ratings/rango
  Recommender-->>Main: [(item_id, score)]
  Main-->>Usuario: Lista de recomendaciones
```

## Requisitos
- Python 3
- Dependencias principales: `numpy`, `scikit-learn`, `matplotlib`

Ejemplo de instalacion:
```
pip install numpy scikit-learn matplotlib
```

## Uso

### Ejecutar el programa
```
python main.py <dataset> <metodo>
```

- `<dataset>`: `movies` | `books`
- `<metodo>`: `simple` | `collaborative` | `content`

Ejemplo:
```
python main.py movies collaborative
```

### Flujo interactivo
1. Introducir un `user_id` valido.
2. Elegir accion:
   - Recomendar: muestra top-N recomendaciones.
   - Evaluar: muestra MAE/RMSE para ese usuario.
   - Comparar: evalua los tres metodos y genera grafico.

## Notas
- Los caches se guardan en `dataset/cache` y aceleran ejecuciones posteriores.
- Los logs se escriben en `logs/` con nombre `log_YYYYMMDD-HHMMSS.txt`.
