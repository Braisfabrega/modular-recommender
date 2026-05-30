from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from datasets import Dataset
from recommenders import Recommender


def mae(predictions: List[float], actuals: List[float]) -> float:
    """Calcula l'Error Absolut Mitjà (MAE) entre les valoracions predites i les reals.

    Parameters
    ----------
    predictions : list of float
        Valors de les valoracions predites pel sistema.
    actuals : list of float
        Valors reals (ground-truth) de les valoracions corresponents.

    Returns
    -------
    float
        El valor del MAE. Retorna ``float('nan')`` quan les llistes d'entrada
        estan buides o tenen longituds diferents.

    Examples
    --------
    >>> mae([4.0, 3.5, 2.0], [4.5, 3.0, 2.5])
    0.5
    """
    if not predictions or len(predictions) != len(actuals):
        return float("nan")
    return sum(abs(p - a) for p, a in zip(predictions, actuals)) / len(predictions)


def rmse(predictions: List[float], actuals: List[float]) -> float:
    """Calcula l'Arrel de l'Error Quadràtic Mitjà (RMSE) entre les valoracions predites i les reals.

    Parameters
    ----------
    predictions : list of float
        Valors de les valoracions predites pel sistema.
    actuals : list of float
        Valors reals (ground-truth) de les valoracions corresponents.

    Returns
    -------
    float
        El valor del RMSE. Retorna ``float('nan')`` quan les llistes d'entrada
        estan buides o tenen longituds diferents.

    Examples
    --------
    >>> rmse([4.0, 3.5, 2.0], [4.5, 3.0, 2.5])
    0.5
    """
    if not predictions or len(predictions) != len(actuals):
        return float("nan")
    mse = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / len(predictions)
    return math.sqrt(mse)


def evaluate_user(
    recommender: Recommender,
    dataset: Dataset,
    user_id: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Avalua la precisió del recomanador per a un únic usuari concret.

    Per a cada ítem que l'usuari ha valorat realment, es demana una predicció al
    recomanador. El MAE i el RMSE es calculen finalment sobre el conjunt d'ítems
    per als quals s'ha pogut obtenir una predicció vàlida.

    Parameters
    ----------
    recommender : Recommender
        Instància entrenada del sistema de recomanació.
    dataset : Dataset
        Conjunt de dades que conté les valoracions reals.
    user_id : str
        Identificador de l'usuari objectiu.

    Returns
    -------
    mae_score : float or None
        L'Error Absolut Mitjà, o ``None`` si no s'ha pogut realitzar cap predicció.
    rmse_score : float or None
        L'Arrel de l'Error Quadràtic Mitjà, o ``None`` si no s'ha pogut realitzar cap predicció.
    """
    actual_ratings = dataset.get_user_ratings(user_id)
    if not actual_ratings:
        return None, None

    preds: List[float] = []
    acts: List[float] = []

    for item_id, actual in actual_ratings.items():
        predicted = recommender.predict_rating(user_id, item_id)
        if predicted is not None:
            preds.append(predicted)
            acts.append(actual)

    if not preds:
        return None, None

    return mae(preds, acts), rmse(preds, acts)


def print_evaluation(
    user_id: str,
    mae_dict: Dict[str, float],
    rmse_dict: Dict[str, float],
) -> None:
    """Mostra per consola una taula comparativa de les mètriques MAE i RMSE entre recomanadors.

    Parameters
    ----------
    user_id : str
        Identificador de l'usuari avaluat.
    mae_dict : dict[str, float]
        Diccionari que mapeja el nom d'un recomanador amb el seu Error Absolut Mitjà.
    rmse_dict : dict[str, float]
        Diccionari que mapeja el nom d'un recomanador amb el seu Arrel de l'Error Quadràtic Mitjà.
        Les claus han de coincidir exactament amb les de *mae_dict*.

    Examples
    --------
    >>> print_evaluation("1", {"SVD": 0.72, "KNN": 0.85}, {"SVD": 0.91, "KNN": 1.02})
    """
    print(f"\nResultats de comparacio per l'usuari {user_id}:")
    print(f"  {'Metode':<15} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8}")
    for name in mae_dict:
        print(f"  {name:<15} {mae_dict[name]:>8.4f} {rmse_dict[name]:>8.4f}")
