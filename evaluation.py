from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

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


def plot_evaluation(
    mae_dict: Dict[str, float],
    rmse_dict: Dict[str, float],
) -> None:
    """Mostra un gràfic de barres agrupades comparant les mètriques MAE i RMSE entre recomanadors.

    Parameters
    ----------
    mae_dict : dict[str, float]
        Diccionari que mapeja el nom d'un recomanador amb el seu Error Absolut Mitjà.
    rmse_dict : dict[str, float]
        Diccionari que mapeja el nom d'un recomanador com el seu Arrel de l'Error Quadràtic Mitjà.
        Les claus d'aquest diccionari han de coincidir exactament amb les de *mae_dict*.

    Notes
    -----
    La funció crida internament a ``plt.show()`` per obrir el gràfic en una finestra independent
    (o renderitzar-lo directament en línia si s'executa en un entorn de Jupyter Notebook).

    Examples
    --------
    >>> plot_evaluation({"SVD": 0.72, "KNN": 0.85}, {"SVD": 0.91, "KNN": 1.02})
    """
    labels = list(mae_dict.keys())
    mae_values = [mae_dict[k] for k in labels]
    rmse_values = [rmse_dict[k] for k in labels]

    x = range(len(labels))
    bar_width = 0.35

    _, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))

    bars_mae = ax.bar(
        [i - bar_width / 2 for i in x],
        mae_values,
        width=bar_width,
        label="MAE",
        color="#4C72B0",
    )
    bars_rmse = ax.bar(
        [i + bar_width / 2 for i in x],
        rmse_values,
        width=bar_width,
        label="RMSE",
        color="#DD8452",
    )

    # Annotate each bar with its numeric value
    for bar in bars_mae:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars_rmse:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Error", fontsize=11)
    ax.set_title("Evaluation: MAE and RMSE by Recommender", fontsize=13)
    ax.legend()
    ax.set_ylim(0, max(rmse_values + mae_values) * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
