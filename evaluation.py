from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from datasets import Dataset
from recommenders import Recommender


def mae(predictions: List[float], actuals: List[float]) -> float:
    """Compute Mean Absolute Error between predicted and actual ratings.

    Parameters
    ----------
    predictions : list of float
        Predicted rating values.
    actuals : list of float
        Corresponding ground-truth rating values.

    Returns
    -------
    float
        MAE value.  Returns ``float('nan')`` when the input lists are empty
        or have mismatched lengths.

    Examples
    --------
    >>> mae([4.0, 3.5, 2.0], [4.5, 3.0, 2.5])
    0.5
    """
    if not predictions or len(predictions) != len(actuals):
        return float("nan")
    return sum(abs(p - a) for p, a in zip(predictions, actuals)) / len(predictions)


def rmse(predictions: List[float], actuals: List[float]) -> float:
    """Compute Root Mean Square Error between predicted and actual ratings.

    Parameters
    ----------
    predictions : list of float
        Predicted rating values.
    actuals : list of float
        Corresponding ground-truth rating values.

    Returns
    -------
    float
        RMSE value.  Returns ``float('nan')`` when the input lists are empty
        or have mismatched lengths.

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
    """Evaluate recommender accuracy for a single user.

    For every item the user has actually rated, the recommender is asked for a
    prediction.  MAE and RMSE are then computed over the items for which a
    prediction was available.

    Parameters
    ----------
    recommender : Recommender
        Trained recommender instance.
    dataset : Dataset
        Dataset that contains the ground-truth ratings.
    user_id : str
        Target user identifier.

    Returns
    -------
    mae_score : float or None
        Mean Absolute Error, or ``None`` when no predictions could be made.
    rmse_score : float or None
        Root Mean Square Error, or ``None`` when no predictions could be made.
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
    """Display a grouped bar chart comparing MAE and RMSE across recommenders.

    Parameters
    ----------
    mae_dict : dict[str, float]
        Mapping from recommender name to its Mean Absolute Error.
    rmse_dict : dict[str, float]
        Mapping from recommender name to its Root Mean Square Error.
        Keys should match those in *mae_dict*.

    Notes
    -----
    The function calls ``plt.show()`` so the chart opens in a window (or is
    rendered inline when running inside a Jupyter notebook).

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
