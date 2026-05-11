from __future__ import annotations

import math
from typing import List, Optional, Tuple

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
