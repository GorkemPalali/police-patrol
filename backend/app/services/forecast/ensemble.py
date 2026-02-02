"""Model ensemble service - combines KDE baseline and spatiotemporal model."""
from typing import Dict, List, Optional

import numpy as np


def _pad_scores(scores: List[float], target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.array([])
    if not scores:
        return np.zeros(target_len)
    arr = np.array(scores, dtype=float)
    if len(arr) >= target_len:
        return arr[:target_len]
    pad_value = float(arr[-1])
    pad = np.full(target_len - len(arr), pad_value)
    return np.concatenate([arr, pad])


def ensemble_forecast(
    kde_scores: List[float],
    spatiotemporal_scores: List[float],
    weights: Optional[Dict[str, float]] = None,
) -> List[float]:
    """
    Combine forecasts from KDE baseline and spatiotemporal model using weighted voting.

    Args:
        kde_scores: Risk scores from KDE baseline
        spatiotemporal_scores: Risk scores from spatiotemporal model
        weights: Model weights (default: KDE 0.4, spatiotemporal 0.6)

    Returns:
        Ensemble forecasted risk scores
    """
    if weights is None:
        weights = {"kde": 0.4, "spatiotemporal": 0.6}

    target_len = max(len(kde_scores), len(spatiotemporal_scores))
    kde_forecast = _pad_scores(kde_scores, target_len)
    ml_forecast = _pad_scores(spatiotemporal_scores, target_len)

    ensemble = weights["kde"] * kde_forecast + weights["spatiotemporal"] * ml_forecast
    return ensemble.tolist()
