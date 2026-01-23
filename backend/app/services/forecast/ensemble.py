"""Model ensemble service - combines KDE, SARIMAX, and spatial models"""
from typing import List, Dict
import numpy as np

from app.services.ml.sarimax_service import forecast_timeseries
from app.services.ml.spatial_service import forecast_spatial_temporal

def ensemble_forecast(
    kde_scores: List[float],
    historical_data: List[Dict],
    spatial_features: np.ndarray,
    temporal_features: np.ndarray,
    weights: Dict[str, float] = None
) -> List[float]:
    """
    Combine forecasts from multiple models using weighted voting.
    
    Args:
        kde_scores: Risk scores from KDE
        historical_data: Historical data for SARIMAX
        spatial_features: Spatial features for spatial model
        temporal_features: Temporal features for spatial model
        weights: Model weights (default: equal weights)
    
    Returns:
        Ensemble forecasted risk scores
    """
    if weights is None:
        weights = {"kde": 0.4, "sarimax": 0.3, "spatial": 0.3}
    
    # Get forecasts from each model
    kde_forecast = np.array(kde_scores) if kde_scores else np.array([0.0])
    
    sarimax_forecast = np.array(
        forecast_timeseries(historical_data, len(kde_forecast))
    )
    
    spatial_forecast = forecast_spatial_temporal(spatial_features, temporal_features)
    if len(spatial_forecast) < len(kde_forecast):
        spatial_forecast = np.tile(spatial_forecast, len(kde_forecast))
    spatial_forecast = spatial_forecast[:len(kde_forecast)]
    
    # Weighted combination
    ensemble = (
        weights["kde"] * kde_forecast +
        weights["sarimax"] * sarimax_forecast +
        weights["spatial"] * spatial_forecast
    )
    
    return ensemble.tolist()



