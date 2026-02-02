from typing import Optional
from pathlib import Path
import numpy as np

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from app.services.ml.spatial_features import create_spatial_features


def _load_spatial_model(model_path: Optional[Path]) -> Optional[dict]:
    if not model_path or not model_path.exists():
        return None
    try:
        import pickle
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _align_feature_dim(features: np.ndarray, target_dim: int) -> np.ndarray:
    if target_dim <= 0:
        return features
    if features.shape[1] == target_dim:
        return features
    if features.shape[1] > target_dim:
        return features[:, :target_dim]
    pad_width = target_dim - features.shape[1]
    return np.pad(features, ((0, 0), (0, pad_width)), mode='constant')


def forecast_spatial_temporal(
    spatial_features: np.ndarray,
    temporal_features: np.ndarray,
    model_path: Optional[Path] = None
) -> np.ndarray:
    """
    Forecast risk using spatial-temporal model.
    
    Args:
        spatial_features: Spatial feature matrix
        temporal_features: Temporal feature matrix
        model_path: Path to pre-trained model (optional)
    
    Returns:
        Forecasted risk scores
    """
    if spatial_features.size == 0:
        return np.array([0.0])
    
    model_dict = _load_spatial_model(model_path)

    # If no model available, use simple spatial average
    if not SKLEARN_AVAILABLE or not model_dict:
        # Simple spatial average with temporal weighting
        spatial_avg = float(np.mean(spatial_features))
        
        # Apply temporal weighting if available
        if temporal_features.size > 0:
            # Use hour of day as weight (higher risk at night)
            if len(temporal_features.shape) > 1:
                hour_sin = temporal_features[:, 0] if temporal_features.shape[1] > 0 else np.array([0.5])
                # Convert sin back to approximate hour (0-23)
                hour_approx = np.arcsin(np.clip(hour_sin, -1, 1)) * 24 / (2 * np.pi)
                # Night hours (22-6) get higher weight
                night_weight = np.where((hour_approx >= 22) | (hour_approx <= 6), 1.2, 1.0)
                return np.array([spatial_avg * np.mean(night_weight)])
        
        return np.array([spatial_avg])
    
    # Use trained model
    try:
        # Combine features
        if len(spatial_features.shape) == 1:
            spatial_features = spatial_features.reshape(1, -1)
        if len(temporal_features.shape) == 1:
            temporal_features = temporal_features.reshape(1, -1)

        # If spatial features are raw coordinates, expand to grid features
        if spatial_features.shape[1] == 2 and model_dict.get('grid_size'):
            spatial_features = create_spatial_features(
                spatial_features,
                np.ones(len(spatial_features)),
                grid_size=int(model_dict.get('grid_size', 10)),
                bounds=tuple(model_dict.get('spatial_bounds', ())) or None
            )

        spatial_features = _align_feature_dim(spatial_features, int(model_dict.get('spatial_dim', 0)))
        temporal_features = _align_feature_dim(temporal_features, int(model_dict.get('temporal_dim', 0)))

        combined = np.hstack([spatial_features, temporal_features])

        # Predict
        predictions = np.dot(combined, model_dict['weights']) + model_dict['intercept']
        return np.clip(predictions, 0.0, 1.0)

    except Exception:
        # Fallback on error
        return np.array([float(np.mean(spatial_features))])
