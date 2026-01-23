"""
Spatial-temporal model training (simplified LSTM approach)
Note: Full GNN implementation would require PyTorch Geometric
"""
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import pickle


def create_spatial_features(
    coordinates: np.ndarray,
    event_counts: np.ndarray,
    grid_size: int = 10
) -> np.ndarray:
    """
    Create spatial feature matrix from coordinates and event counts.
    
    Args:
        coordinates: Array of (lat, lng) coordinates
        event_counts: Array of event counts per location
        grid_size: Size of spatial grid
    
    Returns:
        Spatial feature matrix
    """
    # Simple grid-based features
    # In production, this would use more sophisticated spatial encoding
    features = np.zeros((len(coordinates), grid_size * grid_size))
    
    # Normalize coordinates to [0, 1]
    coords_norm = (coordinates - coordinates.min(axis=0)) / (coordinates.max(axis=0) - coordinates.min(axis=0) + 1e-8)
    
    # Assign to grid cells
    for i, (lat, lng) in enumerate(coords_norm):
        grid_x = int(lat * grid_size)
        grid_y = int(lng * grid_size)
        grid_idx = min(grid_x * grid_size + grid_y, grid_size * grid_size - 1)
        features[i, grid_idx] = event_counts[i]
    
    return features


def create_temporal_features(
    timestamps: np.ndarray
) -> np.ndarray:
    """
    Create temporal feature matrix from timestamps.
    
    Args:
        timestamps: Array of datetime objects or timestamps
    
    Returns:
        Temporal feature matrix (hour, day_of_week, etc.)
    """
    features = []
    
    for ts in timestamps:
        if hasattr(ts, 'hour'):
            hour = ts.hour
            day_of_week = ts.weekday()
        else:
            # Assume it's a timestamp
            from datetime import datetime
            dt = datetime.fromtimestamp(ts)
            hour = dt.hour
            day_of_week = dt.weekday()
        
        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        features.append([hour_sin, hour_cos, day_sin, day_cos])
    
    return np.array(features)


def train_simple_spatial_model(
    spatial_features: np.ndarray,
    temporal_features: np.ndarray,
    target_values: np.ndarray,
    output_path: Optional[Path] = None
) -> dict:
    """
    Train a simple spatial-temporal model.
    
    Args:
        spatial_features: Spatial feature matrix
        temporal_features: Temporal feature matrix
        target_values: Target risk scores
        output_path: Path to save model
    
    Returns:
        Simple model (dictionary with weights)
    """
    # Simple linear combination as placeholder
    # In production, this would be a trained neural network
    
    # Combine features
    combined_features = np.hstack([spatial_features, temporal_features])
    
    # Simple linear regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(combined_features, target_values)
    
    model_dict = {
        'type': 'simple_linear',
        'weights': model.coef_,
        'intercept': model.intercept_,
        'spatial_dim': spatial_features.shape[1],
        'temporal_dim': temporal_features.shape[1]
    }
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(model_dict, f)
    
    return model_dict


def load_spatial_model(model_path: Path) -> dict:
    """Load trained spatial model from file."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)
