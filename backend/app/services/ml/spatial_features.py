"""Spatial-temporal feature engineering utilities."""
from datetime import datetime
from typing import Optional, Tuple
import numpy as np


def create_spatial_features(
    coordinates: np.ndarray,
    event_counts: np.ndarray,
    grid_size: int = 10,
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> np.ndarray:
    """
    Create spatial feature matrix from coordinates and event counts.

    Args:
        coordinates: Array of (lat, lng) coordinates
        event_counts: Array of event counts per location
        grid_size: Size of spatial grid
        bounds: Optional (lat_min, lat_max, lng_min, lng_max) for normalization

    Returns:
        Spatial feature matrix
    """
    if coordinates.size == 0:
        return np.zeros((0, grid_size * grid_size))

    if np.isscalar(event_counts):
        event_counts = np.full(len(coordinates), float(event_counts))
    elif len(event_counts) != len(coordinates):
        event_counts = np.ones(len(coordinates))

    # Normalize coordinates to [0, 1]
    if bounds:
        lat_min, lat_max, lng_min, lng_max = bounds
        mins = np.array([lat_min, lng_min])
        maxs = np.array([lat_max, lng_max])
    else:
        mins = coordinates.min(axis=0)
        maxs = coordinates.max(axis=0)

    coords_norm = (coordinates - mins) / (maxs - mins + 1e-8)
    coords_norm = np.clip(coords_norm, 0.0, 1.0 - 1e-8)

    features = np.zeros((len(coordinates), grid_size * grid_size))
    for i, (lat, lng) in enumerate(coords_norm):
        grid_x = int(lat * grid_size)
        grid_y = int(lng * grid_size)
        grid_idx = min(grid_x * grid_size + grid_y, grid_size * grid_size - 1)
        features[i, grid_idx] = event_counts[i]

    return features


def create_temporal_features(timestamps: np.ndarray) -> np.ndarray:
    """
    Create temporal feature matrix from timestamps.

    Args:
        timestamps: Array of datetime objects or timestamps

    Returns:
        Temporal feature matrix (hour, day_of_week cyclical encoding)
    """
    features = []

    for ts in timestamps:
        if hasattr(ts, 'hour'):
            hour = ts.hour
            day_of_week = ts.weekday()
        else:
            dt = datetime.fromtimestamp(ts)
            hour = dt.hour
            day_of_week = dt.weekday()

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)

        features.append([hour_sin, hour_cos, day_sin, day_cos])

    return np.array(features)


def temporal_feature_vector(dt: datetime) -> np.ndarray:
    """Return a single-row temporal feature vector compatible with the model."""
    return create_temporal_features(np.array([dt]))
