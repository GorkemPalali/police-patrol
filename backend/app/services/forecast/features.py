"""Feature engineering for temporal and spatial features"""
import numpy as np
from datetime import datetime
from typing import Dict


def temporal_features(dt: datetime) -> Dict[str, float]:
    """
    Extract temporal features from datetime.
    Returns sin/cos encoded hour, day of week, and flags.
    """
    hour = dt.hour
    day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
    
    # Cyclical encoding for hour (0-23)
    hour_rad = 2 * np.pi * hour / 24
    hour_sin = np.sin(hour_rad)
    hour_cos = np.cos(hour_rad)
    
    # Cyclical encoding for day of week (0-6)
    day_rad = 2 * np.pi * day_of_week / 7
    day_sin = np.sin(day_rad)
    day_cos = np.cos(day_rad)
    
    # Weekend flag
    is_weekend = 1.0 if day_of_week >= 5 else 0.0
    
    # Time of day categories
    is_night = 1.0 if 22 <= hour or hour < 6 else 0.0
    is_day = 1.0 if 6 <= hour < 18 else 0.0
    is_evening = 1.0 if 18 <= hour < 22 else 0.0
    
    return {
        "hour_sin": float(hour_sin),
        "hour_cos": float(hour_cos),
        "day_sin": float(day_sin),
        "day_cos": float(day_cos),
        "day_of_week": float(day_of_week),
        "is_weekend": is_weekend,
        "is_night": is_night,
        "is_day": is_day,
        "is_evening": is_evening,
    }


def calculate_local_density(
    events: list,
    center_lat: float,
    center_lng: float,
    radius_m: float = 500.0
) -> float:
    """
    Calculate local event density around a point.
    Uses simple distance-based counting.
    """
    from sqlalchemy import text
    from app.db.session import SessionLocal
    
    db = SessionLocal()
    try:
        # Count events within radius using PostGIS
        result = db.execute(
            text("""
                SELECT COUNT(*) as count
                FROM crime_event
                WHERE ST_DWithin(
                    geom,
                    ST_GeogFromText('POINT(:lng :lat)'),
                    :radius
                )
            """),
            {
                "lat": center_lat,
                "lng": center_lng,
                "radius": radius_m
            }
        ).scalar()
        
        # Density = count / area (approximately)
        area_sq_m = np.pi * (radius_m ** 2)
        density = result / area_sq_m if area_sq_m > 0 else 0.0
        
        return float(density)
    finally:
        db.close()



