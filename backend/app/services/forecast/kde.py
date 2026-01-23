"""Adaptive Kernel Density Estimation for spatial risk analysis"""
import numpy as np
from typing import List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.crime_event import CrimeEvent
from app.core.config import get_settings

settings = get_settings()


def compute_adaptive_bandwidth(
    db: Session,
    center_lat: float,
    center_lng: float,
    local_density: float,
    min_bandwidth: float = None,
    max_bandwidth: float = None
) -> float:
    """
    Compute adaptive bandwidth based on local event density.
    Higher density -> smaller bandwidth (more focused)
    Lower density -> larger bandwidth (more spread)
    """
    min_bw = min_bandwidth or settings.kde_min_bandwidth_m
    max_bw = max_bandwidth or settings.kde_max_bandwidth_m
    
    # Normalize density (assume max density of 0.01 events per sq meter)
    max_density = 0.01
    normalized_density = min(local_density / max_density, 1.0)
    
    # Inverse relationship: high density -> low bandwidth
    bandwidth = max_bw - (normalized_density * (max_bw - min_bw))
    
    return max(min_bw, min(max_bw, bandwidth))


def gaussian_kernel(distance: float, bandwidth: float) -> float:
    """Gaussian kernel function"""
    if bandwidth <= 0:
        return 0.0
    
    # Standard Gaussian kernel
    return (1.0 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((distance / bandwidth) ** 2)
    )


def compute_kde_at_point(
    db: Session,
    lat: float,
    lng: float,
    events: List[CrimeEvent],
    bandwidth: float = None
) -> Tuple[float, float]:
    """
    Compute KDE value at a specific point.
    Returns (density_score, confidence)
    """
    if not events:
        return (0.0, 0.0)
    
    # Calculate local density for adaptive bandwidth
    local_density = len(events) / 1000.0  # Rough estimate
    
    if bandwidth is None:
        bandwidth = compute_adaptive_bandwidth(db, lat, lng, local_density)
    
    # Compute weighted KDE
    total_density = 0.0
    total_weight = 0.0
    
    # Get all events with distances in one query for efficiency
    event_ids = [str(e.id) for e in events]
    if not event_ids:
        return (0.0, 0.0)
    
    # Calculate distances for all events at once
    distances_result = db.execute(
        text("""
            SELECT 
                id,
                ST_Distance(
                    geom,
                    ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography
                ) as distance
            FROM crime_event
            WHERE id = ANY(:event_ids::uuid[])
        """),
        {
            "lat": lat,
            "lng": lng,
            "event_ids": event_ids
        }
    ).fetchall()
    
    # Create distance map
    distance_map = {row.id: float(row.distance) for row in distances_result}
    
    # Calculate weighted KDE
    for event in events:
        distance_m = distance_map.get(event.id, float('inf'))
        
        if distance_m < bandwidth * 3:  # 3-sigma cutoff
            # Weight by severity and confidence
            weight = event.severity * event.confidence_score
            kernel_value = gaussian_kernel(distance_m, bandwidth)
            
            total_density += weight * kernel_value
            total_weight += weight
    
    # Normalize
    if total_weight > 0:
        density_score = total_density / total_weight
    else:
        density_score = 0.0
    
    # Confidence based on number of events and distance
    confidence = min(1.0, len(events) / 10.0) if events else 0.0
    
    return (float(density_score), float(confidence))


def compute_kde_grid(
    db: Session,
    bbox: Tuple[float, float, float, float],
    events: List[CrimeEvent],
    grid_size_m: float = None
) -> List[Tuple[float, float, float, float]]:
    """
    Compute KDE for a grid of points.
    Returns list of (lat, lng, density_score, confidence)
    """
    grid_size = grid_size_m or settings.default_grid_size_m
    
    # Convert grid size from meters to degrees (approximately)
    # 1 degree latitude ≈ 111 km
    lat_step = grid_size / 111000.0
    # Longitude step depends on latitude (use average)
    avg_lat = (bbox[0] + bbox[2]) / 2.0
    lng_step = grid_size / (111000.0 * np.cos(np.radians(avg_lat)))
    
    results = []
    
    # Generate grid points
    lat = bbox[0]
    while lat <= bbox[2]:
        lng = bbox[1]
        while lng <= bbox[3]:
            density, confidence = compute_kde_at_point(db, lat, lng, events)
            results.append((lat, lng, density, confidence))
            lng += lng_step
        lat += lat_step
    
    return results

