"""ML-based forecast endpoints"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
from pathlib import Path

from app.db.session import get_db
from app.models.crime_event import CrimeEvent
from app.services.ml.sarimax_service import forecast_timeseries
from app.services.ml.spatial_service import forecast_spatial_temporal
from app.services.ml.spatial_features import create_spatial_features, temporal_feature_vector
from app.services.forecast.ensemble import ensemble_forecast
from app.services.forecast.features import temporal_features
import numpy as np

router = APIRouter(prefix="/ml-forecast", tags=["ML Forecast"])

def _load_spatial_model_metadata(model_path: Path):
    if not model_path.exists():
        return None, None
    try:
        import pickle
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
        grid_size = model_dict.get('grid_size')
        bounds = model_dict.get('spatial_bounds')
        if bounds and len(bounds) == 4:
            bounds = tuple(float(x) for x in bounds)
        else:
            bounds = None
        return grid_size, bounds
    except Exception:
        return None, None


@router.get("/timeseries")
def get_timeseries_forecast(
    db: Session = Depends(get_db),
    forecast_horizon: int = Query(24, ge=1, le=168, description="Hours to forecast ahead"),
    crime_type: Optional[str] = Query(None, description="Filter by crime type"),
):
    """Get time-series forecast using SARIMAX model"""
    
    # Get historical data
    query = db.query(CrimeEvent).order_by(CrimeEvent.event_time.desc()).limit(1000)
    if crime_type:
        query = query.filter(CrimeEvent.crime_type == crime_type)
    
    events = query.all()
    
    if not events:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No historical data available for forecasting"
        )
    
    # Prepare historical data
    historical_data = [
        {
            'risk_score': float(event.severity) / 5.0,  # Normalize to 0-1
            'timestamp': event.event_time
        }
        for event in events
    ]
    
    # Forecast
    model_path = Path("ml/models/sarimax_model.pkl")
    forecast = forecast_timeseries(historical_data, forecast_horizon, model_path)
    
    return {
        "forecast": forecast,
        "horizon_hours": forecast_horizon,
        "historical_points": len(historical_data)
    }


@router.get("/spatial-temporal")
def get_spatial_temporal_forecast(
    db: Session = Depends(get_db),
    lat: float = Query(..., description="Latitude"),
    lng: float = Query(..., description="Longitude"),
    forecast_time: datetime = Query(..., description="Time to forecast for"),
):
    """Get spatial-temporal forecast for a specific location and time"""
    
    # Get nearby events
    from sqlalchemy import text
    events = db.execute(
        text("""
            SELECT
                severity,
                event_time,
                ST_Y(geom::geometry) AS lat,
                ST_X(geom::geometry) AS lng
            FROM crime_event
            WHERE ST_DWithin(
                geom,
                ST_GeogFromText('POINT(:lng :lat)'),
                1000
            )
            ORDER BY event_time DESC
            LIMIT 100
        """),
        {"lat": lat, "lng": lng}
    ).fetchall()
    
    if not events:
        return {"forecast": 0.0, "confidence": 0.0}
    
    coordinates = np.array([[row.lat, row.lng] for row in events])
    event_counts = np.array([float(row.severity) / 5.0 for row in events])
    
    model_path = Path("ml/models/spatial_model.pkl")
    grid_size, bounds = _load_spatial_model_metadata(model_path)
    spatial_features = create_spatial_features(
        coordinates,
        event_counts,
        grid_size=int(grid_size) if grid_size else 10,
        bounds=bounds
    )
    
    temporal_features_array = temporal_feature_vector(forecast_time)
    if len(spatial_features) > 1:
        temporal_features_array = np.repeat(temporal_features_array, len(spatial_features), axis=0)
    
    # Forecast
    forecast = forecast_spatial_temporal(
        spatial_features,
        temporal_features_array,
        model_path
    )
    
    return {
        "forecast": float(np.mean(forecast)) if len(forecast) > 0 else 0.0,
        "confidence": min(1.0, len(events) / 10.0),
        "nearby_events": len(events)
    }


@router.get("/ensemble")
def get_ensemble_forecast(
    db: Session = Depends(get_db),
    start_time: datetime = Query(..., description="Start of forecast window"),
    end_time: datetime = Query(..., description="End of forecast window"),
    lat: Optional[float] = Query(None, description="Latitude (optional)"),
    lng: Optional[float] = Query(None, description="Longitude (optional)"),
):
    """Get ensemble forecast combining KDE, SARIMAX, and spatial models"""
    
    # Get historical data
    events = db.query(CrimeEvent).filter(
        CrimeEvent.event_time <= start_time
    ).order_by(CrimeEvent.event_time.desc()).limit(1000).all()
    
    if not events:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No historical data available"
        )
    
    # Prepare data
    historical_data = [
        {
            'risk_score': float(event.severity) / 5.0,
            'timestamp': event.event_time
        }
        for event in events
    ]
    
    # KDE scores (simplified - would use actual KDE service)
    kde_scores = [d['risk_score'] for d in historical_data[:24]]
    
    # Spatial features
    if lat and lng:
        from sqlalchemy import text
        nearby_events = db.execute(
            text("""
                SELECT
                    severity,
                    ST_Y(geom::geometry) AS lat,
                    ST_X(geom::geometry) AS lng
                FROM crime_event
                WHERE ST_DWithin(
                    geom,
                    ST_GeogFromText('POINT(:lng :lat)'),
                    1000
                )
                AND event_time <= :start_time
                ORDER BY event_time DESC
                LIMIT 200
            """),
            {"lat": lat, "lng": lng, "start_time": start_time}
        ).fetchall()
        
        if nearby_events:
            coordinates = np.array([[row.lat, row.lng] for row in nearby_events])
            event_counts = np.array([float(row.severity) / 5.0 for row in nearby_events])
        else:
            coordinates = np.array([[lat, lng]])
            event_counts = np.array([0.0])
        
        model_path = Path("ml/models/spatial_model.pkl")
        grid_size, bounds = _load_spatial_model_metadata(model_path)
        spatial_features = create_spatial_features(
            coordinates,
            event_counts,
            grid_size=int(grid_size) if grid_size else 10,
            bounds=bounds
        )
        
        temporal_features_array = temporal_feature_vector(start_time)
        if len(spatial_features) > 1:
            temporal_features_array = np.repeat(temporal_features_array, len(spatial_features), axis=0)
    else:
        spatial_features = np.array([[0.0, 0.0]])
        temporal_features_array = np.array([[0.0, 0.0, 0.0, 0.0]])
    
    # Ensemble forecast
    forecast = ensemble_forecast(
        kde_scores,
        historical_data,
        spatial_features,
        temporal_features_array
    )
    
    return {
        "forecast": forecast,
        "time_window": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        },
        "models_used": ["kde", "sarimax", "spatial"]
    }

