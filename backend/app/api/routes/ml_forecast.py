"""ML-based forecast endpoints"""
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.models.crime_event import CrimeEvent
from app.services.forecast.ensemble import ensemble_forecast
from app.services.forecast.spatiotemporal_forecast import (
    predict_spatiotemporal_point,
    predict_spatiotemporal_timeseries,
)
from app.services.utils import get_kucukcekmece_bbox_from_polygon


router = APIRouter(prefix="/ml-forecast", tags=["ML Forecast"])
settings = get_settings()


def _default_location(db: Session) -> Tuple[float, float]:
    bbox = get_kucukcekmece_bbox_from_polygon(db) or settings.kucukcekmece_fallback_bbox
    lat = (bbox[0] + bbox[2]) / 2.0
    lng = (bbox[1] + bbox[3]) / 2.0
    return lat, lng


@router.get("/timeseries")
def get_timeseries_forecast(
    db: Session = Depends(get_db),
    forecast_horizon: int = Query(24, ge=1, le=168, description="Hours to forecast ahead"),
    crime_type: Optional[str] = Query(None, description="Optional crime type filter"),
    lat: Optional[float] = Query(None, description="Latitude (optional)"),
    lng: Optional[float] = Query(None, description="Longitude (optional)"),
):
    """Get time-series forecast using spatiotemporal XGBoost model"""
    if lat is None or lng is None:
        lat, lng = _default_location(db)

    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(hours=forecast_horizon)

    try:
        result = predict_spatiotemporal_timeseries(
            db=db,
            lat=lat,
            lng=lng,
            start_time=start_time,
            end_time=end_time,
            crime_type=crime_type,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Spatiotemporal model artifact not found",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Spatiotemporal forecast failed",
        )

    return {
        "forecast": result["forecast"],
        "horizon_hours": forecast_horizon,
        "historical_points": result["historical_points"],
        "time_bin_hours": result["time_bin_hours"],
        "location": {"lat": lat, "lng": lng},
        "time_window": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
        },
        "model": "spatiotemporal_xgboost",
    }


@router.get("/spatial-temporal")
def get_spatial_temporal_forecast(
    db: Session = Depends(get_db),
    lat: float = Query(..., description="Latitude"),
    lng: float = Query(..., description="Longitude"),
    forecast_time: datetime = Query(..., description="Time to forecast for"),
    crime_type: Optional[str] = Query(None, description="Optional crime type filter"),
):
    """Get spatial-temporal forecast for a specific location and time"""
    try:
        result = predict_spatiotemporal_point(
            db=db,
            lat=lat,
            lng=lng,
            forecast_time=forecast_time,
            crime_type=crime_type,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Spatiotemporal model artifact not found",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Spatiotemporal forecast failed",
        )

    return {
        "forecast": result["forecast"],
        "confidence": result["confidence"],
        "nearby_events": result["nearby_events"],
        "grid_id": result["grid_id"],
        "time_bin_hours": result["time_bin_hours"],
        "model": "spatiotemporal_xgboost",
    }


@router.get("/ensemble")
def get_ensemble_forecast(
    db: Session = Depends(get_db),
    start_time: datetime = Query(..., description="Start of forecast window"),
    end_time: datetime = Query(..., description="End of forecast window"),
    lat: Optional[float] = Query(None, description="Latitude (optional)"),
    lng: Optional[float] = Query(None, description="Longitude (optional)"),
):
    """Get ensemble forecast combining KDE baseline and spatiotemporal model"""
    if end_time <= start_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="end_time must be after start_time",
        )

    if lat is None or lng is None:
        lat, lng = _default_location(db)

    horizon_hours = int((end_time - start_time).total_seconds() / 3600)
    if horizon_hours <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Forecast horizon must be at least 1 hour",
        )

    events = (
        db.query(CrimeEvent)
        .filter(CrimeEvent.event_time <= start_time)
        .order_by(CrimeEvent.event_time.desc())
        .limit(1000)
        .all()
    )
    if not events:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No historical data available",
        )

    risk_scores = [float(event.severity) / 5.0 for event in events]
    kde_scores = risk_scores[:horizon_hours]
    if len(kde_scores) < horizon_hours:
        pad_value = float(np.mean(risk_scores)) if risk_scores else 0.0
        kde_scores.extend([pad_value] * (horizon_hours - len(kde_scores)))

    try:
        ml_result = predict_spatiotemporal_timeseries(
            db=db,
            lat=lat,
            lng=lng,
            start_time=start_time,
            end_time=end_time,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Spatiotemporal model artifact not found",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Spatiotemporal forecast failed",
        )

    forecast = ensemble_forecast(kde_scores, ml_result["forecast"])

    return {
        "forecast": forecast,
        "time_window": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
        },
        "models_used": ["kde", "spatiotemporal"],
    }
