from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Union
import numpy as np

from app.db.session import get_db
from app.models.risk_cell import RiskCell
from app.models.crime_event import CrimeEvent
from app.schemas.risk_cell import RiskMapResponse
from app.services.forecast.risk_cells import generate_risk_cells
from app.services.forecast.features import temporal_features
from app.services.ml.sarimax_service import forecast_timeseries
from app.core.config import get_settings
import logging

router = APIRouter(prefix="/forecast", tags=["Forecast"])
settings = get_settings()
logger = logging.getLogger(__name__)


@router.get("/risk-map", response_model=RiskMapResponse)
def get_risk_map(
    db: Session = Depends(get_db),
    start_time: datetime = Query(..., description="Start of time window"),
    end_time: datetime = Query(..., description="End of time window"),
    station_id: Optional[str] = Query(None, description="Police station ID - filters by station's neighborhoods"),
    min_lat: Optional[float] = Query(None, description="Bounding box min latitude"),
    min_lng: Optional[float] = Query(None, description="Bounding box min longitude"),
    max_lat: Optional[float] = Query(None, description="Bounding box max latitude"),
    max_lng: Optional[float] = Query(None, description="Bounding box max longitude"),
    grid_size_m: Optional[float] = Query(None, description="Grid cell size in meters"),
    threshold: Optional[float] = Query(0.0, ge=0.0, le=1.0, description="Minimum risk score threshold"),
    use_hex: bool = Query(True, description="Use hexagon grid instead of square"),
):  
    if end_time <= start_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="end_time must be after start_time"
        )
    
    from datetime import timezone
    now = datetime.now(timezone.utc)
    
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    
    is_future = start_time > now
    
    # Determine bbox - get from polygon or use fallback
    from app.services.utils import get_kucukcekmece_bbox_from_polygon
    kucukcekmece_bbox = get_kucukcekmece_bbox_from_polygon(db)
    
    if all([min_lat, min_lng, max_lat, max_lng]):
        # Intersect user bbox with Küçükçekmece bbox
        bbox = (
            max(min_lat, kucukcekmece_bbox[0]),
            max(min_lng, kucukcekmece_bbox[1]),
            min(max_lat, kucukcekmece_bbox[2]),
            min(max_lng, kucukcekmece_bbox[3])
        )
    else:
        bbox = kucukcekmece_bbox
    
    # First, update road segment risks using KDE (road-segment-based forecast)
    # Filter by station's neighborhoods if station_id provided
    from app.services.forecast.road_segment_risk import update_road_segment_risks
    try:
        # Update road segment risks for the forecast time window
        # If station_id provided, only update risks for segments within station's neighborhoods
        risk_update_result = update_road_segment_risks(
            db=db,
            time_window_start=start_time,
            time_window_end=end_time,
            station_id=station_id
        )
        logger.info(f"Road segment risks updated: {risk_update_result}")
    except Exception as update_err:
        logger.warning(f"Failed to update road segment risks: {str(update_err)}")
        # Continue with grid-based fallback
    
    # Generate grid-based risk cells as fallback/backward compatibility
    try:
        risk_cells = generate_forecast_risk_cells(
            db=db,
            time_window_start=start_time,
            time_window_end=end_time,
            bbox=bbox,
            grid_size_m=grid_size_m,
            use_hex=use_hex
        )
        
        # Note: generate_forecast_risk_cells already saves cells to DB, but if it didn't, ensure they're saved
        if risk_cells:
            cell_ids = [c.id for c in risk_cells]
            existing_cells = db.query(RiskCell).filter(RiskCell.id.in_(cell_ids)).all()
            existing_ids = {c.id for c in existing_cells}
            
            new_cells = [c for c in risk_cells if c.id not in existing_ids]
            if new_cells:
                try:
                    db.bulk_save_objects(new_cells)
                    db.commit()
                except Exception as save_err:
                    db.rollback()
                    pass
    except Exception as e:
        db.rollback()
        
        try:
            risk_cells = db.query(RiskCell).filter(
                text("time_window && tsrange(:start_time, :end_time, '[]')")
            ).params(
                start_time=start_time,
                end_time=end_time
            ).all()
        except Exception as query_error:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate or query risk cells: {str(e)}"
            )
        
        if not risk_cells:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No risk cells found for the specified time window. Generation failed: {str(e)}"
            )
    
    # Convert to response format
    risk_cell_data = []
    for cell in risk_cells:
        if cell.risk_score >= threshold:
            # Convert geometry to GeoJSON
            geom_json = None
          
            try:
                geom_json = db.execute(
                    text("""
                        SELECT ST_AsGeoJSON(geom)::json as geom_json
                        FROM risk_cell
                        WHERE id = :cell_id
                    """),
                    {"cell_id": cell.id}
                ).scalar()
            except Exception as geom_err:
                print(f"Warning: Could not query geometry for cell {cell.id}: {geom_err}")
                continue
            
            if not geom_json:
                continue  # Skip if geometry conversion failed
            
            time_start = cell.time_window.lower if cell.time_window else start_time
            time_end = cell.time_window.upper if cell.time_window else end_time
            
            risk_cell_data.append({
                "id": str(cell.id),
                "geom": geom_json,
                "risk_score": cell.risk_score,
                "confidence": cell.confidence,
                "time_window_start": time_start.isoformat() if hasattr(time_start, 'isoformat') else str(time_start),
                "time_window_end": time_end.isoformat() if hasattr(time_end, 'isoformat') else str(time_end),
            })
    
    return RiskMapResponse(
        time_window={
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        },
        risk_cells=risk_cell_data,
        grid_size_m=grid_size_m or settings.default_grid_size_m,
        total_cells=len(risk_cell_data)
    )


def get_geometry_center(geom: dict) -> Optional[Tuple[float, float]]:
    """Extract center point from GeoJSON geometry"""
    if not geom or not geom.get('coordinates'):
        return None
    
    try:
        if geom.get('type') == 'Point':
            return (geom['coordinates'][1], geom['coordinates'][0])  # (lat, lng)
        elif geom.get('type') == 'Polygon' and geom.get('coordinates', [])[0]:
            coords = geom['coordinates'][0]
            sum_lat = sum(c[1] for c in coords)
            sum_lng = sum(c[0] for c in coords)
            return (sum_lat / len(coords), sum_lng / len(coords))
    except Exception:
        pass
    return None


def generate_forecast_risk_cells(
    db: Session,
    time_window_start: datetime,
    time_window_end: datetime,
    bbox: Tuple[float, float, float, float],
    grid_size_m: Optional[float] = None,
    use_hex: bool = True
) -> List[RiskCell]:
    """
    Generate risk cells for future dates using ML forecast with spatial KDE.
    Uses historical data to calculate spatial risk distribution.
    """
    from app.services.forecast.risk_cells import create_square_grid, create_hex_grid
    from psycopg2.extras import DateTimeTZRange, DateTimeRange
    import uuid
    import json
    
    grid_size = grid_size_m or settings.default_grid_size_m
    
    # Being sure bbox is within Küçükçekmece boundaries
    from app.services.utils import get_kucukcekmece_bbox_from_polygon
    kucukcekmece_bbox = get_kucukcekmece_bbox_from_polygon(db)
    bbox = (
        max(bbox[0], kucukcekmece_bbox[0]),
        max(bbox[1], kucukcekmece_bbox[1]),
        min(bbox[2], kucukcekmece_bbox[2]),
        min(bbox[3], kucukcekmece_bbox[3])
    )
    
    try:
        if use_hex:
            grid_cells = create_hex_grid(db, bbox, grid_size)
        else:
            grid_cells = create_square_grid(db, bbox, grid_size)
    except Exception as e:
        grid_cells = create_square_grid(db, bbox, grid_size)
    
    if not grid_cells or len(grid_cells) == 0:
        return []
    
    from datetime import timezone
    now = datetime.now(timezone.utc)
    
    # Always use last 30 days of data for risk calculation
    historical_start = now - timedelta(days=30)
    historical_end = now
    
    # Query events within Küçükçekmece polygon boundary
    # Use the database function for polygon filtering
    historical_events = db.query(CrimeEvent).filter(
        CrimeEvent.event_time >= historical_start,
        CrimeEvent.event_time <= historical_end
    ).filter(
        text("""
            is_within_kucukcekmece(geom)
            AND NOT ST_Within(
                geom::geometry,
                ST_MakeEnvelope(28.75, 41.0, 28.78, 41.02, 4326)
            )
        """)
    ).all()
    
    if not historical_events:
        return []  # No historical data for forecast
    
    # Prepare historical data for SARIMAX
    historical_data = [
        {
            'risk_score': float(event.severity) / 5.0,
            'timestamp': event.event_time
        }
        for event in historical_events
    ]
    
    # Calculate forecast horizon (hours)
    forecast_horizon = int((time_window_end - time_window_start).total_seconds() / 3600)
    if forecast_horizon <= 0:
        forecast_horizon = 24
    
    # Get time series forecast
    sarimax_forecast = forecast_timeseries(historical_data, forecast_horizon)
    avg_forecast_risk = float(np.mean(sarimax_forecast)) if sarimax_forecast and len(sarimax_forecast) > 0 else 0.0
    
    # Normalize forecast risk (0-1 range)
    if avg_forecast_risk > 1.0:
        avg_forecast_risk = 1.0
    elif avg_forecast_risk < 0:
        avg_forecast_risk = 0.0
    
    # Apply temporal weighting based on forecast time
    forecast_time = time_window_start
    temporal_feat = temporal_features(forecast_time)
    
    # Temporal adjustment (night hours have higher risk)
    if temporal_feat.get('is_night', 0) > 0:
        risk_multiplier = 1.3
    elif temporal_feat.get('is_evening', 0) > 0:
        risk_multiplier = 1.1
    else:
        risk_multiplier = 1.0
    
    # Calculate risk for each cell using direct PostGIS queries
    cell_risks = []
    for cell in grid_cells:
        try:
            # Get cell center
            center = get_geometry_center(cell["geom"])
            if not center:
                continue
            
            lat, lng = center
            
            # Skip if in lake area
            if 41.0 <= lat <= 41.02 and 28.75 <= lng <= 28.78:
                continue
            
            # Calculate local event density within 500m using PostGIS
            nearby_result = db.execute(
                text("""
                    SELECT 
                        COUNT(*) as count,
                        AVG(severity::float) as avg_severity,
                        AVG(confidence_score) as avg_confidence
                    FROM crime_event
                    WHERE ST_DWithin(
                        geom,
                        ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography,
                        :radius
                    )
                    AND event_time >= :start_time
                    AND event_time <= :end_time
                """),
                {
                    "lat": lat,
                    "lng": lng,
                    "radius": 500,
                    "start_time": historical_start,
                    "end_time": historical_end
                }
            ).first()
            
            # Calculate local risk based on nearby events
            local_risk = 0.0
            local_confidence = 0.0
            
            if nearby_result and nearby_result.count and nearby_result.count > 0:
                event_count = nearby_result.count
                avg_severity = float(nearby_result.avg_severity) if nearby_result.avg_severity else 0.0
                avg_confidence = float(nearby_result.avg_confidence) if nearby_result.avg_confidence else 0.0
                
                # Risk based on event count and severity
                # Normalize: max 10 events = 1.0 risk
                density_factor = min(1.0, event_count / 10.0)
                severity_factor = avg_severity / 5.0  # Normalize severity 1-5 to 0-1
                
                local_risk = (density_factor * 0.6 + severity_factor * 0.4) * avg_confidence
                local_confidence = min(1.0, event_count / 5.0)
            
            # For past dates, prioritize local risk more; for future dates, use forecast
            if time_window_start <= now:
                # Past date: use 80% local risk, 20% temporal average
                spatial_weight = 0.8
                temporal_weight = 0.2
            else:
                # Future date: use 60% local risk, 40% temporal forecast
                spatial_weight = 0.6
                temporal_weight = 0.4
            
            # Combine temporal forecast with spatial local risk
            base_risk = (
                temporal_weight * avg_forecast_risk +
                spatial_weight * local_risk
            )
            
            # Apply temporal multiplier
            risk_score = min(1.0, max(0.0, base_risk * risk_multiplier))
            
            # Lower threshold to ensure cells are created for visualization
            if risk_score >= 0.01:
                cell_risks.append((cell, risk_score, local_confidence))
        except Exception as e:
            continue
    
    # Normalize risk scores to ensure proper distribution
    if cell_risks:
        risk_scores = [r[1] for r in cell_risks]
        max_risk = max(risk_scores) if risk_scores else 1.0
        min_risk = min(risk_scores) if risk_scores else 0.0
        risk_range = max_risk - min_risk if max_risk > min_risk else 1.0
        
        # Generate risk cells
        # Convert to timezone-naive for TSRANGE
        from datetime import timezone
        if time_window_start.tzinfo:
            time_window_start_naive = time_window_start.replace(tzinfo=None)
        else:
            time_window_start_naive = time_window_start
        if time_window_end.tzinfo:
            time_window_end_naive = time_window_end.replace(tzinfo=None)
        else:
            time_window_end_naive = time_window_end
        
        time_range = DateTimeRange(time_window_start_naive, time_window_end_naive, "[]")
        risk_cells = []
        
        for cell, risk_score, confidence in cell_risks:
            try:
                # Normalize risk score to 0-1 range
                if risk_range > 0:
                    normalized_risk = (risk_score - min_risk) / risk_range
                else:
                    normalized_risk = risk_score
                
                # Only create cells with meaningful risk (after normalization)
                # Lower threshold to ensure visualization
                if normalized_risk >= 0.01:
                    # Convert GeoJSON to PostGIS geography
                    geom_dict = cell["geom"]
                    if isinstance(geom_dict, dict):
                        geom_json_str = json.dumps(geom_dict)
                    else:
                        geom_json_str = str(geom_dict)
                    
                    try:
                        # Use ST_GeomFromGeoJSON and convert to geography
                        # GeoAlchemy2 expects WKB bytes or a geography object
                        geom_result = db.execute(
                            text("""
                                SELECT ST_GeomFromGeoJSON(CAST(:geom AS text))::geography as geom
                            """),
                            {"geom": geom_json_str}
                        )
                        geom_wkb = geom_result.scalar()
                        
                        # If geom_wkb is a string (WKB hex), convert to bytes
                        if isinstance(geom_wkb, str):
                            try:
                                # Try to decode as hex
                                geom_wkb = bytes.fromhex(geom_wkb)
                            except ValueError:
                                # If not hex, it might be a PostGIS geography object
                                # Query it again as WKB
                                geom_wkb = db.execute(
                                    text("""
                                        SELECT ST_AsBinary(ST_GeomFromGeoJSON(CAST(:geom AS text))::geography) as geom_wkb
                                    """),
                                    {"geom": geom_json_str}
                                ).scalar()
                        
                        if geom_wkb and geom_wkb is not None:
                            # GeoAlchemy2 Geography expects WKB bytes
                            # But we need to use ST_GeomFromWKB in SQL, not ST_GeogFromText
                            # So we'll use a text representation that GeoAlchemy2 can handle
                            # Convert WKB bytes to hex string for GeoAlchemy2
                            if isinstance(geom_wkb, bytes):
                                # Use WKB hex string - GeoAlchemy2 will convert it
                                # But actually, we should use the WKB directly with proper casting
                                # Use a workaround: save as EWKT text
                                geom_ewkt = db.execute(
                                    text("""
                                        SELECT ST_AsEWKT(ST_GeomFromWKB(:geom_wkb)::geography) as geom_ewkt
                                    """),
                                    {"geom_wkb": geom_wkb}
                                ).scalar()
                                
                                if geom_ewkt:
                                    # GeoAlchemy2 can handle EWKT strings
                                    geom_for_cell = geom_ewkt
                                else:
                                    continue
                            else:
                                # If it's already a string, use it directly
                                geom_for_cell = str(geom_wkb)
                            
                            if geom_for_cell:
                                # Confidence based on forecast horizon and data quality
                                from datetime import timezone
                                now = datetime.now(timezone.utc)
                                if time_window_start.tzinfo is None:
                                    time_window_start = time_window_start.replace(tzinfo=timezone.utc)
                                hours_ahead = (time_window_start - now).total_seconds() / 3600
                                forecast_confidence = max(0.3, 1.0 - (hours_ahead / 168.0))  # Decreases over 7 days
                                
                                # Combine KDE confidence with forecast confidence
                                final_confidence = min(1.0, (confidence * 0.7 + forecast_confidence * 0.3))
                                
                                risk_cell = RiskCell(
                                    id=uuid.uuid4(),
                                    geom=geom_for_cell,
                                    time_window=time_range,
                                    risk_score=normalized_risk,  # Use normalized risk
                                    confidence=final_confidence
                                )
                            risk_cells.append(risk_cell)
                    except Exception as geom_error:
                        # Skip cells with invalid geometry
                        continue
            except Exception as e:
                continue
    
    # Save to database
    if risk_cells:
        try:
            db.bulk_save_objects(risk_cells)
            db.commit()
        except Exception as e:
            db.rollback()
    
    return risk_cells