from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from app.db.session import get_db
from app.models.crime_event import CrimeEvent
from app.schemas.crime_event import CrimeEventCreate, CrimeEventRead, CrimeEventUpdate
from app.services.utils import lat_lng_to_geography, get_point_coordinates, validate_within_boundary

router = APIRouter(prefix="/crimes", tags=["Crimes"])


@router.get("", response_model=List[CrimeEventRead])
def list_crimes(
    db: Session = Depends(get_db),
    start_time: Optional[datetime] = Query(None, description="Filter crimes after this time"),
    end_time: Optional[datetime] = Query(None, description="Filter crimes before this time"),
    crime_type: Optional[str] = Query(None, description="Filter by crime type"),
    min_severity: Optional[int] = Query(None, ge=1, le=5, description="Minimum severity"),
    max_severity: Optional[int] = Query(None, ge=1, le=5, description="Maximum severity"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of results"),
):
    
    from sqlalchemy import text
    from app.services.utils import get_kucukcekmece_boundary
    
    query = db.query(CrimeEvent)
    
    # Küçükçekmece polygon sınırları içinde filtrele
    # Use the database function for consistency
    query = query.filter(
        text("is_within_kucukcekmece(geom)")
    )
    
    if start_time:
        query = query.filter(CrimeEvent.event_time >= start_time)
    if end_time:
        query = query.filter(CrimeEvent.event_time <= end_time)
    if crime_type:
        query = query.filter(CrimeEvent.crime_type == crime_type)
    if min_severity:
        query = query.filter(CrimeEvent.severity >= min_severity)
    if max_severity:
        query = query.filter(CrimeEvent.severity <= max_severity)
    
    events = query.order_by(CrimeEvent.event_time.desc()).limit(limit).all()
    
    results = []
    for event in events:
        # Get coordinates using PostGIS query
        coord_result = db.execute(
            text("""
                SELECT 
                    ST_Y(geom::geometry) as lat,
                    ST_X(geom::geometry) as lng
                FROM crime_event
                WHERE id = :event_id
            """),
            {"event_id": event.id}
        ).first()
        
        if coord_result and coord_result.lat is not None and coord_result.lng is not None:
            lat, lng = float(coord_result.lat), float(coord_result.lng)
        else:
            # Fallback to function
            lat, lng = get_point_coordinates(db, event.geom)
        
        results.append(
            CrimeEventRead(
                id=event.id,
                crime_type=event.crime_type,
                severity=event.severity,
                event_time=event.event_time,
                lat=lat,
                lng=lng,
                street_name=event.street_name,
                confidence_score=event.confidence_score,
                created_at=event.created_at,
            )
        )
    
    return results


@router.get("/snap-to-road")
def snap_crime_to_road(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lng: float = Query(..., ge=-180, le=180, description="Longitude"),
    db: Session = Depends(get_db)
):
    """
    Snap a crime event location to the nearest OSM road segment.
    Returns the snapped coordinates and road segment ID.
    """
    from app.services.forecast.road_segment_risk import snap_crime_to_road_segment
    import logging
    import math
    logger = logging.getLogger(__name__)
    
    try:
        snapped_result = snap_crime_to_road_segment(db, lat, lng, max_distance_m=100.0)
        
        if not snapped_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Yakınlarda OSM yol segmenti bulunamadı (100m içinde)"
            )
        
        snapped_lat, snapped_lng, segment_id = snapped_result
        
        # Calculate distance
        R = 6371000  # Earth radius in meters
        dlat = math.radians(snapped_lat - lat)
        dlng = math.radians(snapped_lng - lng)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat)) * math.cos(math.radians(snapped_lat)) *
             math.sin(dlng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance_m = R * c
        
        return {
            "snapped_lat": snapped_lat,
            "snapped_lng": snapped_lng,
            "road_segment_id": segment_id,
            "original_lat": lat,
            "original_lng": lng,
            "distance_m": round(distance_m, 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error snapping crime to road: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Yol segmentine snap edilirken hata oluştu: {str(e)}"
        )


@router.post("", response_model=CrimeEventRead, status_code=status.HTTP_201_CREATED)
def create_crime(
    payload: CrimeEventCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new crime event"""
    # Validate coordinates are within boundary
    is_valid, error_message = validate_within_boundary(db, payload.lat, payload.lng)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message or "Koordinatlar Küçükçekmece sınırları dışında"
        )
    
    geom = lat_lng_to_geography(payload.lat, payload.lng)
    
    crime_event = CrimeEvent(
        crime_type=payload.crime_type,
        severity=payload.severity,
        event_time=payload.event_time,
        geom=geom,
        street_name=payload.street_name,
        confidence_score=payload.confidence_score,
    )
    
    db.add(crime_event)
    db.commit()
    db.refresh(crime_event)
    
    # Trigger real-time risk update (async, non-blocking)
    def trigger_update_task():
        """Background task to trigger risk update."""
        try:
            import asyncio
            from app.services.realtime.risk_update_service import get_risk_update_service
            
            risk_update_service = get_risk_update_service()
            
            # Create new event loop for background task
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run async task
            if loop.is_running():
                # If loop is already running, schedule task
                asyncio.create_task(risk_update_service.trigger_risk_update(crime_event, db))
            else:
                # Run until complete
                loop.run_until_complete(risk_update_service.trigger_risk_update(crime_event, db))
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error triggering risk update: {str(e)}")
    
    # Add background task
    background_tasks.add_task(trigger_update_task)
    
    return CrimeEventRead(
        id=crime_event.id,
        crime_type=crime_event.crime_type,
        severity=crime_event.severity,
        event_time=crime_event.event_time,
        lat=payload.lat,
        lng=payload.lng,
        street_name=crime_event.street_name,
        confidence_score=crime_event.confidence_score,
        created_at=crime_event.created_at,
    )


@router.patch("/{crime_id}", response_model=CrimeEventRead)
def update_crime(
    crime_id: UUID,
    payload: CrimeEventUpdate,
    db: Session = Depends(get_db)
):
    """Update a crime event"""
    event = db.query(CrimeEvent).filter(CrimeEvent.id == crime_id).first()
    
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Crime event with id {crime_id} not found"
        )
    
    if payload.crime_type is not None:
        event.crime_type = payload.crime_type
    if payload.severity is not None:
        event.severity = payload.severity
    if payload.event_time is not None:
        event.event_time = payload.event_time
    if payload.street_name is not None:
        event.street_name = payload.street_name
    if payload.confidence_score is not None:
        event.confidence_score = payload.confidence_score
    if payload.lat is not None and payload.lng is not None:
        # Validate new coordinates are within boundary
        is_valid, error_message = validate_within_boundary(db, payload.lat, payload.lng)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message or "Koordinatlar Küçükçekmece sınırları dışında"
            )
        event.geom = lat_lng_to_geography(payload.lat, payload.lng)
    
    db.commit()
    db.refresh(event)
    
    lat, lng = get_point_coordinates(db, event.geom)
    
    return CrimeEventRead(
        id=event.id,
        crime_type=event.crime_type,
        severity=event.severity,
        event_time=event.event_time,
        lat=lat,
        lng=lng,
        street_name=event.street_name,
        confidence_score=event.confidence_score,
        created_at=event.created_at,
    )


@router.delete("/{crime_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_crime(crime_id: UUID, db: Session = Depends(get_db)):
    """Delete a crime event"""
    event = db.query(CrimeEvent).filter(CrimeEvent.id == crime_id).first()
    
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Crime event with id {crime_id} not found"
        )
    
    db.delete(event)
    db.commit()
    
    return None