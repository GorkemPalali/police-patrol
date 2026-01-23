from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from app.db.session import get_db
from app.models.police_station import PoliceStation
from app.schemas.station import PoliceStationCreate, PoliceStationRead, PoliceStationUpdate
from app.services.utils import lat_lng_to_geography, get_point_coordinates, geography_to_geojson, validate_within_boundary

router = APIRouter(prefix="/stations", tags=["Stations"])


@router.get("", response_model=List[PoliceStationRead])
def list_stations(
    db: Session = Depends(get_db),
    active_only: bool = True
):
    """List all police stations"""
    query = db.query(PoliceStation)
    
    if active_only:
        query = query.filter(PoliceStation.active == True)
    
    stations = query.all()
    
    results = []
    for station in stations:
        lat, lng = get_point_coordinates(db, station.geom)
        results.append(
            PoliceStationRead(
                id=station.id,
                name=station.name,
                lat=lat,
                lng=lng,
                capacity=station.capacity,
                active=station.active,
            )
        )
    
    return results


@router.get("/{station_id}", response_model=PoliceStationRead)
def get_station(station_id: UUID, db: Session = Depends(get_db)):
    """Get a specific police station by ID"""
    station = db.query(PoliceStation).filter(PoliceStation.id == station_id).first()
    
    if not station:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Police station with id {station_id} not found"
        )
    
    lat, lng = get_point_coordinates(db, station.geom)
    
    return PoliceStationRead(
        id=station.id,
        name=station.name,
        lat=lat,
        lng=lng,
        capacity=station.capacity,
        active=station.active,
    )


@router.post("", response_model=PoliceStationRead, status_code=status.HTTP_201_CREATED)
def create_station(payload: PoliceStationCreate, db: Session = Depends(get_db)):
    """Create a new police station"""
    # Validate coordinates are within boundary
    is_valid, error_message = validate_within_boundary(db, payload.lat, payload.lng)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message or "Koordinatlar Küçükçekmece sınırları dışında"
        )
    
    geom = lat_lng_to_geography(payload.lat, payload.lng)
    
    station = PoliceStation(
        name=payload.name,
        geom=geom,
        capacity=payload.capacity,
        active=payload.active,
    )
    
    db.add(station)
    db.commit()
    db.refresh(station)
    
    return PoliceStationRead(
        id=station.id,
        name=station.name,
        lat=payload.lat,
        lng=payload.lng,
        capacity=station.capacity,
        active=station.active,
    )


@router.patch("/{station_id}", response_model=PoliceStationRead)
def update_station(
    station_id: UUID,
    payload: PoliceStationUpdate,
    db: Session = Depends(get_db)
):
    """Update a police station"""
    station = db.query(PoliceStation).filter(PoliceStation.id == station_id).first()
    
    if not station:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Police station with id {station_id} not found"
        )
    
    # Update fields
    if payload.name is not None:
        station.name = payload.name
    if payload.capacity is not None:
        station.capacity = payload.capacity
    if payload.active is not None:
        station.active = payload.active
    if payload.lat is not None and payload.lng is not None:
        # Validate new coordinates are within boundary
        is_valid, error_message = validate_within_boundary(db, payload.lat, payload.lng)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message or "Koordinatlar Küçükçekmece sınırları dışında"
            )
        station.geom = lat_lng_to_geography(payload.lat, payload.lng)
    
    db.commit()
    db.refresh(station)
    
    lat, lng = get_point_coordinates(db, station.geom)
    
    return PoliceStationRead(
        id=station.id,
        name=station.name,
        lat=lat,
        lng=lng,
        capacity=station.capacity,
        active=station.active,
    )


@router.delete("/{station_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_station(station_id: UUID, db: Session = Depends(get_db)):
    """Delete a police station (soft delete by setting active=False)"""
    station = db.query(PoliceStation).filter(PoliceStation.id == station_id).first()
    
    if not station:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Police station with id {station_id} not found"
        )
    
    station.active = False
    db.commit()
    
    return None
