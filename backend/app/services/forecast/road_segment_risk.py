"""Risk calculation for OSM road segments based on nearby crime events."""

import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import text
from uuid import UUID

from app.models.road_segment import RoadSegment
from app.models.crime_event import CrimeEvent
from app.core.config import get_settings
from app.services.utils import get_kucukcekmece_boundary

logger = logging.getLogger(__name__)
settings = get_settings()

# Only driveable road types from OSM (exclude pedestrian, footway, cycleway, etc.)
DRIVEABLE_ROAD_TYPES = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "service", "unclassified"
]


def snap_crime_to_road_segment(
    db: Session,
    crime_lat: float,
    crime_lng: float,
    max_distance_m: float = 100.0
) -> Optional[Tuple[float, float, int]]:
    """
    Snap a crime event to the nearest road segment within Küçükçekmece boundaries.
    
    Args:
        db: Database session
        crime_lat: Crime event latitude
        crime_lng: Crime event longitude
        max_distance_m: Maximum distance to snap (meters)
    
    Returns:
        Tuple of (snapped_lat, snapped_lng, segment_id) or None if no segment found
    """
    try:
        # Find nearest road segment within Küçükçekmece boundaries
        result = db.execute(
            text("""
                WITH nearest_segment AS (
                    SELECT 
                        rs.id as segment_id,
                        ST_ClosestPoint(
                            rs.geom::geometry,
                            ST_SetSRID(ST_MakePoint(:crime_lng, :crime_lat), 4326)
                        )::geography as snapped_point,
                        ST_Distance(
                            ST_GeogFromText('POINT(:crime_lng :crime_lat)'),
                            rs.geom
                        ) as distance_m
                    FROM road_segment rs
                    WHERE ST_DWithin(
                        ST_GeogFromText('POINT(:crime_lng :crime_lat)'),
                        rs.geom,
                        :max_distance_m
                    )
                    ORDER BY distance_m
                    LIMIT 1
                )
                SELECT 
                    segment_id,
                    ST_Y(snapped_point::geometry) as snapped_lat,
                    ST_X(snapped_point::geometry) as snapped_lng
                FROM nearest_segment
            """),
            {
                "crime_lat": crime_lat,
                "crime_lng": crime_lng,
                "max_distance_m": max_distance_m
            }
        ).first()
        
        if result:
            return (float(result.snapped_lat), float(result.snapped_lng), int(result.segment_id))
        
        return None
        
    except Exception as e:
        logger.error(f"Error snapping crime to road segment: {str(e)}")
        return None


def get_snapped_crimes_for_segment(
    db: Session,
    segment_id: int,
    time_window_start: datetime,
    time_window_end: datetime
) -> List[Dict]:
    """
    Get all crime events snapped to a specific road segment.
    
    Args:
        db: Database session
        segment_id: Road segment ID
        time_window_start: Start of time window
        time_window_end: End of time window
    
    Returns:
        List of dictionaries with crime event data including snapped coordinates
    """
    try:
        # Get segment geometry
        segment_geom = db.execute(
            text("""
                SELECT geom
                FROM road_segment
                WHERE id = :segment_id
            """),
            {"segment_id": segment_id}
        ).scalar()
        
        if not segment_geom:
            return []
        
        # Find all crime events within 100m of this segment and snap them
        crimes = db.execute(
            text("""
                WITH nearby_crimes AS (
                    SELECT 
                        ce.id,
                        ce.severity,
                        ce.confidence_score,
                        ce.event_time,
                        ST_ClosestPoint(
                            :segment_geom::geometry,
                            ce.geom::geometry
                        )::geography as snapped_point,
                        ST_Distance(ce.geom, :segment_geom) as distance_m
                    FROM crime_event ce
                    WHERE ST_DWithin(ce.geom, :segment_geom, 100.0)
                    AND ce.event_time >= :start_time
                    AND ce.event_time <= :end_time
                    AND is_within_kucukcekmece(ce.geom)
                )
                SELECT 
                    id,
                    severity,
                    confidence_score,
                    event_time,
                    ST_Y(snapped_point::geometry) as snapped_lat,
                    ST_X(snapped_point::geometry) as snapped_lng,
                    distance_m
                FROM nearby_crimes
                ORDER BY distance_m
            """),
            {
                "segment_id": segment_id,
                "segment_geom": segment_geom,
                "start_time": time_window_start,
                "end_time": time_window_end
            }
        ).fetchall()
        
        return [
            {
                "id": str(row.id),
                "severity": int(row.severity),
                "confidence_score": float(row.confidence_score),
                "event_time": row.event_time,
                "snapped_lat": float(row.snapped_lat),
                "snapped_lng": float(row.snapped_lng),
                "distance_m": float(row.distance_m)
            }
            for row in crimes
        ]
        
    except Exception as e:
        logger.error(f"Error getting snapped crimes for segment {segment_id}: {str(e)}")
        return []


def calculate_risk_for_road_segment(
    db: Session,
    segment: RoadSegment,
    time_window_start: datetime,
    time_window_end: datetime,
    search_radius_m: float = 100.0
) -> Tuple[float, float]:
    """
    Calculate risk score and confidence for a road segment using KDE on snapped crime events.
    
    Args:
        db: Database session
        segment: RoadSegment object
        time_window_start: Start of time window for forecast
        time_window_end: End of time window for forecast
        search_radius_m: Radius in meters to search for nearby crime events (unused, kept for compatibility)
    
    Returns:
        Tuple of (risk_score, confidence) where both are 0-1
    """
    try:
        # Get segment center point for KDE calculation
        center_result = db.execute(
            text("""
                SELECT 
                    ST_Y(ST_Centroid(geom::geometry)) as lat,
                    ST_X(ST_Centroid(geom::geometry)) as lng
                FROM road_segment
                WHERE id = :segment_id
            """),
            {"segment_id": segment.id}
        ).first()
        
        if not center_result:
            return (0.0, 0.0)
        
        center_lat = float(center_result.lat)
        center_lng = float(center_result.lng)
        
        # For forecast, use historical data from last 30 days
        # This gives us spatial risk distribution
        from datetime import timezone
        now = datetime.now(timezone.utc)
        historical_start = now - timedelta(days=30)
        historical_end = now
        
        # Get crime events snapped to this segment
        snapped_crimes = get_snapped_crimes_for_segment(
            db, segment.id, historical_start, historical_end
        )
        
        if not snapped_crimes:
            return (0.0, 0.0)
        
        # Convert snapped crimes to CrimeEvent-like objects for KDE
        # We need to create temporary CrimeEvent objects with snapped coordinates
        # For KDE calculation, we'll use the snapped coordinates directly
        from app.services.forecast.kde import compute_kde_at_point
        
        # Create a list of CrimeEvent objects with snapped coordinates
        # We'll need to query the actual CrimeEvent objects and update their coordinates
        # for KDE calculation, or we can pass the snapped coordinates directly
        
        # Get actual crime events
        crime_ids = [UUID(crime["id"]) for crime in snapped_crimes]
        crime_events = db.query(CrimeEvent).filter(CrimeEvent.id.in_(crime_ids)).all()
        
        if not crime_events:
            return (0.0, 0.0)
        
        # Create a map of crime_id -> snapped coordinates and event data
        snapped_coords_map = {
            crime["id"]: {
                "lat": crime["snapped_lat"],
                "lng": crime["snapped_lng"],
                "severity": crime["severity"],
                "confidence": crime["confidence_score"]
            }
            for crime in snapped_crimes
        }
        
        # Calculate distances from segment center to each snapped crime point
        # Use Python to calculate distances (Haversine) for simplicity
        import math
        R = 6371000  # Earth radius in meters
        
        def haversine_distance(lat1, lng1, lat2, lng2):
            """Calculate distance between two points using Haversine formula"""
            dlat = math.radians(lat2 - lat1)
            dlng = math.radians(lng2 - lng1)
            a = (math.sin(dlat / 2) ** 2 +
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                 math.sin(dlng / 2) ** 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            return R * c
        
        distance_map = {}
        for crime_id_str, crime_data in snapped_coords_map.items():
            distance_m = haversine_distance(
                center_lat, center_lng,
                crime_data["lat"], crime_data["lng"]
            )
            distance_map[crime_id_str] = distance_m
        
        # Calculate KDE using severity-weighted approach
        from app.services.forecast.kde import gaussian_kernel, compute_adaptive_bandwidth
        
        # Calculate local density for adaptive bandwidth
        local_density = len(crime_events) / 1000.0
        bandwidth = compute_adaptive_bandwidth(db, center_lat, center_lng, local_density)
        
        # Compute weighted KDE
        total_density = 0.0
        total_weight = 0.0
        
        for event in crime_events:
            event_id_str = str(event.id)
            if event_id_str not in distance_map or event_id_str not in snapped_coords_map:
                continue
            
            distance_m = distance_map[event_id_str]
            crime_data = snapped_coords_map[event_id_str]
            
            if distance_m < bandwidth * 3:  # 3-sigma cutoff
                # Weight by severity and confidence (use snapped crime data)
                weight = crime_data["severity"] * crime_data["confidence"]
                kernel_value = gaussian_kernel(distance_m, bandwidth)
                
                total_density += weight * kernel_value
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            density_score = total_density / total_weight
        else:
            density_score = 0.0
        
        # Normalize density_score to 0-1 range for risk_score
        # KDE density can be any positive value, we need to normalize it
        # Use a reasonable max value based on typical KDE results
        max_expected_density = 1.0  # Adjust based on actual KDE results
        risk_score = min(1.0, max(0.0, density_score / max_expected_density))
        
        # Confidence based on number of events and data quality
        avg_confidence = sum(crime["confidence_score"] for crime in snapped_crimes) / len(snapped_crimes) if snapped_crimes else 0.0
        confidence = min(1.0, len(snapped_crimes) / 10.0) * avg_confidence
        
        return (float(risk_score), float(confidence))
        
    except Exception as e:
        logger.error(f"Error calculating risk for road segment {segment.id}: {str(e)}")
        return (0.0, 0.0)


def update_road_segment_risks(
    db: Session,
    time_window_start: datetime,
    time_window_end: datetime,
    min_risk_threshold: float = 0.01,
    limit: Optional[int] = None,
    station_id: Optional[str] = None
) -> dict:
    """
    Update risk scores for road segments within station's neighborhood boundaries.
    If station_id is provided, uses that station's neighborhoods. Otherwise uses Küçükçekmece boundary.
    
    Args:
        db: Database session
        time_window_start: Start of time window for forecast
        time_window_end: End of time window for forecast
        min_risk_threshold: Minimum risk score to update (skip segments with lower risk)
        limit: Optional limit on number of segments to process
        station_id: Optional UUID of police station - if provided, filters by station's neighborhoods
    
    Returns:
        Dictionary with update statistics
    """
    try:
        from app.services.utils import get_station_neighborhood_boundaries, get_kucukcekmece_boundary
        
        # Get boundary based on station_id
        boundary_geom = None
        boundary_filter = ""
        
        if station_id:
            # Use station's neighborhood boundaries
            boundary_geom = get_station_neighborhood_boundaries(db, station_id)
            if boundary_geom:
                # Filter by station's neighborhood boundaries
                boundary_filter = """
                    -- Center point must be within station's neighborhood boundaries
                    ST_Within(
                        ST_Centroid(geom::geometry),
                        (SELECT ST_Union(ab.geom::geometry)::geometry
                         FROM administrative_boundary ab
                         JOIN police_station ps ON ab.name = ANY(ps.neighborhoods)
                         WHERE ps.id = :station_id::uuid
                         AND ab.admin_level = 8)
                    )
                    AND
                    -- At least 30% of segment length must be within boundary
                    (
                        ST_Length(
                            ST_Intersection(
                                geom::geometry,
                                (SELECT ST_Union(ab.geom::geometry)::geometry
                                 FROM administrative_boundary ab
                                 JOIN police_station ps ON ab.name = ANY(ps.neighborhoods)
                                 WHERE ps.id = :station_id::uuid
                                 AND ab.admin_level = 8)
                            )
                        ) / NULLIF(ST_Length(geom::geometry), 0)
                    ) >= 0.3
                """
            else:
                logger.warning(f"Station {station_id} has no neighborhoods defined, falling back to Küçükçekmece boundary")
        
        if not boundary_filter:
            # Fallback to Küçükçekmece boundary
            boundary_exists = db.execute(
                text("""
                    SELECT COUNT(*) > 0
                    FROM administrative_boundary 
                    WHERE name = 'Küçükçekmece' AND admin_level = 6
                """)
            ).scalar()
            
            if boundary_exists:
                boundary_filter = """
                    -- Center point must be within boundary
                    ST_Within(
                        ST_Centroid(geom::geometry),
                        (SELECT geom::geometry FROM administrative_boundary 
                         WHERE name = 'Küçükçekmece' AND admin_level = 6 LIMIT 1)
                    )
                    AND
                    -- At least 50% of segment length must be within boundary
                    (
                        ST_Length(
                            ST_Intersection(
                                geom::geometry,
                                (SELECT geom::geometry FROM administrative_boundary 
                                 WHERE name = 'Küçükçekmece' AND admin_level = 6 LIMIT 1)
                            )
                        ) / NULLIF(ST_Length(geom::geometry), 0)
                    ) >= 0.5
                """
        
        if boundary_filter:
            # Get road segments within boundaries (only driveable roads)
            query = db.query(RoadSegment).filter(
                RoadSegment.road_type.in_(DRIVEABLE_ROAD_TYPES),
                text(boundary_filter).params(station_id=station_id) if station_id else text(boundary_filter)
            )
        else:
            # If no boundary, use all segments (fallback) - but still only driveable roads
            logger.warning("No boundary found, using all driveable road segments")
            query = db.query(RoadSegment).filter(
                RoadSegment.road_type.in_(DRIVEABLE_ROAD_TYPES)
            )
        
        if limit:
            query = query.limit(limit)
        
        segments = query.all()
        
        if not segments:
            logger.warning(f"No road segments found (boundary_exists={boundary_exists})")
            return {
                "total_segments": 0,
                "updated": 0,
                "skipped": 0,
                "errors": 0
            }
        
        logger.info(f"Updating risk scores for {len(segments)} road segments...")
        
        updated = 0
        skipped = 0
        errors = 0
        
        for segment in segments:
            try:
                risk_score, confidence = calculate_risk_for_road_segment(
                    db, segment, time_window_start, time_window_end
                )
                
                if risk_score >= min_risk_threshold:
                    segment.risk_score = risk_score
                    segment.risk_confidence = confidence
                    segment.risk_updated_at = datetime.utcnow()
                    updated += 1
                else:
                    # Reset risk for segments below threshold
                    segment.risk_score = 0.0
                    segment.risk_confidence = 0.0
                    skipped += 1
                    
            except Exception as e:
                logger.error(f"Error updating risk for segment {segment.id}: {str(e)}")
                errors += 1
                continue
        
        db.commit()
        
        logger.info(f"Risk update completed: {updated} updated, {skipped} skipped, {errors} errors")
        
        return {
            "total_segments": len(segments),
            "updated": updated,
            "skipped": skipped,
            "errors": errors
        }
        
    except Exception as e:
        logger.error(f"Error updating road segment risks: {str(e)}")
        db.rollback()
        return {
            "total_segments": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 1,
            "error": str(e)
        }

