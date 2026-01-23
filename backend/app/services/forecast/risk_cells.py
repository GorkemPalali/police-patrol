"""Grid-based risk cell generation using PostGIS"""
from datetime import datetime
from typing import List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text
from psycopg2.extras import DateTimeTZRange
import uuid
import numpy as np

from app.models.risk_cell import RiskCell
from app.models.crime_event import CrimeEvent
from app.services.forecast.kde import compute_kde_grid
from app.core.config import get_settings

settings = get_settings()


def create_hex_grid(
    db: Session,
    bbox: Tuple[float, float, float, float],
    grid_size_m: float = None
) -> List[dict]:
    """
    Create hexagon grid using PostGIS ST_HexagonGrid.
    Returns list of hexagon geometries as GeoJSON.
    Falls back to square grid if hex grid fails.
    """
    grid_size = grid_size_m or settings.default_grid_size_m
    
    try:
        # Try to create hex grid using PostGIS ST_HexagonGrid
        # Note: This requires PostGIS 3.1+ and may not work on all systems
        result = db.execute(
            text("""
                WITH bbox_4326 AS (
                    SELECT ST_MakeEnvelope(
                        :min_lng, :min_lat,
                        :max_lng, :max_lat,
                        4326
                    )::geometry as geom
                ),
                bbox_3857 AS (
                    SELECT ST_Transform(geom, 3857) as geom
                    FROM bbox_4326
                ),
                hexgrid AS (
                    SELECT 
                        (ST_HexagonGrid(:grid_size, bbox_3857.geom)).geom as hex_geom
                    FROM bbox_3857
                ),
                -- Filter out Küçükçekmece Lake (approximate bounds: 41.0-41.02 lat, 28.75-28.78 lng)
                filtered_hexgrid AS (
                    SELECT 
                        ST_Transform(hex_geom, 4326) as hex_geom,
                        ST_Area(ST_Transform(hex_geom, 4326)::geography) as area_sq_m
                    FROM hexgrid
                    WHERE NOT (
                        ST_Intersects(
                            ST_Transform(hex_geom, 4326),
                            ST_MakeEnvelope(28.75, 41.0, 28.78, 41.02, 4326)
                        )
                    )
                )
                SELECT 
                    ST_AsGeoJSON(hex_geom)::json as geom,
                    area_sq_m
                FROM filtered_hexgrid
            """),
            {
                "min_lat": bbox[0],
                "min_lng": bbox[1],
                "max_lat": bbox[2],
                "max_lng": bbox[3],
                "grid_size": grid_size
            }
        ).fetchall()
        
        hexagons = []
        for row in result:
            hexagons.append({
                "geom": row.geom,
                "area_sq_m": float(row.area_sq_m) if row.area_sq_m else 0.0
            })
        
        return hexagons if hexagons else []
    except Exception as e:
        # Fallback to square grid if hex grid fails
        print(f"Hex grid creation failed, using square grid: {e}")
        return create_square_grid(db, bbox, grid_size)


def create_square_grid(
    db: Session,
    bbox: Tuple[float, float, float, float],
    grid_size_m: float = None
) -> List[dict]:
    """
    Create square grid cells.
    Simpler alternative to hex grid.
    """
    grid_size = grid_size_m or settings.default_grid_size_m
    
    # Convert meters to degrees (approximate)
    lat_step = grid_size / 111000.0
    avg_lat = (bbox[0] + bbox[2]) / 2.0
    lng_step = grid_size / (111000.0 * np.cos(np.radians(avg_lat)))
    
    cells = []
    lat = bbox[0]
    
    while lat < bbox[2]:
        lng = bbox[1]
        while lng < bbox[3]:
            # Skip Küçükçekmece Lake area (approximate bounds: 41.0-41.02 lat, 28.75-28.78 lng)
            # Check if cell center is in lake area
            cell_center_lat = lat + lat_step / 2
            cell_center_lng = lng + lng_step / 2
            if 41.0 <= cell_center_lat <= 41.02 and 28.75 <= cell_center_lng <= 28.78:
                lng += lng_step
                continue
            
            # Create square cell
            cell_geom = db.execute(
                text("""
                    SELECT ST_AsGeoJSON(
                        ST_MakeEnvelope(
                            :min_lng, :min_lat,
                            :min_lng + :lng_step, :min_lat + :lat_step,
                            4326
                        )::geography
                    )::json as geom
                """),
                {
                    "min_lat": lat,
                    "min_lng": lng,
                    "lat_step": lat_step,
                    "lng_step": lng_step
                }
            ).scalar()
            
            if cell_geom:
                cells.append({
                    "geom": cell_geom,
                    "center_lat": lat + lat_step / 2,
                    "center_lng": lng + lng_step / 2
                })
            
            lng += lng_step
        lat += lat_step
    
    return cells


def calculate_risk_for_cell(
    db: Session,
    cell_geom: dict,
    events: List[CrimeEvent],
    time_window_start: datetime,
    time_window_end: datetime
) -> Tuple[float, float]:
    """
    Calculate risk score and confidence for a grid cell.
    Returns (risk_score, confidence)
    """
    if not events:
        return (0.0, 0.0)
    
    # Count events in this cell
    geom_wkt = db.execute(
        text("SELECT ST_GeomFromGeoJSON(:geom::text)::geography as geom"),
        {"geom": str(cell_geom)}
    ).scalar()
    
    if not geom_wkt:
        return (0.0, 0.0)
    
    # Count events within cell
    event_count = db.execute(
        text("""
            SELECT COUNT(*) as count
            FROM crime_event
            WHERE ST_Within(
                geom,
                ST_GeomFromGeoJSON(:geom::text)::geography
            )
            AND event_time >= :start_time
            AND event_time <= :end_time
        """),
        {
            "geom": str(cell_geom),
            "start_time": time_window_start,
            "end_time": time_window_end
        }
    ).scalar()
    
    if event_count == 0:
        return (0.0, 0.0)
    
    # Calculate weighted risk
    # Get events in cell with their severity
    cell_events = db.execute(
        text("""
            SELECT severity, confidence_score
            FROM crime_event
            WHERE ST_Within(
                geom,
                ST_GeomFromGeoJSON(:geom::text)::geography
            )
            AND event_time >= :start_time
            AND event_time <= :end_time
        """),
        {
            "geom": str(cell_geom),
            "start_time": time_window_start,
            "end_time": time_window_end
        }
    ).fetchall()
    
    # Weighted average risk
    total_weight = 0.0
    weighted_sum = 0.0
    
    for event in cell_events:
        weight = event.severity * event.confidence_score
        weighted_sum += weight
        total_weight += weight
    
    if total_weight > 0:
        risk_score = weighted_sum / total_weight / 5.0  # Normalize to 0-1
    else:
        risk_score = 0.0
    
    # Confidence based on event count and data quality
    confidence = min(1.0, event_count / 5.0)
    
    return (float(risk_score), float(confidence))


def generate_risk_cells(
    db: Session,
    time_window_start: datetime,
    time_window_end: datetime,
    bbox: Tuple[float, float, float, float] = None,
    grid_size_m: float = None,
    use_hex: bool = True
) -> List[RiskCell]:
    """
    Generate risk cells for a given time window.
    """
    grid_size = grid_size_m or settings.default_grid_size_m
    
    # Default bbox: Istanbul area (based on actual data bounds)
    if bbox is None:
        # Get actual data bounds from crime events
        try:
            bounds_result = db.execute(
                text("""
                    SELECT 
                        ST_YMin(ST_Collect(geom::geometry)) as min_lat,
                        ST_XMin(ST_Collect(geom::geometry)) as min_lng,
                        ST_YMax(ST_Collect(geom::geometry)) as max_lat,
                        ST_XMax(ST_Collect(geom::geometry)) as max_lng
                    FROM crime_event
                    WHERE event_time >= :start_time AND event_time <= :end_time
                """),
                {
                    "start_time": time_window_start,
                    "end_time": time_window_end
                }
            ).first()
            
            if bounds_result and bounds_result.min_lat:
                bbox = (
                    float(bounds_result.min_lat) - 0.01,
                    float(bounds_result.min_lng) - 0.01,
                    float(bounds_result.max_lat) + 0.01,
                    float(bounds_result.max_lng) + 0.01
                )
            else:
                from app.services.utils import get_kucukcekmece_bbox_from_polygon
                bbox = get_kucukcekmece_bbox_from_polygon(db) or settings.kucukcekmece_fallback_bbox
        except Exception:
            from app.services.utils import get_kucukcekmece_bbox_from_polygon
            bbox = get_kucukcekmece_bbox_from_polygon(db) or settings.kucukcekmece_fallback_bbox
    
    # Get events in time window within Küçükçekmece polygon boundaries
    events = db.query(CrimeEvent).filter(
        CrimeEvent.event_time >= time_window_start,
        CrimeEvent.event_time <= time_window_end
    ).filter(
        text("""
            is_within_kucukcekmece(geom)
            AND NOT ST_Within(
                geom::geometry,
                ST_MakeEnvelope(28.75, 41.0, 28.78, 41.02, 4326)
            )
        """)
    ).all()
    
    if not events:
        return []  # No events, no risk cells
    
    # Create grid
    if use_hex:
        grid_cells = create_hex_grid(db, bbox, grid_size)
    else:
        grid_cells = create_square_grid(db, bbox, grid_size)
    
    if not grid_cells:
        return [] # No grid cells created
    
    # Calculate risk for each cell
    time_range = DateTimeTZRange(time_window_start, time_window_end, "[]")
    risk_cells = []
    
    for cell in grid_cells:
        try:
            risk_score, confidence = calculate_risk_for_cell(
                db,
                cell["geom"],
                events,
                time_window_start,
                time_window_end
            )
            
            # Only create cells with non-zero risk
            if risk_score > 0:
                # Convert GeoJSON to PostGIS geography
                geom_wkb = db.execute(
                    text("""
                        SELECT ST_GeomFromGeoJSON(:geom::text)::geography as geom
                    """),
                    {"geom": str(cell["geom"])}
                ).scalar()
                
                if geom_wkb:
                    risk_cell = RiskCell(
                        id=uuid.uuid4(),
                        geom=geom_wkb,
                        time_window=time_range,
                        risk_score=risk_score,
                        confidence=confidence
                    )
                    risk_cells.append(risk_cell)
        except Exception as e:
            # Skip cells that fail to process
            print(f"Error processing cell: {e}")
            continue
    
    # Save to database
    if risk_cells:
        try:
            db.bulk_save_objects(risk_cells)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"Error saving risk cells: {e}")
            # Return cells anyway (they're in memory)
    
    return risk_cells

