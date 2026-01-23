from dataclasses import dataclass
from typing import List, Optional, Tuple
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import text
import json

from app.models.police_station import PoliceStation
from app.models.risk_cell import RiskCell
from app.services.utils import get_point_coordinates, get_kucukcekmece_boundary, get_kucukcekmece_bbox_from_polygon
from app.core.config import get_settings


@dataclass
class RouteRequest:
    station_id: UUID
    risk_threshold: float = 0.7
    max_minutes: int = 90
    end_station_id: Optional[UUID] = None
    start_time: Optional[str] = None  # ISO format datetime string
    end_time: Optional[str] = None  # ISO format datetime string


@dataclass
class RouteWaypoint:
    lat: float
    lng: float
    risk_score: Optional[float] = None


@dataclass
class RouteResult:
    waypoints: List[RouteWaypoint]
    total_distance: float
    total_time: float
    risk_coverage: float
    path: dict  # GeoJSON LineString


def get_station_coordinates(db: Session, station_id: UUID) -> Tuple[float, float]:
    """Get lat/lng coordinates of a police station"""
    station = db.query(PoliceStation).filter(PoliceStation.id == station_id).first()
    if not station:
        raise ValueError(f"Station {station_id} not found")
    
    return get_point_coordinates(db, station.geom)


def get_station_coordinates_snapped(
    db: Session, 
    station_id: UUID, 
    max_distance_m: float = 100.0
) -> Tuple[float, float]:
    """
    Get station coordinates snapped to nearest road network node.
    Falls back to original coordinates if snap fails.
    
    Args:
        db: Database session
        station_id: Police station UUID
        max_distance_m: Maximum distance to snap (meters)
    
    Returns:
        Tuple of (lat, lng) - snapped coordinates or original if snap fails
    """
    # Get original coordinates
    original_lat, original_lng = get_station_coordinates(db, station_id)
    
    # Try to snap to road network
    snapped = _snap_to_road_network(db, original_lat, original_lng, max_distance_m)
    
    if snapped:
        return (snapped[0], snapped[1])  # Return snapped lat, lng
    
    # Fallback to original coordinates
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Could not snap station {station_id} to road network, using original coordinates")
    return (original_lat, original_lng)


def get_high_risk_road_segments(
    db: Session,
    risk_threshold: float,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 100,
    station_id: Optional[str] = None
):
    """
    Get road segments with high risk scores.
    This replaces grid-based risk cells with road-segment-based risk.
    Only uses OSM driveable road segments (motorway, trunk, primary, secondary, tertiary, residential, service, unclassified).
    
    Args:
        db: Database session
        risk_threshold: Minimum risk score threshold
        bbox: Optional bounding box for filtering
        start_time: Optional start time for time window
        end_time: Optional end time for time window
        limit: Maximum number of segments to return
    
    Returns:
        List of RoadSegment objects with risk_score >= threshold (only driveable roads)
    """
    from app.models.road_segment import RoadSegment
    
    # Only use driveable road types from OSM (exclude pedestrian, footway, cycleway, etc.)
    # These are the only types imported from OSM, but we add explicit filter for safety
    driveable_road_types = [
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "residential", "service", "unclassified"
    ]
    
    query = db.query(RoadSegment).filter(
        RoadSegment.risk_score >= risk_threshold,
        RoadSegment.road_type.in_(driveable_road_types)
    )
    
    # Filter by station's neighborhood boundaries if station_id provided, otherwise use Küçükçekmece boundary
    from app.services.utils import get_station_neighborhood_boundaries
    boundary_geom = None
    boundary_filter = ""
    
    if station_id:
        # Use station's neighborhood boundaries
        boundary_geom = get_station_neighborhood_boundaries(db, station_id)
        if boundary_geom:
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
    
    if not boundary_filter:
        # Fallback to Küçükçekmece boundary
        boundary_geom = get_kucukcekmece_boundary(db)
        if boundary_geom:
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
        query = query.filter(
            text(boundary_filter).params(station_id=station_id) if station_id else text(boundary_filter)
        )
    
    # Filter by bbox if provided
    if bbox:
        query = query.filter(
            text("""
                ST_Intersects(
                    geom,
                    ST_MakeEnvelope(:min_lng, :min_lat, :max_lng, :max_lat, 4326)::geography
                )
            """).params(
                min_lat=bbox[0],
                min_lng=bbox[1],
                max_lat=bbox[2],
                max_lng=bbox[3]
            )
        )
    
    # Order by risk score descending
    query = query.order_by(RoadSegment.risk_score.desc())
    
    return query.limit(limit).all()


def get_high_risk_cells(
    db: Session,
    risk_threshold: float,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> List[RiskCell]:
    """Get risk cells above threshold within Küçükçekmece polygon boundaries and time window"""
    from app.services.utils import get_kucukcekmece_boundary
    from datetime import datetime
    
    query = db.query(RiskCell).filter(RiskCell.risk_score >= risk_threshold)
    
    # Filter by time window if provided
    if start_time and end_time:
        try:
            from datetime import timezone
            # Parse ISO format datetime strings
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            
            # Convert to timezone-naive for TSRANGE (PostgreSQL TSRANGE is timezone-naive)
            if start_dt.tzinfo:
                start_dt_naive = start_dt.replace(tzinfo=None)
            else:
                start_dt_naive = start_dt
            if end_dt.tzinfo:
                end_dt_naive = end_dt.replace(tzinfo=None)
            else:
                end_dt_naive = end_dt
            
            # Create TSRANGE for filtering
            # TSRANGE uses [start, end) format (inclusive start, exclusive end)
            # We need to ensure the time_window overlaps with the requested range
            query = query.filter(
                text("time_window && tsrange(:start_time, :end_time, '[]')").params(
                    start_time=start_dt_naive,
                    end_time=end_dt_naive
                )
            )
        except (ValueError, AttributeError) as e:
            # If time parsing fails, log warning but continue without time filter
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to parse time window for risk cells: {str(e)}")
    
    # Filter by polygon intersection
    # If custom bbox provided, we still need to filter by polygon
    # The bbox is only used for initial filtering, final filter is polygon
    boundary_geom = get_kucukcekmece_boundary(db)
    
    if boundary_geom:
        # Use polygon intersection
        query = query.filter(
            text("""
                ST_Intersects(
                    geom,
                    (SELECT geom FROM administrative_boundary 
                     WHERE name = 'Küçükçekmece' AND admin_level = 6 LIMIT 1)
                )
            """)
        )
    else:
        # Fallback to bbox if polygon not available
        from app.core.config import get_settings
        from app.services.utils import get_kucukcekmece_bbox_from_polygon
        settings = get_settings()
        kucukcekmece_bbox = get_kucukcekmece_bbox_from_polygon(db) or settings.kucukcekmece_fallback_bbox
        
        if bbox:
            effective_bbox = (
                max(bbox[0], kucukcekmece_bbox[0]),
                max(bbox[1], kucukcekmece_bbox[1]),
                min(bbox[2], kucukcekmece_bbox[2]),
                min(bbox[3], kucukcekmece_bbox[3])
            )
        else:
            effective_bbox = kucukcekmece_bbox
        
        query = query.filter(
            text("""
                ST_Intersects(
                    geom,
                    ST_MakeEnvelope(:min_lng, :min_lat, :max_lng, :max_lat, 4326)::geography
                )
            """).params(
                min_lat=effective_bbox[0],
                min_lng=effective_bbox[1],
                max_lat=effective_bbox[2],
                max_lng=effective_bbox[3]
            )
        )
    
    return query.limit(100).all()  # Limit to top 100 risk cells


def cluster_risk_cells(
    db: Session,
    risk_cells: List[RiskCell],
    max_clusters: int = 10
) -> List[Tuple[float, float, float]]:
    """
    Cluster risk cells and return cluster centers with average risk.
    Returns list of (lat, lng, avg_risk_score)
    """
    if not risk_cells:
        return []
    
    # Simple approach: group by proximity
    # Get centroids of risk cells
    centroids = []
    for cell in risk_cells:
        centroid = db.execute(
            text("""
                SELECT 
                    ST_Y(ST_Centroid(ST_GeomFromWKB(:geom))) as lat,
                    ST_X(ST_Centroid(ST_GeomFromWKB(:geom))) as lng
            """),
            {"geom": cell.geom.data}
        ).first()
        
        if centroid:
            centroids.append({
                "lat": float(centroid.lat),
                "lng": float(centroid.lng),
                "risk": cell.risk_score
            })
    
    # Simple clustering: group nearby points
    clusters = []
    used = set()
    
    for i, point in enumerate(centroids):
        if i in used:
            continue
        
        cluster_points = [point]
        used.add(i)
        
        # Find nearby points (within 500m)
        for j, other_point in enumerate(centroids):
            if j in used:
                continue
            
            # Calculate distance
            dist = ((point["lat"] - other_point["lat"]) ** 2 + 
                   (point["lng"] - other_point["lng"]) ** 2) ** 0.5 * 111000
            
            if dist < 500:
                cluster_points.append(other_point)
                used.add(j)
        
        # Calculate cluster center and average risk
        avg_lat = sum(p["lat"] for p in cluster_points) / len(cluster_points)
        avg_lng = sum(p["lng"] for p in cluster_points) / len(cluster_points)
        avg_risk = sum(p["risk"] for p in cluster_points) / len(cluster_points)
        
        clusters.append((avg_lat, avg_lng, avg_risk))
        
        if len(clusters) >= max_clusters:
            break
    
    # Try to snap all cluster centroids to road network
    # If snap fails, use original coordinates (snap is optional)
    if clusters:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Clustering {len(centroids)} centroids into {len(clusters)} clusters")
        
        cluster_coords = [(lat, lng) for lat, lng, _ in clusters]
        snapped_coords = snap_risk_cell_centroids(db, cluster_coords, max_distance_m=100.0)
        
        logger.info(f"Snapped {len(snapped_coords)} out of {len(clusters)} clusters to road network")
        
        # Rebuild clusters with snapped coordinates (or original if snap failed)
        final_clusters = []
        for i, (lat, lng, risk) in enumerate(clusters):
            if i < len(snapped_coords):
                # Use snapped coordinates
                final_clusters.append((snapped_coords[i][0], snapped_coords[i][1], risk))
            else:
                # If snap failed for this cluster, use original coordinates
                logger.warning(f"Cluster {i} could not be snapped to road network, using original coordinates")
                final_clusters.append((lat, lng, risk))
        
        logger.info(f"Returning {len(final_clusters)} clusters for route planning")
        return final_clusters
    
    return clusters


def snap_risk_cell_centroids(
    db: Session,
    centroids: List[Tuple[float, float]],
    max_distance_m: float = 100.0
) -> List[Tuple[float, float]]:
    """
    Snap risk cell centroids to nearest road network nodes.
    Filters out centroids that cannot be snapped.
    
    Args:
        db: Database session
        centroids: List of (lat, lng) tuples
        max_distance_m: Maximum distance to snap (meters)
    
    Returns:
        List of snapped (lat, lng) tuples
    """
    snapped_centroids = []
    import logging
    logger = logging.getLogger(__name__)
    
    for lat, lng in centroids:
        snapped = _snap_to_road_network(db, lat, lng, max_distance_m)
        if snapped:
            snapped_centroids.append((snapped[0], snapped[1]))
        else:
            logger.debug(f"Could not snap centroid ({lat}, {lng}) to road network, skipping")
    
    return snapped_centroids


def _snap_to_road_network(
    db: Session,
    lat: float,
    lng: float,
    max_distance_m: float = 100.0
) -> Optional[Tuple[float, float, int]]:
    """
    Snap a point to the nearest road segment vertex.
    Returns (lat, lng, vertex_id) or None if no road found within max_distance_m.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # First try to use road_segment_vertices_pgr table if it exists
        table_exists = db.execute(
            text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'road_segment_vertices_pgr'
                )
            """)
        ).scalar()
        
        if table_exists:
            result = db.execute(
                text("""
                    WITH nearest_vertex AS (
                        SELECT 
                            v.id as vertex_id,
                            ST_Y(v.the_geom::geometry) as lat,
                            ST_X(v.the_geom::geometry) as lng,
                            ST_Distance(
                                ST_GeogFromText('POINT(:lng :lat)'),
                                v.the_geom::geography
                            ) as distance_m
                        FROM road_segment_vertices_pgr v
                        WHERE ST_DWithin(
                            ST_GeogFromText('POINT(:lng :lat)'),
                            v.the_geom::geography,
                            :max_distance_m
                        )
                        ORDER BY distance_m
                        LIMIT 1
                    )
                    SELECT lat, lng, vertex_id
                    FROM nearest_vertex
                """),
                {
                    "lat": lat,
                    "lng": lng,
                    "max_distance_m": max_distance_m
                }
            ).first()
            
            if result:
                return (float(result.lat), float(result.lng), int(result.vertex_id))
        
        # Fallback: Find nearest point on road segments directly
        # Extract start and end points of road segments and find the closest one
        # Only use driveable road segments from OSM that are within Küçükçekmece boundary
        boundary_geom = get_kucukcekmece_boundary(db)
        boundary_filter = ""
        if boundary_geom:
            boundary_filter = """
                AND ST_Within(
                    ST_Centroid(geom::geometry),
                    (SELECT geom::geometry FROM administrative_boundary 
                     WHERE name = 'Küçükçekmece' AND admin_level = 6 LIMIT 1)
                )
                AND (
                    ST_Length(
                        ST_Intersection(
                            geom::geometry,
                            (SELECT geom::geometry FROM administrative_boundary 
                             WHERE name = 'Küçükçekmece' AND admin_level = 6 LIMIT 1)
                        )
                    ) / NULLIF(ST_Length(geom::geometry), 0)
                ) >= 0.5
            """
        
        result = db.execute(
            text(f"""
                WITH segment_points AS (
                    SELECT 
                        id as segment_id,
                        source as vertex_id,
                        ST_Y(ST_StartPoint(geom::geometry)) as lat,
                        ST_X(ST_StartPoint(geom::geometry)) as lng,
                        ST_Distance(
                            ST_GeogFromText('POINT(:lng :lat)'),
                            ST_StartPoint(geom::geometry)::geography
                        ) as distance_m
                    FROM road_segment
                    WHERE source IS NOT NULL
                    AND road_type IN ('motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service', 'unclassified')
                    {boundary_filter}
                    UNION ALL
                    SELECT 
                        id as segment_id,
                        target as vertex_id,
                        ST_Y(ST_EndPoint(geom::geometry)) as lat,
                        ST_X(ST_EndPoint(geom::geometry)) as lng,
                        ST_Distance(
                            ST_GeogFromText('POINT(:lng :lat)'),
                            ST_EndPoint(geom::geometry)::geography
                        ) as distance_m
                    FROM road_segment
                    WHERE target IS NOT NULL
                    AND road_type IN ('motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service', 'unclassified')
                    {boundary_filter}
                ),
                nearest_point AS (
                    SELECT 
                        vertex_id,
                        lat,
                        lng,
                        distance_m
                    FROM segment_points
                    WHERE distance_m <= :max_distance_m
                    ORDER BY distance_m
                    LIMIT 1
                )
                SELECT lat, lng, vertex_id
                FROM nearest_point
            """),
            {
                "lat": lat,
                "lng": lng,
                "max_distance_m": max_distance_m
            }
        ).first()
        
        if result:
            logger.debug(f"Snapped point ({lat}, {lng}) to vertex {result.vertex_id} using fallback method")
            return (float(result.lat), float(result.lng), int(result.vertex_id))
        
        return None
    except Exception as e:
        logger.warning(f"Failed to snap to road network: {str(e)}")
        return None


def _compute_route_with_pgrouting(
    db: Session,
    start_vertex: int,
    end_vertex: int,
    station_id: Optional[str] = None
) -> Optional[Tuple[List[Tuple[float, float]], float, dict]]:
    """
    Compute route between two vertices using pgRouting Dijkstra algorithm.
    Returns (path_coordinates, total_distance_m, geojson_path) or None if route not found.
    path_coordinates: List of (lat, lng) tuples
    geojson_path: GeoJSON LineString dict with [lng, lat] coordinates
    Uses road segments within station's neighborhood boundaries if station_id provided, otherwise Küçükçekmece boundary.
    """
    try:
        # Build pgRouting query with boundary filter
        # Use station's neighborhood boundaries if station_id provided, otherwise Küçükçekmece boundary
        from app.services.utils import get_station_neighborhood_boundaries
        
        boundary_filter = ""
        if station_id:
            # Check if station has neighborhoods
            station_neighborhoods = db.execute(
                text("""
                    SELECT neighborhoods
                    FROM police_station
                    WHERE id = :station_id::uuid
                """),
                {"station_id": station_id}
            ).scalar()
            
            if station_neighborhoods:
                # Use station's neighborhood boundaries
                boundary_filter = """
                    AND ST_Within(
                        ST_Centroid(geom::geometry),
                        (SELECT ST_Union(ab.geom::geometry)::geometry
                         FROM administrative_boundary ab
                         JOIN police_station ps ON ab.name = ANY(ps.neighborhoods)
                         WHERE ps.id = '{station_id}'::uuid
                         AND ab.admin_level = 8)
                    )
                    AND (
                        ST_Length(
                            ST_Intersection(
                                geom::geometry,
                                (SELECT ST_Union(ab.geom::geometry)::geometry
                                 FROM administrative_boundary ab
                                 JOIN police_station ps ON ab.name = ANY(ps.neighborhoods)
                                 WHERE ps.id = '{station_id}'::uuid
                                 AND ab.admin_level = 8)
                            )
                        ) / NULLIF(ST_Length(geom::geometry), 0)
                    ) >= 0.3
                """.format(station_id=station_id)
        
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
                    AND ST_Within(
                        ST_Centroid(geom::geometry),
                        (SELECT geom::geometry FROM administrative_boundary 
                         WHERE name = 'Küçükçekmece' AND admin_level = 6 LIMIT 1)
                    )
                    AND (
                        ST_Length(
                            ST_Intersection(
                                geom::geometry,
                                (SELECT geom::geometry FROM administrative_boundary 
                                 WHERE name = 'Küçükçekmece' AND admin_level = 6 LIMIT 1)
                            )
                        ) / NULLIF(ST_Length(geom::geometry), 0)
                    ) >= 0.5
                """
        
        # Build pgRouting SQL query - filter by boundary if it exists
        # Note: pgRouting requires the query as a string, so we build it dynamically
        pgrouting_query = f"""SELECT id, source, target, cost, reverse_cost 
            FROM road_segment 
            WHERE road_type IN ('motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service', 'unclassified')
            {boundary_filter}"""
        
        # Get route with detailed edge geometries - extract all points from each edge
        # Only use driveable road segments from OSM within Küçükçekmece boundary
        # Note: pgRouting query must be passed as a string literal, not a parameter
        result = db.execute(
            text(f"""
                SELECT 
                    SUM(ST_Length(r.geom::geography)) as total_distance,
                    json_agg(
                        json_build_object(
                            'edge_id', r.id,
                            'geom', ST_AsGeoJSON(r.geom)::json
                        ) ORDER BY di.seq
                    ) as edge_details
                FROM (
                    SELECT 
                        r.geom,
                        r.id,
                        di.seq
                    FROM pgr_dijkstra(
                        '{pgrouting_query.replace("'", "''")}',
                        :start_vertex,
                        :end_vertex,
                        directed := true
                    ) AS di
                    JOIN road_segment r ON di.edge = r.id
                    ORDER BY di.seq
                ) AS route
            """),
            {
                "start_vertex": start_vertex,
                "end_vertex": end_vertex
            }
        ).first()
        
        if not result or not result.edge_details:
            return None
        
        import json
        edge_details = result.edge_details if isinstance(result.edge_details, list) else json.loads(result.edge_details)
        
        # Extract all coordinates from each edge to create detailed path
        path_coords = []  # List of (lat, lng) tuples
        geojson_coords = []  # List of [lng, lat] for GeoJSON
        
        for edge_detail in edge_details:
            edge_geom = edge_detail.get("geom", {})
            if edge_geom.get("type") == "LineString":
                edge_coords = edge_geom.get("coordinates", [])
                # GeoJSON is [lng, lat], convert to (lat, lng) tuples for path_coords
                path_coords.extend([(coord[1], coord[0]) for coord in edge_coords])
                # Add all coordinates from this edge to create detailed path
                geojson_coords.extend(edge_coords)  # Keep as [lng, lat] for GeoJSON
        
        if not path_coords:
            return None
        
        total_distance = float(result.total_distance) if result.total_distance else 0.0
        
        # Fallback: If total_distance is 0 but we have path coordinates, calculate distance from coordinates
        # This can happen if ST_Length returns NULL or 0 for some reason
        if total_distance == 0.0 and len(path_coords) >= 2:
            import math
            R = 6371000  # Earth radius in meters
            calculated_distance = 0.0
            for i in range(len(path_coords) - 1):
                lat1, lng1 = path_coords[i]
                lat2, lng2 = path_coords[i + 1]
                dlat = math.radians(lat2 - lat1)
                dlng = math.radians(lng2 - lng1)
                a = (math.sin(dlat / 2) ** 2 +
                     math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                     math.sin(dlng / 2) ** 2)
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                calculated_distance += R * c
            if calculated_distance > 0:
                total_distance = calculated_distance
                logger.info(f"Calculated distance from GeoJSON coordinates: {total_distance:.0f}m (fallback calculation)")
        
        # Remove duplicate consecutive points
        if geojson_coords:
            cleaned_geojson = [geojson_coords[0]]
            for coord in geojson_coords[1:]:
                if coord != cleaned_geojson[-1]:
                    cleaned_geojson.append(coord)
            geojson_coords = cleaned_geojson
        
        # Create GeoJSON LineString with detailed path
        geojson_path = {
            "type": "LineString",
            "coordinates": geojson_coords
        }
        
        return (path_coords, total_distance, geojson_path)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"pgRouting route computation failed: {str(e)}")
        return None


def compute_route_via_points(
    db: Session,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    waypoints: List[Tuple[float, float]],
    max_distance_m: float,
    waypoint_risk_scores: Optional[List[float]] = None,
    station_id: Optional[str] = None
) -> RouteResult:
    """
    Compute route using pgRouting on OSM road network.
    Falls back to straight-line route if pgRouting is unavailable or fails.
    
    Args:
        db: Database session
        start_lat, start_lng: Start point coordinates
        end_lat, end_lng: End point coordinates
        waypoints: List of (lat, lng) tuples for risk cluster locations
        max_distance_m: Maximum route distance in meters
        waypoint_risk_scores: Optional list of risk scores for each waypoint
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"compute_route_via_points called: start=({start_lat}, {start_lng}), end=({end_lat}, {end_lng}), waypoints={len(waypoints)}")
    
    # Check if pgRouting topology exists
    try:
        has_topology = db.execute(
            text("""
                SELECT COUNT(*) > 0
                FROM road_segment
                WHERE source IS NOT NULL AND target IS NOT NULL
            """)
        ).scalar()
        
        if not has_topology:
            logger.warning("No pgRouting topology found, using fallback straight-line route")
            return _compute_fallback_route(start_lat, start_lng, end_lat, end_lng, waypoints)
        
        # Try to use pgRouting for route computation
        # All points should already be snapped, but ensure they are
        all_points = [(start_lat, start_lng)] + waypoints + [(end_lat, end_lng)]
        
        # If no waypoints and start=end, try to create a patrol route from nearby road segments
        if len(waypoints) == 0 and abs(start_lat - end_lat) < 0.0001 and abs(start_lng - end_lng) < 0.0001:
            logger.info("No waypoints and start=end, attempting to create patrol route from nearby road network")
            # Find nearby road segments to create a patrol loop
            nearby_vertices = db.execute(
                text("""
                    SELECT 
                        v.id as vertex_id,
                        ST_Y(v.the_geom::geometry) as lat,
                        ST_X(v.the_geom::geometry) as lng,
                        ST_Distance(
                            ST_GeogFromText('POINT(:lng :lat)'),
                            v.the_geom::geography
                        ) as distance_m
                    FROM road_segment_vertices_pgr v
                    WHERE ST_DWithin(
                        ST_GeogFromText('POINT(:lng :lat)'),
                        v.the_geom::geography,
                        1000.0
                    )
                    ORDER BY distance_m
                    LIMIT 8
                """),
                {"lat": start_lat, "lng": start_lng}
            ).all()
            
            if nearby_vertices:
                # Use these vertices as waypoints for patrol route
                for vertex in nearby_vertices:
                    waypoints.append((float(vertex.lat), float(vertex.lng)))
                logger.info(f"Found {len(waypoints)} nearby road vertices for patrol route")
                all_points = [(start_lat, start_lng)] + waypoints + [(end_lat, end_lng)]
        
        # Snap all points to road network (ensure they are on network)
        snapped_points = []
        for lat, lng in all_points:
            snapped = _snap_to_road_network(db, lat, lng, max_distance_m=100.0)
            if snapped:
                snapped_points.append(snapped)
            else:
                # If can't snap, use original point (will use fallback for this segment)
                snapped_points.append((lat, lng, None))
                logger.warning(f"Could not snap point ({lat}, {lng}) to road network")
        
        # Compute route segments between consecutive points using pgRouting
        full_path_coords = []  # List of (lat, lng) tuples
        full_geojson_coords = []  # List of [lng, lat] for GeoJSON
        route_waypoints = []
        total_distance = 0.0
        
        # If we only have start and end (no waypoints), try to create a loop route
        loop_route_created = False
        if len(snapped_points) == 2 and abs(snapped_points[0][0] - snapped_points[1][0]) < 0.0001 and abs(snapped_points[0][1] - snapped_points[1][1]) < 0.0001:
            logger.info("Start and end are the same, creating patrol loop route")
            # Create a loop route: find nearby vertices and create a circular path
            start_point = snapped_points[0]
            if start_point[2] is not None:
                # Find nearby vertices connected to start vertex
                nearby_edges = db.execute(
                    text("""
                        SELECT DISTINCT
                            CASE WHEN rs.source = :start_vertex THEN rs.target ELSE rs.source END as connected_vertex,
                            ST_Y(v.the_geom::geometry) as lat,
                            ST_X(v.the_geom::geometry) as lng
                        FROM road_segment rs
                        JOIN road_segment_vertices_pgr v ON (
                            (rs.source = :start_vertex AND v.id = rs.target) OR
                            (rs.target = :start_vertex AND v.id = rs.source)
                        )
                        WHERE (rs.source = :start_vertex OR rs.target = :start_vertex)
                        AND rs.road_type IN ('motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service', 'unclassified')
                        LIMIT 4
                    """),
                    {"start_vertex": start_point[2]}
                ).all()
                
                if nearby_edges:
                    # Create a loop: start -> nearby vertices -> back to start
                    loop_vertices = [start_point]
                    for edge in nearby_edges[:3]:  # Use up to 3 nearby vertices
                        loop_vertices.append((float(edge.lat), float(edge.lng), int(edge.connected_vertex)))
                    loop_vertices.append(start_point)  # Return to start
                    
                    # Compute route through loop vertices
                    for i in range(len(loop_vertices) - 1):
                        v1 = loop_vertices[i]
                        v2 = loop_vertices[i + 1]
                        if v1[2] is not None and v2[2] is not None:
                            route_result = _compute_route_with_pgrouting(db, v1[2], v2[2], station_id=station_id)
                            if route_result:
                                segment_path, segment_distance, segment_geojson = route_result
                                full_path_coords.extend(segment_path)
                                full_geojson_coords.extend(segment_geojson["coordinates"])
                                total_distance += segment_distance
                                route_waypoints.append(RouteWaypoint(
                                    lat=v1[0],
                                    lng=v1[1],
                                    risk_score=None
                                ))
                    
                    if total_distance > 0:
                        route_waypoints.append(RouteWaypoint(
                            lat=start_point[0],
                            lng=start_point[1],
                            risk_score=None
                        ))
                        loop_route_created = True
        
        # Normal route computation if loop route was not created
        if not loop_route_created:
            for i in range(len(snapped_points) - 1):
                point1 = snapped_points[i]
                point2 = snapped_points[i + 1]
                
                # If both points are snapped to vertices, use pgRouting
                if point1[2] is not None and point2[2] is not None:
                    route_result = _compute_route_with_pgrouting(db, point1[2], point2[2], station_id=station_id)
                    if route_result:
                        segment_path, segment_distance, segment_geojson = route_result
                        # segment_path is (lat, lng) tuples
                        # segment_geojson is GeoJSON with [lng, lat] coordinates
                        full_path_coords.extend(segment_path)
                        # Add all coordinates from the segment (detailed path)
                        full_geojson_coords.extend(segment_geojson["coordinates"])
                        total_distance += segment_distance
                        
                        # Add waypoint at segment start
                        # i=0 is start station, i=1 to len(waypoints) are risk clusters
                        risk_score = None
                        if waypoint_risk_scores and i > 0 and (i - 1) < len(waypoint_risk_scores):
                            risk_score = waypoint_risk_scores[i - 1]
                        
                        route_waypoints.append(RouteWaypoint(
                            lat=point1[0],
                            lng=point1[1],
                            risk_score=risk_score
                        ))
                        continue
            
            # Fallback: straight line for this segment if pgRouting fails
            import math
            R = 6371000  # Earth radius in meters
            lat1, lng1 = point1[0], point1[1]
            lat2, lng2 = point2[0], point2[1]
            
            dlat = math.radians(lat2 - lat1)
            dlng = math.radians(lng2 - lng1)
            a = (math.sin(dlat / 2) ** 2 +
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                 math.sin(dlng / 2) ** 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            segment_distance = R * c
            total_distance += segment_distance
            
            # Add straight line segment
            full_path_coords.append((lat1, lng1))
            full_path_coords.append((lat2, lng2))
            full_geojson_coords.append([lng1, lat1])
            full_geojson_coords.append([lng2, lat2])
            
            # Add waypoint with risk score if available
            risk_score = None
            if waypoint_risk_scores and i > 0 and (i - 1) < len(waypoint_risk_scores):
                risk_score = waypoint_risk_scores[i - 1]
            
            route_waypoints.append(RouteWaypoint(
                lat=lat1,
                lng=lng1,
                risk_score=risk_score
            ))
        
        # Add final point (end station)
        if snapped_points:
            final_point = snapped_points[-1]
            route_waypoints.append(RouteWaypoint(
                lat=final_point[0],
                lng=final_point[1],
                risk_score=None
            ))
        
        # Remove duplicate consecutive points from GeoJSON coordinates
        if full_geojson_coords:
            cleaned_geojson = [full_geojson_coords[0]]
            for coord in full_geojson_coords[1:]:
                # Only add if different from last point (avoid duplicates)
                if coord != cleaned_geojson[-1]:
                    cleaned_geojson.append(coord)
            full_geojson_coords = cleaned_geojson
        
        # Ensure we have at least 2 points for a valid LineString
        if len(full_geojson_coords) < 2:
            logger.warning(f"Route path has only {len(full_geojson_coords)} points, adding start and end points")
            # If we don't have enough points, add start and end
            if snapped_points:
                start_point = snapped_points[0]
                end_point = snapped_points[-1]
                full_geojson_coords = [
                    [start_point[1], start_point[0]],  # [lng, lat]
                    [end_point[1], end_point[0]]
                ]
            else:
                # Fallback: use original points
                full_geojson_coords = [
                    [start_lng, start_lat],
                    [end_lng, end_lat]
                ]
        
        # Fallback: If total_distance is still 0 but we have coordinates, calculate from GeoJSON
        if total_distance == 0.0 and len(full_geojson_coords) >= 2:
            import math
            R = 6371000  # Earth radius in meters
            calculated_distance = 0.0
            for i in range(len(full_geojson_coords) - 1):
                lng1, lat1 = full_geojson_coords[i]
                lng2, lat2 = full_geojson_coords[i + 1]
                dlat = math.radians(lat2 - lat1)
                dlng = math.radians(lng2 - lng1)
                a = (math.sin(dlat / 2) ** 2 +
                     math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                     math.sin(dlng / 2) ** 2)
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                calculated_distance += R * c
            if calculated_distance > 0:
                total_distance = calculated_distance
                logger.info(f"Calculated total distance from GeoJSON coordinates: {total_distance:.0f}m (fallback)")
        
        # Estimate time (assuming average speed of 30 km/h)
        total_time = (total_distance / 1000.0) / 30.0 * 60.0 if total_distance > 0 else 0.0
        
        # Create GeoJSON LineString with detailed path from OSM edges
        path = {
            "type": "LineString",
            "coordinates": full_geojson_coords  # Already in [lng, lat] format
        }
        
        # Calculate risk coverage (simplified)
        risk_coverage = min(1.0, len(waypoints) / 10.0)
        
        logger.info(f"Route computed using OSM edges: {len(full_geojson_coords)} GeoJSON points, {len(route_waypoints)} waypoints, {total_distance:.0f}m, {total_time:.1f}min")
        
        # Debug: Log if path is too short or distance is 0
        if len(full_geojson_coords) < 10:
            logger.warning(f"Route path has only {len(full_geojson_coords)} points, may appear as dots on map")
        if total_distance == 0.0:
            logger.warning(f"Route total_distance is 0 - check if route segments were found and distance calculated correctly")
        
        return RouteResult(
            waypoints=route_waypoints,
            total_distance=total_distance,
            total_time=total_time,
            risk_coverage=risk_coverage,
            path=path
        )
        
    except Exception as e:
        logger.warning(f"pgRouting route computation failed: {str(e)}, using fallback")
        return _compute_fallback_route(start_lat, start_lng, end_lat, end_lng, waypoints)


def _compute_fallback_route(
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    waypoints: List[Tuple[float, float]]
) -> RouteResult:
    """
    Fallback route computation using straight lines between points.
    Used when pgRouting is unavailable or fails.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"_compute_fallback_route called: start=({start_lat}, {start_lng}), end=({end_lat}, {end_lng}), waypoints={len(waypoints)}")
    
    all_points = [(start_lat, start_lng)] + waypoints + [(end_lat, end_lng)]
    
    # If start and end are the same and no waypoints, create a small circular route
    if len(all_points) == 2 and abs(start_lat - end_lat) < 0.0001 and abs(start_lng - end_lng) < 0.0001:
        logger.warning("Start and end are the same with no waypoints, creating a small circular route")
        # Create a small square around the point (about 100m radius)
        import math
        R = 6371000  # Earth radius in meters
        lat_offset = 100.0 / R * 180.0 / math.pi  # ~100m in degrees
        lng_offset = 100.0 / (R * math.cos(math.radians(start_lat))) * 180.0 / math.pi
        
        all_points = [
            (start_lat, start_lng),
            (start_lat + lat_offset, start_lng),
            (start_lat + lat_offset, start_lng + lng_offset),
            (start_lat, start_lng + lng_offset),
            (start_lat, start_lng)  # Return to start
        ]
    
    route_waypoints = []
    total_distance = 0.0
    
    for i in range(len(all_points) - 1):
        lat1, lng1 = all_points[i]
        lat2, lng2 = all_points[i + 1]
        
        # Calculate distance (Haversine approximation)
        import math
        R = 6371000  # Earth radius in meters
        
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        
        total_distance += distance
        
        route_waypoints.append(RouteWaypoint(
            lat=lat1,
            lng=lng1,
            risk_score=None
        ))
    
    # Add final point
    route_waypoints.append(RouteWaypoint(
        lat=end_lat,
        lng=end_lng,
        risk_score=None
    ))
    
    # Estimate time (assuming average speed of 30 km/h)
    total_time = (total_distance / 1000.0) / 30.0 * 60.0
    
    # Create GeoJSON LineString
    coordinates = [[lng, lat] for lat, lng in all_points]
    path = {
        "type": "LineString",
        "coordinates": coordinates
    }
    
    # Calculate risk coverage (simplified)
    risk_coverage = min(1.0, len(waypoints) / 10.0)
    
    return RouteResult(
        waypoints=route_waypoints,
        total_distance=total_distance,
        total_time=total_time,
        risk_coverage=risk_coverage,
        path=path
    )


def compute_route(
    db: Session,
    request: RouteRequest
) -> RouteResult:
    """
    Main route optimization function.
    """
    # Get start station coordinates (snapped to road network)
    start_lat, start_lng = get_station_coordinates_snapped(db, request.station_id, max_distance_m=100.0)
    
    # Determine end station (snapped to road network)
    if request.end_station_id:
        end_lat, end_lng = get_station_coordinates_snapped(db, request.end_station_id, max_distance_m=100.0)
    else:
        # Return to start station
        end_lat, end_lng = start_lat, start_lng
    
    # Get high-risk cells within Küçükçekmece polygon boundaries
    from app.core.config import get_settings
    from app.services.utils import get_kucukcekmece_bbox_from_polygon, get_kucukcekmece_boundary
    settings = get_settings()
    
    # Check if start station is within Küçükçekmece polygon
    boundary_geom = get_kucukcekmece_boundary(db)
    if boundary_geom:
        # Check if station is within polygon
        # ST_Within requires geometry type, so cast geography to geometry
        station_within = db.execute(
            text("""
                SELECT ST_Within(
                    ST_GeomFromText('POINT(:lng :lat)', 4326),
                    (SELECT geom::geometry FROM administrative_boundary 
                     WHERE name = 'Küçükçekmece' AND admin_level = 6 LIMIT 1)
                )
            """),
            {"lat": start_lat, "lng": start_lng}
        ).scalar()
        
        if station_within:
            # Use polygon boundary (no bbox needed, polygon filter will be applied in get_high_risk_cells)
            bbox = None
        else:
            # Station outside Küçükçekmece, use 5km radius
            bbox = (
                start_lat - 0.045,  # ~5km
                start_lng - 0.045,
                start_lat + 0.045,
                start_lng + 0.045
            )
    else:
        # Fallback to bbox if polygon not available
        kucukcekmece_bbox = get_kucukcekmece_bbox_from_polygon(db) or settings.kucukcekmece_fallback_bbox
        if (kucukcekmece_bbox[0] <= start_lat <= kucukcekmece_bbox[2] and
            kucukcekmece_bbox[1] <= start_lng <= kucukcekmece_bbox[3]):
            bbox = kucukcekmece_bbox
        else:
            bbox = (
                start_lat - 0.045,
                start_lng - 0.045,
                start_lat + 0.045,
                start_lng + 0.045
            )
    
    import logging
    logger = logging.getLogger(__name__)
    
    # Use road-segment-based risk instead of grid-based risk cells
    # This ensures routes follow actual road network
    from app.models.road_segment import RoadSegment
    
    high_risk_segments = get_high_risk_road_segments(
        db,
        request.risk_threshold,
        bbox,
        request.start_time,
        request.end_time,
        limit=100,
        station_id=str(request.station_id)
    )
    
    logger.info(f"Found {len(high_risk_segments)} road segments with risk >= {request.risk_threshold}")
    
    # Extract waypoints from road segment centers
    # These are already on the road network, so no snapping needed
    waypoints = []
    waypoint_risk_scores = []
    
    for segment in high_risk_segments:
        # Get segment center point
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
        
        if center_result:
            waypoints.append((float(center_result.lat), float(center_result.lng)))
            waypoint_risk_scores.append(segment.risk_score)
    
    # Limit to max 10 waypoints for route optimization
    if len(waypoints) > 10:
        # Sort by risk score and take top 10
        sorted_indices = sorted(range(len(waypoint_risk_scores)), 
                               key=lambda i: waypoint_risk_scores[i], 
                               reverse=True)[:10]
        waypoints = [waypoints[i] for i in sorted_indices]
        waypoint_risk_scores = [waypoint_risk_scores[i] for i in sorted_indices]
    
    logger.info(f"Created {len(waypoints)} waypoints from high-risk road segments")
    
    # Use old risk_cells approach as fallback if no road segments found
    if not waypoints:
        logger.warning("No high-risk road segments found, falling back to grid-based risk cells")
        risk_cells = get_high_risk_cells(
            db, 
            request.risk_threshold, 
            bbox,
            request.start_time,
            request.end_time
        )
        
        # Cluster risk cells
        risk_clusters = cluster_risk_cells(db, risk_cells, max_clusters=10)
        
        logger.info(f"Found {len(risk_cells)} risk cells above threshold {request.risk_threshold}, clustered into {len(risk_clusters)} clusters")
        
        # Extract waypoints (cluster centers) - these are the risk forecast locations for patrol
        waypoints = [(lat, lng) for lat, lng, _ in risk_clusters]
        waypoint_risk_scores = [risk for _, _, risk in risk_clusters]
        logger.info(f"Created {len(waypoints)} waypoints from risk clusters (fallback)")
    
    # If still no waypoints found, create a patrol route using nearby road segments
    # This ensures we always have a meaningful route even without risk data
    if not waypoints:
        logger.warning(f"No waypoints found (no risk data), creating patrol route from nearby road segments")
        
        # Get nearby road segments within station's neighborhoods to create a patrol route
        # Find road segments near the station (within 2km radius) to create waypoints
        nearby_segments = db.execute(
            text("""
                SELECT 
                    rs.id,
                    ST_Y(ST_Centroid(rs.geom::geometry)) as lat,
                    ST_X(ST_Centroid(rs.geom::geometry)) as lng,
                    ST_Distance(
                        ST_GeogFromText('POINT(:lng :lat)'),
                        rs.geom
                    ) as distance_m
                FROM road_segment rs
                WHERE rs.road_type IN ('motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service', 'unclassified')
                AND ST_DWithin(
                    ST_GeogFromText('POINT(:lng :lat)'),
                    rs.geom,
                    2000.0
                )
                ORDER BY distance_m
                LIMIT 20
            """),
            {"lat": start_lat, "lng": start_lng}
        ).all()
        
        if nearby_segments:
            # Select evenly distributed segments for waypoints (max 8 waypoints)
            import math
            num_waypoints = min(8, len(nearby_segments))
            step = max(1, len(nearby_segments) // num_waypoints)
            
            for i in range(0, len(nearby_segments), step):
                if len(waypoints) >= num_waypoints:
                    break
                seg = nearby_segments[i]
                waypoints.append((float(seg.lat), float(seg.lng)))
                waypoint_risk_scores.append(0.0)  # No risk data
            
            logger.info(f"Created {len(waypoints)} waypoints from nearby road segments (no risk data available)")
    
    # Final fallback: If still no waypoints, create a small circular patrol route
    if not waypoints:
        logger.warning("No waypoints found even after all fallbacks, creating minimal circular patrol route")
        # This will be handled by compute_route_via_points with empty waypoints list
        # which will create a small circular route in _compute_fallback_route
    
    # Limit waypoints based on max time
    # Rough estimate: each waypoint adds ~10 minutes
    max_waypoints = max(1, request.max_minutes // 10)
    waypoints = waypoints[:max_waypoints]
    
    # Compute route
    # max_distance_m: maximum total route distance
    # Assuming average speed of 30 km/h = 500 m/min
    max_distance_m = request.max_minutes * 500
    
    # waypoint_risk_scores is already set above (either from road segments or clusters)
    route = compute_route_via_points(
        db,
        start_lat,
        start_lng,
        end_lat,
        end_lng,
        waypoints,
        max_distance_m,
        waypoint_risk_scores=waypoint_risk_scores if waypoint_risk_scores else None,
        station_id=str(request.station_id)
    )
    
    return route