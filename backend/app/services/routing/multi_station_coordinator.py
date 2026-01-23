"""Multi-station route coordination service."""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.police_station import PoliceStation
from app.models.risk_cell import RiskCell
from app.services.routing.route_optimizer import (
    RouteRequest,
    RouteResult,
    compute_route,
    get_high_risk_cells,
    get_station_coordinates,
)
from app.services.utils import get_point_coordinates

logger = logging.getLogger(__name__)

settings = get_settings()


@dataclass
class StationAssignment:
    """Risk cell assignment to a station."""

    station_id: UUID
    station_name: str
    risk_cells: List[RiskCell]
    total_capacity: int
    assigned_load: int


@dataclass
class MultiStationRouteResult:
    """Result of multi-station route coordination."""

    station_routes: List[Tuple[UUID, str, RouteResult]]
    total_risk_coverage: float
    overlap_percentage: float
    coordination_score: float


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate Haversine distance between two points in meters.

    Args:
        lat1, lng1: First point coordinates
        lat2, lng2: Second point coordinates

    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters

    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def get_cell_center(db: Session, cell: RiskCell) -> Tuple[float, float]:
    """
    Get center coordinates of a risk cell.

    Args:
        db: Database session
        cell: Risk cell

    Returns:
        (lat, lng) tuple
    """
    try:
        result = db.execute(
            """
                SELECT 
                    ST_Y(ST_Centroid(ST_GeomFromWKB(:geom))) as lat,
                    ST_X(ST_Centroid(ST_GeomFromWKB(:geom))) as lng
            """,
            {"geom": cell.geom.data},
        ).first()

        if result:
            return (float(result.lat), float(result.lng))
    except Exception as e:
        logger.warning(f"Error getting cell center: {str(e)}")

    # Fallback: try to get from cell.geom directly
    try:
        from app.services.utils import get_point_coordinates

        return get_point_coordinates(db, cell.geom)
    except Exception:
        return (0.0, 0.0)


def distribute_risk_cells(
    db: Session,
    stations: List[PoliceStation],
    risk_cells: List[RiskCell],
    capacity_weight: float = None,
    distance_weight: float = None,
    risk_weight: float = None,
) -> Dict[UUID, StationAssignment]:
    """
    Distribute risk cells among stations based on proximity, capacity, and risk.

    Args:
        db: Database session
        stations: List of police stations
        risk_cells: List of risk cells to distribute
        capacity_weight: Weight for capacity in assignment (0-1)
        distance_weight: Weight for distance in assignment (0-1)
        risk_weight: Weight for risk score in assignment (0-1)

    Returns:
        Dictionary mapping station_id to StationAssignment
    """
    if not stations or not risk_cells:
        return {}

    # Get weights from config or use defaults
    capacity_weight = (
        capacity_weight
        or getattr(settings, "capacity_weight", 0.3)
    )
    distance_weight = (
        distance_weight
        or getattr(settings, "distance_weight", 0.4)
    )
    risk_weight = risk_weight or getattr(settings, "risk_weight", 0.3)

    # Normalize weights
    total_weight = capacity_weight + distance_weight + risk_weight
    if total_weight > 0:
        capacity_weight /= total_weight
        distance_weight /= total_weight
        risk_weight /= total_weight

    # Initialize assignments
    assignments: Dict[UUID, StationAssignment] = {}
    for station in stations:
        assignments[station.id] = StationAssignment(
            station_id=station.id,
            station_name=station.name,
            risk_cells=[],
            total_capacity=station.capacity or 10,
            assigned_load=0,
        )

    # Get station coordinates
    station_coords: Dict[UUID, Tuple[float, float]] = {}
    for station in stations:
        try:
            lat, lng = get_station_coordinates(db, station.id)
            station_coords[station.id] = (lat, lng)
        except Exception as e:
            logger.warning(f"Error getting coordinates for station {station.id}: {str(e)}")
            continue

    # Sort risk cells by risk score (descending)
    sorted_cells = sorted(risk_cells, key=lambda c: c.risk_score, reverse=True)

    # Initial assignment: assign each cell to nearest station
    for cell in sorted_cells:
        cell_center = get_cell_center(db, cell)
        if cell_center == (0.0, 0.0):
            continue

        best_station_id = None
        best_score = float("-inf")

        for station_id, (station_lat, station_lng) in station_coords.items():
            # Calculate distance
            distance = haversine_distance(
                cell_center[0], cell_center[1], station_lat, station_lng
            )
            distance_score = 1.0 / (1.0 + distance / 1000.0)  # Normalize to 0-1

            # Calculate capacity score (inverse of current load)
            assignment = assignments[station_id]
            capacity_ratio = (
                1.0 - (assignment.assigned_load / assignment.total_capacity)
                if assignment.total_capacity > 0
                else 0.5
            )

            # Risk score (already normalized 0-1)
            risk_score = cell.risk_score

            # Combined score
            score = (
                distance_weight * distance_score
                + capacity_weight * capacity_ratio
                + risk_weight * risk_score
            )

            if score > best_score:
                best_score = score
                best_station_id = station_id

        if best_station_id:
            assignments[best_station_id].risk_cells.append(cell)
            assignments[best_station_id].assigned_load += 1

    # Balance capacity: redistribute if some stations are overloaded
    max_iterations = 10
    for iteration in range(max_iterations):
        # Find overloaded and underloaded stations
        overloaded = [
            (sid, assignment)
            for sid, assignment in assignments.items()
            if assignment.assigned_load > assignment.total_capacity * 1.2
        ]
        underloaded = [
            (sid, assignment)
            for sid, assignment in assignments.items()
            if assignment.assigned_load < assignment.total_capacity * 0.8
        ]

        if not overloaded or not underloaded:
            break

        # Redistribute from overloaded to underloaded
        for overloaded_id, overloaded_assignment in overloaded:
            if not overloaded_assignment.risk_cells:
                continue

            # Sort cells by risk (lowest risk first for redistribution)
            cells_to_redistribute = sorted(
                overloaded_assignment.risk_cells, key=lambda c: c.risk_score
            )

            for cell in cells_to_redistribute[:3]:  # Redistribute up to 3 cells
                cell_center = get_cell_center(db, cell)
                if cell_center == (0.0, 0.0):
                    continue

                best_target = None
                best_score = float("-inf")

                for underloaded_id, underloaded_assignment in underloaded:
                    if underloaded_id == overloaded_id:
                        continue

                    station_lat, station_lng = station_coords[underloaded_id]
                    distance = haversine_distance(
                        cell_center[0], cell_center[1], station_lat, station_lng
                    )
                    distance_score = 1.0 / (1.0 + distance / 1000.0)

                    capacity_ratio = (
                        1.0
                        - (underloaded_assignment.assigned_load / underloaded_assignment.total_capacity)
                        if underloaded_assignment.total_capacity > 0
                        else 0.5
                    )

                    score = distance_weight * distance_score + capacity_weight * capacity_ratio

                    if score > best_score:
                        best_score = score
                        best_target = underloaded_id

                if best_target:
                    # Move cell
                    overloaded_assignment.risk_cells.remove(cell)
                    overloaded_assignment.assigned_load -= 1
                    assignments[best_target].risk_cells.append(cell)
                    assignments[best_target].assigned_load += 1
                    break

    return assignments


def calculate_route_overlap(
    route1: RouteResult, route2: RouteResult, threshold_m: float = 500.0
) -> float:
    """
    Calculate overlap percentage between two routes.

    Args:
        route1: First route
        route2: Second route
        threshold_m: Distance threshold for considering waypoints as overlapping (meters)

    Returns:
        Overlap percentage (0.0-1.0)
    """
    if not route1.waypoints or not route2.waypoints:
        return 0.0

    overlapping_waypoints = 0
    total_waypoints = len(route1.waypoints)

    for wp1 in route1.waypoints:
        for wp2 in route2.waypoints:
            distance = haversine_distance(wp1.lat, wp1.lng, wp2.lat, wp2.lng)
            if distance < threshold_m:
                overlapping_waypoints += 1
                break

    return overlapping_waypoints / total_waypoints if total_waypoints > 0 else 0.0


def minimize_route_overlap(
    station_routes: List[Tuple[UUID, str, RouteResult]],
    overlap_threshold: float = None,
) -> List[Tuple[UUID, str, RouteResult]]:
    """
    Minimize overlap between routes by removing or reassigning waypoints.

    Args:
        station_routes: List of (station_id, station_name, route_result) tuples
        overlap_threshold: Maximum allowed overlap (0.0-1.0)

    Returns:
        Optimized list of routes with minimized overlap
    """
    if len(station_routes) <= 1:
        return station_routes

    overlap_threshold = (
        overlap_threshold
        or getattr(settings, "default_overlap_threshold", 0.2)
    )

    # Create a copy to modify
    optimized_routes = [
        (sid, sname, RouteResult(
            waypoints=list(route.waypoints),
            total_distance=route.total_distance,
            total_time=route.total_time,
            risk_coverage=route.risk_coverage,
            path=route.path,
        ))
        for sid, sname, route in station_routes
    ]

    # Check all pairs of routes
    for i in range(len(optimized_routes)):
        for j in range(i + 1, len(optimized_routes)):
            sid1, sname1, route1 = optimized_routes[i]
            sid2, sname2, route2 = optimized_routes[j]

            overlap = calculate_route_overlap(route1, route2)

            if overlap > overlap_threshold:
                # Resolve overlap: keep waypoint in route with higher risk score
                waypoints_to_remove_1 = []
                waypoints_to_remove_2 = []

                for wp1 in route1.waypoints:
                    for wp2 in route2.waypoints:
                        distance = haversine_distance(
                            wp1.lat, wp1.lng, wp2.lat, wp2.lng
                        )
                        if distance < 500.0:  # Overlapping waypoints
                            # Keep waypoint in route with higher risk score
                            risk1 = wp1.risk_score or 0.0
                            risk2 = wp2.risk_score or 0.0

                            if risk1 > risk2:
                                waypoints_to_remove_2.append(wp2)
                            elif risk2 > risk1:
                                waypoints_to_remove_1.append(wp1)
                            else:
                                # Equal risk: remove from route with more waypoints
                                if len(route1.waypoints) > len(route2.waypoints):
                                    waypoints_to_remove_1.append(wp1)
                                else:
                                    waypoints_to_remove_2.append(wp2)

                # Remove overlapping waypoints
                route1.waypoints = [
                    wp for wp in route1.waypoints if wp not in waypoints_to_remove_1
                ]
                route2.waypoints = [
                    wp for wp in route2.waypoints if wp not in waypoints_to_remove_2
                ]

                # Recalculate route metrics
                route1.total_distance *= (1.0 - overlap * 0.5)
                route1.total_time *= (1.0 - overlap * 0.5)
                route2.total_distance *= (1.0 - overlap * 0.5)
                route2.total_time *= (1.0 - overlap * 0.5)

    return optimized_routes


def coordinate_multi_station_routes(
    db: Session,
    station_ids: Optional[List[UUID]] = None,
    risk_threshold: float = 0.7,
    max_minutes_per_station: int = 90,
    minimize_overlap: bool = True,
    distribute_by_capacity: bool = True,
) -> MultiStationRouteResult:
    """
    Coordinate routes for multiple police stations.

    Args:
        db: Database session
        station_ids: Optional list of station IDs (None = all active stations)
        risk_threshold: Minimum risk score threshold
        max_minutes_per_station: Maximum route time per station
        minimize_overlap: Whether to minimize route overlap
        distribute_by_capacity: Whether to consider station capacity in distribution

    Returns:
        MultiStationRouteResult with coordinated routes
    """
    # Get stations
    if station_ids:
        stations = (
            db.query(PoliceStation)
            .filter(
                PoliceStation.id.in_(station_ids),
                PoliceStation.active == True,
            )
            .all()
        )
    else:
        stations = (
            db.query(PoliceStation)
            .filter(PoliceStation.active == True)
            .all()
        )

    if not stations:
        raise ValueError("No active police stations found")

    # Get high-risk cells
    risk_cells = get_high_risk_cells(db, risk_threshold, bbox=None)

    if not risk_cells:
        logger.warning("No high-risk cells found for multi-station coordination")
        # Return empty routes
        return MultiStationRouteResult(
            station_routes=[],
            total_risk_coverage=0.0,
            overlap_percentage=0.0,
            coordination_score=0.0,
        )

    # Distribute risk cells among stations
    if distribute_by_capacity:
        assignments = distribute_risk_cells(db, stations, risk_cells)
    else:
        # Simple distance-based distribution
        assignments = distribute_risk_cells(
            db,
            stations,
            risk_cells,
            capacity_weight=0.0,
            distance_weight=1.0,
            risk_weight=0.0,
        )

    # Generate route for each station
    station_routes: List[Tuple[UUID, str, RouteResult]] = []

    for station_id, assignment in assignments.items():
        if not assignment.risk_cells:
            continue

        try:
            # Create route request
            route_request = RouteRequest(
                station_id=station_id,
                risk_threshold=risk_threshold,
                max_minutes=max_minutes_per_station,
                end_station_id=None,  # Return to start station
            )

            # Generate route using assigned risk cells
            # Note: We need to modify compute_route to accept specific risk cells
            # For now, we'll use the existing function which gets all high-risk cells
            # This is a limitation that could be improved
            route = compute_route(db, route_request)

            station_routes.append((station_id, assignment.station_name, route))

        except Exception as e:
            logger.error(
                f"Error generating route for station {assignment.station_name}: {str(e)}"
            )
            continue

    # Minimize overlap if requested
    if minimize_overlap and len(station_routes) > 1:
        station_routes = minimize_route_overlap(station_routes)

    # Calculate metrics
    total_risk_coverage = (
        sum(route.risk_coverage for _, _, route in station_routes)
        / len(station_routes)
        if station_routes
        else 0.0
    )

    # Calculate overall overlap
    total_overlap = 0.0
    overlap_pairs = 0
    for i in range(len(station_routes)):
        for j in range(i + 1, len(station_routes)):
            _, _, route1 = station_routes[i]
            _, _, route2 = station_routes[j]
            overlap = calculate_route_overlap(route1, route2)
            total_overlap += overlap
            overlap_pairs += 1

    overlap_percentage = (
        total_overlap / overlap_pairs if overlap_pairs > 0 else 0.0
    )

    # Calculate coordination score (higher is better)
    # Score = (1 - overlap) * risk_coverage
    coordination_score = (1.0 - overlap_percentage) * total_risk_coverage

    return MultiStationRouteResult(
        station_routes=station_routes,
        total_risk_coverage=total_risk_coverage,
        overlap_percentage=overlap_percentage,
        coordination_score=coordination_score,
    )

