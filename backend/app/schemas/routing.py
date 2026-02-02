from pydantic import BaseModel, Field
from uuid import UUID
from typing import Optional, List, Any


class RouteWaypoint(BaseModel):
    lat: float
    lng: float
    risk_score: Optional[float] = None


class RouteResponse(BaseModel):
    waypoints: List[RouteWaypoint]
    total_distance: float
    total_time: float
    risk_coverage: float  # 0.0 to 1.0
    path: Any  # GeoJSON LineString


class RouteOptimizeRequest(BaseModel):
    station_id: UUID
    risk_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_minutes: int = Field(default=90, ge=1, le=180)
    end_station_id: Optional[UUID] = None  # If None, returns to start station
    start_time: Optional[str] = None  # ISO format datetime string for forecast time window start
    end_time: Optional[str] = None  # ISO format datetime string for forecast time window end


class StationRoute(BaseModel):
    """Route information for a single station."""

    station_id: UUID
    station_name: str
    waypoints: List[RouteWaypoint]
    total_distance: float
    total_time: float
    risk_coverage: float
    path: Any  # GeoJSON LineString


class MultiStationRouteRequest(BaseModel):
    """Request for multi-station route optimization."""

    station_ids: Optional[List[UUID]] = Field(
        default=None, description="List of station IDs (None = all active stations)"
    )
    risk_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_minutes_per_station: int = Field(default=90, ge=1, le=180)
    minimize_overlap: bool = Field(default=True, description="Minimize route overlap")
    distribute_by_capacity: bool = Field(
        default=True, description="Consider station capacity in distribution"
    )


class MultiStationRouteResponse(BaseModel):
    """Response for multi-station route optimization."""

    routes: List[StationRoute]
    total_stations: int
    total_risk_coverage: float
    overlap_percentage: float
    coordination_score: float  # 0.0-1.0, how well routes are coordinated

