import logging
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from datetime import datetime

from app.db.session import get_db

logger = logging.getLogger(__name__)
from app.schemas.routing import (
    MultiStationRouteRequest,
    MultiStationRouteResponse,
    RouteOptimizeRequest,
    RouteResponse,
    RouteWaypoint,
    StationRoute,
)
from app.services.routing.multi_station_coordinator import (
    coordinate_multi_station_routes,
)
from app.services.routing.route_optimizer import compute_route, RouteRequest

router = APIRouter(prefix="/routing", tags=["Routing"])


@router.post("/update-road-risks")
def update_road_risks(
    start_time: datetime = Query(..., description="Start of time window"),
    end_time: datetime = Query(..., description="End of time window"),
    db: Session = Depends(get_db),
):
    """
    Update risk scores for all road segments based on nearby crime events.
    This should be called before route optimization to ensure road segments have current risk data.
    """
    from datetime import datetime
    from app.services.forecast.road_segment_risk import update_road_segment_risks
    
    try:
        result = update_road_segment_risks(
            db=db,
            time_window_start=start_time,
            time_window_end=end_time
        )
        return {
            "success": True,
            "message": "Road segment risks updated successfully",
            **result
        }
    except Exception as e:
        logger.error(f"Failed to update road segment risks: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update road segment risks: {str(e)}"
        )


@router.post("/optimize", response_model=RouteResponse)
def optimize_route(
    body: RouteOptimizeRequest,
    db: Session = Depends(get_db)
):
    """Optimize patrol route based on road segment risks (KDE-based)"""
    
    try:
        logger.info(f"Route optimization request: station_id={body.station_id}, risk_threshold={body.risk_threshold}, max_minutes={body.max_minutes}, start_time={body.start_time}, end_time={body.end_time}")
        
        # Update road segment risks for the requested time window (KDE-based)
        if body.start_time and body.end_time:
            from datetime import datetime
            from app.services.forecast.road_segment_risk import update_road_segment_risks
            
            try:
                # Parse ISO format datetime strings
                start_dt = datetime.fromisoformat(body.start_time.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(body.end_time.replace('Z', '+00:00'))
                
                # Update road segment risks using KDE - filtered by station's neighborhoods
                risk_update_result = update_road_segment_risks(
                    db=db,
                    time_window_start=start_dt,
                    time_window_end=end_dt,
                    station_id=str(body.station_id)
                )
                logger.info(f"Road segment risks updated for route optimization: {risk_update_result}")
            except Exception as update_err:
                logger.warning(f"Failed to update road segment risks before route optimization: {str(update_err)}")
                # Continue with existing risk scores
        
        route_request = RouteRequest(
            station_id=body.station_id,
            risk_threshold=body.risk_threshold,
            max_minutes=body.max_minutes,
            end_station_id=body.end_station_id,
            start_time=body.start_time,
            end_time=body.end_time
        )
        
        result = compute_route(db, route_request)
        
        logger.info(f"Route optimization result: {len(result.waypoints)} waypoints, {result.total_distance:.0f}m, {result.total_time:.1f}min, path_coords={len(result.path.get('coordinates', [])) if result.path else 0}")
        
        # Convert to response format
        waypoints = [
            RouteWaypoint(
                lat=wp.lat,
                lng=wp.lng,
                risk_score=wp.risk_score
            )
            for wp in result.waypoints
        ]
        
        return RouteResponse(
            waypoints=waypoints,
            total_distance=result.total_distance,
            total_time=result.total_time,
            risk_coverage=result.risk_coverage,
            path=result.path
        )
        
    except ValueError as e:
        logger.warning(f"Route optimization validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Route optimization failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Route optimization failed: {str(e)}"
        )


@router.post("/optimize-multi", response_model=MultiStationRouteResponse)
def optimize_multi_station_route(
    body: MultiStationRouteRequest,
    db: Session = Depends(get_db),
):
    """Optimize coordinated routes for multiple police stations"""
    
    try:
        # Check if multi-station coordination is enabled
        from app.core.config import get_settings
        settings = get_settings()
        
        if not getattr(settings, "multi_station_coordination_enabled", True):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Multi-station coordination is disabled"
            )
        
        # Validate station IDs if provided
        if body.station_ids:
            from app.models.police_station import PoliceStation
            from app.services.utils import validate_within_boundary
            from app.services.routing.route_optimizer import get_station_coordinates
            
            valid_stations = []
            for station_id in body.station_ids:
                station = db.query(PoliceStation).filter(
                    PoliceStation.id == station_id,
                    PoliceStation.active == True
                ).first()
                
                if not station:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Station {station_id} not found or inactive"
                    )
                
                # Validate station is within Küçükçekmece boundary
                lat, lng = get_station_coordinates(db, station_id)
                is_valid, error_message = validate_within_boundary(db, lat, lng)
                if not is_valid:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Station {station.name} is outside Küçükçekmece boundary: {error_message}"
                    )
                
                valid_stations.append(station_id)
            
            if not valid_stations:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid active stations found"
                )
        
        # Coordinate routes
        result = coordinate_multi_station_routes(
            db=db,
            station_ids=body.station_ids,
            risk_threshold=body.risk_threshold,
            max_minutes_per_station=body.max_minutes_per_station,
            minimize_overlap=body.minimize_overlap,
            distribute_by_capacity=body.distribute_by_capacity,
        )
        
        # Convert to response format
        station_routes = []
        for station_id, station_name, route_result in result.station_routes:
            waypoints = [
                RouteWaypoint(
                    lat=wp.lat,
                    lng=wp.lng,
                    risk_score=wp.risk_score,
                )
                for wp in route_result.waypoints
            ]
            
            station_routes.append(
                StationRoute(
                    station_id=station_id,
                    station_name=station_name,
                    waypoints=waypoints,
                    total_distance=route_result.total_distance,
                    total_time=route_result.total_time,
                    risk_coverage=route_result.risk_coverage,
                    path=route_result.path,
                )
            )
        
        return MultiStationRouteResponse(
            routes=station_routes,
            total_stations=len(station_routes),
            total_risk_coverage=result.total_risk_coverage,
            overlap_percentage=result.overlap_percentage,
            coordination_score=result.coordination_score,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-station route optimization failed: {str(e)}"
        )
    