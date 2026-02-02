"""OSM import API endpoints."""

import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.session import get_db
from app.services.osm.osm_service import import_osm_data, get_osm_import_status
from app.services.osm.routing_topology import RoutingTopology
from app.services.osm.boundary_service import BoundaryService
from app.services.osm.boundary_importer import BoundaryImporter
from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/osm", tags=["OSM"])


class OSMImportRequest(BaseModel):
    """Request model for OSM import."""

    clear_existing: bool = False
    create_topology: bool = True
    highway_tags: Optional[List[str]] = None


@router.get("/status")
def get_status(db: Session = Depends(get_db)):
    """
    Get OSM import and routing topology status.

    Returns:
        Dictionary with import status and statistics
    """
    try:
        status = get_osm_import_status(db)
        return status
    except Exception as e:
        logger.error(f"Failed to get OSM status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.post("/import")
def import_osm(
    request: OSMImportRequest,
    db: Session = Depends(get_db),
):
    """
    Import OSM data and create routing topology.

    Args:
        request: OSM import request parameters
        db: Database session

    Returns:
        Dictionary with import results
    """
    try:
        logger.info(f"Starting OSM import (clear_existing={request.clear_existing})")

        result = import_osm_data(
            db=db,
            clear_existing=request.clear_existing,
            create_topology=request.create_topology,
            highway_tags=request.highway_tags,
        )

        if result["success"]:
            return {
                "success": True,
                "message": "OSM import completed successfully",
                **result,
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"OSM import failed: {', '.join(result.get('errors', []))}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import OSM data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.post("/refresh-topology")
def refresh_topology(
    force: bool = Query(False, description="Force recreation of topology"),
    db: Session = Depends(get_db),
):
    """
    Refresh pgRouting topology.

    Args:
        force: If True, drop existing topology before creating
        db: Database session

    Returns:
        Dictionary with topology creation results
    """
    try:
        settings = get_settings()
        topology_tolerance = getattr(settings, "osm_topology_tolerance", 0.0001)

        topology_service = RoutingTopology(db, tolerance=topology_tolerance)
        result = topology_service.create_topology(force_recreate=force)

        if result.get("success"):
            return {
                "success": True,
                "message": "Topology refreshed successfully",
                **result,
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Topology refresh failed: {result.get('error', 'Unknown error')}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to refresh topology: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Topology refresh failed: {str(e)}")


@router.post("/clean-boundary")
def clean_road_segments_boundary(
    db: Session = Depends(get_db),
):
    """
    Manually clean road segments that are outside Küçükçekmece boundary.
    This is only needed if data was imported without proper Overpass API filtering.
    Overpass API already filters by relation/polygon/bbox, so this is typically not needed.
    
    Uses optimized filtering: motorway/trunk only need intersection, others need center OR 30% length.
    
    Returns:
        Dictionary with cleaning statistics
    """
    try:
        from app.services.osm.osm_importer import OSMImporter
        
        importer = OSMImporter(db, clear_existing=False)
        cleanup_stats = importer.clean_segments_outside_boundary()
        
        return {
            "success": True,
            "message": "Road segments cleaned successfully",
            **cleanup_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to clean road segments: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clean road segments: {str(e)}"
        )


@router.post("/clear-all")
def clear_all_osm_data(
    db: Session = Depends(get_db),
):
    """
    Clear all OSM road segment data from database.
    This will delete all road segments and related topology data.
    
    Returns:
        Dictionary with deletion statistics
    """
    try:
        from app.models.road_segment import RoadSegment
        from sqlalchemy import text
        
        # Get count before deletion
        total_before = db.query(RoadSegment).count()
        
        # Delete all road segments
        deleted_count = db.query(RoadSegment).delete()
        
        # Also delete topology vertices if they exist
        try:
            db.execute(text("DROP TABLE IF EXISTS road_segment_vertices_pgr CASCADE"))
        except Exception as e:
            logger.warning(f"Failed to drop topology vertices table: {str(e)}")
        
        db.commit()
        
        logger.info(f"Cleared {deleted_count} road segments from database")
        
        return {
            "success": True,
            "message": "All OSM data cleared successfully",
            "deleted_segments": deleted_count,
            "total_before": total_before
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to clear OSM data: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear OSM data: {str(e)}"
        )


@router.post("/import-motorway")
def import_motorway_only(
    db: Session = Depends(get_db),
    clear_existing: bool = Query(True, description="Clear existing data before import"),
    create_topology: bool = Query(True, description="Create pgRouting topology after import"),
):
    """
    Import only motorway road segments from OSM within Küçükçekmece boundary.
    Uses Overpass API to fetch motorway data filtered by relation ID.
    
    Args:
        clear_existing: If True, clear existing road segments before import
        create_topology: If True, create pgRouting topology after import
    
    Returns:
        Dictionary with import results
    """
    try:
        # Import only motorway highways
        result = import_osm_data(
            db=db,
            clear_existing=clear_existing,
            create_topology=create_topology,
            highway_tags=["motorway"]  # Only motorway
        )
        
        if result.get("success"):
            return {
                "success": True,
                "message": "Motorway data imported successfully",
                **result
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Import failed: {result.get('errors', ['Unknown error'])}"
            )
        
    except Exception as e:
        logger.error(f"Failed to import motorway data: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to import motorway data: {str(e)}"
        )


@router.get("/topology-status")
def get_topology_status(db: Session = Depends(get_db)):
    """
    Get routing topology status.

    Returns:
        Dictionary with topology status information
    """
    try:
        settings = get_settings()
        topology_tolerance = getattr(settings, "osm_topology_tolerance", 0.0001)

        topology_service = RoutingTopology(db, tolerance=topology_tolerance)
        status = topology_service.get_topology_status()

        return status

    except Exception as e:
        logger.error(f"Failed to get topology status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get topology status: {str(e)}"
        )


@router.post("/import-boundary")
def import_boundary(
    force: bool = Query(False, description="Force re-import even if boundary exists"),
    db: Session = Depends(get_db),
):
    """
    Import Küçükçekmece boundary from OSM.

    Args:
        force: If True, re-import even if boundary already exists
        db: Database session

    Returns:
        Dictionary with import results
    """
    try:
        settings = get_settings()

        # Check if boundary already exists
        importer = BoundaryImporter(db)
        boundary_exists = importer.boundary_exists(
            settings.kucukcekmece_boundary_name,
            settings.kucukcekmece_boundary_admin_level
        )

        if boundary_exists and not force:
            return {
                "success": True,
                "message": "Boundary already exists. Use force=true to re-import",
                "boundary_exists": True,
            }

        # Fetch boundary from OSM
        logger.info("Fetching Küçükçekmece boundary from OSM...")
        boundary_service = BoundaryService(
            api_url=getattr(settings, "overpass_api_url", "https://overpass-api.de/api/interpreter")
        )

        # Use fallback bbox to limit search area
        bbox = settings.kucukcekmece_fallback_bbox
        xml_data = boundary_service.fetch_boundary_by_name(
            name=settings.kucukcekmece_boundary_name,
            admin_level=settings.kucukcekmece_boundary_admin_level,
            bbox=bbox,
        )

        # Import boundary
        result = importer.import_boundary(
            name=settings.kucukcekmece_boundary_name,
            admin_level=settings.kucukcekmece_boundary_admin_level,
            xml_data=xml_data,
            update_existing=force,
        )

        if result["success"]:
            return {
                "success": True,
                "message": "Boundary imported successfully",
                **result,
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Boundary import failed: {result.get('error', 'Unknown error')}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import boundary: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Boundary import failed: {str(e)}")


@router.get("/boundary-status")
def get_boundary_status(db: Session = Depends(get_db)):
    """
    Get Küçükçekmece boundary status.

    Returns:
        Dictionary with boundary status information
    """
    try:
        settings = get_settings()
        importer = BoundaryImporter(db)

        boundary_exists = importer.boundary_exists(
            settings.kucukcekmece_boundary_name,
            settings.kucukcekmece_boundary_admin_level
        )

        if not boundary_exists:
            return {
                "boundary_loaded": False,
                "message": "Küçükçekmece boundary not loaded. Use POST /api/v1/osm/import-boundary to import it.",
            }

        boundary = importer.get_boundary(
            settings.kucukcekmece_boundary_name,
            settings.kucukcekmece_boundary_admin_level
        )

        return {
            "boundary_loaded": True,
            "name": boundary.name,
            "admin_level": boundary.admin_level,
            "created_at": boundary.created_at.isoformat() if boundary.created_at else None,
            "updated_at": boundary.updated_at.isoformat() if boundary.updated_at else None,
        }

    except Exception as e:
        logger.error(f"Failed to get boundary status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get boundary status: {str(e)}"
        )


@router.get("/road-network")
def get_road_network(
    min_lat: Optional[float] = Query(None, description="Bounding box min latitude"),
    min_lng: Optional[float] = Query(None, description="Bounding box min longitude"),
    max_lat: Optional[float] = Query(None, description="Bounding box max latitude"),
    max_lng: Optional[float] = Query(None, description="Bounding box max longitude"),
    limit: int = Query(10000, ge=1, le=50000, description="Maximum number of road segments"),
    db: Session = Depends(get_db),
):
    """
    Get OSM driveable road network as GeoJSON.
    Returns road segments within Küçükçekmece boundaries or specified bbox.
    Only includes driveable roads (motorway, trunk, primary, secondary, tertiary, residential, service, unclassified).
    """
    try:
        from sqlalchemy import text
        from app.models.road_segment import RoadSegment
        from app.services.utils import get_kucukcekmece_boundary
        
        # Only driveable road types
        driveable_road_types = [
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "residential", "service", "unclassified"
        ]
        
        query = db.query(RoadSegment).filter(
            RoadSegment.road_type.in_(driveable_road_types)
        )
        
        # Filter by boundary - temporarily disabled due to boundary parser issue
        # Boundary polygon is incomplete (only 18 points, too small area)
        # TODO: Fix boundary parser to merge all outer ways correctly
        # For now, return all driveable roads (they were already filtered by Overpass API using relation_id)
        # boundary_geom = get_kucukcekmece_boundary(db)
        # if boundary_geom:
        #     query = query.filter(...)
        
        # Filter by bbox if provided
        if all([min_lat, min_lng, max_lat, max_lng]):
            query = query.filter(
                text("""
                    ST_Intersects(
                        geom,
                        ST_MakeEnvelope(:min_lng, :min_lat, :max_lng, :max_lat, 4326)::geography
                    )
                """).params(
                    min_lat=min_lat,
                    min_lng=min_lng,
                    max_lat=max_lat,
                    max_lng=max_lng
                )
            )
        
        segments = query.limit(limit).all()
        
        # Convert to GeoJSON FeatureCollection
        features = []
        for segment in segments:
            try:
                geom_json = db.execute(
                    text("""
                        SELECT ST_AsGeoJSON(geom)::json as geom_json
                        FROM road_segment
                        WHERE id = :segment_id
                    """),
                    {"segment_id": segment.id}
                ).scalar()
                
                if geom_json:
                    features.append({
                        "type": "Feature",
                        "properties": {
                            "id": segment.id,
                            "road_type": segment.road_type,
                            "speed_limit": segment.speed_limit,
                            "one_way": segment.one_way,
                            "risk_score": segment.risk_score if segment.risk_score else 0.0,
                        },
                        "geometry": geom_json
                    })
            except Exception as e:
                logger.debug(f"Failed to convert segment {segment.id} to GeoJSON: {str(e)}")
                continue
        
        return {
            "type": "FeatureCollection",
            "features": features,
            "total": len(features)
        }
        
    except Exception as e:
        logger.error(f"Failed to get road network: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get road network: {str(e)}"
        )

