"""Main OSM import service orchestrator."""

import logging
from typing import Dict, Optional, Tuple
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.services.osm.overpass_client import OverpassClient
from app.services.osm.osm_parser import OSMParser
from app.services.osm.osm_importer import OSMImporter
from app.services.osm.routing_topology import RoutingTopology

logger = logging.getLogger(__name__)


def import_osm_data(
    db: Session,
    clear_existing: bool = False,
    create_topology: bool = True,
    highway_tags: Optional[list] = None,
) -> Dict[str, any]:
    """
    Main function to import OSM data and create routing topology.
    Uses only OSM relation ID for filtering - Overpass API handles the filtering.

    Args:
        db: Database session
        clear_existing: If True, clear existing road segments before import
        create_topology: If True, create pgRouting topology after import
        highway_tags: List of highway tag values to filter. If None, uses defaults

    Returns:
        Dictionary with import results and statistics
    """
    settings = get_settings()

    # Get OSM relation ID from Küçükçekmece boundary - only method supported
    from app.services.utils import get_kucukcekmece_boundary
    from app.models.administrative_boundary import AdministrativeBoundary
    
    boundary_geom = get_kucukcekmece_boundary(db)
    relation_id = None
    
    # Get OSM relation ID from database
    if boundary_geom:
        boundary_record = db.query(AdministrativeBoundary).filter(
            AdministrativeBoundary.name == settings.kucukcekmece_boundary_name,
            AdministrativeBoundary.admin_level == settings.kucukcekmece_boundary_admin_level
        ).first()
        
        if boundary_record and boundary_record.osm_id:
            relation_id = boundary_record.osm_id
            logger.info(f"Using OSM relation ID: {relation_id} for Küçükçekmece boundary")
        else:
            logger.warning("Küçükçekmece boundary found but no OSM relation ID. Import will fail.")
            return {
                "success": False,
                "relation_id": None,
                "steps": {},
                "errors": ["Küçükçekmece boundary must have OSM relation ID. Please import boundary first."],
            }
    else:
        logger.warning("Küçükçekmece boundary not found. Please import boundary first.")
        return {
            "success": False,
            "relation_id": None,
            "steps": {},
            "errors": ["Küçükçekmece boundary not found. Please import boundary first using POST /api/v1/osm/import-boundary"],
        }

    logger.info(f"Starting OSM import for Küçükçekmece (relation_id={relation_id})")

    result = {
        "success": False,
        "relation_id": relation_id,
        "steps": {},
        "errors": [],
    }

    try:
        # Step 1: Initialize Overpass client
        logger.info("Step 1: Initializing Overpass API client...")
        overpass_url = getattr(settings, "overpass_api_url", "https://overpass-api.de/api/interpreter")
        alternative_urls = getattr(settings, "overpass_api_alternatives", [])
        client = OverpassClient(api_url=overpass_url)

        # Check API status
        api_status = client.check_api_status()
        if api_status["status"] != "available":
            error_msg = f"Overpass API unavailable: {api_status.get('error', 'Unknown error')}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            return result

        result["steps"]["api_status"] = api_status
        logger.info(f"Overpass API is available (response time: {api_status.get('response_time_ms', 0):.2f}ms)")

        # Step 2: Fetch OSM data - Overpass API filters by relation_id
        logger.info("Step 2: Fetching OSM data from Overpass API using relation_id...")
        logger.info(f"Using relation_id: {relation_id}")
        try:
            # Overpass API filters by relation_id - no manual filtering needed
            xml_data = client.fetch_osm_data(
                bbox=None,
                highway_tags=highway_tags,
                alternative_urls=alternative_urls,
                relation_id=relation_id,
                polygon_coords=None
            )
            result["steps"]["osm_data_fetched"] = {
                "size_bytes": len(xml_data),
                "success": True,
            }
            logger.info(f"Fetched {len(xml_data)} bytes of OSM data")
        except Exception as e:
            error_msg = f"Failed to fetch OSM data: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            return result

        # Step 3: Parse OSM data
        logger.info("Step 3: Parsing OSM XML data...")
        try:
            parser = OSMParser()
            road_segments = parser.parse_xml(xml_data)
            result["steps"]["osm_parsed"] = {
                "segments_found": len(road_segments),
                "success": True,
            }
            logger.info(f"Parsed {len(road_segments)} road segments")
        except Exception as e:
            error_msg = f"Failed to parse OSM data: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            return result

        if len(road_segments) == 0:
            logger.warning("No road segments found in OSM data")
            result["success"] = True
            result["steps"]["import"] = {"imported": 0, "message": "No segments to import"}
            return result

        # Step 4: Import to database
        logger.info("Step 4: Importing road segments to database...")
        try:
            importer = OSMImporter(db, clear_existing=clear_existing)
            import_stats = importer.import_road_segments(road_segments)
            result["steps"]["import"] = import_stats
            logger.info(
                f"Import completed: {import_stats['imported']} imported, "
                f"{import_stats['skipped']} skipped, {import_stats['errors']} errors"
            )
            
            # No post-import cleanup needed - Overpass API already filtered the data
            # cleanup_segments_outside_boundary() is available as a manual endpoint if needed
        except Exception as e:
            error_msg = f"Failed to import road segments: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            db.rollback()
            return result

        # Step 5: Create topology
        if create_topology:
            logger.info("Step 5: Creating pgRouting topology...")
            try:
                topology_tolerance = getattr(settings, "osm_topology_tolerance", 0.0001)
                topology_service = RoutingTopology(db, tolerance=topology_tolerance)
                topology_result = topology_service.create_topology(force_recreate=clear_existing)
                result["steps"]["topology"] = topology_result
                if topology_result.get("success"):
                    logger.info("Topology created successfully")
                else:
                    logger.warning(f"Topology creation had issues: {topology_result.get('error')}")
            except Exception as e:
                error_msg = f"Failed to create topology: {str(e)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                # Don't fail the whole import if topology fails
                result["steps"]["topology"] = {"success": False, "error": str(e)}

        # Get final statistics
        try:
            final_stats = importer.get_import_statistics()
            result["statistics"] = final_stats
        except Exception as e:
            logger.warning(f"Failed to get final statistics: {str(e)}")

        result["success"] = True
        logger.info("OSM import completed successfully")
        return result

    except Exception as e:
        error_msg = f"Unexpected error during OSM import: {str(e)}"
        logger.error(error_msg, exc_info=True)
        result["errors"].append(error_msg)
        db.rollback()
        return result


def get_osm_import_status(db: Session) -> Dict[str, any]:
    """
    Get status of OSM import and routing topology.

    Args:
        db: Database session

    Returns:
        Dictionary with import status information
    """
    try:
        # Check if data exists
        importer = OSMImporter(db)
        data_exists = importer.check_data_exists()

        if not data_exists:
            return {
                "data_imported": False,
                "topology_created": False,
                "message": "No OSM data imported yet",
            }

        # Get import statistics
        stats = importer.get_import_statistics()

        # Get topology status
        topology_tolerance = getattr(get_settings(), "osm_topology_tolerance", 0.0001)
        topology_service = RoutingTopology(db, tolerance=topology_tolerance)
        topology_status = topology_service.get_topology_status()

        return {
            "data_imported": True,
            "statistics": stats,
            "topology": topology_status,
        }

    except Exception as e:
        logger.error(f"Failed to get OSM import status: {str(e)}")
        return {
            "error": str(e),
        }

