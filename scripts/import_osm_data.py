#!/usr/bin/env python3
"""Docker init script for automatic OSM data import."""

import sys
import os
import logging
import time

# Add backend to path - Docker container uses /app as working directory
backend_path = os.path.join(os.path.dirname(__file__), "..", "backend")
if os.path.exists(backend_path):
    sys.path.insert(0, backend_path)
# Also try /app (Docker container path)
if os.path.exists("/app"):
    sys.path.insert(0, "/app")

from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.core.config import get_settings
from app.services.osm.osm_service import import_osm_data, get_osm_import_status
from app.services.osm.osm_importer import OSMImporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def wait_for_database(max_retries: int = 30, retry_delay: int = 2) -> bool:
    """
    Wait for database to be available.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        True if database is available, False otherwise
    """
    logger.info("Waiting for database to be available...")
    db = SessionLocal()
    
    for i in range(max_retries):
        try:
            from sqlalchemy import text
            db.execute(text("SELECT 1"))
            db.close()
            logger.info("Database is available")
            return True
        except Exception as e:
            if i < max_retries - 1:
                logger.info(f"Database not ready yet (attempt {i+1}/{max_retries}), waiting {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Database not available after {max_retries} attempts: {str(e)}")
                db.close()
                return False
    
    db.close()
    return False


def main():
    """Main function for OSM import."""
    settings = get_settings()

    # Check if import should run on startup
    if not getattr(settings, "osm_import_on_startup", True):
        logger.info("OSM import on startup is disabled, skipping...")
        return 0

    # Wait for database
    if not wait_for_database():
        logger.error("Failed to connect to database, exiting...")
        return 1

    # Check if data already exists
    db = SessionLocal()
    try:
        importer = OSMImporter(db)
        data_exists = importer.check_data_exists()

        if data_exists:
            logger.info("OSM data already exists in database, skipping import...")
            logger.info("To re-import, use the API endpoint or set clear_existing=True")
            
            # Check topology status
            status = get_osm_import_status(db)
            if status.get("topology", {}).get("has_topology"):
                logger.info("Routing topology is already created")
            else:
                logger.warning("OSM data exists but topology is not created")
                logger.info("Creating topology...")
                from app.services.osm.routing_topology import RoutingTopology
                topology_tolerance = getattr(settings, "osm_topology_tolerance", 0.0001)
                topology_service = RoutingTopology(db, tolerance=topology_tolerance)
                topology_result = topology_service.create_topology()
                if topology_result.get("success"):
                    logger.info("Topology created successfully")
                else:
                    logger.error(f"Failed to create topology: {topology_result.get('error')}")
                    return 1
            
            return 0

        # Import OSM data
        logger.info("Starting OSM data import...")
        result = import_osm_data(
            db=db,
            bbox=None,  # Use default from config
            clear_existing=False,
            create_topology=True,
            highway_tags=getattr(settings, "osm_highway_tags", None),
        )

        if result["success"]:
            logger.info("OSM import completed successfully")
            logger.info(f"Imported {result['steps'].get('import', {}).get('imported', 0)} road segments")
            return 0
        else:
            logger.error(f"OSM import failed: {result.get('errors', [])}")
            return 1

    except Exception as e:
        logger.error(f"Unexpected error during OSM import: {str(e)}", exc_info=True)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

