#!/usr/bin/env python3
"""Script to import Küçükçekmece boundary from OSM."""

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
from app.services.osm.boundary_service import BoundaryService
from app.services.osm.boundary_importer import BoundaryImporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def wait_for_database(max_retries: int = 30, retry_delay: int = 2) -> bool:
    """Wait for database to be available."""
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
    """Main function for boundary import."""
    settings = get_settings()

    # Wait for database
    if not wait_for_database():
        logger.error("Failed to connect to database, exiting...")
        return 1

    db = SessionLocal()
    try:
        # Check if boundary already exists
        importer = BoundaryImporter(db)
        boundary_exists = importer.boundary_exists(
            settings.kucukcekmece_boundary_name,
            settings.kucukcekmece_boundary_admin_level
        )

        if boundary_exists:
            logger.info("Küçükçekmece boundary already exists in database, skipping import...")
            logger.info("To re-import, delete the existing boundary first or use the API endpoint")
            return 0

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
        logger.info("Importing boundary to database...")
        result = importer.import_boundary(
            name=settings.kucukcekmece_boundary_name,
            admin_level=settings.kucukcekmece_boundary_admin_level,
            xml_data=xml_data,
            update_existing=False,
        )

        if result["success"]:
            logger.info("Boundary import completed successfully")
            logger.info(f"Boundary ID: {result.get('boundary_id')}")
            logger.info(f"Points: {result.get('points_count', 0)}")
            return 0
        else:
            logger.error(f"Boundary import failed: {result.get('error')}")
            return 1

    except Exception as e:
        logger.error(f"Unexpected error during boundary import: {str(e)}", exc_info=True)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

