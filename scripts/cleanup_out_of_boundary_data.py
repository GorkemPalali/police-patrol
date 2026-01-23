#!/usr/bin/env python3
"""Script to clean up data outside Küçükçekmece boundary."""

import sys
import os
import logging
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.session import SessionLocal
from app.core.config import get_settings
from app.models.crime_event import CrimeEvent
from app.models.police_station import PoliceStation
from app.models.risk_cell import RiskCell

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


def find_out_of_boundary_data(db: Session) -> dict:
    """
    Find all data outside Küçükçekmece boundary.

    Returns:
        Dictionary with counts of out-of-boundary records per table
    """
    stats = {
        "crime_event": 0,
        "police_station": 0,
        "risk_cell": 0,
    }

    try:
        # Check crime_event
        result = db.execute(
            text("""
                SELECT COUNT(*) 
                FROM crime_event 
                WHERE NOT is_within_kucukcekmece(geom)
            """)
        ).scalar()
        stats["crime_event"] = result or 0

        # Check police_station
        result = db.execute(
            text("""
                SELECT COUNT(*) 
                FROM police_station 
                WHERE NOT is_within_kucukcekmece(geom)
            """)
        ).scalar()
        stats["police_station"] = result or 0

        # Check risk_cell (using ST_Intersects for polygons)
        result = db.execute(
            text("""
                SELECT COUNT(*) 
                FROM risk_cell 
                WHERE NOT ST_Intersects(
                    geom::geometry,
                    (SELECT geom::geometry FROM administrative_boundary 
                     WHERE name = 'Küçükçekmece' AND admin_level = 8 LIMIT 1)
                )
            """)
        ).scalar()
        stats["risk_cell"] = result or 0

        return stats

    except Exception as e:
        logger.error(f"Failed to find out-of-boundary data: {str(e)}")
        return stats


def cleanup_out_of_boundary_data(db: Session, dry_run: bool = True) -> dict:
    """
    Clean up out-of-boundary data.

    Args:
        db: Database session
        dry_run: If True, only report without deleting

    Returns:
        Dictionary with cleanup statistics
    """
    stats = find_out_of_boundary_data(db)
    cleanup_stats = {
        "found": stats.copy(),
        "deleted": {"crime_event": 0, "police_station": 0, "risk_cell": 0},
    }

    if dry_run:
        logger.info("DRY RUN MODE - No data will be deleted")
        return cleanup_stats

    try:
        # Delete out-of-boundary crime_event records
        if stats["crime_event"] > 0:
            result = db.execute(
                text("""
                    DELETE FROM crime_event 
                    WHERE NOT is_within_kucukcekmece(geom)
                """)
            )
            cleanup_stats["deleted"]["crime_event"] = result.rowcount
            logger.info(f"Deleted {result.rowcount} out-of-boundary crime_event records")

        # Delete out-of-boundary police_station records
        if stats["police_station"] > 0:
            result = db.execute(
                text("""
                    DELETE FROM police_station 
                    WHERE NOT is_within_kucukcekmece(geom)
                """)
            )
            cleanup_stats["deleted"]["police_station"] = result.rowcount
            logger.info(f"Deleted {result.rowcount} out-of-boundary police_station records")

        # Delete out-of-boundary risk_cell records
        if stats["risk_cell"] > 0:
            result = db.execute(
                text("""
                    DELETE FROM risk_cell 
                    WHERE NOT ST_Intersects(
                        geom::geometry,
                        (SELECT geom::geometry FROM administrative_boundary 
                         WHERE name = 'Küçükçekmece' AND admin_level = 8 LIMIT 1)
                    )
                """)
            )
            cleanup_stats["deleted"]["risk_cell"] = result.rowcount
            logger.info(f"Deleted {result.rowcount} out-of-boundary risk_cell records")

        db.commit()
        logger.info("Cleanup completed successfully")

    except Exception as e:
        db.rollback()
        logger.error(f"Cleanup failed: {str(e)}")
        raise

    return cleanup_stats


def main():
    """Main function for cleanup."""
    import argparse

    parser = argparse.ArgumentParser(description="Clean up data outside Küçükçekmece boundary")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report out-of-boundary data without deleting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Actually delete out-of-boundary data (requires --dry-run to be False)",
    )

    args = parser.parse_args()

    # Wait for database
    if not wait_for_database():
        logger.error("Failed to connect to database, exiting...")
        return 1

    db = SessionLocal()
    try:
        # Check if boundary exists
        boundary_exists = db.execute(
            text("""
                SELECT COUNT(*) > 0
                FROM administrative_boundary
                WHERE name = 'Küçükçekmece' AND admin_level = 8
            """)
        ).scalar()

        if not boundary_exists:
            logger.warning("Küçükçekmece boundary not found in database!")
            logger.warning("Please import the boundary first using:")
            logger.warning("  python3 scripts/import_kucukcekmece_boundary.py")
            logger.warning("  OR")
            logger.warning("  curl -X POST http://localhost:8000/api/v1/osm/import-boundary")
            return 1

        # Find out-of-boundary data
        logger.info("Scanning for out-of-boundary data...")
        stats = find_out_of_boundary_data(db)

        total_out_of_boundary = sum(stats.values())
        if total_out_of_boundary == 0:
            logger.info("No out-of-boundary data found. Database is clean!")
            return 0

        # Report findings
        logger.info("=" * 60)
        logger.info("OUT-OF-BOUNDARY DATA REPORT")
        logger.info("=" * 60)
        logger.info(f"Crime Events: {stats['crime_event']}")
        logger.info(f"Police Stations: {stats['police_station']}")
        logger.info(f"Risk Cells: {stats['risk_cell']}")
        logger.info(f"TOTAL: {total_out_of_boundary}")
        logger.info("=" * 60)

        # Perform cleanup
        dry_run = args.dry_run or not args.force
        if dry_run:
            logger.info("\nDRY RUN MODE - No data will be deleted")
            logger.info("To actually delete, run with --force flag")
            return 0

        # Confirm deletion
        logger.warning(f"\nWARNING: This will delete {total_out_of_boundary} records!")
        logger.warning("Press Ctrl+C to cancel, or wait 5 seconds to continue...")
        time.sleep(5)

        # Perform cleanup
        cleanup_stats = cleanup_out_of_boundary_data(db, dry_run=False)

        logger.info("\n" + "=" * 60)
        logger.info("CLEANUP SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Crime Events deleted: {cleanup_stats['deleted']['crime_event']}")
        logger.info(f"Police Stations deleted: {cleanup_stats['deleted']['police_station']}")
        logger.info(f"Risk Cells deleted: {cleanup_stats['deleted']['risk_cell']}")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.info("\nCleanup cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during cleanup: {str(e)}", exc_info=True)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)




