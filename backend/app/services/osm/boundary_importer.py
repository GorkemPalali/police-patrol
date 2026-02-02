"""Boundary importer service for importing OSM boundaries into database."""

import logging
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.administrative_boundary import AdministrativeBoundary
from app.services.osm.boundary_parser import BoundaryParser

logger = logging.getLogger(__name__)


class BoundaryImporter:
    """Service for importing administrative boundaries into database."""

    def __init__(self, db: Session):
        """
        Initialize boundary importer.

        Args:
            db: Database session
        """
        self.db = db
        self.parser = BoundaryParser()

    def import_boundary(
        self,
        name: str,
        admin_level: int,
        xml_data: str,
        update_existing: bool = True,
    ) -> dict:
        """
        Import boundary from OSM XML data.

        Args:
            name: Name of the administrative boundary
            admin_level: Administrative level
            xml_data: OSM XML data
            update_existing: If True, update existing boundary; if False, skip if exists

        Returns:
            Dictionary with import results
        """
        try:
            # Parse boundary and extract relation ID
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_data)
            
            # Extract relation ID from XML
            osm_id = None
            relations = root.findall("relation")
            for relation in relations:
                tags = {tag.get("k"): tag.get("v") for tag in relation.findall("tag")}
                if tags.get("boundary") == "administrative":
                    osm_id = int(relation.get("id"))
                    break
            
            # Parse boundary
            polygon_rings = self.parser.parse_boundary_xml(xml_data)
            if not polygon_rings or len(polygon_rings) == 0:
                return {
                    "success": False,
                    "error": "Failed to parse boundary from OSM data",
                }

            # Use the outer ring (first ring)
            outer_ring = polygon_rings[0]
            if len(outer_ring) < 3:
                return {
                    "success": False,
                    "error": "Boundary polygon must have at least 3 points",
                }

            # Convert to WKT
            wkt = self.parser.coordinates_to_wkt(outer_ring)

            # Check if boundary already exists
            existing = (
                self.db.query(AdministrativeBoundary)
                .filter(
                    AdministrativeBoundary.name == name,
                    AdministrativeBoundary.admin_level == admin_level,
                )
                .first()
            )

            if existing:
                if not update_existing:
                    logger.info(f"Boundary {name} (admin_level={admin_level}) already exists, skipping")
                    return {
                        "success": True,
                        "message": "Boundary already exists, skipped",
                        "boundary_id": str(existing.id),
                    }

                # Update existing
                logger.info(f"Updating existing boundary {name} (admin_level={admin_level})")
                self.db.execute(
                    text("""
                        UPDATE administrative_boundary
                        SET geom = ST_GeogFromText(:wkt),
                            osm_id = :osm_id,
                            updated_at = NOW()
                        WHERE id = :id
                    """),
                    {
                        "wkt": f"SRID=4326;{wkt}",
                        "osm_id": osm_id,
                        "id": str(existing.id),
                    },
                )
                boundary_id = str(existing.id)
            else:
                # Create new
                logger.info(f"Creating new boundary {name} (admin_level={admin_level}, osm_id={osm_id})")
                result = self.db.execute(
                    text("""
                        INSERT INTO administrative_boundary (id, name, admin_level, osm_id, geom, created_at, updated_at)
                        VALUES (gen_random_uuid(), :name, :admin_level, :osm_id, ST_GeogFromText(:wkt), NOW(), NOW())
                        RETURNING id
                    """),
                    {
                        "name": name,
                        "admin_level": admin_level,
                        "osm_id": osm_id,
                        "wkt": f"SRID=4326;{wkt}",
                    },
                )
                boundary_id = str(result.scalar())

            self.db.commit()

            return {
                "success": True,
                "message": "Boundary imported successfully",
                "boundary_id": boundary_id,
                "points_count": len(outer_ring),
            }

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to import boundary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
            }

    def get_boundary(self, name: str, admin_level: int) -> Optional[AdministrativeBoundary]:
        """
        Get boundary from database.

        Args:
            name: Name of the boundary
            admin_level: Administrative level

        Returns:
            AdministrativeBoundary or None
        """
        return (
            self.db.query(AdministrativeBoundary)
            .filter(
                AdministrativeBoundary.name == name,
                AdministrativeBoundary.admin_level == admin_level,
            )
            .first()
        )

    def boundary_exists(self, name: str, admin_level: int) -> bool:
        """
        Check if boundary exists in database.

        Args:
            name: Name of the boundary
            admin_level: Administrative level

        Returns:
            True if exists, False otherwise
        """
        return self.get_boundary(name, admin_level) is not None




