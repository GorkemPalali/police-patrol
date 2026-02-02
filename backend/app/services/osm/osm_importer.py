"""Database import service for OSM road segments."""

import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from geoalchemy2 import Geography

from app.models.road_segment import RoadSegment
from app.services.osm.osm_parser import RoadSegmentData

logger = logging.getLogger(__name__)


class OSMImporter:
    """Service for importing OSM road segments into database."""

    def __init__(self, db: Session, clear_existing: bool = False):
        """
        Initialize OSM importer.

        Args:
            db: Database session
            clear_existing: If True, clear existing road segments before import
        """
        self.db = db
        self.clear_existing = clear_existing

    def import_road_segments(
        self, road_segments: List[RoadSegmentData], batch_size: int = 1000
    ) -> dict:
        """
        Import road segments into database.

        Args:
            road_segments: List of RoadSegmentData objects
            batch_size: Number of segments to insert per batch

        Returns:
            Dictionary with import statistics
        """
        stats = {
            "total": len(road_segments),
            "imported": 0,
            "skipped": 0,
            "errors": 0,
        }

        try:
            # Clear existing data if requested
            if self.clear_existing:
                logger.info("Clearing existing road segments...")
                deleted_count = self.db.query(RoadSegment).delete()
                self.db.commit()
                logger.info(f"Deleted {deleted_count} existing road segments")

            # Import in batches
            for i in range(0, len(road_segments), batch_size):
                batch = road_segments[i : i + batch_size]
                batch_stats = self._import_batch(batch)
                stats["imported"] += batch_stats["imported"]
                stats["skipped"] += batch_stats["skipped"]
                stats["errors"] += batch_stats["errors"]

                logger.info(
                    f"Imported batch {i // batch_size + 1}: "
                    f"{batch_stats['imported']} segments, "
                    f"{batch_stats['skipped']} skipped, "
                    f"{batch_stats['errors']} errors"
                )

            self.db.commit()
            logger.info(
                f"Import completed: {stats['imported']} imported, "
                f"{stats['skipped']} skipped, {stats['errors']} errors"
            )

            return stats

        except Exception as e:
            self.db.rollback()
            logger.error(f"Import failed: {str(e)}")
            raise

    def _import_batch(self, road_segments: List[RoadSegmentData]) -> dict:
        """
        Import a batch of road segments.

        Args:
            road_segments: List of RoadSegmentData objects

        Returns:
            Dictionary with batch statistics
        """
        stats = {"imported": 0, "skipped": 0, "errors": 0}

        for segment_data in road_segments:
            try:
                # Convert coordinates to LineString WKT
                # RoadSegmentData.geom_coordinates already uses (lon, lat) order - standard GeoJSON/PostGIS format
                coords_wkt = ", ".join(
                    [f"{lon} {lat}" for lon, lat in segment_data.geom_coordinates]
                )
                linestring_wkt = f"LINESTRING({coords_wkt})"

                # No boundary filtering here - Overpass API already filters by relation/polygon/bbox
                # The data fetched from Overpass API is already filtered, so we trust it and import directly

                # Check if segment already exists
                existing = self.db.query(RoadSegment).filter(RoadSegment.id == segment_data.osm_id).first()

                # Use raw SQL to insert geometry properly
                geom_sql = text(f"ST_GeogFromText('SRID=4326;{linestring_wkt}')")

                if existing:
                    # Update existing segment using raw SQL
                    self.db.execute(
                        text("""
                            UPDATE road_segment 
                            SET geom = ST_GeogFromText(:wkt),
                                road_type = :road_type,
                                speed_limit = :speed_limit,
                                one_way = :one_way
                            WHERE id = :id
                        """),
                        {
                            "wkt": f"SRID=4326;{linestring_wkt}",
                            "road_type": segment_data.road_type,
                            "speed_limit": segment_data.speed_limit,
                            "one_way": segment_data.one_way,
                            "id": segment_data.osm_id,
                        },
                    )
                    stats["imported"] += 1
                else:
                    # Insert new segment using raw SQL
                    self.db.execute(
                        text("""
                            INSERT INTO road_segment (id, geom, road_type, speed_limit, one_way)
                            VALUES (:id, ST_GeogFromText(:wkt), :road_type, :speed_limit, :one_way)
                        """),
                        {
                            "id": segment_data.osm_id,
                            "wkt": f"SRID=4326;{linestring_wkt}",
                            "road_type": segment_data.road_type,
                            "speed_limit": segment_data.speed_limit,
                            "one_way": segment_data.one_way,
                        },
                    )
                    stats["imported"] += 1

            except Exception as e:
                logger.warning(f"Failed to import segment {segment_data.osm_id}: {str(e)}")
                stats["errors"] += 1
                continue

        return stats

    def get_import_statistics(self) -> dict:
        """
        Get statistics about imported road segments.

        Returns:
            Dictionary with statistics
        """
        try:
            total_count = self.db.query(RoadSegment).count()

            # Count by road type
            road_type_counts = (
                self.db.query(RoadSegment.road_type, text("COUNT(*)"))
                .group_by(RoadSegment.road_type)
                .all()
            )

            # Count one-way roads
            one_way_count = self.db.query(RoadSegment).filter(RoadSegment.one_way == True).count()

            return {
                "total_segments": total_count,
                "road_types": {rt: count for rt, count in road_type_counts},
                "one_way_count": one_way_count,
                "two_way_count": total_count - one_way_count,
            }

        except Exception as e:
            logger.error(f"Failed to get import statistics: {str(e)}")
            return {"error": str(e)}

    def check_data_exists(self) -> bool:
        """
        Check if road segment data already exists in database.

        Returns:
            True if data exists, False otherwise
        """
        try:
            count = self.db.query(RoadSegment).count()
            return count > 0
        except Exception:
            return False

    def clean_segments_outside_boundary(self) -> dict:
        """
        Remove road segments that are outside Küçükçekmece boundary.
        This ensures only segments significantly within the boundary remain.
        
        Returns:
            Dictionary with cleaning statistics
        """
        stats = {
            "total_before": 0,
            "deleted": 0,
            "kept": 0,
            "errors": 0,
        }
        
        try:
            # Get total count before cleaning
            stats["total_before"] = self.db.query(RoadSegment).count()
            
            # Check if boundary exists
            boundary_exists = self.db.execute(
                text("""
                    SELECT COUNT(*) > 0
                    FROM administrative_boundary 
                    WHERE name = 'Küçükçekmece' AND admin_level = 6
                """)
            ).scalar()
            
            if not boundary_exists:
                logger.warning("Küçükçekmece boundary not found, skipping cleanup")
                stats["kept"] = stats["total_before"]
                return stats
            
            # Optimized cleanup: motorway/trunk only need intersection, others need center OR 30% length
            deleted_count = self.db.execute(
                text("""
                    WITH boundary_geom AS (
                        SELECT geom FROM administrative_boundary 
                        WHERE name = 'Küçükçekmece' AND admin_level = 6 LIMIT 1
                    )
                    DELETE FROM road_segment rs
                    WHERE NOT (
                        -- Motorway/trunk: only check intersection (they often cross boundaries)
                        (rs.road_type IN ('motorway', 'trunk') AND ST_Intersects(
                            rs.geom::geometry,
                            (SELECT geom::geometry FROM boundary_geom)
                        ))
                        OR
                        -- Other roads: center within OR at least 30% length within
                        (rs.road_type NOT IN ('motorway', 'trunk') AND (
                            ST_Within(
                                ST_Centroid(rs.geom::geometry),
                                (SELECT geom::geometry FROM boundary_geom)
                            )
                            OR
                            (
                                ST_Length(
                                    ST_Intersection(
                                        rs.geom::geometry,
                                        (SELECT geom::geometry FROM boundary_geom)
                                    )
                                ) / NULLIF(ST_Length(rs.geom::geometry), 0)
                            ) >= 0.3
                        ))
                    )
                """)
            ).rowcount
            
            self.db.commit()
            stats["deleted"] = deleted_count
            stats["kept"] = stats["total_before"] - deleted_count
            
            logger.info(
                f"Cleaned road segments: {stats['deleted']} deleted, "
                f"{stats['kept']} kept (out of {stats['total_before']} total)"
            )
            
            return stats
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to clean segments outside boundary: {str(e)}")
            stats["errors"] = 1
            return stats

