"""Risk update service for real-time risk map updates."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.crime_event import CrimeEvent
from app.services.realtime.risk_cache import get_risk_cache
from app.services.realtime.websocket_manager import get_websocket_manager
from app.services.utils import get_kucukcekmece_bbox_from_polygon

logger = logging.getLogger(__name__)

settings = get_settings()


class RiskUpdateService:
    """Service for triggering and broadcasting risk updates."""

    def __init__(self):
        """Initialize the risk update service."""
        self.websocket_manager = get_websocket_manager()
        self.risk_cache = get_risk_cache()

    async def trigger_risk_update(
        self, crime_event: CrimeEvent, db: Session
    ):
        """
        Trigger risk update when a new crime event is added.

        Args:
            crime_event: The newly added crime event
            db: Database session
        """
        if not getattr(settings, "realtime_enabled", True):
            logger.debug("Real-time updates disabled, skipping risk update")
            return

        if not getattr(settings, "risk_update_broadcast_enabled", True):
            logger.debug("Risk update broadcast disabled, skipping")
            return

        try:
            # Get active time windows (current time + next hour)
            active_windows = self._get_active_time_windows()

            # Get bbox from crime event location
            bbox = self._get_bbox_from_crime_event(crime_event, db)

            # Calculate and broadcast risk for each active window
            for start_time, end_time in active_windows:
                await self.calculate_and_broadcast_risk(
                    start_time=start_time,
                    end_time=end_time,
                    bbox=bbox,
                    db=db,
                )

        except Exception as e:
            logger.error(f"Error triggering risk update: {str(e)}", exc_info=True)

    async def calculate_and_broadcast_risk(
        self,
        start_time: datetime,
        end_time: datetime,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        grid_size_m: Optional[float] = None,
        use_hex: bool = True,
        db: Optional[Session] = None,
    ):
        """
        Calculate risk map and broadcast to connected clients.

        Args:
            start_time: Start time of the time window
            end_time: End time of the time window
            bbox: Optional bounding box
            grid_size_m: Optional grid size in meters
            use_hex: Use hex grid (default: True)
            db: Database session (required if not cached)
        """
        if not getattr(settings, "realtime_enabled", True):
            return

        try:
            # Check cache first
            cached_data = self.risk_cache.get_cached_risk_map(
                start_time=start_time,
                end_time=end_time,
                bbox=bbox,
                grid_size_m=grid_size_m,
                use_hex=use_hex,
            )

            if cached_data:
                logger.debug("Using cached risk map data for broadcast")
                await self.websocket_manager.broadcast_risk_update(cached_data)
                return

            # If not cached and db is provided, calculate
            if db is None:
                logger.warning(
                    "Cannot calculate risk map: database session not provided"
                )
                return

            # Import here to avoid circular dependencies
            from app.api.routes.forecast import generate_forecast_risk_cells
            from app.models.risk_cell import RiskCell
            from sqlalchemy import text

            # Calculate risk cells
            risk_cells = generate_forecast_risk_cells(
                db=db,
                time_window_start=start_time,
                time_window_end=end_time,
                bbox=bbox or get_kucukcekmece_bbox_from_polygon(db),
                grid_size_m=grid_size_m,
                use_hex=use_hex,
            )

            if not risk_cells:
                logger.warning("No risk cells generated")
                return

            # Convert to response format
            risk_cell_data = []
            for cell in risk_cells:
                if cell.risk_score >= 0.0:  # Include all cells
                    # Get coordinates from cell
                    from app.services.utils import get_point_coordinates
                    lat, lng = get_point_coordinates(db, cell.geom)

                    risk_cell_data.append({
                        "id": str(cell.id),
                        "lat": lat,
                        "lng": lng,
                        "risk_score": cell.risk_score,
                        "confidence": cell.confidence,
                    })

            risk_response = type("RiskMapResponse", (), {
                "cells": risk_cell_data
            })()

            # Convert response to dict for broadcasting
            risk_data = {
                "cells": risk_response.cells,
                "time_window": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "bbox": bbox,
                "grid_size_m": grid_size_m or settings.default_grid_size_m,
                "use_hex": use_hex,
            }

            # Cache the result
            self.risk_cache.set_cached_risk_map(
                risk_data=risk_data,
                start_time=start_time,
                end_time=end_time,
                bbox=bbox,
                grid_size_m=grid_size_m,
                use_hex=use_hex,
            )

            # Broadcast to connected clients
            await self.websocket_manager.broadcast_risk_update(risk_data)

            logger.info(
                f"Broadcasted risk update for window {start_time} - {end_time}"
            )

        except Exception as e:
            logger.error(
                f"Error calculating and broadcasting risk: {str(e)}",
                exc_info=True,
            )

    def _get_active_time_windows(
        self, window_hours: int = 1
    ) -> List[Tuple[datetime, datetime]]:
        """
        Get active time windows for risk updates.

        Args:
            window_hours: Number of hours ahead to consider (default: 1)

        Returns:
            List of (start_time, end_time) tuples
        """
        now = datetime.now(timezone.utc)

        # Current window: now to now + window_hours
        windows = [
            (
                now,
                now + timedelta(hours=window_hours),
            )
        ]

        return windows

    def _get_bbox_from_crime_event(
        self, crime_event: CrimeEvent, db: Session
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Get bounding box from crime event location.

        Args:
            crime_event: Crime event
            db: Database session

        Returns:
            Bounding box tuple or None
        """
        try:
            # Get crime event coordinates
            from app.services.utils import get_point_coordinates

            lat, lng = get_point_coordinates(db, crime_event.geom)

            if lat == 0.0 and lng == 0.0:
                # Fallback to Küçükçekmece bbox
                return get_kucukcekmece_bbox_from_polygon(db)

            # Create a small bbox around the crime event (e.g., 1km radius)
            # Approximate: 0.01 degrees ≈ 1km
            radius = 0.01

            bbox = (
                lat - radius,
                lng - radius,
                lat + radius,
                lng + radius,
            )

            # Intersect with Küçükçekmece bbox
            kucukcekmece_bbox = get_kucukcekmece_bbox_from_polygon(db)
            if kucukcekmece_bbox:
                bbox = (
                    max(bbox[0], kucukcekmece_bbox[0]),
                    max(bbox[1], kucukcekmece_bbox[1]),
                    min(bbox[2], kucukcekmece_bbox[2]),
                    min(bbox[3], kucukcekmece_bbox[3]),
                )

            return bbox

        except Exception as e:
            logger.error(f"Error getting bbox from crime event: {str(e)}")
            return get_kucukcekmece_bbox_from_polygon(db)

    def invalidate_cache_for_bbox(
        self, bbox: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Invalidate cache for a specific bounding box.

        Args:
            bbox: Optional bounding box (invalidates all if None)
        """
        self.risk_cache.invalidate_cache(bbox=bbox)


# Singleton instance
_service: Optional[RiskUpdateService] = None


def get_risk_update_service() -> RiskUpdateService:
    """
    Get the singleton risk update service instance.

    Returns:
        RiskUpdateService instance
    """
    global _service
    if _service is None:
        _service = RiskUpdateService()
    return _service

