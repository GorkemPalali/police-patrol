"""Real-time WebSocket endpoints for risk updates."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.services.realtime.risk_cache import get_risk_cache
from app.services.realtime.risk_update_service import get_risk_update_service
from app.services.realtime.websocket_manager import get_websocket_manager
from app.services.utils import get_kucukcekmece_bbox_from_polygon

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/realtime", tags=["Realtime"])
settings = get_settings()


@router.websocket("/risk-updates")
async def websocket_risk_updates(
    websocket: WebSocket,
    start_time: Optional[datetime] = Query(None, description="Start time of time window"),
    end_time: Optional[datetime] = Query(None, description="End time of time window"),
    min_lat: Optional[float] = Query(None, description="Bounding box min latitude"),
    min_lng: Optional[float] = Query(None, description="Bounding box min longitude"),
    max_lat: Optional[float] = Query(None, description="Bounding box max latitude"),
    max_lng: Optional[float] = Query(None, description="Bounding box max longitude"),
    db: Session = Depends(get_db),
):
    """
    WebSocket endpoint for real-time risk map updates.

    When a client connects:
    1. Sends current risk map data (from cache or calculated)
    2. Listens for new risk updates and broadcasts them
    3. Sends heartbeat messages periodically
    """
    if not getattr(settings, "realtime_enabled", True):
        await websocket.close(code=1003, reason="Real-time updates disabled")
        return

    websocket_manager = get_websocket_manager()
    risk_cache = get_risk_cache()
    risk_update_service = get_risk_update_service()

    # Connect client
    client_id = await websocket_manager.connect(websocket)

    try:
        # Determine time window
        now = datetime.now(timezone.utc)
        if start_time is None:
            start_time = now
        if end_time is None:
            end_time = now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)

        # Ensure timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        # Determine bounding box
        bbox = None
        if all([min_lat, min_lng, max_lat, max_lng]):
            bbox = (min_lat, min_lng, max_lat, max_lng)
        else:
            bbox = get_kucukcekmece_bbox_from_polygon(db)

        # Send initial risk map data
        try:
            cached_data = risk_cache.get_cached_risk_map(
                start_time=start_time,
                end_time=end_time,
                bbox=bbox,
            )

            if cached_data:
                await websocket_manager.send_personal_message(
                    {
                        "type": "risk_update",
                        "data": cached_data,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    client_id,
                )
            else:
                    # Calculate and send initial risk map
                # Note: This will broadcast to all clients, including the new one
                await risk_update_service.calculate_and_broadcast_risk(
                    start_time=start_time,
                    end_time=end_time,
                    bbox=bbox,
                    db=db,
                )
        except Exception as e:
            logger.error(f"Error sending initial risk map: {str(e)}")
            await websocket_manager.send_error(
                client_id, f"Error loading initial risk map: {str(e)}"
            )

        # Start heartbeat task
        heartbeat_interval = getattr(settings, "websocket_heartbeat_interval", 30)
        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(websocket_manager, client_id, heartbeat_interval)
        )

        # Listen for messages (client can send ping/pong or other commands)
        try:
            while True:
                try:
                    # Wait for message with timeout
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    # Handle client messages if needed
                    logger.debug(f"Received message from client {client_id}: {data}")
                except asyncio.TimeoutError:
                    # Timeout is expected, continue loop
                    continue
                except WebSocketDisconnect:
                    break

        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected")
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}", exc_info=True)
        await websocket_manager.send_error(client_id, f"Server error: {str(e)}")
    finally:
        await websocket_manager.disconnect(client_id)


async def _heartbeat_loop(
    websocket_manager, client_id: str, interval: int
):
    """
    Send periodic heartbeat messages to a client.

    Args:
        websocket_manager: WebSocket manager instance
        client_id: Client ID
        interval: Heartbeat interval in seconds
    """
    try:
        while True:
            await asyncio.sleep(interval)
            await websocket_manager.send_heartbeat(client_id)
    except asyncio.CancelledError:
        logger.debug(f"Heartbeat loop cancelled for client {client_id}")
    except Exception as e:
        logger.error(f"Error in heartbeat loop: {str(e)}")

