"""WebSocket connection manager for real-time risk updates."""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time risk updates."""

    def __init__(self):
        """Initialize the WebSocket manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: WebSocket connection
            client_id: Optional client ID (generated if not provided)

        Returns:
            Client ID
        """
        await websocket.accept()

        if client_id is None:
            client_id = str(uuid.uuid4())

        async with self._lock:
            self.active_connections[client_id] = websocket
            self.connection_metadata[client_id] = {
                "connected_at": datetime.now(timezone.utc).isoformat(),
                "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            }

        logger.info(f"WebSocket client connected: {client_id}")
        return client_id

    async def disconnect(self, client_id: str):
        """
        Disconnect a WebSocket client.

        Args:
            client_id: Client ID to disconnect
        """
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
            if client_id in self.connection_metadata:
                del self.connection_metadata[client_id]

        logger.info(f"WebSocket client disconnected: {client_id}")

    async def send_personal_message(self, message: dict, client_id: str):
        """
        Send a message to a specific client.

        Args:
            message: Message dictionary to send
            client_id: Target client ID
        """
        if client_id not in self.active_connections:
            logger.warning(f"Client {client_id} not found in active connections")
            return

        websocket = self.active_connections[client_id]
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message to client {client_id}: {str(e)}")
            await self.disconnect(client_id)

    async def broadcast(self, message: dict, exclude_client: Optional[str] = None):
        """
        Broadcast a message to all connected clients.

        Args:
            message: Message dictionary to broadcast
            exclude_client: Optional client ID to exclude from broadcast
        """
        disconnected_clients = []

        async with self._lock:
            clients_to_send = [
                (client_id, ws)
                for client_id, ws in self.active_connections.items()
                if client_id != exclude_client
            ]

        for client_id, websocket in clients_to_send:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {str(e)}")
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)

    async def broadcast_risk_update(self, risk_data: dict):
        """
        Broadcast a risk update to all connected clients.

        Args:
            risk_data: Risk map data to broadcast
        """
        message = {
            "type": "risk_update",
            "data": risk_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await self.broadcast(message)
        logger.info(f"Broadcasted risk update to {len(self.active_connections)} clients")

    async def send_heartbeat(self, client_id: str):
        """
        Send a heartbeat message to a client.

        Args:
            client_id: Client ID
        """
        message = {
            "type": "heartbeat",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await self.send_personal_message(message, client_id)

        # Update last heartbeat time
        async with self._lock:
            if client_id in self.connection_metadata:
                self.connection_metadata[client_id]["last_heartbeat"] = (
                    datetime.now(timezone.utc).isoformat()
                )

    async def send_error(self, client_id: str, error_message: str):
        """
        Send an error message to a client.

        Args:
            client_id: Client ID
            error_message: Error message
        """
        message = {
            "type": "error",
            "data": {"message": error_message},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await self.send_personal_message(message, client_id)

    def get_connection_count(self) -> int:
        """
        Get the number of active connections.

        Returns:
            Number of active connections
        """
        return len(self.active_connections)

    def get_client_ids(self) -> Set[str]:
        """
        Get all active client IDs.

        Returns:
            Set of client IDs
        """
        return set(self.active_connections.keys())


# Singleton instance
_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """
    Get the singleton WebSocket manager instance.

    Returns:
        WebSocketManager instance
    """
    global _manager
    if _manager is None:
        _manager = WebSocketManager()
    return _manager




