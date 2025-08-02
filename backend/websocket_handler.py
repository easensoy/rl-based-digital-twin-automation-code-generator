import json
import logging
from typing import List, Dict, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class WebSocketConnectionManager:
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        logger.info("WebSocket Connection Manager initialized")
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        try:
            if websocket in self.active_connections:
                await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        if not self.active_connections:
            return
        
        message_text = json.dumps(message)
        disconnected_clients = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.append(connection)
        
        for client in disconnected_clients:
            self.disconnect(client)
        
        if disconnected_clients:
            logger.info(f"Removed {len(disconnected_clients)} disconnected clients")
    
    def get_connection_count(self) -> int:
        return len(self.active_connections)
    
    async def send_training_update(self, episode: int, reward: float, epsilon: float):
        message = {
            "type": "training_update",
            "data": {
                "episode": episode,
                "reward": reward,
                "epsilon": epsilon
            }
        }
        await self.broadcast(message)
    
    async def send_performance_metrics(self, metrics: Dict[str, float]):
        message = {
            "type": "performance_metrics",
            "data": metrics
        }
        await self.broadcast(message)
    
    async def send_system_status(self, status: Dict[str, Any]):
        message = {
            "type": "system_status",
            "data": status
        }
        await self.broadcast(message)