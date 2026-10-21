"""
WebSocket Routes — ClipMind AI API
=========================================
Real-time pipeline progress streaming to the React dashboard.
Each connected client receives JSON progress events per agent transition
(sent by api/routes/jobs.py through manager.broadcast).
"""

from __future__ import annotations

import asyncio
import json
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect


router = APIRouter(prefix="/api/ws", tags=["websocket"])

# job_id → set of connected WebSocket clients
_connections: Dict[str, Set[WebSocket]] = {}


class ConnectionManager:
    def connect(self, job_id: str, ws: WebSocket):
        if job_id not in _connections:
            _connections[job_id] = set()
        _connections[job_id].add(ws)

    def disconnect(self, job_id: str, ws: WebSocket):
        if job_id in _connections:
            _connections[job_id].discard(ws)

    async def broadcast(self, job_id: str, message: dict):
        dead = set()
        for ws in list(_connections.get(job_id, [])):
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.disconnect(job_id, ws)


manager = ConnectionManager()

@router.websocket("/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job progress.
    Client connects and receives:
      { type, job_id, agent, status, progress_pct, timestamp }
    """
    await websocket.accept()
    manager.connect(job_id, websocket)

    # Send initial connection acknowledgement
    await websocket.send_text(json.dumps({
        "type": "connected",
        "job_id": job_id,
        "message": "Connected to ClipMind AI pipeline stream",
    }))

    try:
        # Keep connection alive, receiving optional ping messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                # Send keep-alive
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
    except WebSocketDisconnect:
        manager.disconnect(job_id, websocket)
