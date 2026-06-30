"""
WebSocket Routes — AI Video Director API
=========================================
Real-time pipeline progress streaming to the React dashboard.
Each connected client receives JSON progress events per agent transition.
"""

from __future__ import annotations

import asyncio
import datetime
import json
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.models import WSProgressMessage


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

AGENT_ORDER = [
    "scene_agent", "quality_agent", "highlight_agent",
    "caption_agent", "brand_rag_agent", "story_planner_agent",
    "editing_agent", "export_agent",
]


def _pct(agent: str, status: str) -> float:
    idx = AGENT_ORDER.index(agent) if agent in AGENT_ORDER else 0
    base = (idx / len(AGENT_ORDER)) * 100
    return base + (100 / len(AGENT_ORDER) if status == "done" else 0)


async def broadcast_progress(job_id: str, agent: str, status: str, attempt: int = 0, msg: str = ""):
    """Called from pipeline progress callback (via run_in_executor)."""
    payload = {
        "type": "progress",
        "job_id": job_id,
        "agent": agent,
        "status": status,
        "attempt": attempt,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "progress_pct": _pct(agent, status),
        "message": msg,
    }
    await manager.broadcast(job_id, payload)


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
        "message": "Connected to AI Video Director pipeline stream",
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
