"""
FastAPI Application — ClipMind AI
========================================
Main entry point for the backend API.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # before importing routes/agents, which read env vars at import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import jobs, upload, ws
from graphs.workflow import AGENT_ORDER

# ── Ensure directories exist ──────────────────────────────────────────────────
for d in [os.getenv("UPLOAD_DIR", "uploads"), os.getenv("OUTPUT_DIR", "outputs"), "temp", "logs/failed_runs", "pipeline"]:
    Path(d).mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("ClipMind AI API starting up...")
    yield
    print("ClipMind AI API shutting down")


app = FastAPI(
    title="ClipMind AI",
    description="Multi-agent AI pipeline for automated video reel generation",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    if request.url.path.startswith("/outputs/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    else:
        response.headers["Cache-Control"] = "no-store, max-age=0"

    if "x-xss-protection" in response.headers:
        del response.headers["x-xss-protection"]
    if "Expires" in response.headers:
        del response.headers["Expires"]

    ct = response.headers.get("Content-Type", "")
    if ct == "application/json":
        response.headers["Content-Type"] = "application/json; charset=utf-8"

    return response


# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(ws.router)

# ── Static outputs (video/thumbnail download) ──────────────────────────────────
app.mount("/outputs", StaticFiles(directory=os.getenv("OUTPUT_DIR", "outputs")), name="outputs")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ClipMind AI"}


@app.get("/")
async def root():
    return {
        "message": "ClipMind AI API",
        "docs": "/docs",
        "pipeline_agents": AGENT_ORDER,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "true").lower() == "true",
        log_level="info",
    )
