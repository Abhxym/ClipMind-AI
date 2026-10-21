"""
FastAPI Application — AI Video Director
========================================
Main entry point for the backend API.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import jobs, upload, ws

load_dotenv()

# ── Ensure directories exist ──────────────────────────────────────────────────
for d in ["uploads", "outputs", "temp", "logs/failed_runs", "pipeline"]:
    Path(d).mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AI Video Director API starting up...")
    yield
    print("AI Video Director API shutting down")


app = FastAPI(
    title="AI Video Director",
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

from fastapi import Request
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
Path("outputs").mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "AI Video Director"}


@app.get("/")
async def root():
    return {
        "message": "AI Video Director API",
        "docs": "/docs",
        "pipeline_agents": [
            "input_agent", "preprocess_agent", "whisper_agent",
            "scene_agent", "metadata_agent", "embedding_agent",
            "rag_agent", "planner_agent", "caption_agent",
            "editing_agent", "reviewer_agent", "export_agent",
            "evaluation_agent",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
        log_level="info",
    )
