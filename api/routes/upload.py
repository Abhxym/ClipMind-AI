"""
Upload Routes — AI Video Director API
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.models import UploadResponse


router = APIRouter(prefix="/api/upload", tags=["upload"])

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".pdf"}


@router.post("/video", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload a single video clip. Returns server path for use in job creation."""
    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, f"Unsupported video format: {ext}")

    # Stream to disk
    file_id = hashlib.md5(f"{file.filename}{time.time()}".encode()).hexdigest()[:12]
    dest = UPLOAD_DIR / f"{file_id}{ext}"

    size = 0
    async with aiofiles.open(str(dest), "wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            size += len(chunk)
            if size > MAX_SIZE_MB * 1024 * 1024:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {MAX_SIZE_MB}MB limit")
            await out.write(chunk)

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        size_bytes=size,
        server_path=str(dest),
    )


@router.post("/brand-pdf", response_model=UploadResponse)
async def upload_brand_pdf(file: UploadFile = File(...)):
    """Upload a brand style guide PDF for RAG indexing."""
    ext = Path(file.filename).suffix.lower()
    if ext != ".pdf":
        raise HTTPException(400, "Only PDF files accepted for brand guide")

    file_id = hashlib.md5(f"{file.filename}{time.time()}".encode()).hexdigest()[:12]
    dest = UPLOAD_DIR / f"brand_{file_id}.pdf"

    size = 0
    async with aiofiles.open(str(dest), "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            await out.write(chunk)

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        size_bytes=size,
        server_path=str(dest),
    )
