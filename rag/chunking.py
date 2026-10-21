"""
Text Chunking — ClipMind AI RAG
=======================================
Handles PDF → text → chunks with configurable strategy.
Supports chunk sizes 256 / 512 / 1024 for benchmark comparison.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))


def _load_pdf(path: str) -> str:
    """Extract raw text from a PDF file."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF {path}: {e}")


def _load_text(path: str) -> str:
    """Read a plain text or markdown file."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_document(path: str) -> str:
    """Load any supported document format and return raw text."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    elif ext in (".txt", ".md", ".rst"):
        return _load_text(path)
    elif ext == ".docx":
        try:
            import docx
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            raise RuntimeError(f"Failed to read DOCX {path}: {e}")
    else:
        # Try as plain text
        return _load_text(path)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping word-level chunks.

    Args:
        text:       Raw document text
        chunk_size: Target words per chunk (256 / 512 / 1024)
        overlap:    Overlapping words between consecutive chunks

    Returns:
        List of text chunk strings
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def chunk_document(
    path: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    End-to-end: load document → extract text → split into chunks.
    Used by vector_db.py to index documents.
    """
    text = load_document(path)
    chunks = chunk_text(text, chunk_size, overlap)
    return chunks
