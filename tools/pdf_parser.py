"""
PDF Parser Tool — AI Video Director
=====================================
Utility for extracting text from PDF brand guidelines and semantic chunking.
Used by: rag_agent
"""
from __future__ import annotations

import os
from typing import Any, Dict, List


def parse_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF file."""
    try:
        import PyPDF2
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n\n"
        return text.strip()
    except ImportError:
        return "[Error: PyPDF2 not installed. Cannot parse PDF.]"
    except Exception as e:
        return f"[Error parsing PDF: {e}]"


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks.
    Respects sentence boundaries if possible.
    """
    if not text:
        return []

    words = text.replace('\n', ' ').split()
    chunks = []
    
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        
        # Try to find a period to break nicely, if not at the very end
        if end < len(words):
            for i in range(end - 1, max(start, end - chunk_overlap), -1):
                if words[i].endswith('.'):
                    end = i + 1
                    break

        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "text": chunk_text,
            "type": "brand_doc",
            "start_idx": start,
            "end_idx": end
        })
        
        start = end - chunk_overlap
        if start < 0 or end == len(words):
            start = end

    return chunks


def process_brand_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Parse a PDF and return a list of text chunks ready for embedding."""
    chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "64"))
    
    full_text = parse_pdf(pdf_path)
    if full_text.startswith("[Error"):
        return [{"text": full_text, "type": "error"}]
        
    chunks = chunk_text(full_text, chunk_size, chunk_overlap)
    return chunks
