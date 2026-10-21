"""
Embedding Agent — ClipMind AI
=====================================
Pipeline 7 + 8: Generate embeddings and store in FAISS vector DB.
  - Converts transcript segments + scene descriptions into Sentence-Transformer
    embeddings (EMBEDDING_MODEL, default all-MiniLM-L6-v2)
  - Stores in a FAISS index (persisted to vector_db/)
  - Tags each vector with metadata: timestamp, clip, type
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from memory.state import VideoState
from tools.embeddings import EMBEDDING_MODEL, embed_texts

VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", "vector_db"))


def _build_documents(state: VideoState) -> List[Dict[str, Any]]:
    """Create a flat list of text documents to embed."""
    docs = []

    # From transcript
    for seg in state.get("transcript", []):
        text = seg.get("text", "").strip()
        if text and not text.startswith("["):
            docs.append({
                "text": text,
                "type": "transcript",
                "clip_path": seg.get("clip_path", ""),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
            })

    # From scene descriptions
    for sc in state.get("objects", []):
        desc = sc.get("description", "").strip()
        labels = ", ".join(sc.get("object_labels", []))
        text = f"{desc}. Objects: {labels}".strip(" .")
        if text:
            docs.append({
                "text": text,
                "type": "scene",
                "clip_path": sc.get("clip_path", ""),
                "global_index": sc.get("global_index", 0),
                "start": sc.get("start", 0),
                "end": sc.get("end", 0),
                "excitement_score": sc.get("excitement_score", 0),
            })

    return docs


def embedding_agent_node(state: VideoState) -> VideoState:
    """
    LangGraph Node: Embedding + Vector DB Agent.
    Reads:  transcript, objects
    Writes: logs (embeddings stored to disk — consumed by rag_agent)
    """
    logs = ["[embedding_agent] Building embedding store"]
    errors = []

    docs = _build_documents(state)
    logs.append(f"[embedding_agent] Documents to embed: {len(docs)}")

    if not docs:
        logs.append("[embedding_agent] No documents to embed, skipping FAISS build")
        return {**state, "current_agent": "embedding_agent", "errors": [], "logs": logs}

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    job_id = state.get("job_id", "default")
    index_dir = VECTOR_DB_DIR / job_id
    index_dir.mkdir(exist_ok=True)

    texts = [d["text"] for d in docs]
    with open(index_dir / "docs.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)

    embeddings = embed_texts(texts)
    if embeddings is None:
        logs.append("[embedding_agent] sentence-transformers unavailable — saved raw docs for keyword retrieval")
        return {**state, "current_agent": "embedding_agent", "errors": errors, "logs": logs}

    try:
        import faiss

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product = cosine on normalised vecs
        index.add(embeddings)
        faiss.write_index(index, str(index_dir / "index.faiss"))
        logs.append(f"[embedding_agent] FAISS index saved: {len(docs)} vectors @ dim={dim} ({EMBEDDING_MODEL})")
    except ImportError:
        logs.append("[embedding_agent] faiss-cpu not installed — saved raw docs for keyword retrieval")
    except Exception as exc:
        errors.append({"agent": "embedding_agent", "type": "FAISSError", "message": str(exc)})

    return {
        **state,
        "current_agent": "embedding_agent",
        "errors": errors,
        "logs": logs,
    }
