"""
Embeddings Tool — ClipMind AI
======================================
Utility functions for text embedding generation.
Used by: embedding_agent, rag_agent
"""
from __future__ import annotations

import os
from typing import List, Optional

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_model_cache = None


def get_model(model_name: Optional[str] = None):
    """
    Load (and cache) a SentenceTransformer embedding model.
    Returns None if sentence-transformers is not installed.
    """
    global _model_cache
    target = model_name or EMBEDDING_MODEL

    if _model_cache is not None:
        return _model_cache

    try:
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer(target)
        return _model_cache
    except ImportError:
        return None
    except Exception:
        return None


def embed_texts(texts: List[str], normalize: bool = True) -> Optional[list]:
    """
    Embed a list of texts using the configured SentenceTransformer model.
    Returns a numpy array of shape (len(texts), dim), or None on failure.
    """
    model = get_model()
    if model is None:
        return None
    try:
        import numpy as np
        embeddings = model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
        return np.array(embeddings, dtype="float32")
    except Exception:
        return None


def cosine_similarity(vec_a, vec_b) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    Assumes vectors are already L2-normalized (returns dot product).
    """
    try:
        import numpy as np
        a = np.array(vec_a, dtype="float32")
        b = np.array(vec_b, dtype="float32")
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except Exception:
        return 0.0


def keyword_score(query: str, document: str) -> float:
    """
    Simple keyword overlap score (Jaccard-style).
    Used as a fallback when embeddings are unavailable.
    """
    query_tokens = set(query.lower().split())
    doc_tokens = set(document.lower().split())
    if not query_tokens or not doc_tokens:
        return 0.0
    intersection = query_tokens & doc_tokens
    union = query_tokens | doc_tokens
    return len(intersection) / len(union)
