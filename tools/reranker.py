"""
Reranker Tool — ClipMind AI
=====================================
Cross-encoder reranking utility for RAG retrieval.
Used by: rag_agent
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

_reranker_cache = None


def get_reranker(model_name: Optional[str] = None):
    """
    Load (and cache) a CrossEncoder reranker model.
    Returns None if sentence-transformers is not installed.
    """
    global _reranker_cache
    target = model_name or RERANKER_MODEL

    if _reranker_cache is not None:
        return _reranker_cache

    try:
        from sentence_transformers import CrossEncoder
        _reranker_cache = CrossEncoder(target)
        return _reranker_cache
    except ImportError:
        return None
    except Exception:
        return None


def rerank(
    query: str,
    documents: List[Dict[str, Any]],
    text_key: str = "text",
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Rerank a list of document dicts by relevance to the query.

    Args:
        query:     The search query string.
        documents: List of dicts, each containing at least `text_key`.
        text_key:  Key in each dict that holds the text to score.
        top_k:     Number of top results to return.

    Returns:
        Top-k documents sorted by reranking score (descending),
        each augmented with a "rerank_score" field.
    """
    if not documents:
        return []

    reranker = get_reranker()

    if reranker is None:
        # Fallback: return as-is, truncated to top_k
        return [dict(d) for d in documents[:top_k]]

    try:
        pairs: List[Tuple[str, str]] = [(query, d.get(text_key, "")) for d in documents]
        scores = reranker.predict(pairs)

        scored = sorted(
            zip(scores, documents),
            key=lambda x: float(x[0]),
            reverse=True,
        )

        results = []
        for score, doc in scored[:top_k]:
            d = dict(doc)
            d["rerank_score"] = round(float(score), 4)
            results.append(d)
        return results

    except Exception:
        return [dict(d) for d in documents[:top_k]]


def rerank_texts(
    query: str,
    texts: List[str],
    top_k: int = 3,
) -> List[Tuple[int, float]]:
    """
    Rerank a list of plain text strings.

    Returns:
        List of (original_index, score) tuples sorted by score descending.
    """
    if not texts:
        return []

    reranker = get_reranker()
    if reranker is None:
        return [(i, 0.0) for i in range(min(top_k, len(texts)))]

    try:
        pairs = [(query, t) for t in texts]
        scores = reranker.predict(pairs)
        indexed = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
        return [(i, round(float(s), 4)) for i, s in indexed[:top_k]]
    except Exception:
        return [(i, 0.0) for i in range(min(top_k, len(texts)))]
