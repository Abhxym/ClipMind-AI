"""
BGE Cross-Encoder Reranker — AI Video Director RAG
===================================================
Takes Top-K retrieved chunks and reranks them using a
BAAI/bge-reranker-base cross-encoder model.

This is the interviewers' favourite component — it visually
shows precision improvement from retrieval → reranking.
"""

from __future__ import annotations

import os
from typing import List

from logs.agent_logs import AgentLogger


RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
_reranker_instance = None


def _get_reranker():
    """Lazy-load the cross-encoder model (heavy, only load once)."""
    global _reranker_instance
    if _reranker_instance is None:
        from sentence_transformers import CrossEncoder
        _reranker_instance = CrossEncoder(RERANKER_MODEL)
    return _reranker_instance


class BGEReranker:
    """
    BGE Cross-Encoder Reranker.

    Usage:
        reranker = BGEReranker()
        results = reranker.rerank(query, documents, top_k=3)

    Each result: {"text": str, "score": float, "original_index": int}
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3,
    ) -> List[dict]:
        """
        Rerank documents using cross-encoder relevance scoring.

        Args:
            query:     The search query
            documents: List of retrieved document chunks (strings)
            top_k:     Number of top results to return

        Returns:
            List of dicts sorted by descending relevance score.
            Each: {"text": str, "score": float, "original_index": int}
        """
        if not documents:
            return []

        try:
            model = _get_reranker()
            pairs = [[query, doc] for doc in documents]
            scores = model.predict(pairs)

            scored = [
                {
                    "text": doc,
                    "score": float(score),
                    "original_index": i,
                }
                for i, (doc, score) in enumerate(zip(documents, scores))
            ]
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]

        except Exception as e:
            # Fallback: return first top_k without reranking
            return [
                {"text": doc, "score": 1.0, "original_index": i}
                for i, doc in enumerate(documents[:top_k])
            ]

    def compute_precision_lift(
        self,
        query: str,
        retrieved: List[str],
        relevant_keywords: List[str],
        top_k_before: int,
        top_k_after: int,
    ) -> dict:
        """
        Compute precision@K before and after reranking.
        Used in evaluation/benchmark.py.

        A document is considered 'relevant' if it contains
        at least one of the relevant_keywords.
        """
        def precision_at_k(docs: List[str], k: int) -> float:
            top = docs[:k]
            hits = sum(
                1 for d in top
                if any(kw.lower() in d.lower() for kw in relevant_keywords)
            )
            return hits / k if k > 0 else 0.0

        p_before = precision_at_k(retrieved, top_k_before)
        reranked = self.rerank(query, retrieved, top_k=top_k_after)
        reranked_texts = [r["text"] for r in reranked]
        p_after = precision_at_k(reranked_texts, top_k_after)

        return {
            "precision_before": p_before,
            "precision_after": p_after,
            "lift": p_after - p_before,
            "top_k_before": top_k_before,
            "top_k_after": top_k_after,
        }
