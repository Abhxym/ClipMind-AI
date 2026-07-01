"""
RAG Agent — AI Video Director
================================
Pipeline Step 7: Retrieval-Augmented Generation for brand context.
  - Reads the FAISS vector index built by the embedding agent
  - Retrieves the top-K most relevant brand/style docs for the user prompt
  - Falls back to keyword matching if FAISS is unavailable
  - Populates retrieved_brand_docs for the planner agent
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from memory.state import VideoState

VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", "vector_db"))
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "10"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "3"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


def _load_docs(job_id: str) -> List[Dict[str, Any]]:
    """Load the docs.json file saved by the embedding agent."""
    index_dir = VECTOR_DB_DIR / job_id
    docs_path = index_dir / "docs.json"
    if docs_path.exists():
        try:
            with open(docs_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Also check rag/ directory for brand PDFs processed by the brand RAG pipeline
    rag_paths = [
        Path("rag") / "faiss_index" / "docs.json",
        Path("rag") / "chroma_db" / "docs.json",
    ]
    for p in rag_paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return []


def _faiss_search(query: str, job_id: str, top_k: int) -> List[Dict[str, Any]]:
    """Search FAISS index for query. Returns top-k matching docs."""
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        index_dir = VECTOR_DB_DIR / job_id
        index_path = index_dir / "index.faiss"
        docs_path = index_dir / "docs.json"

        if not index_path.exists() or not docs_path.exists():
            return []

        model = SentenceTransformer(EMBEDDING_MODEL)
        query_vec = model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")

        index = faiss.read_index(str(index_path))
        with open(docs_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        k = min(top_k, len(docs))
        scores, indices = index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(docs):
                doc = dict(docs[idx])
                doc["relevance_score"] = round(float(score), 4)
                results.append(doc)
        return results

    except ImportError:
        return []
    except Exception:
        return []


def _keyword_search(query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Simple keyword scoring fallback."""
    query_terms = set(query.lower().split())
    scored = []
    for doc in docs:
        text = doc.get("text", "").lower()
        score = sum(1 for term in query_terms if term in text)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, doc in scored[:top_k]:
        d = dict(doc)
        d["relevance_score"] = score
        results.append(d)
    return results


def _rerank_with_model(query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Optional: rerank with a cross-encoder model."""
    try:
        from sentence_transformers import CrossEncoder
        reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        reranker = CrossEncoder(reranker_model)

        pairs = [(query, d.get("text", "")) for d in docs]
        scores = reranker.predict(pairs)
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored_docs[:top_k]:
            d = dict(doc)
            d["rerank_score"] = round(float(score), 4)
            results.append(d)
        return results
    except Exception:
        return docs[:top_k]


def rag_agent_node(state: VideoState) -> VideoState:
    """
    LangGraph Node: RAG Agent.
    Reads:  user_prompt, job_id
    Writes: retrieved_brand_docs
    """
    logs: List[str] = ["[rag_agent] Retrieving relevant brand/style context"]
    errors: List[Dict[str, Any]] = []

    job_id = state.get("job_id", "default")
    user_prompt = state.get("user_prompt", "")

    # Build a rich query combining prompt and scene context
    scene_descs = state.get("scene_descriptions", [])
    scene_context = " ".join(d.get("description", "") for d in scene_descs[:3])
    query = f"{user_prompt}. {scene_context}".strip()

    # First try FAISS
    retrieved = _faiss_search(query, job_id, TOP_K_RETRIEVE)
    logs.append(f"[rag_agent] FAISS search returned {len(retrieved)} results")

    # If no FAISS results, fallback to keyword search on loaded docs
    if not retrieved:
        all_docs = _load_docs(job_id)
        logs.append(f"[rag_agent] Keyword fallback on {len(all_docs)} docs")
        retrieved = _keyword_search(query, all_docs, TOP_K_RETRIEVE)

    # Rerank if we have enough candidates
    if len(retrieved) > TOP_K_RERANK:
        reranked = _rerank_with_model(query, retrieved, TOP_K_RERANK)
        logs.append(f"[rag_agent] Reranked to top {len(reranked)}")
        retrieved = reranked

    for doc in retrieved:
        logs.append(
            f"[rag_agent] doc [{doc.get('type','?')}] score={doc.get('relevance_score', 0):.3f}: "
            f"\"{doc.get('text','')[:80]}...\""
        )

    if not retrieved:
        logs.append("[rag_agent] No brand docs found — proceeding without RAG context")

    return {
        **state,
        "retrieved_brand_docs": retrieved,
        "current_agent": "rag_agent",
        "errors": errors,
        "logs": logs,
    }
