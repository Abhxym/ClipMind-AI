"""
RAG Agent — ClipMind AI
================================
Pipeline Step 7: Retrieval-Augmented Generation for brand/style context.
  - If a brand PDF was uploaded: chunk it, embed it into an in-memory FAISS
    index and retrieve the top-K chunks for the brief + scene context
  - Otherwise: search the per-job index of transcript + scene documents
    built by the embedding agent
  - Falls back to keyword matching if FAISS / embeddings are unavailable
  - Reranks the candidates with a BGE cross-encoder down to TOP_K_RERANK
  - Populates retrieved_brand_docs for the planner agent
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from memory.state import VideoState
from tools.embeddings import embed_texts
from tools.pdf_parser import process_brand_pdf
from tools.reranker import rerank

VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", "vector_db"))
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "10"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "3"))


def _load_docs(job_id: str) -> List[Dict[str, Any]]:
    """Load the docs.json file saved by the embedding agent."""
    docs_path = VECTOR_DB_DIR / job_id / "docs.json"
    if docs_path.exists():
        try:
            with open(docs_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _faiss_search(query: str, job_id: str, top_k: int) -> List[Dict[str, Any]]:
    """Search FAISS index for query. Returns top-k matching docs."""
    try:
        import faiss
        import numpy as np
        index_dir = VECTOR_DB_DIR / job_id
        index_path = index_dir / "index.faiss"
        docs_path = index_dir / "docs.json"

        if not index_path.exists() or not docs_path.exists():
            return []

        query_vec = embed_texts([query])
        if query_vec is None:
            return []

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

    retrieved = []
    brand_pdf_path = state.get("brand_pdf_path")

    # 1. Ingest and search attached Brand PDF if present
    if brand_pdf_path and Path(brand_pdf_path).exists():
        logs.append(f"[rag_agent] Processing brand PDF: {Path(brand_pdf_path).name}")
        chunks = process_brand_pdf(brand_pdf_path)
        if chunks and chunks[0].get("type") == "error":
            logs.append(f"[rag_agent] Error parsing PDF: {chunks[0].get('text')}")
        elif chunks:
            texts = [c["text"] for c in chunks]
            embeddings = embed_texts(texts)
            if embeddings is not None:
                try:
                    import faiss
                    dim = embeddings.shape[1]
                    pdf_index = faiss.IndexFlatIP(dim)
                    pdf_index.add(embeddings)
                    query_vec = embed_texts([query])

                    k = min(TOP_K_RETRIEVE, len(chunks))
                    scores, indices = pdf_index.search(query_vec, k)
                    
                    for score, idx in zip(scores[0], indices[0]):
                        if 0 <= idx < len(chunks):
                            doc = dict(chunks[idx])
                            doc["relevance_score"] = round(float(score), 4)
                            retrieved.append(doc)
                    logs.append(f"[rag_agent] FAISS retrieved {len(retrieved)} chunks from attached PDF")
                except Exception as e:
                    logs.append(f"[rag_agent] PDF FAISS search error: {e}")
            
            if not retrieved:
                logs.append("[rag_agent] Keyword fallback on PDF chunks")
                retrieved = _keyword_search(query, chunks, TOP_K_RETRIEVE)

    # 2. Fallback to default FAISS if no attached PDF
    if not retrieved:
        retrieved = _faiss_search(query, job_id, TOP_K_RETRIEVE)
        if retrieved:
            logs.append(f"[rag_agent] FAISS search returned {len(retrieved)} results from project index")

    # 3. Fallback to keyword search on loaded docs if FAISS failed
    if not retrieved:
        all_docs = _load_docs(job_id)
        if all_docs:
            logs.append(f"[rag_agent] Keyword fallback on {len(all_docs)} docs")
            retrieved = _keyword_search(query, all_docs, TOP_K_RETRIEVE)

    # Rerank if we have enough candidates
    if len(retrieved) > TOP_K_RERANK:
        reranked = rerank(query, retrieved, top_k=TOP_K_RERANK)
        logs.append(f"[rag_agent] Reranked to top {len(reranked)}")
        retrieved = reranked

    for doc in retrieved:
        logs.append(
            f"[rag_agent] doc [{doc.get('type','?')}] score={float(doc.get('relevance_score', 0)):.3f}: "
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
