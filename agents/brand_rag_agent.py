"""
Brand RAG Agent — AI Video Director
=====================================
Retrieves brand style context from a vector database.

Pipeline:
  PDF/text upload → chunking → embeddings → FAISS/Chroma storage
  → query (user_prompt + tone) → Top-10 retrieval
  → BGE cross-encoder reranker → Top-3 chunks → BrandStyle dict

This is the showpiece RAG component with a visible reranker lift.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from logs.agent_logs import AgentLogger
from memory.langgraph_state import BrandStyle, VideoDirectorState
from rag.reranker import BGEReranker
from rag.vector_db import VectorDatabase


TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "10"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "3"))


# ── Default Brand Style ────────────────────────────────────────────────────────

DEFAULT_BRAND_STYLE = BrandStyle(
    primary_color="#FF4B4B",
    secondary_color="#FFD700",
    caption_tone="energetic",
    font_preference="Inter",
    transition_style="fast-cut",
    music_preference="upbeat",
    logo_position="bottom-right",
    raw_chunks=[],
)

# Tone → brand defaults mapping (fallback when no PDF uploaded)
TONE_DEFAULTS = {
    "energetic":     {"transition_style": "fast-cut",  "music_preference": "upbeat",    "caption_tone": "energetic"},
    "cinematic":     {"transition_style": "smooth",    "music_preference": "cinematic",  "caption_tone": "cinematic"},
    "casual":        {"transition_style": "cut",       "music_preference": "lo-fi",      "caption_tone": "casual"},
    "professional":  {"transition_style": "fade",      "music_preference": "corporate",  "caption_tone": "professional"},
    "emotional":     {"transition_style": "fade",      "music_preference": "ambient",    "caption_tone": "emotional"},
    "humorous":      {"transition_style": "zoom",      "music_preference": "pop",        "caption_tone": "casual"},
}


def _parse_brand_from_chunks(chunks: List[str], tone: str) -> BrandStyle:
    """
    Heuristically extract brand parameters from retrieved text chunks.
    Falls back to tone-based defaults for any missing fields.
    """
    combined = " ".join(chunks).lower()
    tone_defaults = TONE_DEFAULTS.get(tone, TONE_DEFAULTS["energetic"])

    def _extract(keywords: List[str], default: str) -> str:
        for kw in keywords:
            if kw in combined:
                idx = combined.index(kw)
                snippet = combined[idx : idx + 40].split()[1:3]
                if snippet:
                    return " ".join(snippet)
        return default

    # Try to detect explicit color mentions
    primary = "#FF4B4B"
    if "primary color" in combined or "brand color" in combined:
        import re
        hex_colors = re.findall(r"#[0-9a-fA-F]{6}", " ".join(chunks))
        if hex_colors:
            primary = hex_colors[0]

    return BrandStyle(
        primary_color=primary,
        secondary_color="#FFD700",
        caption_tone=tone_defaults["caption_tone"],
        font_preference="Inter",
        transition_style=tone_defaults["transition_style"],
        music_preference=tone_defaults["music_preference"],
        logo_position="bottom-right",
        raw_chunks=chunks,
    )


# ── LangGraph Node ─────────────────────────────────────────────────────────────

def brand_rag_agent(state: VideoDirectorState) -> VideoDirectorState:
    """
    LangGraph node: Brand RAG Agent.
    Reads:  state.brand_pdf_path, state.user_prompt, state.overall_tone
    Writes: state.brand_style, state.retrieved_doc_ids, state.reranked_doc_ids, state.rag_query
    """
    job_id = state["job_id"]
    log = AgentLogger("brand_rag_agent", job_id)
    log.start("Loading brand style via RAG")

    # If no brand PDF, use tone-based defaults
    if not state.get("brand_pdf_path"):
        log.info("No brand PDF uploaded — using tone-based defaults")
        tone = state.get("overall_tone", "energetic")
        brand_style = _parse_brand_from_chunks([], tone)
        return {
            **state,
            "brand_style": brand_style,
            "retrieved_doc_ids": [],
            "reranked_doc_ids": [],
            "rag_query": "",
            "current_agent": "brand_rag_agent",
            "completed_agents": ["brand_rag_agent"],
            "agent_timings": {
                **state.get("agent_timings", {}),
                "brand_rag_agent": log.elapsed_ms(),
            },
        }

    # Index the brand PDF if needed
    db = VectorDatabase()
    pdf_path = state["brand_pdf_path"]
    log.info(f"Indexing brand PDF: {Path(pdf_path).name}")
    db.index_document(pdf_path, doc_id=f"brand_{job_id}")

    # Build RAG query from user prompt + tone
    tone = state.get("overall_tone", "energetic")
    query = (
        f"{state['user_prompt']} {tone} style transitions captions music colors"
    )
    log.info(f"RAG query: {query[:80]}")

    # Retrieve Top-K
    retrieved = db.search(query, top_k=TOP_K_RETRIEVE)
    log.info(f"Retrieved {len(retrieved)} chunks (Top-{TOP_K_RETRIEVE})")

    retrieved_ids = [r["id"] for r in retrieved]
    retrieved_texts = [r["text"] for r in retrieved]

    # Rerank with BGE cross-encoder
    reranker = BGEReranker()
    reranked = reranker.rerank(query, retrieved_texts, top_k=TOP_K_RERANK)
    reranked_texts = [r["text"] for r in reranked]
    reranked_ids = [retrieved_ids[r["original_index"]] for r in reranked]

    log.info(
        f"Reranker lift: P@{TOP_K_RETRIEVE}→{TOP_K_RERANK} | "
        f"scores: {[f'{r[\"score\"]:.3f}' for r in reranked]}"
    )

    brand_style = _parse_brand_from_chunks(reranked_texts, tone)
    log.success(
        f"Brand style: transitions={brand_style['transition_style']}, "
        f"music={brand_style['music_preference']}"
    )

    return {
        **state,
        "brand_style": brand_style,
        "retrieved_doc_ids": retrieved_ids,
        "reranked_doc_ids": reranked_ids,
        "rag_query": query,
        "current_agent": "brand_rag_agent",
        "completed_agents": ["brand_rag_agent"],
        "agent_timings": {
            **state.get("agent_timings", {}),
            "brand_rag_agent": log.elapsed_ms(),
        },
    }
