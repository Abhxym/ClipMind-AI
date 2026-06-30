# AI Video Director — Tests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestRAGChunking:
    def test_chunk_text_basic(self):
        from rag.chunking import chunk_text
        text = " ".join(["word"] * 1000)
        chunks = chunk_text(text, chunk_size=256, overlap=64)
        assert len(chunks) > 0
        assert all(len(c.split()) <= 256 + 10 for c in chunks)

    def test_chunk_overlap(self):
        from rag.chunking import chunk_text
        text = " ".join([str(i) for i in range(100)])
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        # Check overlap: last words of chunk[0] appear in start of chunk[1]
        if len(chunks) > 1:
            end_words = set(chunks[0].split()[-5:])
            start_words = set(chunks[1].split()[:5])
            assert len(end_words & start_words) > 0

    def test_empty_text(self):
        from rag.chunking import chunk_text
        assert chunk_text("") == []


class TestReranker:
    def test_reranker_returns_top_k(self):
        from rag.reranker import BGEReranker
        reranker = BGEReranker()
        docs = [f"Document about topic {i}" for i in range(10)]
        results = reranker.rerank("topic 5", docs, top_k=3)
        assert len(results) == 3
        assert all("text" in r and "score" in r and "original_index" in r for r in results)

    def test_reranker_scores_descending(self):
        from rag.reranker import BGEReranker
        reranker = BGEReranker()
        docs = ["Paris is the capital of France", "The sky is blue", "Football is a sport"]
        results = reranker.rerank("France capital city", docs, top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestState:
    def test_create_initial_state(self):
        from memory.langgraph_state import create_initial_state
        state = create_initial_state(
            job_id="test-123",
            raw_clips=["clip1.mp4", "clip2.mp4"],
            user_prompt="30-second reel",
            target_duration=30,
            platform="instagram",
        )
        assert state["job_id"] == "test-123"
        assert len(state["raw_clips"]) == 2
        assert state["platform"] == "instagram"
        assert state["scenes"] == []
        assert state["errors"] == []


class TestEvaluation:
    def test_precision_at_k(self):
        from evaluation.metrics import retrieval_precision_at_k
        docs = ["brand color palette", "music style upbeat", "weather is cold"]
        p = retrieval_precision_at_k(docs, ["color", "music"], k=3)
        assert 0.0 <= p <= 1.0

    def test_hallucination_empty(self):
        from evaluation.metrics import hallucination_check
        assert hallucination_check([], "") == 0.0

    def test_latency_breakdown(self):
        from evaluation.metrics import latency_breakdown
        timings = {"scene_agent": 500, "quality_agent": 300, "export_agent": 100}
        result = latency_breakdown(timings)
        assert result["total_ms"] == 900
        assert result["bottleneck_agent"] == "scene_agent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
