# ClipMind AI — Tests
# Fast unit tests: no API keys, video files or model downloads required
# (the reranker tests fall back to unscored order if the BGE model is unavailable).

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
        from memory.state import create_initial_state
        state = create_initial_state(
            job_id="test-123",
            user_prompt="30-second reel",
            uploaded_videos=["clip1.mp4", "clip2.mp4"],
            target_duration=30,
            platform="instagram",
        )
        assert state["job_id"] == "test-123"
        assert len(state["uploaded_videos"]) == 2
        assert state["platform"] == "instagram"
        assert state["story_plan"] == []
        assert state["errors"] == []
        assert state["review_passed"] is False
        assert state["agent_timings"] == {}

    def test_merge_timings_adds_repeated_runs(self):
        from memory.state import merge_timings
        assert merge_timings({"editing_agent": 100.0}, {"editing_agent": 50.0, "export_agent": 5.0}) == {
            "editing_agent": 150.0,
            "export_agent": 5.0,
        }


class TestWorkflow:
    def test_review_routing(self):
        from graphs.workflow import review_routing
        from agents.reviewer_agent import MAX_RETRIES
        assert review_routing({"review_passed": True, "retry_count": {"editing_agent": 1}}) == "export_agent"
        assert review_routing({"review_passed": False, "retry_count": {"editing_agent": 1}}) == "editing_agent"
        assert review_routing({"review_passed": False, "retry_count": {"editing_agent": MAX_RETRIES}}) == "export_agent"

    def test_graph_has_all_agents(self):
        from graphs.workflow import AGENT_ORDER, build_workflow
        graph = build_workflow().compile()
        assert set(AGENT_ORDER) <= set(graph.get_graph().nodes)
        assert len(AGENT_ORDER) == 13


class TestTools:
    def test_pdf_chunking_terminates_and_overlaps(self):
        from tools.pdf_parser import chunk_text
        text = " ".join(f"w{i}." if i % 7 == 0 else f"w{i}" for i in range(300))
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
        assert chunks and chunks[-1]["end_idx"] == 300
        assert all(c["end_idx"] > c["start_idx"] for c in chunks)

    def test_drawtext_escaping(self):
        from tools.ffmpeg import _escape_drawtext
        out = _escape_drawtext("It's 100%: go")
        assert "'" not in out
        assert r"\:" in out and r"\%" in out

    def test_planner_clamps_segments_to_clip(self, monkeypatch):
        import agents.planner_agent as planner
        monkeypatch.setattr(planner, "_plan_with_gemini", lambda state: (
            [{"clip_index": 5, "start": 2.0, "end": 40.0}, {"clip_index": "x"}], ""
        ))
        state = {
            "uploaded_videos": ["a.mp4", "b.mp4"],
            "objects": [{"clip_index": 1, "start": 0.0, "end": 8.0}],
            "target_duration": 10,
        }
        out = planner.planner_agent_node(state)
        assert len(out["story_plan"]) == 1
        seg = out["story_plan"][0]
        assert seg["clip_index"] == 1 and seg["end"] == 8.0 and seg["duration"] == 6.0


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
        timings = {"scene_agent": 500, "planner_agent": 300, "export_agent": 100}
        result = latency_breakdown(timings)
        assert result["total_ms"] == 900
        assert result["bottleneck_agent"] == "scene_agent"

    def test_generate_report_from_state(self):
        from evaluation.metrics import generate_report
        from memory.state import create_initial_state
        state = create_initial_state("job-1", "reel", ["a.mp4"])
        state["agent_timings"] = {"scene_agent": 1200.0, "export_agent": 300.0}
        state["evaluation_score"] = {"duration_accuracy": 0.9, "edit_success_rate": 1.0}
        report = generate_report(state)
        assert report["bottleneck_agent"] == "scene_agent"
        assert report["duration_accuracy"] == 0.9
        assert report["success"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
