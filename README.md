# 🎬 AI Video Director

> **Multi-agent AI platform that transforms raw video clips into polished reels — fully automated.**

Built to demonstrate every requirement in the Foto Owl JD: LangGraph, RAG, FAISS/ChromaDB, BGE Reranker, Gemini, FFmpeg, OpenCV, Whisper, FastAPI, and a React dashboard.

---

## 🏗️ Architecture

```
User Prompt + Raw Clips
        │
  ┌─────▼──────┐
  │  FastAPI   │  ← Upload endpoint, WebSocket progress stream
  └─────┬──────┘
        │  LangGraph StateGraph (SQLite checkpointing)
        ▼
  Scene Extraction Agent   ← OpenCV histogram scene detection
        ▼
  Quality Analysis Agent   ← Blur (Laplacian), brightness, motion (optical flow)
        ▼
  Highlight Detection Agent ← Face/smile detection, audio peaks, Whisper transcript
        ▼
  Caption Generator Agent  ← Gemini 1.5 Flash (multimodal: thumbnails + text)
        ▼
  Brand RAG Agent          ← FAISS/Chroma → Top-10 → BGE Reranker → Top-3
        ▼
  Story Planner Agent      ← Gemini 1.5 Pro → narrative arc JSON
        ▼
  Video Editing Agent      ← FFmpeg: trim, transitions, subtitle burn, music mix
        ▼
  Export Agent             ← Final MP4 + thumbnail + evaluation metrics
        ▼
  React Dashboard          ← Live agent graph + video preview + metrics panel
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | **LangGraph** + LangChain |
| LLM | **Gemini 1.5 Flash** (captions) + **Gemini 1.5 Pro** (planning) |
| Vector DB | **FAISS** (primary) + **ChromaDB** (benchmark comparison) |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Reranker | **BGE Reranker** `BAAI/bge-reranker-base` (cross-encoder) |
| Speech-to-Text | **OpenAI Whisper** (`base` model) |
| Video Processing | **OpenCV** + **FFmpeg** + **MoviePy** |
| API | **FastAPI** + WebSockets (real-time progress) |
| Frontend | **React** + **Vite** + TypeScript |
| Tracing | **LangSmith** |
| Persistence | SQLite (LangGraph checkpoints) |
| Evaluation | Custom metrics → CSV benchmark reports |

---

## 📁 Project Structure

```
AI-Video-Director/
├── agents/
│   ├── scene_agent.py          # OpenCV scene extraction
│   ├── quality_agent.py        # Blur/brightness/motion scoring
│   ├── highlight_agent.py      # Face/smile/Whisper/audio peaks
│   ├── caption_agent.py        # Gemini Flash captions
│   ├── brand_rag_agent.py      # FAISS + BGE Reranker
│   ├── story_planner_agent.py  # Gemini Pro narrative arc
│   ├── editing_agent.py        # FFmpeg editing pipeline
│   └── export_agent.py         # Final export + metrics
├── rag/
│   ├── vector_db.py            # FAISS + ChromaDB dual backend
│   ├── chunking.py             # 256/512/1024 token chunking
│   ├── reranker.py             # BGE cross-encoder reranker
│   └── brand_kb/               # Sample brand style guides
├── memory/
│   └── langgraph_state.py      # TypedDict state schema
├── pipeline/
│   └── graph.py                # LangGraph StateGraph assembly
├── evaluation/
│   ├── metrics.py              # Scene accuracy, ROUGE, P@K, latency
│   └── benchmark.py            # FAISS vs Chroma, chunk sizes, top-K
├── logs/
│   ├── agent_logs.py           # Loguru structured logger
│   └── failed_runs/            # JSON failure snapshots
├── api/
│   ├── main.py                 # FastAPI app
│   └── routes/                 # upload, jobs, websocket
├── frontend/                   # React + Vite dashboard
└── tests/                      # Pytest suite
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone <your-repo-url>
cd AI-Video-Director

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here    # optional
LANGCHAIN_TRACING_V2=true
```

> **How to get your Gemini API key:**
> 1. Go to [aistudio.google.com](https://aistudio.google.com)
> 2. Click **Get API key** → Create API key
> 3. Copy and paste into `.env`

> **How to get your LangSmith key (optional but impressive):**
> 1. Sign up at [smith.langchain.com](https://smith.langchain.com)
> 2. Settings → API Keys → Create
> 3. Copy into `.env`

### 3. Install FFmpeg (required for video editing)

**Windows:**
```powershell
winget install ffmpeg
# OR download from https://ffmpeg.org/download.html and add to PATH
```

**macOS:**
```bash
brew install ffmpeg
```

### 4. Start the Backend

```bash
python -m api.main
# API running at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 5. Start the Frontend

```bash
cd frontend
npm install
npm run dev
# Dashboard at http://localhost:5173
```

---

## 🔬 RAG Benchmark

Run the full benchmark comparing FAISS vs ChromaDB, chunk sizes, top-K, and reranker impact:

```bash
# Generate sample brand guide first
python rag/brand_kb/sample_brand_guide.py

# Run benchmark
python -m evaluation.benchmark \
  --brand-pdf rag/brand_kb/sample_brand_guide.txt \
  --query "energetic transitions music captions color palette"
```

Example output:
```
📊 RAG Benchmark Results
Backend  | Chunk | K  | No Rerank | Reranked | Lift   | ms
---------|-------|----|-----------|-----------|---------|---------
faiss    | 512   | 10 | 0.300     | 0.667    | +0.367 | 234ms
chroma   | 512   | 10 | 0.300     | 0.667    | +0.367 | 891ms
faiss    | 256   | 5  | 0.400     | 0.600    | +0.200 | 198ms
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Scene Selection Accuracy | IoU of auto-selected vs manually labelled scenes |
| Caption Quality (ROUGE-L) | F-measure against reference captions |
| Retrieval Precision@K | Fraction of top-K chunks containing relevant keywords |
| Reranker Lift | Δ P@K before vs after BGE reranking |
| End-to-End Latency | Per-agent timing breakdown |
| Hallucination Rate | Caption terms absent from transcript |
| Error Count | Agent-level failure tracking with retry logs |

---

## 🧩 JD Coverage

| Requirement | Implementation |
|-------------|----------------|
| LangGraph | `pipeline/graph.py` — 8-node StateGraph with conditional edges |
| RAG | `agents/brand_rag_agent.py` + `rag/` — PDF → chunks → FAISS → reranker |
| Vector DB | `rag/vector_db.py` — FAISS + ChromaDB with unified interface |
| BGE Reranker | `rag/reranker.py` — cross-encoder, Top-10→Top-3 |
| Agent Memory | `memory/langgraph_state.py` — TypedDict flowing through all nodes |
| Evaluation | `evaluation/metrics.py + benchmark.py` — CSV reports |
| Failure Debugging | `logs/failed_runs/*.json` — structured error snapshots + retry |
| FastAPI | `api/` — upload, jobs, WebSocket stream |
| FFmpeg | `agents/editing_agent.py` — trim, transitions, subtitles, music |
| OpenCV | `agents/scene_agent.py + quality_agent.py + highlight_agent.py` |
| Whisper | `agents/highlight_agent.py` — full audio transcription |
| Gemini | `agents/caption_agent.py + story_planner_agent.py` |
| LangSmith | `pipeline/graph.py` — auto-configured tracing |

---

## 📄 License

MIT — built as a portfolio project demonstrating production-grade AI agent architecture.
