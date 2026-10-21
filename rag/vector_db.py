"""
Vector Database — AI Video Director RAG
========================================
Dual-backend: FAISS (fast local ANN) and ChromaDB (persistent, filterable).
Switch with VECTOR_DB_BACKEND env var.

Both backends expose the same interface:
  db.index_document(path, doc_id)
  db.search(query, top_k) → List[{"id", "text", "score"}]
  db.search_by_text(texts, query, top_k)
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from rag.chunking import chunk_document


BACKEND = os.getenv("VECTOR_DB_BACKEND", "faiss")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "rag/faiss_index"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "rag/chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "brand_knowledge")


# ── Embedding Model (shared singleton) ────────────────────────────────────────

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _embed(texts: List[str]) -> np.ndarray:
    model = _get_embed_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ── FAISS Backend ──────────────────────────────────────────────────────────────

class FAISSVectorDB:
    """
    In-memory FAISS index with pickle persistence.
    Stores: index + list of (id, text) pairs.
    """

    def __init__(self, index_path: Path = FAISS_INDEX_PATH):
        import faiss
        self.index_path = index_path
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._index_file = index_path / "index.faiss"
        self._meta_file = index_path / "meta.pkl"

        if self._index_file.exists() and self._meta_file.exists():
            self._index = faiss.read_index(str(self._index_file))
            with open(self._meta_file, "rb") as f:
                self._meta: List[dict] = pickle.load(f)
        else:
            dim = 384  # all-MiniLM-L6-v2 output dim
            self._index = faiss.IndexFlatIP(dim)   # Inner product = cosine (normalised)
            self._meta: List[dict] = []

    def _save(self):
        import faiss
        faiss.write_index(self._index, str(self._index_file))
        with open(self._meta_file, "wb") as f:
            pickle.dump(self._meta, f)

    def add_texts(self, texts: List[str], doc_id: str):
        embeddings = _embed(texts)
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            self._index.add(emb.reshape(1, -1).astype(np.float32))
            self._meta.append({"id": chunk_id, "text": text, "doc_id": doc_id})
        self._save()

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        if self._index.ntotal == 0:
            return []
        q_emb = _embed([query]).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._meta):
                results.append({
                    "id": self._meta[idx]["id"],
                    "text": self._meta[idx]["text"],
                    "score": float(score),
                })
        return results


# ── ChromaDB Backend ───────────────────────────────────────────────────────────

class ChromaVectorDB:
    """
    Persistent ChromaDB backend with sentence-transformer embeddings.
    """

    def __init__(
        self,
        db_path: str = CHROMA_DB_PATH,
        collection_name: str = CHROMA_COLLECTION,
    ):
        import chromadb
        from chromadb.config import Settings
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_texts(self, texts: List[str], doc_id: str):
        embeddings = _embed(texts).tolist()
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(texts))]
        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"doc_id": doc_id} for _ in texts],
        )

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        q_emb = _embed([query]).tolist()
        results = self._collection.query(
            query_embeddings=q_emb,
            n_results=min(top_k, max(1, self._collection.count())),
        )
        output = []
        for i, (doc, dist, doc_id) in enumerate(zip(
            results["documents"][0],
            results["distances"][0],
            results["ids"][0],
        )):
            output.append({
                "id": doc_id,
                "text": doc,
                "score": 1.0 - float(dist),   # convert distance to similarity
            })
        return output


# ── Unified Interface ──────────────────────────────────────────────────────────

class VectorDatabase:
    """
    Unified vector database interface.
    Delegates to FAISS or Chroma based on VECTOR_DB_BACKEND env var.

    Usage:
        db = VectorDatabase()
        db.index_document("brand_guide.pdf", doc_id="brand_001")
        results = db.search("color palette transitions", top_k=10)
    """

    def __init__(self, backend: Optional[str] = None):
        backend = backend or BACKEND
        if backend == "chroma":
            self._db = ChromaVectorDB()
        else:
            self._db = FAISSVectorDB()
        self._backend = backend
        self._indexed_docs: set = set()

    def index_document(
        self,
        path: str,
        doc_id: Optional[str] = None,
        chunk_size: int = int(os.getenv("CHUNK_SIZE", "512")),
    ):
        """Chunk + embed + store a document."""
        doc_id = doc_id or hashlib.md5(path.encode()).hexdigest()[:12]
        if doc_id in self._indexed_docs:
            return  # Already indexed this session

        chunks = chunk_document(path, chunk_size=chunk_size)
        if not chunks:
            return
        self._db.add_texts(chunks, doc_id)
        self._indexed_docs.add(doc_id)

    def index_texts(self, texts: List[str], doc_id: str):
        """Directly index a list of text strings."""
        if not texts:
            return
        self._db.add_texts(texts, doc_id)
        self._indexed_docs.add(doc_id)

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """Return top_k most similar chunks."""
        return self._db.search(query, top_k)

    @property
    def backend(self) -> str:
        return self._backend
