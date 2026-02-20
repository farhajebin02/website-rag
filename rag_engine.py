"""
RAG Engine — Text chunking, FAISS vector indexing, and Groq-powered QA.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from groq import Groq
except ImportError:
    Groq = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A text chunk with its source metadata."""
    text: str
    source_url: str
    source_title: str
    chunk_index: int


@dataclass
class QueryResult:
    """Result from a RAG query."""
    answer: str
    sources: List[dict] = field(default_factory=list)
    context_chunks: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Text splitter
# ---------------------------------------------------------------------------

class RecursiveTextSplitter:
    """
    Splits text into overlapping chunks, trying to break at paragraph /
    sentence / word boundaries in that order.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = self._split_recursive(text, self.SEPARATORS)
        # merge tiny trailing chunks
        merged: list[str] = []
        for c in chunks:
            c = c.strip()
            if not c:
                continue
            if merged and len(merged[-1]) + len(c) < self.chunk_size // 2:
                merged[-1] += " " + c
            else:
                merged.append(c)
        return merged

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        sep = separators[0] if separators else " "
        remaining_seps = separators[1:] if len(separators) > 1 else []

        parts = text.split(sep)
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If a single part exceeds chunk_size, recurse with finer sep
                if len(part) > self.chunk_size and remaining_seps:
                    chunks.extend(self._split_recursive(part, remaining_seps))
                else:
                    current = part
                    continue
                current = ""

        if current:
            chunks.append(current)

        # Add overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped: list[str] = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-self.chunk_overlap :]
                overlapped.append(prev_tail + " " + chunks[i])
            return overlapped

        return chunks


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    End-to-end RAG pipeline: ingest documents → build FAISS index → query.
    """

    EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
    GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

    def __init__(self, groq_api_key: Optional[str] = None):
        # Embedding model (lazy-loaded)
        self._embed_model: Optional[SentenceTransformer] = None
        self._embed_dim: int = 384  # MiniLM-L6 output dim

        # FAISS index + metadata store
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []

        # Groq
        self._groq_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self._groq_client = None

        # Splitter
        self.splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=50)

        self._ready = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._ready and self.index is not None and self.index.ntotal > 0

    @property
    def stats(self) -> dict:
        source_urls = set(c.source_url for c in self.chunks)
        return {
            "total_chunks": self.index.ntotal if self.index else 0,
            "total_documents": len(source_urls),
        }

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_embed_model(self):
        if self._embed_model is None:
            logger.info("Loading embedding model: %s …", self.EMBED_MODEL_NAME)
            self._embed_model = SentenceTransformer(self.EMBED_MODEL_NAME)
            self._embed_dim = self._embed_model.get_sentence_embedding_dimension()
            logger.info("Embedding model loaded (dim=%d).", self._embed_dim)

    def _load_groq(self):
        if self._groq_client is None and Groq and self._groq_key:
            self._groq_client = Groq(api_key=self._groq_key)
            logger.info("Groq client configured: %s", self.GROQ_MODEL_NAME)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_documents(self, documents: list[dict]) -> dict:
        """
        Process a list of documents (dicts with url, title, content) into
        the FAISS index.

        Returns
        -------
        dict with ingestion stats.
        """
        self._load_embed_model()

        all_chunks: list[Chunk] = []
        for doc in documents:
            text = doc.get("content", "")
            url = doc.get("url", "")
            title = doc.get("title", "")
            parts = self.splitter.split(text)
            for i, part in enumerate(parts):
                all_chunks.append(
                    Chunk(text=part, source_url=url, source_title=title, chunk_index=i)
                )

        if not all_chunks:
            return {"chunks": 0, "documents": 0}

        # Encode
        texts = [c.text for c in all_chunks]
        logger.info("Encoding %d chunks …", len(texts))
        embeddings = self._embed_model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True, batch_size=64,
        )
        embeddings = np.array(embeddings, dtype="float32")

        # Build FAISS index (inner-product on normalised vectors = cosine sim)
        self.index = faiss.IndexFlatIP(self._embed_dim)
        self.index.add(embeddings)
        self.chunks = all_chunks
        self._ready = True

        stats = self.stats
        logger.info(
            "Index built — %d chunks from %d documents.",
            stats["total_chunks"], stats["total_documents"],
        )
        return stats

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Return the top-k most relevant chunks for a query."""
        if not self.is_ready:
            return []

        q_emb = self._embed_model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(q_emb, top_k)
        results: list[Chunk] = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

    # ------------------------------------------------------------------
    # Generation (Gemini)
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 5) -> QueryResult:
        """
        Full RAG pipeline: retrieve context → generate answer with Gemini.
        """
        self._load_groq()

        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved:
            return QueryResult(
                answer="I don't have any information to answer that question. Please scrape a website first.",
                sources=[],
            )

        # Build context block
        context_parts: list[str] = []
        sources: list[dict] = []
        seen_urls: set[str] = set()
        for i, chunk in enumerate(retrieved, 1):
            context_parts.append(f"[Source {i}] ({chunk.source_url})\n{chunk.text}")
            if chunk.source_url not in seen_urls:
                seen_urls.add(chunk.source_url)
                sources.append({
                    "url": chunk.source_url,
                    "title": chunk.source_title,
                })

        context = "\n\n---\n\n".join(context_parts)

        prompt = (
            "You are a helpful assistant that answers questions based ONLY on the "
            "provided context. If the context does not contain enough information, "
            "say so clearly. Cite sources using [Source N] notation.\n\n"
            f"### Context\n\n{context}\n\n"
            f"### Question\n\n{question}\n\n"
            "### Answer\n"
        )

        # Call Groq
        if self._groq_client:
            try:
                response = self._groq_client.chat.completions.create(
                    model=self.GROQ_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1024,
                )
                answer = response.choices[0].message.content
            except Exception as exc:
                logger.error("Groq generation failed: %s", exc)
                answer = f"Error generating answer: {exc}"
        else:
            # Fallback: no LLM available, return raw context
            answer = (
                "⚠️ No LLM configured. Here are the most relevant passages:\n\n"
                + context
            )

        return QueryResult(
            answer=answer,
            sources=sources,
            context_chunks=[c.text for c in retrieved],
        )
