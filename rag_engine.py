"""
RAG Engine — Text chunking, FAISS vector indexing, and Groq-powered QA.
"""

import hashlib
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

        # Incremental indexing — hash cache & embedding cache
        self._page_hashes: dict[str, str] = {}       # {url: sha256_hex}
        self._page_embeddings: dict[str, tuple[list[Chunk], np.ndarray]] = {}  # {url: (chunks, embeddings)}

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

    @staticmethod
    def _content_hash(text: str) -> str:
        """Return the SHA-256 hex digest of *text*."""
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    def ingest_documents(self, documents: list[dict]) -> dict:
        """
        Process a list of documents (dicts with url, title, content) into
        the FAISS index.

        Uses **smart incremental indexing**: pages whose content has not
        changed since the last ingest are skipped — their cached chunks
        and embeddings are reused directly.

        Returns
        -------
        dict with ingestion stats (includes ``skipped_documents``).
        """
        self._load_embed_model()

        new_or_changed_docs: list[dict] = []
        skipped = 0

        for doc in documents:
            url = doc.get("url", "")
            content = doc.get("content", "")
            page_hash = self._content_hash(content)

            if url in self._page_hashes and self._page_hashes[url] == page_hash:
                # Content unchanged — skip re-embedding
                skipped += 1
                logger.debug("Skipping unchanged page: %s", url)
                continue

            new_or_changed_docs.append(doc)
            self._page_hashes[url] = page_hash

        if skipped:
            logger.info(
                "Incremental indexing: %d page(s) unchanged — skipped.",
                skipped,
            )

        # ---- Chunk & embed only the NEW / CHANGED documents ----
        if new_or_changed_docs:
            new_chunks: list[Chunk] = []
            for doc in new_or_changed_docs:
                text = doc.get("content", "")
                url = doc.get("url", "")
                title = doc.get("title", "")
                parts = self.splitter.split(text)
                for i, part in enumerate(parts):
                    new_chunks.append(
                        Chunk(text=part, source_url=url, source_title=title, chunk_index=i)
                    )

            texts = [c.text for c in new_chunks]
            logger.info("Encoding %d new chunks …", len(texts))
            new_embeddings = self._embed_model.encode(
                texts, show_progress_bar=False, normalize_embeddings=True, batch_size=64,
            )
            new_embeddings = np.array(new_embeddings, dtype="float32")

            # Cache embeddings per URL
            for doc in new_or_changed_docs:
                url = doc.get("url", "")
                url_chunks = [c for c in new_chunks if c.source_url == url]
                url_indices = [i for i, c in enumerate(new_chunks) if c.source_url == url]
                if url_indices:
                    self._page_embeddings[url] = (
                        url_chunks,
                        new_embeddings[url_indices],
                    )
        elif not self._page_embeddings:
            # Nothing new and no cache — nothing to index
            return {"total_chunks": 0, "total_documents": 0, "skipped_documents": 0}

        # ---- Rebuild FAISS index from ALL cached embeddings ----
        # Keep only pages that are present in the current scrape set
        current_urls = {doc.get("url", "") for doc in documents}
        # Prune stale URLs from caches
        stale = set(self._page_hashes.keys()) - current_urls
        for url in stale:
            self._page_hashes.pop(url, None)
            self._page_embeddings.pop(url, None)

        all_chunks: list[Chunk] = []
        all_embeddings: list[np.ndarray] = []
        for url in current_urls:
            if url in self._page_embeddings:
                chunks, embs = self._page_embeddings[url]
                all_chunks.extend(chunks)
                all_embeddings.append(embs)

        if not all_chunks:
            return {"total_chunks": 0, "total_documents": 0, "skipped_documents": skipped}

        combined = np.vstack(all_embeddings).astype("float32")
        self.index = faiss.IndexFlatIP(self._embed_dim)
        self.index.add(combined)
        self.chunks = all_chunks
        self._ready = True

        stats = self.stats
        stats["skipped_documents"] = skipped
        logger.info(
            "Index built — %d chunks from %d documents (%d skipped).",
            stats["total_chunks"], stats["total_documents"], skipped,
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
    # Generation (Groq)
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 5, chat_history: list | None = None) -> QueryResult:
        """
        Full RAG pipeline: retrieve context → generate answer with Groq.
        Accepts optional chat_history (list of {"role", "content"} dicts)
        for multi-turn conversations.
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

        # Call Groq
        if self._groq_client:
            try:
                # Build messages: system → history → current question
                system_msg = {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions based ONLY on the "
                        "provided context. If the context does not contain enough information, "
                        "say so clearly. Cite sources using [Source N] notation."
                    ),
                }

                user_msg = {
                    "role": "user",
                    "content": f"### Context\n\n{context}\n\n### Question\n\n{question}",
                }

                messages = [system_msg]
                # Append previous conversation turns
                if chat_history:
                    for turn in chat_history[-10:]:  # keep last 10 turns to stay within token limits
                        messages.append({
                            "role": turn.get("role", "user"),
                            "content": turn.get("content", ""),
                        })
                messages.append(user_msg)

                response = self._groq_client.chat.completions.create(
                    model=self.GROQ_MODEL_NAME,
                    messages=messages,
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
