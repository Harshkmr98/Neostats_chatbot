"""Simple in-memory vector store for Retrieval-Augmented Generation (RAG)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from models.embeddings import embed_texts
from config.config import RAG_CHUNK_SIZE, RAG_TOP_K


@dataclass
class SimpleDocument:
    """Represents a single text chunk in the knowledge base."""

    id: int
    text: str


@dataclass
class SimpleVectorStore:
    """Minimal in-memory vector store for small to medium knowledge bases."""

    documents: List[SimpleDocument] = field(default_factory=list)
    embeddings: np.ndarray | None = None  # shape: (n_docs, dim)

    def add_texts(self, texts: List[str]) -> None:
        """Add new texts to the store and compute their embeddings.

        Args:
            texts: List of text chunks to index.
        """
        if not texts:
            return

        start_id = len(self.documents)
        new_docs = [SimpleDocument(id=start_id + i, text=t) for i, t in enumerate(texts)]
        new_embs = np.asarray(embed_texts(texts), dtype="float32")

        self.documents.extend(new_docs)
        if self.embeddings is None:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
    ) -> List[Tuple[SimpleDocument, float]]:
        """Return the top-k most similar documents to the query.

        Args:
            query: The user query string.
            k: Number of results to return. Defaults to RAG_TOP_K.

        Returns:
            List of (document, similarity_score) pairs sorted by descending score.
        """
        if not self.documents or self.embeddings is None:
            return []

        k = k or RAG_TOP_K

        query_emb = np.asarray(embed_texts([query])[0], dtype="float32")

        doc_embs = self.embeddings
        dot_products = doc_embs @ query_emb
        norms = np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb)
        norms = np.where(norms == 0, 1e-10, norms)
        scores = dot_products / norms

        idxs = np.argsort(scores)[::-1][:k]
        return [(self.documents[int(i)], float(scores[int(i)])) for i in idxs]


def chunk_text(text: str, chunk_size: int | None = None) -> List[str]:
    """Naive character-based text chunker for uploaded files.

    Args:
        text: Input document text.
        chunk_size: Target chunk size in characters. Defaults to RAG_CHUNK_SIZE.

    Returns:
        List of text chunks.
    """
    size = chunk_size or RAG_CHUNK_SIZE
    text = text.strip()
    if not text:
        return []

    return [text[i : i + size] for i in range(0, len(text), size)]
