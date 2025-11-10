"""Simple in-memory vector store for RAG."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from models.embeddings import embed_texts
from config.config import RAG_CHUNK_SIZE, RAG_TOP_K

@dataclass
class SimpleDocument:
    text: str
    source: str | None = None

@dataclass
class SimpleVectorStore:
    documents: list[SimpleDocument] = field(default_factory=list)
    embeddings: np.ndarray | None = None

    def add_texts(self, texts: List[str], source: str | None = None) -> None:
        if not texts:
            return
        self.documents.extend(SimpleDocument(t, source) for t in texts)
        embs = np.array(embed_texts(texts), dtype=float)
        if self.embeddings is None:
            self.embeddings = embs
        else:
            self.embeddings = np.vstack([self.embeddings, embs])

    def similarity_search(self, query: str, k: int | None = None) -> List[Tuple[SimpleDocument, float]]:
        if not self.documents or self.embeddings is None:
            return []
        q_emb = np.array(embed_texts([query])[0], dtype=float)
        sim = self.embeddings @ q_emb / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-9)
        idx = np.argsort(-sim)[: (k or RAG_TOP_K)]
        return [(self.documents[i], float(sim[i])) for i in idx]

def chunk_text(text: str, chunk_size: int | None = None) -> List[str]:
    size = chunk_size or RAG_CHUNK_SIZE
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts: List[str] = []
    buf = []
    curr = 0
    for line in text.split("\n"):
        if curr + len(line) + 1 > size and buf:
            parts.append("\n".join(buf))
            buf = []
            curr = 0
        buf.append(line)
        curr += len(line) + 1
    if buf:
        parts.append("\n".join(buf))
    return parts
