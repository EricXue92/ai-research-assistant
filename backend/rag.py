"""
rag.py — Retrieval-Augmented Generation logic

This file handles:
1. Extracting text from PDFs
2. Splitting text into chunks (with source tracking for multi-doc support)
3. Building a searchable vector store using FAISS
"""

import io
import faiss
import numpy as np
from dataclasses import dataclass
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Lazy-loaded embedding model — initialized on first use, not at import time
_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


@dataclass
class Chunk:
    """A piece of text with metadata about where it came from."""
    text: str
    source: str       # PDF filename
    chunk_index: int  # position within that document


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Read a PDF from raw bytes and return all its text."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n".join(pages)


def chunk_document(text: str, source: str, chunk_size: int = 150, overlap: int = 20) -> list[Chunk]:
    """
    Split text into overlapping Chunk objects tagged with their source filename.

    Why overlap? So sentences spanning chunk boundaries aren't lost.
    """
    words = text.split()
    chunks = []
    i = 0
    index = 0
    while i < len(words):
        chunk_text = " ".join(words[i: i + chunk_size])
        chunks.append(Chunk(text=chunk_text, source=source, chunk_index=index))
        i += chunk_size - overlap
        index += 1
    return chunks


class VectorStore:
    """
    Stores text chunks as vectors so we can find the most relevant ones
    for any given question — the 'retrieval' part of RAG.

    Supports multiple documents: call add_document() for each PDF.
    """

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.index = None  # FAISS index

    def reset(self):
        """Clear all documents and reset the index."""
        self.chunks = []
        self.index = None

    def add_document(self, chunks: list[Chunk]):
        """
        Add one document's chunks to the index without clearing existing ones.
        This enables multi-document search across all loaded PDFs.
        """
        if not chunks:
            return

        embeddings = _get_model().encode([c.text for c in chunks], show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)

        if self.index is None:
            # First document — create the index
            dim = embeddings.shape[1]  # 384
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 4) -> list[Chunk]:
        """
        Find the top_k most relevant chunks for a given question.
        Returns Chunk objects so callers know which document each came from.
        """
        if self.index is None or not self.chunks:
            return []

        query_vector = _get_model().encode([query])
        _, indices = self.index.search(
            np.array(query_vector, dtype=np.float32), top_k
        )

        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
