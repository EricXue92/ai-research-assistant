"""
rag.py — Retrieval-Augmented Generation logic

This file handles 3 things:
1. Extracting text from a PDF
2. Splitting the text into small chunks
3. Building a vector store so we can search by meaning (not just keywords)
"""

import io
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Load a small but powerful embedding model (downloads once, ~80MB)
# This model converts text into vectors of 384 numbers
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Read a PDF from raw bytes and return all its text."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 20) -> list[str]:
    """
    Split text into overlapping chunks of words.

    Why overlap? So that a sentence spanning two chunks isn't lost.
    Example with chunk_size=5, overlap=2:
      words = [A, B, C, D, E, F, G]
      chunk1 = [A, B, C, D, E]
      chunk2 = [D, E, F, G]   ← D and E are repeated for continuity
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap  # step forward, keeping some overlap
    return chunks


class VectorStore:
    """
    Stores text chunks as vectors so we can find the most relevant ones
    for any given question — this is the 'retrieval' part of RAG.
    """

    def __init__(self):
        self.chunks = []   # the original text chunks
        self.index = None  # FAISS index (the searchable vector database)

    def build(self, chunks: list[str]):
        """Convert chunks to vectors and store them in a FAISS index."""
        self.chunks = chunks

        # encode() turns each chunk into a 384-dimensional vector
        embeddings = model.encode(chunks, show_progress_bar=False)

        # FAISS IndexFlatL2 finds nearest vectors by Euclidean distance
        dim = embeddings.shape[1]  # 384
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings, dtype=np.float32))

    def search(self, query: str, top_k: int = 4) -> list[str]:
        """
        Find the top_k most relevant chunks for a given question.

        Steps:
          1. Embed the question into a vector
          2. Search FAISS for the closest chunk vectors
          3. Return those chunks as text
        """
        if self.index is None:
            return []

        query_vector = model.encode([query])
        _, indices = self.index.search(
            np.array(query_vector, dtype=np.float32), top_k
        )

        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
