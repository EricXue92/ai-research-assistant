"""
tests/test_rag.py — Unit tests for rag.py core logic

Tests chunking, VectorStore add/search/reset.
No API key or network access required.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from rag import chunk_document, VectorStore, Chunk


# ── chunk_document ──────────────────────────────────────────────

def test_chunk_document_basic():
    text = " ".join([f"word{i}" for i in range(300)])
    chunks = chunk_document(text, source="test.pdf")
    assert len(chunks) > 0
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.source == "test.pdf"
        assert len(c.text) > 0


def test_chunk_document_index_increments():
    text = " ".join([f"word{i}" for i in range(300)])
    chunks = chunk_document(text, source="test.pdf")
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_document_overlap_means_more_chunks():
    text = " ".join([f"word{i}" for i in range(300)])
    chunks_with_overlap    = chunk_document(text, source="test.pdf", chunk_size=150, overlap=20)
    chunks_without_overlap = chunk_document(text, source="test.pdf", chunk_size=150, overlap=0)
    assert len(chunks_with_overlap) >= len(chunks_without_overlap)


def test_chunk_document_short_text():
    chunks = chunk_document("hello world", source="tiny.pdf")
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_chunk_document_empty_text():
    chunks = chunk_document("", source="empty.pdf")
    assert chunks == []


def test_chunk_document_source_preserved():
    text = " ".join([f"word{i}" for i in range(200)])
    chunks = chunk_document(text, source="my_paper.pdf")
    assert all(c.source == "my_paper.pdf" for c in chunks)


# ── VectorStore ──────────────────────────────────────────────────

def test_vectorstore_starts_empty():
    store = VectorStore()
    assert store.index is None
    assert store.chunks == []


def test_vectorstore_search_empty_returns_empty():
    store = VectorStore()
    results = store.search("anything")
    assert results == []


def test_vectorstore_add_and_search():
    store = VectorStore()
    chunks = chunk_document(
        "Machine learning is a subset of artificial intelligence. "
        "Neural networks learn from data. Deep learning uses many layers.",
        source="ml.pdf",
    )
    store.add_document(chunks)
    results = store.search("neural networks")
    assert len(results) > 0
    assert all(isinstance(r, Chunk) for r in results)


def test_vectorstore_search_returns_relevant_chunk():
    store = VectorStore()
    chunks = chunk_document(
        "The Eiffel Tower is in Paris France. "
        "Python is a programming language. "
        "The Louvre is a famous museum in Paris.",
        source="facts.pdf",
    )
    store.add_document(chunks)
    results = store.search("Paris landmark")
    texts = " ".join(r.text for r in results)
    assert "Paris" in texts or "Eiffel" in texts or "Louvre" in texts


def test_vectorstore_reset():
    store = VectorStore()
    chunks = chunk_document("Some text about science.", source="sci.pdf")
    store.add_document(chunks)
    assert store.index is not None
    store.reset()
    assert store.index is None
    assert store.chunks == []


def test_vectorstore_multi_document():
    store = VectorStore()
    chunks_a = chunk_document("Python is great for data science.", source="a.pdf")
    chunks_b = chunk_document("JavaScript is popular for web development.", source="b.pdf")
    store.add_document(chunks_a)
    store.add_document(chunks_b)

    sources = {c.source for c in store.chunks}
    assert "a.pdf" in sources
    assert "b.pdf" in sources


def test_vectorstore_add_empty_chunks_does_nothing():
    store = VectorStore()
    store.add_document([])
    assert store.index is None
    assert store.chunks == []


def test_vectorstore_search_top_k():
    store = VectorStore()
    chunks = chunk_document(
        " ".join([f"sentence about topic {i} and details" for i in range(50)]),
        source="big.pdf",
    )
    store.add_document(chunks)
    results = store.search("topic", top_k=3)
    assert len(results) <= 3
