"""
tests/test_api.py — Integration tests for FastAPI endpoints

Uses httpx AsyncClient with TestClient — no server process needed.
Claude API calls are mocked so no ANTHROPIC_API_KEY required.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# Mock the Claude client before importing main (avoids API key check at import)
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")

with patch("anthropic.Anthropic"):
    from main import app, store

client = TestClient(app)


def _make_minimal_pdf() -> bytes:
    """Return a minimal valid single-page PDF with extractable text."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f\n"
        b"0000000009 00000 n\n0000000058 00000 n\n"
        b"0000000115 00000 n\n0000000266 00000 n\n"
        b"0000000360 00000 n\n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n450\n%%EOF\n"
    )


# ── /upload endpoint ────────────────────────────────────────────

def test_upload_rejects_non_pdf():
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"some text", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_rejects_empty_pdf():
    with patch("main.extract_text_from_pdf", return_value="   "):
        response = client.post(
            "/upload",
            files={"file": ("empty.pdf", b"%PDF-1.4", "application/pdf")},
        )
    assert response.status_code == 400


def test_upload_valid_pdf_succeeds():
    with patch("main.extract_text_from_pdf", return_value="Hello world test document content"):
        response = client.post(
            "/upload",
            files={"file": ("paper.pdf", _make_minimal_pdf(), "application/pdf")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "num_chunks" in data
    assert data["num_chunks"] > 0


# ── /ask endpoint ───────────────────────────────────────────────

def test_ask_without_upload_returns_400():
    store.reset()
    response = client.post("/ask", json={"text": "What is this about?"})
    assert response.status_code == 400


def test_ask_after_upload_streams_response():
    with patch("main.extract_text_from_pdf", return_value="Machine learning is fascinating."):
        client.post(
            "/upload",
            files={"file": ("ml.pdf", _make_minimal_pdf(), "application/pdf")},
        )

    def mock_stream(*args, **kwargs):
        yield "This "
        yield "is "
        yield "an answer."

    with patch("main.stream_answer", return_value=mock_stream()):
        response = client.post("/ask", json={"text": "What is machine learning?"})

    assert response.status_code == 200
    assert "answer" in response.text.lower() or len(response.text) > 0


# ── /summarize endpoint ─────────────────────────────────────────

def test_summarize_without_upload_returns_400():
    store.reset()
    import main as m
    m.full_text_cache = ""
    response = client.post("/summarize")
    assert response.status_code == 400


def test_summarize_after_upload_streams_response():
    with patch("main.extract_text_from_pdf", return_value="Deep learning uses neural networks."):
        client.post(
            "/upload",
            files={"file": ("dl.pdf", _make_minimal_pdf(), "application/pdf")},
        )

    def mock_summary(*args, **kwargs):
        yield "Summary: "
        yield "Deep learning overview."

    with patch("main.summarize", return_value=mock_summary()):
        response = client.post("/summarize")

    assert response.status_code == 200
