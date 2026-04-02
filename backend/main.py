"""
main.py — FastAPI backend

Exposes two endpoints:
  POST /upload  — accepts a PDF, runs RAG processing, stores vectors
  POST /ask     — accepts a question, returns a streaming answer
  POST /summarize — returns a streaming summary of the uploaded doc
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import extract_text_from_pdf, chunk_document, VectorStore
from claude_client import stream_answer, summarize

app = FastAPI(title="AI Research Assistant API")

# Allow the Streamlit frontend (running on port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# A single in-memory store — holds the current document's vectors
# (In production you'd use a persistent database per user/session)
store = VectorStore()
full_text_cache = ""  # cache the raw text for summarization


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    1. Read the uploaded PDF bytes
    2. Extract text
    3. Split into chunks
    4. Build FAISS vector index
    """
    global full_text_cache

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    text = extract_text_from_pdf(content)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    full_text_cache = text
    store.reset()
    chunks = chunk_document(text, source=file.filename)
    store.add_document(chunks)

    return {
        "message": f"✅ Processed '{file.filename}' — {len(chunks)} chunks indexed.",
        "num_chunks": len(chunks),
    }


class Question(BaseModel):
    text: str


@app.post("/ask")
def ask_question(q: Question):
    """
    1. Search the vector store for relevant chunks
    2. Stream Claude's answer back using Server-Sent Events
    """
    if store.index is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet.")

    relevant_chunks = store.search(q.text)

    # StreamingResponse sends data chunk by chunk as Claude generates it
    return StreamingResponse(
        stream_answer(relevant_chunks, q.text),
        media_type="text/plain",
    )


@app.post("/summarize")
def summarize_document():
    """Stream a summary of the currently uploaded document."""
    if not full_text_cache:
        raise HTTPException(status_code=400, detail="No document uploaded yet.")

    return StreamingResponse(
        summarize(full_text_cache),
        media_type="text/plain",
    )
