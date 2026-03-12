# AI Research Assistant

An AI-powered tool to analyze research papers using **RAG** (Retrieval-Augmented Generation) and **streaming** responses via the Claude API.

## Features
- Upload any research paper (PDF)
- Get an instant AI-generated summary
- Ask questions and get accurate, document-grounded answers in real time

## Tech Stack
- **Backend:** FastAPI + FAISS + sentence-transformers
- **Frontend:** Streamlit
- **AI:** Claude API (Anthropic) with streaming

## Architecture
```
User uploads PDF
      ↓
FastAPI extracts text → splits into chunks → embeds with sentence-transformers → stores in FAISS
      ↓
User asks a question
      ↓
Question is embedded → FAISS finds top 4 relevant chunks
      ↓
Chunks + question sent to Claude API → answer streamed back to UI
```

## Setup

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Set your API key
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Start the backend
```bash
cd backend
uvicorn main:app --reload
```

### 4. Start the frontend (new terminal)
```bash
cd frontend
streamlit run app.py
```

Then open http://localhost:8501 in your browser.
