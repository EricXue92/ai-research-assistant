# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

Always activate the venv first:
```bash
source venv/bin/activate
export ANTHROPIC_API_KEY=your-key-here
```

**Local development (two terminals):**
```bash
# Terminal 1 — backend API
cd backend && uvicorn main:app --reload

# Terminal 2 — Streamlit UI
streamlit run app.py
```

**Deployed version** uses only `app.py` (no FastAPI) — imports backend modules directly.

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Architecture

There are **two deployment modes** that must stay in sync:

| Mode | Entry point | How backend is called |
|---|---|---|
| Local dev | `frontend/app.py` | HTTP via `httpx` to FastAPI on `:8000` |
| Production (Streamlit Cloud) | `app.py` (root) | Direct Python imports from `backend/` |

`app.py` (root) adds `backend/` to `sys.path` and imports `rag`, `claude_client`, and `export` directly — no HTTP layer.

### Data flow

```
PDF upload → extract_text_from_pdf() → chunk_document() → VectorStore.add_document()
                                                                    ↓ (FAISS index)
User question → VectorStore.search() → relevant Chunk objects
                                              ↓
                              stream_answer(chunks, question, chat_history)
                                              ↓
                                     Claude API (streaming)
                                              ↓
                                    st.write_stream() → UI
```

### Key types

- `Chunk` (dataclass in `rag.py`): `text`, `source` (filename), `chunk_index`. Flows through the entire pipeline — from `VectorStore.search()` return value to `stream_answer()` context and citation rendering.
- `VectorStore`: in-memory FAISS index. `add_document()` is additive (multi-doc); `reset()` clears everything. Module-level `SentenceTransformer` model is shared.

### Session state keys (`app.py`)

| Key | Type | Purpose |
|---|---|---|
| `store` | `VectorStore` | Shared FAISS index across all loaded docs |
| `loaded_docs` | `dict[str, str]` | Maps filename → full extracted text |
| `chat_history` | `list[dict]` | `{"question": ..., "answer": ...}` pairs |
| `citations` | `list[list[Chunk]]` | Parallel to `chat_history`, chunks used per answer |
| `summary` | `str` | Cached summary of selected document |

### macOS proxy issue

On macOS with a SOCKS proxy, `httpx` picks up system proxy settings. The frontend httpx client uses `trust_env=False` to bypass it for localhost calls. The Anthropic client does **not** use `trust_env=False` — it must go through the system proxy to reach the Anthropic API.

## Model

Claude model is set in `backend/claude_client.py`. Current: `claude-sonnet-4-6`. To check which models are available for an API key: `python3 check_models.py` (requires `ANTHROPIC_API_KEY` to be exported).

## Deployment

Deployed on Streamlit Community Cloud using `app.py` as the entry point. `ANTHROPIC_API_KEY` is set as a secret in the Streamlit Cloud dashboard, not in any file.
