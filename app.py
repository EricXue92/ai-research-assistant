"""
app.py — Single-file Streamlit app for deployment

Combines the frontend UI and backend logic into one file.
No FastAPI needed — RAG and Claude are called directly.
"""

import sys
import os

# Allow imports from the backend/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import streamlit as st
from rag import extract_text_from_pdf, chunk_text, VectorStore
from claude_client import stream_answer, summarize

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Research Assistant", page_icon="📄", layout="wide")
st.title("📄 AI Research Assistant")
st.caption("Upload a research paper and ask questions about it — powered by Claude")

# ── Session state ─────────────────────────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store = VectorStore()
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "full_text" not in st.session_state:
    st.session_state.full_text = ""

# ── Sidebar: upload ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner("Extracting text and building vector index..."):
            text = extract_text_from_pdf(uploaded_file.getvalue())
            if not text.strip():
                st.error("Could not extract text from this PDF.")
            else:
                chunks = chunk_text(text)
                st.session_state.store.build(chunks)
                st.session_state.full_text = text
                st.session_state.doc_loaded = True
                st.session_state.chat_history = []
                st.session_state.summary = ""
                st.success(f"✅ Processed '{uploaded_file.name}' — {len(chunks)} chunks indexed.")

    if st.session_state.doc_loaded:
        st.divider()
        st.header("2. Summarize")
        if st.button("Generate Summary"):
            st.session_state.summary = ""
            with st.spinner("Summarizing..."):
                # Collect the full streamed summary
                st.session_state.summary = "".join(
                    summarize(st.session_state.full_text)
                )

        if st.session_state.summary:
            st.markdown(st.session_state.summary)

# ── Main area: Q&A ────────────────────────────────────────────────────────────
if not st.session_state.doc_loaded:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    st.header("3. Ask Questions")

    # Show chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])

    question = st.chat_input("Ask anything about the document...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            relevant_chunks = st.session_state.store.search(question)
            # st.write_stream() natively handles generators and streams token by token
            full_answer = st.write_stream(stream_answer(relevant_chunks, question))

        st.session_state.chat_history.append(
            {"question": question, "answer": full_answer}
        )
