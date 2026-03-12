"""
app.py — AI Research Assistant (Streamlit)

Features:
1. Multi-document support — upload and search across multiple PDFs
2. Citation highlighting — see which source chunks each answer used
3. Chat memory — Claude remembers previous Q&A in the conversation
4. Export — download the full Q&A session as a PDF report
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import streamlit as st
from rag import extract_text_from_pdf, chunk_document, VectorStore, Chunk
from claude_client import stream_answer, summarize
from export import export_chat_to_pdf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Research Assistant", page_icon="📄", layout="wide")
st.title("📄 AI Research Assistant")
st.caption("Upload research papers and ask questions — powered by Claude")

# ── Session state ─────────────────────────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store = VectorStore()
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = {}       # {filename: full_text}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # [{"question": ..., "answer": ...}]
if "citations" not in st.session_state:
    st.session_state.citations = []         # parallel list of list[Chunk] per answer
if "summary" not in st.session_state:
    st.session_state.summary = ""


def render_citations(chunks: list[Chunk]):
    """Show the source chunks used to generate an answer."""
    if not chunks:
        return
    with st.expander(f"📎 Sources ({len(chunks)} chunks used)", expanded=False):
        for chunk in chunks:
            st.markdown(f"**{chunk.source}** — chunk {chunk.chunk_index + 1}")
            st.caption(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))


doc_loaded = bool(st.session_state.loaded_docs)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── 1. Upload ─────────────────────────────────────────────────────────────
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose one or more PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process Documents", type="primary"):
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.loaded_docs:
                st.warning(f"'{uploaded_file.name}' already loaded.")
                continue
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                text = extract_text_from_pdf(uploaded_file.getvalue())
                if not text.strip():
                    st.error(f"Could not extract text from '{uploaded_file.name}'.")
                    continue
                chunks = chunk_document(text, source=uploaded_file.name)
                st.session_state.store.add_document(chunks)
                st.session_state.loaded_docs[uploaded_file.name] = text
                st.success(f"✅ '{uploaded_file.name}' — {len(chunks)} chunks indexed.")

    # Show loaded documents
    if st.session_state.loaded_docs:
        st.markdown("**Loaded documents:**")
        for name in st.session_state.loaded_docs:
            st.markdown(f"- {name}")

        if st.button("🗑️ Clear all documents"):
            st.session_state.store.reset()
            st.session_state.loaded_docs = {}
            st.session_state.chat_history = []
            st.session_state.citations = []
            st.session_state.summary = ""
            st.rerun()

    # ── 2. Summarize ──────────────────────────────────────────────────────────
    if doc_loaded:
        st.divider()
        st.header("2. Summarize")
        doc_names = list(st.session_state.loaded_docs.keys())
        selected_doc = st.selectbox("Select document", doc_names)

        if st.button("Generate Summary"):
            st.session_state.summary = ""
            with st.spinner("Summarizing..."):
                st.session_state.summary = "".join(
                    summarize(st.session_state.loaded_docs[selected_doc])
                )

        if st.session_state.summary:
            st.markdown(st.session_state.summary)

    # ── 3. Export ─────────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.divider()
        st.header("3. Export")
        pdf_bytes = export_chat_to_pdf(
            st.session_state.chat_history,
            st.session_state.citations,
            list(st.session_state.loaded_docs.keys()),
        )
        st.download_button(
            label="⬇️ Download Q&A as PDF",
            data=pdf_bytes,
            file_name="qa_report.pdf",
            mime="application/pdf",
        )

# ── Main area: Q&A ────────────────────────────────────────────────────────────
if not doc_loaded:
    st.info("👈 Upload one or more PDFs in the sidebar to get started.")
else:
    st.header("Ask Questions")

    # Render full chat history with citations
    for i, entry in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            if i < len(st.session_state.citations):
                render_citations(st.session_state.citations[i])

    # New question input
    question = st.chat_input("Ask anything about the documents...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            relevant_chunks = st.session_state.store.search(question)
            # st.write_stream handles the generator and returns the full answer string
            full_answer = st.write_stream(
                stream_answer(
                    relevant_chunks,
                    question,
                    chat_history=st.session_state.chat_history,
                )
            )
            render_citations(relevant_chunks)

        st.session_state.chat_history.append(
            {"question": question, "answer": full_answer}
        )
        st.session_state.citations.append(relevant_chunks)
