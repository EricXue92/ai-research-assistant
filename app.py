"""
app.py — AI Research Assistant (Streamlit)

Features:
1. Multi-document support — upload and search across multiple PDFs
2. Citation highlighting — see which source chunks each answer used
3. Chat memory — Claude remembers previous Q&A in the conversation
4. Export — download the full Q&A session as a PDF report
5. Language selector — English or Chinese UI and responses
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import streamlit as st
from rag import extract_text_from_pdf, chunk_document, VectorStore, Chunk
from claude_client import stream_answer, summarize
from export import export_chat_to_pdf

# ── Translations ──────────────────────────────────────────────────────────────
T = {
    "English": {
        "title": "AI Research Assistant",
        "caption": "Upload research papers and ask questions — powered by Claude",
        "upload_header": "1. Upload Documents",
        "upload_label": "Choose one or more PDFs",
        "process_btn": "Process Documents",
        "already_loaded": "already loaded.",
        "processing": "Processing",
        "extract_error": "Could not extract text from",
        "indexed": "chunks indexed.",
        "loaded_docs": "Loaded documents:",
        "clear_btn": "Clear all documents",
        "summarize_header": "2. Summarize",
        "select_doc": "Select document",
        "summary_btn": "Generate Summary",
        "summarizing": "Summarizing...",
        "export_header": "3. Export",
        "export_btn": "Download Q&A as PDF",
        "ask_header": "Ask Questions",
        "chat_placeholder": "Ask anything about the documents...",
        "upload_prompt": "Upload one or more PDFs in the sidebar to get started.",
        "sources_label": "Sources",
        "chunks_used": "chunks used",
        "chunk_label": "chunk",
        "lang_name": "English",
    },
    "中文": {
        "title": "AI 论文助手",
        "caption": "上传研究论文并提问 — 由 Claude 驱动",
        "upload_header": "1. 上传文件",
        "upload_label": "选择一个或多个 PDF",
        "process_btn": "处理文件",
        "already_loaded": "已加载，跳过。",
        "processing": "正在处理",
        "extract_error": "无法提取文本：",
        "indexed": "个片段已索引。",
        "loaded_docs": "已加载文件：",
        "clear_btn": "清除所有文件",
        "summarize_header": "2. 总结",
        "select_doc": "选择文件",
        "summary_btn": "生成摘要",
        "summarizing": "正在总结...",
        "export_header": "3. 导出",
        "export_btn": "下载问答报告 (PDF)",
        "ask_header": "提问",
        "chat_placeholder": "对文档提任何问题...",
        "upload_prompt": "请在左侧上传 PDF 文件开始使用。",
        "sources_label": "来源",
        "chunks_used": "个片段",
        "chunk_label": "片段",
        "lang_name": "Chinese",
    },
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Research Assistant", page_icon="📄", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store = VectorStore()
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "citations" not in st.session_state:
    st.session_state.citations = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "lang" not in st.session_state:
    st.session_state.lang = "English"

# Shorthand for current translations
t = T[st.session_state.lang]
doc_loaded = bool(st.session_state.loaded_docs)


def render_citations(chunks: list[Chunk]):
    if not chunks:
        return
    with st.expander(f"📎 {t['sources_label']} ({len(chunks)} {t['chunks_used']})", expanded=False):
        for chunk in chunks:
            st.markdown(f"**{chunk.source}** — {t['chunk_label']} {chunk.chunk_index + 1}")
            st.caption(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Language selector
    selected_lang = st.radio("🌐 Language / 语言", ["English", "中文"], horizontal=True,
                             index=0 if st.session_state.lang == "English" else 1)
    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.session_state.summary = ""  # reset summary when language changes
        st.rerun()

    st.divider()

    # ── 1. Upload ─────────────────────────────────────────────────────────────
    st.header(t["upload_header"])
    uploaded_files = st.file_uploader(
        t["upload_label"],
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files and st.button(t["process_btn"], type="primary"):
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.loaded_docs:
                st.warning(f"'{uploaded_file.name}' {t['already_loaded']}")
                continue
            with st.spinner(f"{t['processing']} '{uploaded_file.name}'..."):
                text = extract_text_from_pdf(uploaded_file.getvalue())
                if not text.strip():
                    st.error(f"{t['extract_error']} '{uploaded_file.name}'.")
                    continue
                chunks = chunk_document(text, source=uploaded_file.name)
                st.session_state.store.add_document(chunks)
                st.session_state.loaded_docs[uploaded_file.name] = text
                st.success(f"✅ '{uploaded_file.name}' — {len(chunks)} {t['indexed']}")

    if st.session_state.loaded_docs:
        st.markdown(f"**{t['loaded_docs']}**")
        for name in st.session_state.loaded_docs:
            st.markdown(f"- {name}")

        if st.button(f"🗑️ {t['clear_btn']}"):
            st.session_state.store.reset()
            st.session_state.loaded_docs = {}
            st.session_state.chat_history = []
            st.session_state.citations = []
            st.session_state.summary = ""
            st.rerun()

    # ── 2. Summarize ──────────────────────────────────────────────────────────
    if doc_loaded:
        st.divider()
        st.header(t["summarize_header"])
        doc_names = list(st.session_state.loaded_docs.keys())
        selected_doc = st.selectbox(t["select_doc"], doc_names)

        if st.button(t["summary_btn"]):
            st.session_state.summary = ""
            with st.spinner(t["summarizing"]):
                st.session_state.summary = "".join(
                    summarize(
                        st.session_state.loaded_docs[selected_doc],
                        lang=t["lang_name"],
                    )
                )

        if st.session_state.summary:
            st.markdown(st.session_state.summary)

    # ── 3. Export ─────────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.divider()
        st.header(t["export_header"])
        pdf_bytes = export_chat_to_pdf(
            st.session_state.chat_history,
            st.session_state.citations,
            list(st.session_state.loaded_docs.keys()),
        )
        st.download_button(
            label=f"⬇️ {t['export_btn']}",
            data=pdf_bytes,
            file_name="qa_report.pdf",
            mime="application/pdf",
        )

# ── Title (changes with language) ─────────────────────────────────────────────
st.title(f"📄 {t['title']}")
st.caption(t["caption"])

# ── Main area: Q&A ────────────────────────────────────────────────────────────
if not doc_loaded:
    st.info(f"👈 {t['upload_prompt']}")
else:
    st.header(t["ask_header"])

    for i, entry in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            if i < len(st.session_state.citations):
                render_citations(st.session_state.citations[i])

    question = st.chat_input(t["chat_placeholder"])

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            relevant_chunks = st.session_state.store.search(question)
            full_answer = st.write_stream(
                stream_answer(
                    relevant_chunks,
                    question,
                    chat_history=st.session_state.chat_history,
                    lang=t["lang_name"],
                )
            )
            render_citations(relevant_chunks)

        st.session_state.chat_history.append(
            {"question": question, "answer": full_answer}
        )
        st.session_state.citations.append(relevant_chunks)
