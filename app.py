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
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import streamlit as st
from rag import extract_text_from_pdf, chunk_document, VectorStore, Chunk
from claude_client import stream_answer, summarize, MODELS
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
st.set_page_config(page_title="AI Research Assistant", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }
/* Hide sidebar collapse/expand buttons */
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
    pointer-events: none !important;
}
/* Hide the entire sidebar header row (contains only the collapse button) */
[data-testid="stSidebarHeader"] {
    display: none !important;
}
/* Catch newer Streamlit versions' collapse button */
button[kind="header"] { display: none !important; }
[data-testid="stSidebarNav"] > button { display: none !important; }
section[data-testid="stSidebar"] > div:first-child > button { display: none !important; }

/* ── Global typography ── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* ── Main content area ── */
.block-container {
    padding: 2rem 3rem 2rem 3rem;
    max-width: 900px;
}

/* ── Title ── */
h1 {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #1E293B !important;
    margin-bottom: 0.1rem !important;
}

/* ── Section headers ── */
h2, h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 1.5rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #F8FAFC;
    border-right: 1px solid #E2E8F0;
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 0.75rem !important;
    color: #94A3B8 !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    border: 1px solid #E2E8F0 !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: #2563EB !important;
    color: #2563EB !important;
    background: #EFF6FF !important;
}
/* Primary button */
.stButton > button[kind="primary"] {
    background: #2563EB !important;
    border-color: #2563EB !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
    color: white !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #CBD5E1 !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
    background: #FFFFFF !important;
    pointer-events: auto !important;
    cursor: pointer !important;
    position: relative !important;
    z-index: 1 !important;
}
[data-testid="stFileUploader"] * {
    pointer-events: auto !important;
}

/* ── Expander (citations) ── */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    background: #F8FAFC !important;
}

/* ── Info / success / warning boxes ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border: none !important;
}

/* ── Divider ── */
hr {
    border-color: #E2E8F0 !important;
    margin: 1rem 0 !important;
}

/* ── Caption text ── */
[data-testid="stCaptionContainer"] p {
    color: #94A3B8 !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

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
if "model_label" not in st.session_state:
    st.session_state.model_label = "Sonnet 4.6 — Balanced"

# ── Language selector (must come before computing t) ──────────────────────────
with st.sidebar:
    selected_lang = st.radio("🌐 Language / 语言", ["English", "中文"], horizontal=True,
                             index=0 if st.session_state.lang == "English" else 1)
    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.session_state.summary = ""

    st.session_state.model_label = st.selectbox(
        "🤖 Model",
        options=list(MODELS.keys()),
        index=list(MODELS.keys()).index(st.session_state.model_label),
        key="model_selector",
    )
    st.divider()

# Shorthand for current translations — computed after language is set
t = T[st.session_state.lang]
doc_loaded = bool(st.session_state.loaded_docs)


def render_citations(chunks: list[Chunk]):
    if not chunks:
        return
    with st.expander(f"📎 {t['sources_label']} ({len(chunks)} {t['chunks_used']})", expanded=False):
        for chunk in chunks:
            st.markdown(f"**{chunk.source}** — {t['chunk_label']} {chunk.chunk_index + 1}")
            st.caption(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))


# ── Sidebar (continued) ───────────────────────────────────────────────────────
with st.sidebar:

    # ── 1. Upload ─────────────────────────────────────────────────────────────
    st.header(t["upload_header"])
    uploaded_files = st.file_uploader(
        t["upload_label"],
        type="pdf",
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files and st.button(t["process_btn"], type="primary", key="process_btn"):
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.loaded_docs:
                st.warning(f"'{uploaded_file.name}' {t['already_loaded']}")
                continue
            if len(uploaded_file.getvalue()) > 50 * 1024 * 1024:
                st.error(f"'{uploaded_file.name}' exceeds 50MB limit.")
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

        if st.button(f"🗑️ {t['clear_btn']}", key="clear_btn"):
            st.session_state.store.reset()
            st.session_state.loaded_docs = {}
            st.session_state.chat_history = []
            st.session_state.citations = []
            st.session_state.summary = ""
            st.rerun()

    # ── 2. Summarize ──────────────────────────────────────────────────────────
    if st.session_state.loaded_docs:
        st.divider()
        st.header(t["summarize_header"])
        doc_names = list(st.session_state.loaded_docs.keys())
        selected_doc = st.selectbox(t["select_doc"], doc_names, key="select_doc")

        if st.button(t["summary_btn"], key="summary_btn"):
            st.session_state.summary = ""
            with st.spinner(t["summarizing"]):
                st.session_state.summary = "".join(
                    summarize(
                        st.session_state.loaded_docs[selected_doc],
                        lang=t["lang_name"],
                        model=MODELS[st.session_state.model_label],
                    )
                )

        if st.session_state.summary:
            st.markdown(st.session_state.summary)

    # ── 3. Export ─────────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.divider()
        st.header(t["export_header"])
        try:
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
        except Exception as e:
            st.warning(f"PDF export unavailable: {e}")

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
            gen = stream_answer(
                relevant_chunks,
                question,
                chat_history=st.session_state.chat_history,
                lang=t["lang_name"],
                model=MODELS[st.session_state.model_label],
            )
            try:
                full_answer = str(st.write_stream(gen))
            finally:
                gen.close()
            render_citations(relevant_chunks)

        st.session_state.chat_history.append(
            {"question": question, "answer": full_answer}
        )
        st.session_state.citations.append(relevant_chunks)
