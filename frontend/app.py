"""
app.py — Streamlit frontend

A clean UI that:
1. Lets the user upload a PDF
2. Shows a streaming summary
3. Lets the user ask questions and see answers stream in real time
"""

import httpx
import streamlit as st

API_URL = "http://localhost:8000"

# Bypass system proxy for localhost — prevents SOCKS proxy errors on macOS
NO_PROXY_CLIENT = httpx.Client(trust_env=False, timeout=60)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Research Assistant", page_icon="📄", layout="wide")
st.title("📄 AI Research Assistant")
st.caption("Upload a research paper and ask questions about it — powered by Claude")

# ── Session state (persists across re-runs) ───────────────────────────────────
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"question": ..., "answer": ...}
if "summary" not in st.session_state:
    st.session_state.summary = ""

# ── Sidebar: upload ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner("Extracting text and building vector index..."):
            try:
                response = NO_PROXY_CLIENT.post(
                    f"{API_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(data["message"])
                    st.session_state.doc_loaded = True
                    st.session_state.chat_history = []  # reset chat for new doc
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            except httpx.ConnectError:
                st.error("Cannot connect to backend. Make sure it's running on port 8000.")

    if st.session_state.doc_loaded:
        st.divider()
        st.header("2. Summarize")
        if st.button("Generate Summary"):
            st.session_state.summary = ""
            with st.spinner("Summarizing..."):
                full_summary = ""
                try:
                    with NO_PROXY_CLIENT.stream("POST", f"{API_URL}/summarize") as r:
                        for chunk in r.iter_text():
                            full_summary += chunk
                except httpx.RemoteProtocolError:
                    pass
                except Exception as e:
                    st.error(f"Error: {e}")
                st.session_state.summary = full_summary

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

    # Chat input box (appears at the bottom like ChatGPT)
    question = st.chat_input("Ask anything about the document...")

    if question:
        # Show user's question immediately
        with st.chat_message("user"):
            st.write(question)

        # Stream Claude's answer
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            full_answer = ""
            try:
                with NO_PROXY_CLIENT.stream(
                    "POST",
                    f"{API_URL}/ask",
                    json={"text": question},
                ) as r:
                    for chunk in r.iter_text():
                        full_answer += chunk
                        answer_placeholder.markdown(full_answer + "▌")  # blinking cursor effect
                answer_placeholder.markdown(full_answer)  # remove cursor when done
            except httpx.RemoteProtocolError:
                answer_placeholder.markdown(full_answer)  # show whatever we received
            except Exception as e:
                st.error(f"Error: {e}")

        # Save to chat history
        st.session_state.chat_history.append(
            {"question": question, "answer": full_answer}
        )
