"""
claude_client.py — Claude API integration with streaming and chat memory

Features:
- Streaming responses (token by token)
- Chat memory (Claude remembers previous Q&A pairs)
- Citation-aware context (chunks are labeled with their source document)
"""

from typing import Iterator
import os
import unicodedata
import anthropic
from dotenv import load_dotenv
from rag import Chunk


def _safe_text(text: str) -> str:
    """Normalize Unicode so the HTTP layer never hits an ASCII encode error."""
    return unicodedata.normalize("NFKC", text)

load_dotenv()  # loads .env from project root (or parent dirs)

_api_key = os.getenv("ANTHROPIC_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY is not set. "
        "Run: export ANTHROPIC_API_KEY=your-key-here"
    )

client = anthropic.Anthropic(api_key=_api_key)


MODELS = {
    "Sonnet 4.6 — Balanced": "claude-sonnet-4-6",
    "Haiku 4.5 — Fast": "claude-haiku-4-5-20251001",
    "Opus 4.6 — Most Powerful": "claude-opus-4-6",
}

def stream_answer(
    context_chunks: list[Chunk],
    question: str,
    chat_history: list[dict] | None = None,
    max_history_pairs: int = 6,
    lang: str = "English",
    model: str = "claude-sonnet-4-6",
) -> Iterator[str]:
    """
    Stream Claude's answer for a question given relevant document chunks.

    Args:
        context_chunks:    The most relevant chunks retrieved from the vector store.
        question:          The user's current question.
        chat_history:      Previous Q&A pairs for memory (each is {"question": ..., "answer": ...}).
        max_history_pairs: How many past exchanges to include (prevents context overflow).
    """
    # Label each chunk with its source so Claude can reference them
    context_parts = []
    for chunk in context_chunks:
        context_parts.append(f"[Source: {chunk.source}]\n{_safe_text(chunk.text)}")
    context = "\n\n---\n\n".join(context_parts)

    # System prompt holds the document context — stays constant across turns
    system_prompt = f"""You are a helpful research assistant.
Answer questions using ONLY the context provided below.
If the answer is not in the context, say "I couldn't find that in the document."
Be concise and cite the source document name when relevant.
You MUST respond in {lang}.

Document Context:
{context}"""

    # Build message list: inject chat history for memory, then add current question
    messages = []
    if chat_history:
        for pair in chat_history[-max_history_pairs:]:
            messages.append({"role": "user", "content": pair["question"]})
            messages.append({"role": "assistant", "content": pair["answer"]})
    messages.append({"role": "user", "content": question})

    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def summarize(text: str, lang: str = "English", model: str = "claude-sonnet-4-6") -> Iterator[str]:
    """Summarize a document. Streams the summary token by token."""
    prompt = f"""Provide a concise summary of this document. Include:
- Main topic
- Key findings or arguments
- Methodology (if any)
- Conclusions

You MUST respond in {lang}.

Document:
{_safe_text(text[:12000])}

Summary:"""

    with client.messages.stream(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text
