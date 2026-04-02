"""
claude_client.py — Claude API integration with streaming and chat memory

Features:
- Streaming responses (token by token via callback)
- Chat memory (Claude remembers previous Q&A pairs)
- Citation-aware context (chunks are labeled with their source document)
"""

from typing import Callable
import os
import unicodedata
import anthropic
from dotenv import load_dotenv
from rag import Chunk


def _safe_text(text: str) -> str:
    """Normalize Unicode so the HTTP layer never hits an ASCII encode error."""
    return unicodedata.normalize("NFKC", text)

load_dotenv()

# Client is initialized lazily so a missing key never crashes the app at import time.
# The SDK reads ANTHROPIC_API_KEY from the environment automatically.
_client: anthropic.Anthropic | None = None

def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


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
    on_token: Callable[[str], object] | None = None,
) -> str:
    """Stream Claude's answer. Calls on_token(full_text_so_far) per token. Returns full answer."""
    context_parts = []
    for chunk in context_chunks:
        context_parts.append(f"[Source: {chunk.source}]\n{_safe_text(chunk.text)}")
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = f"""You are a helpful research assistant.
Answer questions using ONLY the context provided below.
If the answer is not in the context, say "I couldn't find that in the document."
Be concise and cite the source document name when relevant.
You MUST respond in {lang}.

Document Context:
{context}"""

    messages = []
    if chat_history:
        for pair in chat_history[-max_history_pairs:]:
            messages.append({"role": "user", "content": pair["question"]})
            messages.append({"role": "assistant", "content": pair["answer"]})
    messages.append({"role": "user", "content": question})

    full_answer = ""
    with _get_client().messages.stream(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for token in stream.text_stream:
            full_answer += token
            if on_token:
                on_token(full_answer)
    return full_answer


def summarize(text: str, lang: str = "English", model: str = "claude-sonnet-4-6") -> str:
    """Summarize a document. Returns the full summary."""
    prompt = f"""Provide a concise summary of this document. Include:
- Main topic
- Key findings or arguments
- Methodology (if any)
- Conclusions

You MUST respond in {lang}.

Document:
{_safe_text(text[:12000])}

Summary:"""

    response = _get_client().messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
