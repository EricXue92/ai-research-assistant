"""
claude_client.py — Claude API integration with streaming

This file sends the retrieved context + user question to Claude,
and streams the response back token by token.
"""

import os
from typing import Iterator
import anthropic

# The Anthropic client reads ANTHROPIC_API_KEY from your environment automatically
client = anthropic.Anthropic()


def stream_answer(context_chunks: list[str], question: str) -> Iterator[str]:
    """
    Given relevant text chunks and a question, ask Claude to answer
    and yield the response word by word (streaming).

    Why streaming? So the UI shows text appearing in real time
    instead of freezing for 10 seconds then dumping everything at once.
    """
    # Join the retrieved chunks into one context block
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful research assistant.
Answer the question using ONLY the context provided below.
If the answer is not found in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""

    # client.messages.stream() opens a streaming connection to Claude
    # Instead of waiting for the full response, we get tokens one by one
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:  # yields one token at a time
            yield text


def summarize(text: str) -> Iterator[str]:
    """
    Summarize a full document (used for short documents that fit in context).
    Streams the summary back token by token.
    """
    prompt = f"""Please provide a concise summary of the following document.
Include: main topic, key findings or arguments, methodology (if any), and conclusions.

Document:
{text[:12000]}

Summary:"""

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text
