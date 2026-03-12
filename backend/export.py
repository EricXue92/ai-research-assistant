"""
export.py — Generate a PDF report of the Q&A session using fpdf2
"""

from datetime import datetime
from fpdf import FPDF
from rag import Chunk


def _safe(text: str) -> str:
    """Replace common Unicode chars with ASCII equivalents for fpdf2 core fonts."""
    return (
        text
        .replace("\u2014", "-")    # em dash
        .replace("\u2013", "-")    # en dash
        .replace("\u2018", "'")    # left single quote
        .replace("\u2019", "'")    # right single quote
        .replace("\u201c", '"')    # left double quote
        .replace("\u201d", '"')    # right double quote
        .replace("\u2026", "...")  # ellipsis
        .replace("\u2022", "*")    # bullet
        .replace("\u00a0", " ")    # non-breaking space
        .replace("\u00b7", ".")    # middle dot
        .encode("latin-1", errors="replace")
        .decode("latin-1")
    )


def export_chat_to_pdf(
    chat_history: list[dict],
    citations: list[list[Chunk]],
    doc_names: list[str],
) -> bytes:
    """
    Generate a downloadable PDF report of the Q&A session.

    Args:
        chat_history: List of {"question": ..., "answer": ...} dicts.
        citations:    Parallel list of chunks used for each answer.
        doc_names:    Names of all loaded documents.

    Returns:
        PDF file as bytes.
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Title ─────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "AI Research Assistant - Q&A Report", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, _safe(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── Documents analyzed ────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Documents Analyzed:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for name in doc_names:
        pdf.cell(0, 6, _safe(f"  • {name}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── Q&A pairs ─────────────────────────────────────────────────────────────
    for i, entry in enumerate(chat_history):
        # Question
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 7, _safe(f"Q{i + 1}: {entry['question']}"))

        # Answer
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _safe(entry["answer"]))

        # Citations
        if i < len(citations) and citations[i]:
            pdf.set_font("Helvetica", "I", 9)
            sources = list({c.source for c in citations[i]})  # unique source names
            pdf.multi_cell(0, 5, _safe(f"Sources: {', '.join(sources)}"))

        pdf.ln(5)

    return bytes(pdf.output())
