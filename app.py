"""
app.py — Milestone 5 (part 2): Gradio Query Interface
======================================================

A simple web UI for our grounded RAG assistant. The user types a question,
clicks "Ask" (or presses Enter), and sees:

    - the grounded answer
    - the source documents the answer drew from
    - the raw retrieved chunks (source, chunk index, distance, preview)

This file does NOT contain any RAG logic itself — it just calls ask() from
query.py and formats the result for display. Keeping the logic in query.py
means the CLI test and the web UI share the exact same pipeline.

Run it with:

    python app.py

Then open the local URL it prints (usually http://localhost:7860).
"""

import gradio as gr

# Reuse the end-to-end grounded-answer function from Milestone 5 part 1.
from query import ask


# ---------------------------------------------------------------------------
# Helper: format the retrieved chunks for human-readable display
# ---------------------------------------------------------------------------
def format_chunks(chunks):
    """
    Turn the list of retrieved-chunk dicts into a readable multi-line string
    showing each chunk's source, index, distance, and a short text preview.
    """
    lines = []
    for rank, chunk in enumerate(chunks, start=1):
        distance = round(chunk["distance"], 3)
        # A short preview keeps the panel readable; newlines flattened to spaces.
        preview = chunk["text"][:200].replace("\n", " ")
        lines.append(
            f"[Rank {rank}] source={chunk['source']} | "
            f"chunk_index={chunk['chunk_index']} | distance={distance}\n"
            f"    {preview}..."
        )
    # Blank line between chunks for readability.
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# The function Gradio calls when the user asks a question
# ---------------------------------------------------------------------------
def handle_query(question):
    """
    Run the question through the RAG pipeline and return three strings, one for
    each output box: (answer, sources, retrieved chunks).
    """
    # Guard against empty input so we don't waste an API call.
    if not question or not question.strip():
        return "Please enter a question.", "", ""

    result = ask(question)

    # The answer comes straight from the grounded model response.
    answer = result["answer"]

    # Sources are the de-duplicated source filenames from chunk metadata.
    sources = "\n".join(f"• {source}" for source in result["sources"])

    # The retrieved chunks panel shows the evidence behind the answer.
    chunks = format_chunks(result["retrieved_chunks"])

    return answer, sources, chunks


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------
with gr.Blocks(title="UF Dining — Unofficial Guide") as demo:
    gr.Markdown(
        "# 🐊 UF Dining — Unofficial Guide\n"
        "Ask a question about eating at the University of Florida. Answers are "
        "grounded **only** in the collected documents (official dining pages and "
        "student Reddit threads). If the documents don't cover your question, the "
        "assistant will say so instead of guessing."
    )

    # Input row: the question textbox and the Ask button.
    question_box = gr.Textbox(
        label="Your question",
        placeholder="e.g. Are UF freshman meal plans worth it?",
        lines=2,
    )
    ask_button = gr.Button("Ask", variant="primary")

    # Output boxes.
    answer_box = gr.Textbox(label="Answer", lines=8)
    sources_box = gr.Textbox(label="Sources (from retrieved documents)", lines=4)
    chunks_box = gr.Textbox(label="Retrieved chunks (evidence)", lines=12)

    outputs = [answer_box, sources_box, chunks_box]

    # Wire up BOTH the button click and pressing Enter in the textbox.
    ask_button.click(handle_query, inputs=question_box, outputs=outputs)
    question_box.submit(handle_query, inputs=question_box, outputs=outputs)


# Launch the web server when run directly.
if __name__ == "__main__":
    demo.launch()
