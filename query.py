"""
query.py — Milestone 5 (part 1): Grounded Generation
=====================================================

This module connects retrieval (Milestone 4) to an LLM to produce answers that
are GROUNDED in our own documents — meaning the model is instructed to use only
the retrieved chunks, not its general training knowledge.

Pipeline position:

    ... Retrieval (retriever.py: retrieve())
      -> Generation (Groq llama-3.3-70b-versatile, grounded prompt)   <-- THIS FILE
      -> Query Interface (Gradio, app.py)

Two design choices keep this honest:
  1. A strict system prompt that tells the model to answer ONLY from the context
     and to say a fixed "I don't have enough information" sentence otherwise.
  2. Source attribution is built PROGRAMMATICALLY from the retrieved chunks'
     metadata — we never trust the LLM to invent or remember citations.

Run a quick CLI test with:

    python query.py
"""

import os

# python-dotenv reads key=value pairs from a local .env file into os.environ.
from dotenv import load_dotenv

# The Groq SDK is OpenAI-compatible and gives us a free-tier hosted LLM.
from groq import Groq

# Reuse the retrieval function we already built and tested in Milestone 4.
# NOTE: importing retriever.py runs its module-level setup (it loads the
# embedding model and (re)builds the ChromaDB collection once). That's expected.
from retriever import retrieve


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Load environment variables from .env (must contain GROQ_API_KEY=...).
load_dotenv()

# The Groq model to use. llama-3.3-70b-versatile is free-tier and capable.
GROQ_MODEL = "llama-3.3-70b-versatile"

# How many chunks to retrieve and feed to the model as context.
TOP_K = 5

# The exact sentence the model must use when the context is insufficient.
# Keeping it as a constant lets the UI or tests check for it reliably.
INSUFFICIENT_INFO_MESSAGE = (
    "I don’t have enough information in the provided documents to answer that."
)

# The system prompt is where grounding is ENFORCED, not merely suggested.
# It tells the model: only use the context, no outside knowledge, and gives it
# an explicit escape hatch (the fixed sentence above) when the context falls short.
SYSTEM_PROMPT = (
    "You are a grounded RAG assistant. Answer only using the provided context. "
    "Do not use outside knowledge. If the context does not contain enough "
    f"information, say exactly: {INSUFFICIENT_INFO_MESSAGE}"
)

# Create the Groq client once. It reads the API key we loaded from .env.
# If the key is missing, we raise a clear, friendly error instead of a cryptic one.
_api_key = os.getenv("GROQ_API_KEY")
if not _api_key or _api_key == "your_key_here":
    raise RuntimeError(
        "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
        "from https://console.groq.com"
    )
client = Groq(api_key=_api_key)


# ---------------------------------------------------------------------------
# Helper: turn retrieved chunks into a single context block
# ---------------------------------------------------------------------------
def build_context(chunks):
    """
    Format the retrieved chunks into one labeled text block for the prompt.

    Each chunk is wrapped with its source filename and chunk index so the model
    can see (and optionally mention) where each piece of information came from.
    """
    blocks = []
    for chunk in chunks:
        # A clear header per chunk keeps sources visually separated for the model.
        header = f"[Source: {chunk['source']} | chunk {chunk['chunk_index']}]"
        blocks.append(f"{header}\n{chunk['text']}")

    # Separate chunks with a blank line so they don't blur together.
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Helper: unique source filenames, preserving the retrieval order
# ---------------------------------------------------------------------------
def unique_sources(chunks):
    """
    Return the list of source filenames from the chunks with duplicates removed,
    keeping the order in which they first appeared (most relevant first).

    We build citations from this metadata — NOT from the LLM's text — so the
    sources are always real, traceable files.
    """
    seen = set()
    ordered = []
    for chunk in chunks:
        source = chunk["source"]
        if source not in seen:
            seen.add(source)
            ordered.append(source)
    return ordered


# ---------------------------------------------------------------------------
# The main entry point: ask a grounded question
# ---------------------------------------------------------------------------
def ask(question):
    """
    Answer `question` using only our retrieved documents.

    Returns a dictionary:
        {
            "answer": "<the model's grounded answer>",
            "sources": ["09_reddit_meal_plan_or_none.txt", ...],  # de-duped, ordered
            "retrieved_chunks": [ {text, source, chunk_index, distance}, ... ],
        }
    """
    # 1. Retrieve the most relevant chunks for this question.
    chunks = retrieve(question, top_k=TOP_K)

    # 2. Build the context block the model is allowed to use.
    context = build_context(chunks)

    # 3. The user message pairs the context with the question and reminds the
    #    model (again) to stay within the context.
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    # 4. Call the LLM. temperature=0 keeps answers focused and repeatable, which
    #    suits a grounded, factual assistant.
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    # 5. Pull the answer text out of the response object.
    answer = response.choices[0].message.content.strip()

    # 6. Build the source list PROGRAMMATICALLY from metadata (not from the LLM).
    sources = unique_sources(chunks)

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": chunks,
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
def main():
    """
    Test grounded generation on two questions:
      1. An IN-SCOPE question our documents cover (meal plans).
      2. An OUT-OF-SCOPE question our documents do NOT cover (parking garages),
         which should trigger the fixed "not enough information" response.
    """
    test_questions = [
        "What do students say about whether UF freshman meal plans are worth it?",
        "What is the best parking garage at UF?",
    ]

    for question in test_questions:
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        result = ask(question)

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES (from retrieved chunk metadata):")
        for source in result["sources"]:
            print(f"  • {source}")

        print("\nRETRIEVED CHUNKS:")
        for rank, chunk in enumerate(result["retrieved_chunks"], start=1):
            distance = round(chunk["distance"], 3)
            preview = chunk["text"][:120].replace("\n", " ")
            print(f"  [Rank {rank}] {chunk['source']} "
                  f"(chunk {chunk['chunk_index']}, distance {distance})")
            print(f"      {preview}...")


if __name__ == "__main__":
    main()
