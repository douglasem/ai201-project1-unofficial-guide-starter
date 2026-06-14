"""
ingest.py — Milestone 3: Document Ingestion and Chunking
=========================================================

This script handles the first half of the RAG pipeline for the
"Unofficial Guide to Eating at UF" project:

    Raw .txt documents  ->  cleaned text  ->  overlapping chunks (with metadata)

It does NOT do embeddings, vector storage, or generation yet. Those come in
later milestones. The functions here (especially `chunk_text`) are written to
be reusable so we can import them later instead of rewriting them.

Run it from the VS Code terminal with:

    python ingest.py
"""

# We use Python's built-in `pathlib` to work with file paths in a clean,
# cross-platform way (works the same on Mac, Windows, and Linux).
from pathlib import Path

# `re` is Python's regular-expression module. We use it to normalize whitespace.
import re


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Keeping these "settings" at the top makes them easy to find and tweak later.

# Folder that holds our raw .txt documents (relative to this script).
DOCUMENTS_DIR = Path(__file__).parent / "documents"

# Chunking settings, straight from our planning.md strategy.
CHUNK_SIZE = 500     # target characters per chunk
CHUNK_OVERLAP = 100  # characters shared between neighboring chunks


# ---------------------------------------------------------------------------
# Step 1: Load documents
# ---------------------------------------------------------------------------
def load_documents(documents_dir):
    """
    Read every .txt file in `documents_dir` and return a list of dictionaries.

    Each dictionary looks like:
        {"source": "01_uf_dining_who_we_are.txt", "text": "<raw file contents>"}

    We return the filename as `source` so we can track where each chunk came
    from later (this becomes our metadata).
    """
    documents = []

    # `sorted(...)` so the files are processed in a predictable order
    # (01, 02, 03, ...) every time we run the script.
    for file_path in sorted(documents_dir.glob("*.txt")):
        # Open the file using UTF-8 so special characters (curly quotes,
        # emoji from Reddit posts, etc.) load correctly.
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,  # just the filename, e.g. "05_uf_dining_faq.txt"
            "text": text,
        })

    return documents


# ---------------------------------------------------------------------------
# Step 2: Clean text
# ---------------------------------------------------------------------------
def clean_text(text):
    """
    Normalize whitespace and remove empty lines from a block of text.

    Why we do this:
      - Raw files have inconsistent spacing, tabs, and lots of blank lines.
      - Messy whitespace wastes characters inside our fixed-size chunks and
        can hurt the quality of embeddings later.

    The result is a single, tidy block of text where lines are separated by a
    single newline and there are no blank lines.
    """
    cleaned_lines = []

    # Go through the file one line at a time.
    for line in text.splitlines():
        # Replace any run of whitespace (spaces, tabs, etc.) within the line
        # with a single space, then trim the ends.
        line = re.sub(r"\s+", " ", line).strip()

        # Keep the line only if it still has content (this drops empty lines).
        if line:
            cleaned_lines.append(line)

    # Re-join the surviving lines with single newlines.
    return "\n".join(cleaned_lines)


# ---------------------------------------------------------------------------
# Step 3: Chunk text (the reusable core function)
# ---------------------------------------------------------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split `text` into overlapping chunks of roughly `chunk_size` characters.

    Parameters:
        text       : the cleaned text to split
        chunk_size : target number of characters per chunk (default 500)
        overlap    : number of characters each chunk shares with the previous
                     one (default 100)

    How the sliding window works:
        - Take characters [0 : 500]            -> chunk 1
        - Move the start forward by (500 - 100 = 400) characters
        - Take characters [400 : 900]          -> chunk 2  (overlaps chunk 1)
        - ...and so on until we reach the end of the text.

    The overlap means an idea that lands near a boundary (like a meal-plan
    recommendation and the reason behind it) is less likely to be cut in half.

    Returns a list of chunk strings.
    """
    # Defensive check: overlap must be smaller than chunk_size, otherwise the
    # window would never move forward and we'd loop forever.
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    # How far we advance the window each step.
    step = chunk_size - overlap

    # Keep slicing until the start position passes the end of the text.
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        # Only keep non-empty chunks (the final slice can be blank/whitespace).
        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


# ---------------------------------------------------------------------------
# Step 4: Turn documents into chunks WITH metadata
# ---------------------------------------------------------------------------
def build_chunks(documents):
    """
    Take the list of loaded documents and produce a flat list of chunk
    dictionaries, each carrying its own metadata.

    Each chunk dictionary looks like:
        {
            "text": "<the chunk text>",
            "source": "07_reddit_food_tier_list.txt",
            "chunk_index": 3,   # position of this chunk within its source file
        }

    `chunk_index` resets to 0 for each new document, so it tells us "this is
    the Nth chunk of that specific file."
    """
    all_chunks = []

    for document in documents:
        # Clean first, then chunk the cleaned text.
        cleaned = clean_text(document["text"])
        chunks = chunk_text(cleaned)

        # `enumerate` gives us both the index (0, 1, 2, ...) and the chunk.
        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": document["source"],
                "chunk_index": index,
            })

    return all_chunks


# ---------------------------------------------------------------------------
# Step 5: Run everything and print a summary
# ---------------------------------------------------------------------------
def main():
    """Load, clean, chunk, and print a human-readable summary."""
    # 1. Load the raw documents.
    documents = load_documents(DOCUMENTS_DIR)

    # 2 & 3. Clean and chunk them (with metadata).
    chunks = build_chunks(documents)

    # 4. Print the headline numbers.
    print("=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Documents loaded : {len(documents)}")
    print(f"Total chunks     : {len(chunks)}")
    print("=" * 60)

    # 5. Print 5 representative sample chunks spread across the corpus.
    #    Instead of just the first 5 (which would all come from one file),
    #    we space the samples out evenly so we see a variety of sources.
    print("\n5 SAMPLE CHUNKS")
    print("-" * 60)

    num_samples = min(5, len(chunks))
    if num_samples > 0:
        # `step` evenly spaces the sample indices across all chunks.
        step = max(1, len(chunks) // num_samples)

        for i in range(num_samples):
            sample = chunks[i * step]
            print(f"\nSample {i + 1}")
            print(f"  source       : {sample['source']}")
            print(f"  chunk_index  : {sample['chunk_index']}")
            print(f"  length       : {len(sample['text'])} chars")
            # Show a short preview so the terminal isn't flooded.
            preview = sample["text"][:200]
            print(f"  text preview : {preview}...")

    print("\n" + "-" * 60)
    print("Done. The chunk_text() and build_chunks() functions are ready")
    print("to be imported by later milestones (embeddings + vector store).")


# This standard Python idiom means: only run main() when we execute the file
# directly (python ingest.py). If a later milestone does `import ingest`, this
# block will NOT run automatically — so importing chunk_text() stays clean.
if __name__ == "__main__":
    main()
