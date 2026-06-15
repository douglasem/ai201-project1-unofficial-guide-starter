"""
retriever.py — Milestone 4: Embedding and Retrieval
====================================================

This script takes the chunks produced by our Milestone 3 ingestion pipeline,
turns them into vector embeddings, stores them in a local ChromaDB collection,
and provides a reusable `retrieve()` function for semantic search.

Pipeline position:

    ... Chunking (ingest.py)
      -> Embedding + Vector Store   <-- THIS FILE
         (sentence-transformers all-MiniLM-L6-v2 + ChromaDB)
      -> Retrieval (top-5 semantic search with source metadata)   <-- THIS FILE
      -> Generation (Groq)          <-- later milestone, NOT here

It does NOT call Groq, generate answers, or build a Gradio UI yet.

Run it from the VS Code terminal with:

    python retriever.py
"""

from pathlib import Path

# ChromaDB is our local vector database. It stores embeddings + metadata and
# does the nearest-neighbor search for us.
import chromadb

# SentenceTransformer turns text into a list of numbers (a vector/embedding)
# that captures its meaning, so similar text ends up "close" in vector space.
from sentence_transformers import SentenceTransformer

# Reuse the functions we already built and tested in Milestone 3.
# Because ingest.py guards its demo code behind `if __name__ == "__main__"`,
# importing these functions does NOT run ingest.py's printing logic.
from ingest import load_documents, build_chunks


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Folder with our raw .txt documents (relative to this script).
DOCUMENTS_DIR = Path(__file__).parent / "documents"

# Name of the embedding model. all-MiniLM-L6-v2 runs locally, needs no API key,
# and has no rate limits — a solid default for this project.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Where ChromaDB will persist its data on disk, so we don't have to re-embed
# every time. This folder is created automatically on first run.
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

# The name of our collection (think of it like a table) inside ChromaDB.
COLLECTION_NAME = "uf_dining_chunks"

# How many chunks to retrieve per query by default.
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Load the embedding model ONCE
# ---------------------------------------------------------------------------
# Loading a model is slow, so we do it a single time at import and reuse it
# everywhere (both for embedding the chunks and for embedding queries).
print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads it)...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------------------------
# Step 1: Build (or rebuild) the vector store
# ---------------------------------------------------------------------------
def build_vector_store():
    """
    Load documents, chunk them, embed every chunk, and store everything in a
    fresh ChromaDB collection.

    Returns the ready-to-query ChromaDB collection object.
    """
    # 1a. Get our chunks from the Milestone 3 pipeline.
    documents = load_documents(DOCUMENTS_DIR)
    chunks = build_chunks(documents)
    print(f"Loaded {len(documents)} documents and built {len(chunks)} chunks.")

    # 1b. Create a PersistentClient. Unlike an in-memory client, this saves the
    #     database to CHROMA_DB_DIR so it survives between runs.
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # 1c. RESET the collection so re-running this script gives a clean rebuild
    #     instead of piling duplicate chunks on top of old ones.
    #     delete_collection raises if it doesn't exist yet, so we guard it.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # Collection didn't exist yet — that's fine on the first run.

    # 1d. Create the collection. We tell ChromaDB to use cosine distance
    #     ("hnsw:space": "cosine"), which is the natural fit for sentence
    #     embeddings: distance ~0 means very similar, ~1 means unrelated.
    #     (ChromaDB's default is squared-L2, which is less intuitive to read.)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 1e. Prepare the four parallel lists ChromaDB's add() expects. Each list
    #     is indexed the same way: position i describes the same chunk.
    documents_text = [chunk["text"] for chunk in chunks]

    # Metadata travels with each vector and comes back on every query — this is
    # how we attribute answers to a source file later.
    metadatas = [
        {"source": chunk["source"], "chunk_index": chunk["chunk_index"]}
        for chunk in chunks
    ]

    # IDs must be unique strings. Combining source + chunk index guarantees
    # uniqueness and stays human-readable for debugging.
    ids = [f"{chunk['source']}::chunk_{chunk['chunk_index']}" for chunk in chunks]

    # 1f. Embed every chunk in one batch (much faster than one at a time).
    #     .tolist() converts the NumPy array into plain Python lists, which is
    #     the format ChromaDB wants.
    print("Embedding chunks (this can take a moment)...")
    embeddings = embedding_model.encode(
        documents_text,
        show_progress_bar=True,
    ).tolist()

    # 1g. Add everything to the collection. Because we pass our own embeddings,
    #     ChromaDB stores them directly instead of computing its own.
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents_text,
        metadatas=metadatas,
    )

    print(f"Stored {collection.count()} chunks in ChromaDB collection "
          f"'{COLLECTION_NAME}'.")
    return collection


# ---------------------------------------------------------------------------
# Module-level collection so retrieve() can be called repeatedly
# ---------------------------------------------------------------------------
# We build the store once when this module is imported/run, then reuse it.
collection = build_vector_store()


# ---------------------------------------------------------------------------
# Step 2: The reusable retrieval function
# ---------------------------------------------------------------------------
def retrieve(query, top_k=DEFAULT_TOP_K):
    """
    Find the `top_k` chunks most semantically similar to `query`.

    Returns a list of dictionaries, each shaped like:
        {
            "text": "<chunk text>",
            "source": "07_reddit_food_tier_list.txt",
            "chunk_index": 3,
            "distance": 0.21,   # lower = more similar (cosine distance)
        }
    """
    # 2a. Embed the query with the SAME model used for the chunks. Mixing models
    #     would make the vectors live in incompatible spaces and break search.
    query_embedding = embedding_model.encode(query).tolist()

    # 2b. Ask ChromaDB for the nearest neighbors. We request documents,
    #     metadatas, and distances back so we can build our result dicts.
    results = collection.query(
        query_embeddings=[query_embedding],  # a list because you CAN batch queries
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # 2c. IMPORTANT — ChromaDB's results are NESTED ONE LEVEL PER QUERY.
    #     Because query_embeddings could hold many queries, each key maps to a
    #     LIST OF LISTS:
    #         results["documents"]  -> [[doc_for_q0_rank0, doc_for_q0_rank1, ...]]
    #         results["distances"]  -> [[dist_for_q0_rank0, ...]]
    #     We sent exactly ONE query, so we take index [0] to get the inner list
    #     for our single query. (IDs are always returned, so no need to request
    #     them in `include`.)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # 2d. Zip the parallel inner lists together into clean result dictionaries.
    retrieved = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        retrieved.append({
            "text": text,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "distance": distance,
        })

    return retrieved


# ---------------------------------------------------------------------------
# Step 3: Test retrieval against real evaluation queries
# ---------------------------------------------------------------------------
def main():
    """Run a few evaluation-plan queries and print the retrieved chunks."""
    test_queries = [
        "What do students say about whether UF freshman meal plans are worth it?",
        "What are common student complaints about UF campus dining?",
        "What are some alternatives to relying completely on a meal plan?",
    ]

    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = retrieve(query, top_k=DEFAULT_TOP_K)

        # enumerate(..., start=1) gives us a 1-based rank for display.
        for rank, result in enumerate(results, start=1):
            # Round the distance to 3 decimals for readable output.
            distance = round(result["distance"], 3)
            preview = result["text"][:300].replace("\n", " ")

            print(f"\n[Rank {rank}] distance={distance} "
                  f"| source={result['source']} "
                  f"| chunk_index={result['chunk_index']}")
            print(f"    {preview}...")

    print("\n" + "-" * 70)
    print("Retrieval test complete. Lower cosine distance = more relevant.")
    print("Inspect the results above: are the chunks actually on-topic?")


# Only run the test queries when executed directly (python retriever.py).
# A later milestone can `from retriever import retrieve` without triggering this.
if __name__ == "__main__":
    main()
