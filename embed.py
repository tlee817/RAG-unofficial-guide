"""
embed.py — Embed chunks with all-MiniLM-L6-v2 and store in ChromaDB.
           Also exposes a retrieve() function for query-time retrieval.

Pipeline position (from planning.md architecture):
  clean.py → ingest.py → embed.py (this file) → query.py

Usage:
  python embed.py          # build the vector store from scratch
  python embed.py --query "Is Eggert a hard professor?"   # test retrieval
"""

import argparse
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import ingest_all

# ── configuration ────────────────────────────────────────────────────────────

CHROMA_DIR  = "./chroma_db"        # where ChromaDB persists data on disk
COLLECTION  = "ucla_reviews"       # name of the collection inside ChromaDB
MODEL_NAME  = "all-MiniLM-L6-v2"  # 256-token max, 384-dim embeddings
TOP_K       = 5                    # chunks returned per query

# ── setup ────────────────────────────────────────────────────────────────────

def get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """
    Get-or-create the ChromaDB collection.

    chromadb.PersistentClient(path=...) opens (or creates) a local SQLite-backed
    store at that path.  get_or_create_collection() returns an existing collection
    by name, or makes a new empty one — so it's safe to call on repeat runs.
    """
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity, per planning.md
    )


# ── build ─────────────────────────────────────────────────────────────────────

def build_vector_store() -> None:
    """Embed all chunks and upsert them into ChromaDB."""
    print("Loading chunks from ingest pipeline…")
    chunks = ingest_all()
    print(f"  {len(chunks)} chunks ready\n")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = get_collection(client)

    # Check if already populated to avoid re-embedding unnecessarily.
    # collection.count() returns the number of items currently stored.
    existing = collection.count()
    if existing > 0:
        print(f"Collection already contains {existing} items.")
        answer = input("Re-embed and overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Skipping embed — using existing store.")
            return
        client.delete_collection(COLLECTION)
        collection = get_collection(client)

    texts      = [c["text"]        for c in chunks]
    ids        = [f"chunk_{i}"     for i in range(len(chunks))]
    metadatas  = [
        {
            "source":      c["source"],
            "professor":   c["professor"],
            "course":      c["course"],
            "chunk_index": c["chunk_index"],
            "chunk_mode":  c["chunk_mode"],
        }
        for c in chunks
    ]

    print(f"Embedding {len(texts)} chunks…")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_list=True)

    # collection.upsert() inserts new items or updates existing ones by id.
    # We pass:
    #   ids        — unique string identifier per chunk (required by ChromaDB)
    #   embeddings — the pre-computed vectors (list of lists of floats)
    #   documents  — the raw text, stored alongside so we can return it at query time
    #   metadatas  — arbitrary key-value pairs attached to each item for filtering
    print("Storing in ChromaDB…")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"\nDone. {collection.count()} chunks stored in {CHROMA_DIR}/")


# ── retrieve ──────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed query, run cosine similarity search, return top_k chunks.

    Returns a list of dicts:
      text       — the chunk text
      source     — filename it came from
      professor  — parsed from filename
      course     — parsed from filename
      distance   — cosine distance (lower = more similar; 0.0 is identical)
    """
    model      = SentenceTransformer(MODEL_NAME)
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = get_collection(client)

    # collection.query() takes:
    #   query_embeddings — the embedded query vector (must be a list of lists)
    #   n_results        — how many nearest neighbours to return
    #   include          — which fields to include in the response
    query_vec = model.encode([query], convert_to_list=True)
    results   = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # results is a dict of parallel lists-of-lists (one list per query).
    # Since we sent one query, we unpack index [0] from each.
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text":      doc,
            "source":    meta["source"],
            "professor": meta["professor"],
            "course":    meta["course"],
            "distance":  round(dist, 4),
        })
    return hits


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Test a retrieval query")
    args = parser.parse_args()

    if args.query:
        print(f"Query: {args.query}\n")
        hits = retrieve(args.query)
        for i, h in enumerate(hits, 1):
            print(f"--- Result {i} | {h['source']} | distance={h['distance']} ---")
            print(h["text"][:400])
            print()
    else:
        build_vector_store()
