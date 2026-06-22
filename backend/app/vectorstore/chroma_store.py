"""
ChromaDB persistent vector store for the Traditional RAG pipeline.

Manages a local ChromaDB collection that stores chunk embeddings
alongside the Neo4j graph (which serves the Graph/Agentic RAG pipelines).
"""
from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings

from app.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

log = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


# ── Lifecycle ──────────────────────────────────────────────────────────────
def get_chroma_client() -> chromadb.ClientAPI:
    """Return (and lazily create) the persistent ChromaDB client."""
    global _client
    if _client is None:
        log.info("Initializing ChromaDB (persist_dir=%s) …", CHROMA_PERSIST_DIR)
        _client = chromadb.Client(Settings(
            persist_directory=CHROMA_PERSIST_DIR,
            is_persistent=True,
            anonymized_telemetry=False,
        ))
        log.info("ChromaDB client ready ✓")
    return _client


def get_collection() -> chromadb.Collection:
    """Return (and lazily create) the chunks collection."""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "ChromaDB collection '%s' ready (%d documents).",
            CHROMA_COLLECTION_NAME,
            _collection.count(),
        )
    return _collection


# ── Write ──────────────────────────────────────────────────────────────────
def add_documents(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> int:
    """
    Add chunk documents and their embeddings to ChromaDB.
    Idempotent — uses chunk_index as the unique ID.

    Returns the number of documents added.
    """
    collection = get_collection()

    ids = [f"chunk_{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "chunk_index": c["chunk_index"],
            "page_start": c.get("page_start", 0),
            "page_end": c.get("page_end", 0),
            "section": c.get("section", ""),
        }
        for c in chunks
    ]

    # ChromaDB upsert handles deduplication by ID
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    log.info("Upserted %d documents into ChromaDB.", len(ids))
    return len(ids)


# ── Search ─────────────────────────────────────────────────────────────────
def similarity_search(
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Find the top-K most similar chunks using cosine distance.

    Returns list of dicts:
        {text, page_start, page_end, section, chunk_index, score}
    """
    collection = get_collection()

    if collection.count() == 0:
        log.warning("ChromaDB collection is empty. Run ingestion first.")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict[str, Any]] = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 1.0
            # ChromaDB returns cosine distance; convert to similarity score
            score = 1.0 - distance

            chunks.append({
                "text": doc,
                "page_start": meta.get("page_start", 0),
                "page_end": meta.get("page_end", 0),
                "section": meta.get("section", ""),
                "chunk_index": meta.get("chunk_index", i),
                "score": round(score, 4),
            })

    log.info("ChromaDB search returned %d results.", len(chunks))
    return chunks


def get_collection_stats() -> dict[str, Any]:
    """Return basic collection statistics."""
    try:
        collection = get_collection()
        return {
            "name": CHROMA_COLLECTION_NAME,
            "count": collection.count(),
            "persist_dir": CHROMA_PERSIST_DIR,
        }
    except Exception:
        return {"name": CHROMA_COLLECTION_NAME, "count": 0, "persist_dir": CHROMA_PERSIST_DIR}
