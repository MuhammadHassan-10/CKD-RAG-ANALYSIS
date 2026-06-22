"""
Hybrid retriever — combines vector similarity search with graph traversal
to produce rich, context-aware retrieval for the RAG chain.
"""
from __future__ import annotations

import logging
from typing import Any

from app.retrieval.vector_search import vector_search
from app.retrieval.graph_search import graph_search

log = logging.getLogger(__name__)


async def hybrid_retrieve(
    question: str,
    vector_top_k: int = 5,
) -> dict[str, Any]:
    """
    Run both vector and graph retrieval in parallel, then merge results.

    Returns:
        {
            "context_text": str,          # Combined chunk texts for the prompt
            "graph_triples": str,         # Formatted entity-relationship triples
            "sources": [...],             # Deduplicated source chunks with metadata
            "raw_triples": [...],         # Raw triple dicts for the frontend
            "matched_entities": [str],    # Entities found in the graph
        }
    """
    # Run both retrievals (graph_search is async, vector_search is async)
    import asyncio
    vector_task = asyncio.create_task(vector_search(question, vector_top_k))
    graph_task = asyncio.create_task(graph_search(question))

    vector_results = await vector_task
    graph_results = await graph_task

    # ── Merge and deduplicate chunks ─────────────────────────────────────
    seen_indices = set()
    all_chunks: list[dict[str, Any]] = []

    # Vector results first (highest relevance)
    for chunk in vector_results:
        idx = chunk.get("chunk_index")
        if idx is not None and idx not in seen_indices:
            seen_indices.add(idx)
            chunk["retrieval_method"] = "vector"
            all_chunks.append(chunk)

    # Then graph-connected chunks (may overlap)
    for chunk in graph_results.get("related_chunks", []):
        idx = chunk.get("chunk_index")
        if idx is not None and idx not in seen_indices:
            seen_indices.add(idx)
            chunk["retrieval_method"] = "graph"
            chunk["score"] = 0.0  # graph chunks don't have cosine score
            all_chunks.append(chunk)

    # ── Format context text ──────────────────────────────────────────────
    context_parts: list[str] = []
    for i, chunk in enumerate(all_chunks[:10]):  # cap at 10 chunks
        source_info = f"[Page {chunk.get('page_start', '?')}]"
        if chunk.get("section"):
            source_info += f" [{chunk['section']}]"
        context_parts.append(f"{source_info}\n{chunk['text']}")

    context_text = "\n\n---\n\n".join(context_parts)

    # ── Format graph triples ─────────────────────────────────────────────
    triples = graph_results.get("triples", [])
    triple_lines = []
    for t in triples:
        triple_lines.append(f"• {t['source']} —[{t['relationship']}]→ {t['target']}")
    graph_triples_text = "\n".join(triple_lines) if triple_lines else "(No graph relationships found)"

    # ── Build sources list ───────────────────────────────────────────────
    sources = []
    for chunk in all_chunks[:10]:
        sources.append({
            "text": chunk.get("text", "")[:300],  # truncate for display
            "page": chunk.get("page_start"),
            "section": chunk.get("section", ""),
            "score": chunk.get("score", 0.0),
        })

    log.info(
        "Hybrid retrieval: %d total chunks (%d vector, %d graph), %d triples.",
        len(all_chunks),
        len(vector_results),
        len(graph_results.get("related_chunks", [])),
        len(triples),
    )

    return {
        "context_text": context_text,
        "graph_triples": graph_triples_text,
        "sources": sources,
        "raw_triples": triples,
        "matched_entities": graph_results.get("matched_entities", []),
    }
