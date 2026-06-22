"""
Vector similarity search against the Neo4j vector index on Chunk embeddings.
"""
from __future__ import annotations

import logging
from typing import Any

from app.config import VECTOR_INDEX_NAME, VECTOR_TOP_K
from app.database import get_driver
from app.ingestion.embedder import embed_single

log = logging.getLogger(__name__)


async def vector_search(
    query: str,
    top_k: int = VECTOR_TOP_K,
) -> list[dict[str, Any]]:
    """
    Embed the query and find the top-K most similar Chunk nodes
    using the Neo4j vector index.

    Returns list of dicts with keys: text, page_start, page_end, section, score.
    """
    query_embedding = embed_single(query)

    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(
            f"""
            CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
            YIELD node, score
            RETURN node.text AS text,
                   node.page_start AS page_start,
                   node.page_end AS page_end,
                   node.section AS section,
                   node.chunk_index AS chunk_index,
                   score
            ORDER BY score DESC
            """,
            {
                "index_name": VECTOR_INDEX_NAME,
                "top_k": top_k,
                "embedding": query_embedding,
            },
        )
        records = await result.data()

    log.info("Vector search returned %d results for query: '%s…'",
             len(records), query[:60])
    return records
