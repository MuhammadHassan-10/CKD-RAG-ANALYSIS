"""
Neo4j Knowledge Graph builder.
Writes extracted entities, relationships, and chunk embeddings into Neo4j Aura.
Uses MERGE to deduplicate entities across chunks.
"""
from __future__ import annotations

import logging
from typing import Any

from app.database import get_driver

log = logging.getLogger(__name__)


async def write_chunks_to_graph(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> int:
    """
    Write Chunk nodes with their text and embedding vectors to Neo4j.
    Returns the number of chunks written.
    """
    driver = await get_driver()
    count = 0
    async with driver.session() as session:
        for chunk, embedding in zip(chunks, embeddings):
            await session.run(
                """
                MERGE (c:Chunk {chunk_index: $chunk_index})
                SET c.text = $text,
                    c.page_start = $page_start,
                    c.page_end = $page_end,
                    c.section = $section,
                    c.embedding = $embedding
                """,
                {
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "section": chunk["section"],
                    "embedding": embedding,
                },
            )
            count += 1
    log.info("Wrote %d chunk nodes.", count)
    return count


async def write_entities_to_graph(
    extraction_results: list[dict[str, Any]],
) -> tuple[int, int]:
    """
    Write entity nodes and relationship edges to Neo4j.
    Links each entity back to its source chunk.
    Returns (entity_count, relationship_count).
    """
    driver = await get_driver()
    entity_count = 0
    rel_count = 0

    async with driver.session() as session:
        for result in extraction_results:
            chunk_index = result["chunk_index"]

            # ── Create entity nodes ──────────────────────────────────────
            for entity in result["entities"]:
                name = entity["name"].strip()
                etype = entity["type"]
                desc = entity.get("description", "")

                # MERGE entity by name + label
                # We use a generic label AND the specific type label
                await session.run(
                    f"""
                    MERGE (e:Entity:{etype} {{name: $name}})
                    ON CREATE SET e.description = $description,
                                  e.entity_type = $etype
                    ON MATCH SET e.description = CASE
                        WHEN size(e.description) < size($description)
                        THEN $description ELSE e.description END
                    WITH e
                    MATCH (c:Chunk {{chunk_index: $chunk_index}})
                    MERGE (e)-[:MENTIONED_IN]->(c)
                    """,
                    {
                        "name": name,
                        "description": desc,
                        "etype": etype,
                        "chunk_index": chunk_index,
                    },
                )
                entity_count += 1

            # ── Create relationships ─────────────────────────────────────
            for rel in result["relationships"]:
                source_name = rel["source"].strip()
                target_name = rel["target"].strip()
                rel_type = rel["type"]

                # Sanitize relationship type for Cypher (must be valid identifier)
                safe_rel = "".join(c if c.isalnum() or c == "_" else "_" for c in rel_type)

                try:
                    await session.run(
                        f"""
                        MATCH (a:Entity {{name: $source}})
                        MATCH (b:Entity {{name: $target}})
                        MERGE (a)-[r:{safe_rel}]->(b)
                        SET r.source_chunk = $chunk_index
                        """,
                        {
                            "source": source_name,
                            "target": target_name,
                            "chunk_index": chunk_index,
                        },
                    )
                    rel_count += 1
                except Exception as exc:
                    log.warning("Failed to create relationship %s -[%s]-> %s: %s",
                                source_name, rel_type, target_name, exc)

    log.info("Wrote %d entity nodes, %d relationships.", entity_count, rel_count)
    return entity_count, rel_count


async def get_graph_stats() -> dict[str, Any]:
    """Return counts of nodes and relationships by label/type."""
    driver = await get_driver()
    async with driver.session() as session:
        # Total nodes
        result = await session.run("MATCH (n) RETURN count(n) as cnt")
        record = await result.single()
        total_nodes = record["cnt"] if record else 0

        # Total relationships
        result = await session.run("MATCH ()-[r]->() RETURN count(r) as cnt")
        record = await result.single()
        total_rels = record["cnt"] if record else 0

        # Nodes by label
        result = await session.run(
            "CALL db.labels() YIELD label "
            "CALL (label) { MATCH (n) WHERE label IN labels(n) RETURN count(n) AS cnt } "
            "RETURN label, cnt"
        )
        records = await result.data()
        node_labels = {r["label"]: r["cnt"] for r in records}

        # Relationships by type
        result = await session.run(
            "CALL db.relationshipTypes() YIELD relationshipType AS type "
            "CALL (type) { MATCH ()-[r]->() WHERE type(r) = type RETURN count(r) AS cnt } "
            "RETURN type, cnt"
        )
        records = await result.data()
        rel_types = {r["type"]: r["cnt"] for r in records}

    return {
        "total_nodes": total_nodes,
        "total_relationships": total_rels,
        "node_labels": node_labels,
        "relationship_types": rel_types,
    }
