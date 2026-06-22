"""
Graph-based context expansion.
Extracts entities from the user query, looks them up in the graph,
and traverses 1-2 hops to collect related entities and their source chunks.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from groq import AsyncGroq

from app.config import GROQ_API_KEY, GROQ_MODEL, GRAPH_TRAVERSAL_DEPTH, FULLTEXT_INDEX_NAME
from app.database import get_driver

log = logging.getLogger(__name__)

ENTITY_EXTRACT_PROMPT = """Extract the key medical entities from this question.
Return a JSON array of entity names only. Be concise and use standard medical terminology.
Example: ["CKD", "GFR", "albuminuria"]

Question: {question}

Return ONLY a JSON array, nothing else."""


async def _extract_query_entities(question: str) -> list[str]:
    """Use Groq to identify medical entities in the user question."""
    client = AsyncGroq(api_key=GROQ_API_KEY)
    try:
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Extract medical entity names from questions. Return only a JSON array."},
                {"role": "user", "content": ENTITY_EXTRACT_PROMPT.format(question=question)},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        content = response.choices[0].message.content or "[]"
        # Clean and parse
        cleaned = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`")
        entities = json.loads(cleaned)
        if isinstance(entities, list):
            return [str(e).strip() for e in entities if e]
    except Exception as exc:
        log.warning("Query entity extraction failed: %s", exc)
    return []


async def graph_search(
    question: str,
    max_triples: int = 30,
) -> dict[str, Any]:
    """
    Extract entities from the question, look them up in the graph,
    and traverse relationships to build context.

    Returns:
        {
            "triples": [{"source": str, "relationship": str, "target": str}, ...],
            "related_chunks": [{"text": str, "page_start": int, ...}, ...],
            "matched_entities": [str, ...]
        }
    """
    entities = await _extract_query_entities(question)
    log.info("Query entities: %s", entities)

    if not entities:
        return {"triples": [], "related_chunks": [], "matched_entities": []}

    driver = await get_driver()
    all_triples: list[dict[str, str]] = []
    all_chunks: list[dict[str, Any]] = []
    matched: list[str] = []

    async with driver.session() as session:
        for entity_name in entities:
            # Fuzzy match via fulltext index
            try:
                fuzzy_result = await session.run(
                    f"""
                    CALL db.index.fulltext.queryNodes($index_name, $query)
                    YIELD node, score
                    WHERE score > 0.5
                    RETURN node.name AS name, labels(node) AS labels, score
                    ORDER BY score DESC
                    LIMIT 3
                    """,
                    {"index_name": FULLTEXT_INDEX_NAME, "query": entity_name},
                )
                fuzzy_records = await fuzzy_result.data()
            except Exception:
                fuzzy_records = []

            # Also try exact match
            exact_result = await session.run(
                """
                MATCH (e:Entity {name: $name})
                RETURN e.name AS name, labels(e) AS labels
                LIMIT 1
                """,
                {"name": entity_name},
            )
            exact_records = await exact_result.data()

            # Combine matches
            match_names = set()
            for r in exact_records:
                match_names.add(r["name"])
            for r in fuzzy_records:
                match_names.add(r["name"])

            if not match_names:
                continue

            matched.extend(match_names)

            # Traverse relationships for each matched entity
            for name in match_names:
                # 1-hop relationships
                rel_result = await session.run(
                    """
                    MATCH (e:Entity {name: $name})-[r]-(related:Entity)
                    RETURN e.name AS source,
                           type(r) AS relationship,
                           related.name AS target,
                           related.entity_type AS target_type
                    LIMIT $limit
                    """,
                    {"name": name, "limit": max_triples},
                )
                rel_records = await rel_result.data()

                for rec in rel_records:
                    all_triples.append({
                        "source": rec["source"],
                        "relationship": rec["relationship"],
                        "target": rec["target"],
                    })

                # 2-hop if enabled
                if GRAPH_TRAVERSAL_DEPTH >= 2:
                    hop2_result = await session.run(
                        """
                        MATCH (e:Entity {name: $name})-[r1]-(mid:Entity)-[r2]-(far:Entity)
                        WHERE far.name <> e.name
                        RETURN mid.name AS source,
                               type(r2) AS relationship,
                               far.name AS target
                        LIMIT $limit
                        """,
                        {"name": name, "limit": 15},
                    )
                    hop2_records = await hop2_result.data()
                    for rec in hop2_records:
                        all_triples.append({
                            "source": rec["source"],
                            "relationship": rec["relationship"],
                            "target": rec["target"],
                        })

                # Collect related chunks through entities
                chunk_result = await session.run(
                    """
                    MATCH (e:Entity {name: $name})-[:MENTIONED_IN]->(c:Chunk)
                    RETURN c.text AS text,
                           c.page_start AS page_start,
                           c.page_end AS page_end,
                           c.section AS section,
                           c.chunk_index AS chunk_index
                    LIMIT 5
                    """,
                    {"name": name},
                )
                chunk_records = await chunk_result.data()
                all_chunks.extend(chunk_records)

    # Deduplicate triples
    seen_triples = set()
    unique_triples = []
    for t in all_triples:
        key = (t["source"], t["relationship"], t["target"])
        if key not in seen_triples:
            seen_triples.add(key)
            unique_triples.append(t)

    # Deduplicate chunks by chunk_index
    seen_chunks = set()
    unique_chunks = []
    for c in all_chunks:
        if c["chunk_index"] not in seen_chunks:
            seen_chunks.add(c["chunk_index"])
            unique_chunks.append(c)

    log.info("Graph search: %d triples, %d related chunks, %d matched entities.",
             len(unique_triples), len(unique_chunks), len(matched))

    return {
        "triples": unique_triples[:max_triples],
        "related_chunks": unique_chunks,
        "matched_entities": list(set(matched)),
    }
