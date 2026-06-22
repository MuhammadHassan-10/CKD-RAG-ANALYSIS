"""
Neo4j Aura connection manager — provides a singleton async driver,
creates vector & fulltext indexes on startup, and exposes query helpers.
"""
from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from app.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    VECTOR_INDEX_NAME,
    FULLTEXT_INDEX_NAME,
    EMBEDDING_DIMENSION,
)

log = logging.getLogger(__name__)

_driver: AsyncDriver | None = None


# ── Driver lifecycle ──────────────────────────────────────────────────────────
async def get_driver() -> AsyncDriver:
    """Return (and lazily create) the singleton Neo4j async driver."""
    global _driver
    if _driver is None:
        log.info("Connecting to Neo4j at %s …", NEO4J_URI)
        _driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )
        # Quick connectivity check
        async with _driver.session() as session:
            result = await session.run("RETURN 1 AS ok")
            record = await result.single()
            assert record and record["ok"] == 1
        log.info("Neo4j connection verified ✓")
    return _driver


async def close_driver() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
        log.info("Neo4j driver closed.")


# ── Index creation ─────────────────────────────────────────────────────────────
async def ensure_indexes() -> None:
    """Create vector and fulltext indexes if they don't already exist."""
    driver = await get_driver()
    async with driver.session() as session:
        # Vector index on Chunk.embedding
        try:
            await session.run(
                f"""
                CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
                FOR (c:Chunk)
                ON (c.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {EMBEDDING_DIMENSION},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """
            )
            log.info("Vector index '%s' ensured.", VECTOR_INDEX_NAME)
        except Exception as exc:
            log.warning("Vector index creation note: %s", exc)

        # Fulltext index on entity names for fuzzy lookup
        try:
            await session.run(
                f"""
                CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS
                FOR (n:Disease|Symptom|Treatment|Medication|Biomarker|LabTest|RiskFactor|Guideline|Recommendation|CKDStage|Population|Organ)
                ON EACH [n.name]
                """
            )
            log.info("Fulltext index '%s' ensured.", FULLTEXT_INDEX_NAME)
        except Exception as exc:
            log.warning("Fulltext index creation note: %s", exc)


# ── Query helpers ──────────────────────────────────────────────────────────────
async def run_query(
    query: str,
    parameters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute a read query and return list of record dicts."""
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(query, parameters or {})
        records = await result.data()
        return records


async def run_write(
    query: str,
    parameters: dict[str, Any] | None = None,
) -> None:
    """Execute a write query inside an implicit transaction."""
    driver = await get_driver()
    async with driver.session() as session:
        await session.run(query, parameters or {})
