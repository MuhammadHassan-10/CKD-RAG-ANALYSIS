"""
LLM-based entity & relationship extraction using Groq (Llama 3.3 70B).

Sends each text chunk to the LLM with a structured medical ontology prompt
and parses the JSON response into entity/relationship dicts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from groq import AsyncGroq

from app.config import GROQ_API_KEY, GROQ_MODEL, EXTRACTION_DELAY_SECONDS

log = logging.getLogger(__name__)

# ── Medical domain schema ──────────────────────────────────────────────────────
ENTITY_TYPES = [
    "Disease", "Symptom", "Treatment", "Medication", "Biomarker",
    "LabTest", "RiskFactor", "Guideline", "Recommendation",
    "CKDStage", "Population", "Organ",
]

RELATIONSHIP_TYPES = [
    "TREATS", "CAUSES", "INDICATES", "MEASURED_BY", "RISK_FACTOR_FOR",
    "RECOMMENDS", "CONTRAINDICATES", "STAGE_OF", "ASSOCIATED_WITH",
    "MONITORS", "AFFECTS", "DIAGNOSED_BY", "PROGRESSES_TO",
    "DEFINED_BY", "USED_FOR", "MENTIONED_IN",
]

EXTRACTION_PROMPT = """You are a medical knowledge-graph extraction engine.

Given the following text from the KDIGO 2024 Clinical Practice Guideline for CKD,
extract ALL medical entities and their relationships.

### ALLOWED ENTITY TYPES
{entity_types}

### ALLOWED RELATIONSHIP TYPES
{relationship_types}

### RULES
1. Extract as many entities and relationships as possible from the text.
2. Each entity must have a "name" (canonical, concise) and a "type" from the allowed list.
3. Each relationship must reference two entity names and a "type" from the allowed list.
4. Use consistent naming: prefer standard medical terminology (e.g., "CKD" not "chronic kidney disease" for short references).
5. Return ONLY valid JSON. No markdown, no code fences, no commentary.

### OUTPUT FORMAT
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "brief description"}},
    ...
  ],
  "relationships": [
    {{"source": "entity_name", "target": "entity_name", "type": "REL_TYPE"}},
    ...
  ]
}}

### TEXT TO PROCESS
{text}
"""


def _build_prompt(text: str) -> str:
    return EXTRACTION_PROMPT.format(
        entity_types=", ".join(ENTITY_TYPES),
        relationship_types=", ".join(RELATIONSHIP_TYPES),
        text=text,
    )


def _parse_json_response(content: str) -> dict[str, Any]:
    """Robustly parse JSON from LLM output, stripping markdown fences if present."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", content)
    cleaned = cleaned.strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    log.warning("Failed to parse LLM JSON response. Returning empty result.")
    return {"entities": [], "relationships": []}


async def extract_entities_from_chunk(
    client: AsyncGroq,
    chunk_text: str,
    retries: int = 2,
) -> dict[str, Any]:
    """
    Send a single chunk to Groq and return parsed entities + relationships.
    Includes retry logic and rate-limit delays.
    """
    prompt = _build_prompt(chunk_text)

    for attempt in range(retries + 1):
        try:
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You extract medical entities and relationships from clinical guideline text. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            content = response.choices[0].message.content or ""
            result = _parse_json_response(content)

            # Validate structure
            entities = result.get("entities", [])
            relationships = result.get("relationships", [])

            # Filter to allowed types
            valid_entities = [
                e for e in entities
                if isinstance(e, dict) and e.get("type") in ENTITY_TYPES and e.get("name")
            ]
            valid_rels = [
                r for r in relationships
                if isinstance(r, dict) and r.get("type") in RELATIONSHIP_TYPES
                and r.get("source") and r.get("target")
            ]

            return {"entities": valid_entities, "relationships": valid_rels}

        except Exception as exc:
            log.warning("Extraction attempt %d failed: %s", attempt + 1, exc)
            if attempt < retries:
                await asyncio.sleep(EXTRACTION_DELAY_SECONDS * (attempt + 1))
            else:
                log.error("All extraction attempts failed for chunk.")
                return {"entities": [], "relationships": []}

    return {"entities": [], "relationships": []}


async def extract_all(
    chunks: list[dict[str, Any]],
    on_progress: Any = None,
) -> list[dict[str, Any]]:
    """
    Process all chunks through the entity extractor.
    Returns list of {"chunk_index": int, "entities": [...], "relationships": [...]}.
    """
    client = AsyncGroq(api_key=GROQ_API_KEY)
    results: list[dict[str, Any]] = []

    total = len(chunks)
    for i, chunk in enumerate(chunks):
        log.info("Extracting entities from chunk %d / %d …", i + 1, total)
        extracted = await extract_entities_from_chunk(client, chunk["text"])
        results.append({
            "chunk_index": chunk["chunk_index"],
            "entities": extracted["entities"],
            "relationships": extracted["relationships"],
        })

        if on_progress:
            on_progress(i + 1, total)

        # Rate limiting
        if i < total - 1:
            await asyncio.sleep(EXTRACTION_DELAY_SECONDS)

    total_entities = sum(len(r["entities"]) for r in results)
    total_rels = sum(len(r["relationships"]) for r in results)
    log.info("Extraction complete: %d entities, %d relationships from %d chunks.",
             total_entities, total_rels, total)
    return results
