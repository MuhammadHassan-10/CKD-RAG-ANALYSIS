"""
Recursive character-based text splitter.
Splits at paragraph → sentence → word boundaries with configurable
chunk size and overlap. Each chunk preserves page/section metadata.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.ingestion.pdf_parser import PageContent

log = logging.getLogger(__name__)


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    page_start: int
    page_end: int
    section: str


# Separators ordered from coarsest to finest
_SEPARATORS = ["\n\n", "\n", ". ", ", ", " "]


def _split_text(text: str, max_len: int) -> list[str]:
    """Recursively split text, preferring natural boundaries."""
    if len(text) <= max_len:
        return [text]

    for sep in _SEPARATORS:
        parts = text.split(sep)
        if len(parts) == 1:
            continue

        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= max_len:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If a single part exceeds max_len, recurse with a finer sep
                if len(part) > max_len:
                    chunks.extend(_split_text(part, max_len))
                else:
                    current = part
                    continue
                current = ""
        if current:
            chunks.append(current)
        return chunks

    # Absolute fallback: hard split
    return [text[i : i + max_len] for i in range(0, len(text), max_len)]


def _add_overlap(fragments: list[str], overlap: int) -> list[str]:
    """Re-introduce overlap between consecutive fragments."""
    if overlap <= 0 or len(fragments) <= 1:
        return fragments
    result: list[str] = [fragments[0]]
    for i in range(1, len(fragments)):
        prefix = fragments[i - 1][-overlap:]
        result.append(prefix + fragments[i])
    return result


def chunk_pages(pages: list[PageContent]) -> list[TextChunk]:
    """
    Take a list of parsed pages and return overlapping text chunks.
    """
    # Concatenate all page texts, recording page boundaries
    full_text = ""
    page_map: list[tuple[int, int, str]] = []  # (start_char, page_no, section)
    for p in pages:
        start = len(full_text)
        full_text += p.text + "\n\n"
        page_map.append((start, p.page, p.section))

    # Split into raw fragments
    raw_fragments = _split_text(full_text, CHUNK_SIZE)
    fragments = _add_overlap(raw_fragments, CHUNK_OVERLAP)

    # Build TextChunk objects with metadata
    chunks: list[TextChunk] = []
    char_pos = 0
    for idx, frag in enumerate(fragments):
        frag_clean = frag.strip()
        if len(frag_clean) < 50:
            char_pos += len(frag)
            continue

        # Determine which page(s) this chunk spans
        frag_start = full_text.find(frag_clean[:80], max(0, char_pos - CHUNK_OVERLAP))
        if frag_start == -1:
            frag_start = char_pos
        frag_end = frag_start + len(frag_clean)

        page_start = pages[0].page
        page_end = pages[0].page
        section = pages[0].section
        for map_start, pg, sec in page_map:
            if map_start <= frag_start:
                page_start = pg
                section = sec
            if map_start <= frag_end:
                page_end = pg

        chunks.append(
            TextChunk(
                text=frag_clean,
                chunk_index=len(chunks),
                page_start=page_start,
                page_end=page_end,
                section=section,
            )
        )
        char_pos = frag_end

    log.info("Created %d chunks (size=%d, overlap=%d).", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks
