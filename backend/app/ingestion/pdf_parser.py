"""
PDF text extraction using PyMuPDF (fitz).
Extracts text page by page, attempts to identify section headings,
and cleans up common artefacts from the KDIGO guideline PDF.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF

log = logging.getLogger(__name__)


@dataclass
class PageContent:
    page: int
    text: str
    section: str = ""


# Patterns typical of KDIGO headers / footers to strip
_HEADER_RE = re.compile(
    r"(Kidney International.*?Supplement|www\.kidney-international\.org|"
    r"S\d+\s*$|VOLUME \d+.*?APRIL \d+)",
    re.IGNORECASE,
)

# Pattern to detect section headings (CHAPTER, SECTION, numbered headings)
_HEADING_RE = re.compile(
    r"^(?:CHAPTER\s+\d+|SECTION\s+\d+|(?:\d+\.)+\s+[A-Z])[^\n]{3,80}$",
    re.MULTILINE,
)


def _clean_text(raw: str) -> str:
    """Remove common PDF artefacts and normalize whitespace."""
    text = _HEADER_RE.sub("", raw)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize whitespace within lines (keep newlines)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _extract_section(text: str, prev_section: str) -> str:
    """Try to find the most prominent heading on the page."""
    matches = _HEADING_RE.findall(text)
    if matches:
        return matches[0].strip()
    return prev_section


def parse_pdf(file_path: str) -> list[PageContent]:
    """
    Extract cleaned text from every page of the PDF.

    Returns a list of PageContent objects with page number (1-indexed),
    cleaned text, and best-guess section heading.
    """
    log.info("Opening PDF: %s", file_path)
    doc = fitz.open(file_path)
    pages: list[PageContent] = []
    current_section = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text")
        if not raw_text or len(raw_text.strip()) < 30:
            continue  # skip near-empty pages (images, cover, etc.)

        cleaned = _clean_text(raw_text)
        current_section = _extract_section(cleaned, current_section)

        pages.append(
            PageContent(
                page=page_num + 1,  # 1-indexed
                text=cleaned,
                section=current_section,
            )
        )

    doc.close()
    log.info("Extracted text from %d pages.", len(pages))
    return pages
