import asyncio
import logging
import os
from app.config import DEFAULT_PDF_PATH
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import embed_texts
from app.vectorstore.chroma_store import add_documents, get_collection

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def populate_chroma_only(file_path: str):
    if not os.path.exists(file_path):
        log.error("PDF not found: %s", file_path)
        return

    log.info("1. Parsing PDF...")
    pages = parse_pdf(file_path)
    
    log.info("2. Chunking pages...")
    chunks = chunk_pages(pages)
    chunk_dicts = [
        {
            "chunk_index": c.chunk_index,
            "text": c.text,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "section": c.section,
        }
        for c in chunks
    ]
    log.info("Created %d chunks.", len(chunk_dicts))
    
    log.info("3. Computing embeddings...")
    texts = [c["text"] for c in chunk_dicts]
    embeddings = embed_texts(texts)
    
    log.info("4. Writing to ChromaDB...")
    num_added = add_documents(chunk_dicts, embeddings)
    
    col = get_collection()
    log.info("Done! ChromaDB now has %d documents.", col.count())

if __name__ == "__main__":
    populate_chroma_only(DEFAULT_PDF_PATH)
