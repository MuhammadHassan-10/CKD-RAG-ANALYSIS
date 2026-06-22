"""
FastAPI application — serves the Dual RAG chat API, comparison/evaluation
endpoints, and frontend static files.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import DEFAULT_PDF_PATH
from app.database import get_driver, close_driver, ensure_indexes
from app.models import (
    ChatRequest,
    ChatResponse,
    CompareRequest,
    CompareResponse,
    IngestRequest,
    IngestStatus,
    GraphStats,
)
from app.chat.chain import generate_response
from app.ingestion.graph_builder import get_graph_stats
from app.evaluation.evaluator import compare_pipelines, get_comparison_history

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Global ingestion state ─────────────────────────────────────────────────────
_ingest_status = IngestStatus(status="idle")
_ingest_lock = asyncio.Lock()


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    log.info("Starting Dual RAG backend …")

    # Initialize Neo4j
    try:
        await get_driver()
        await ensure_indexes()
        log.info("Neo4j connected and indexes ensured ✓")
    except Exception as exc:
        log.error("Neo4j startup failed: %s", exc)

    # Initialize ChromaDB
    try:
        from app.vectorstore.chroma_store import get_collection
        collection = get_collection()
        log.info("ChromaDB ready ✓ (%d documents)", collection.count())
    except Exception as exc:
        log.error("ChromaDB startup failed: %s", exc)

    yield
    await close_driver()
    log.info("Backend shut down.")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="KDIGO CKD Dual RAG Assistant",
    description="Traditional + Agentic RAG chatbot with evaluation dashboard for KDIGO 2024 CKD Guidelines",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API routes ─────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer a question using the selected RAG pipeline."""
    try:
        response = await generate_response(request.question, rag_type=request.rag_type)
        return response
    except Exception as exc:
        log.error("Chat error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    """Run both RAG pipelines on the same question and compare with evaluation metrics."""
    try:
        result = await compare_pipelines(request.question)
        return result
    except Exception as exc:
        log.error("Compare error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/compare/history")
async def compare_history():
    """Return past comparison results."""
    return get_comparison_history()


@app.post("/api/ingest")
async def ingest(request: IngestRequest | None = None):
    """
    Trigger the full ingestion pipeline:
    PDF → chunks → embeddings → Neo4j graph + ChromaDB vectors
    """
    global _ingest_status

    if _ingest_lock.locked():
        return JSONResponse(
            status_code=409,
            content={"detail": "Ingestion already running."},
        )

    file_path = (request and request.file_path) or DEFAULT_PDF_PATH
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"PDF not found: {file_path}")

    # Run ingestion in background
    asyncio.create_task(_run_ingestion(file_path))
    return {"status": "started", "file": file_path}


async def _run_ingestion(file_path: str):
    """Background ingestion task — writes to both Neo4j and ChromaDB."""
    global _ingest_status
    async with _ingest_lock:
        try:
            _ingest_status = IngestStatus(status="running", message="Parsing PDF…")

            # 1. Parse PDF
            from app.ingestion.pdf_parser import parse_pdf
            pages = parse_pdf(file_path)
            _ingest_status.message = f"Parsed {len(pages)} pages. Chunking…"
            _ingest_status.progress = 10

            # 2. Chunk
            from app.ingestion.chunker import chunk_pages
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
            _ingest_status.chunks_count = len(chunk_dicts)
            _ingest_status.message = f"{len(chunk_dicts)} chunks. Computing embeddings…"
            _ingest_status.progress = 20

            # 3. Embed
            from app.ingestion.embedder import embed_texts
            texts = [c["text"] for c in chunk_dicts]
            embeddings = embed_texts(texts)
            _ingest_status.message = "Embeddings computed. Writing to Neo4j & ChromaDB…"
            _ingest_status.progress = 35

            # 4a. Write chunks to Neo4j graph
            from app.ingestion.graph_builder import write_chunks_to_graph
            await write_chunks_to_graph(chunk_dicts, embeddings)
            _ingest_status.progress = 40

            # 4b. Write chunks to ChromaDB
            from app.vectorstore.chroma_store import add_documents
            add_documents(chunk_dicts, embeddings)
            _ingest_status.message = "Chunks stored in Neo4j + ChromaDB. Extracting entities…"
            _ingest_status.progress = 45

            # 5. Extract entities
            from app.ingestion.entity_extractor import extract_all

            def on_progress(current: int, total: int):
                global _ingest_status
                pct = 45 + int((current / total) * 45)
                _ingest_status.progress = pct
                _ingest_status.message = f"Extracting entities: {current}/{total} chunks…"

            extraction_results = await extract_all(chunk_dicts, on_progress=on_progress)
            _ingest_status.message = "Entities extracted. Writing to graph…"
            _ingest_status.progress = 92

            # 6. Write entities to graph
            from app.ingestion.graph_builder import write_entities_to_graph
            ent_count, rel_count = await write_entities_to_graph(extraction_results)
            _ingest_status.entities_count = ent_count
            _ingest_status.relationships_count = rel_count

            _ingest_status.status = "completed"
            _ingest_status.progress = 100
            _ingest_status.message = (
                f"Done! {ent_count} entities, {rel_count} relationships, "
                f"{len(chunk_dicts)} chunks (Neo4j + ChromaDB)."
            )
            log.info("Ingestion complete ✓")

        except Exception as exc:
            log.error("Ingestion failed: %s", exc, exc_info=True)
            _ingest_status.status = "failed"
            _ingest_status.message = f"Error: {str(exc)}"


@app.get("/api/status")
async def status():
    """Return the current ingestion status."""
    return _ingest_status


@app.get("/api/graph-stats", response_model=GraphStats)
async def graph_stats():
    """Return node/relationship counts from Neo4j."""
    try:
        stats = await get_graph_stats()
        return GraphStats(**stats)
    except Exception as exc:
        log.error("Graph stats error: %s", exc)
        return GraphStats()


@app.get("/api/chroma-stats")
async def chroma_stats():
    """Return ChromaDB collection statistics."""
    try:
        from app.vectorstore.chroma_store import get_collection_stats
        return get_collection_stats()
    except Exception as exc:
        log.error("ChromaDB stats error: %s", exc)
        return {"name": "kdigo_chunks", "count": 0}


# ── Serve frontend ─────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def serve_index():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        return FileResponse(index_path)
