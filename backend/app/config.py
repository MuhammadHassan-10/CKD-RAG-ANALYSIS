"""
Application configuration — loads environment variables and defines constants.
"""
import os
from dotenv import load_dotenv

# Load .env from the backend directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


# ── Groq LLM ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.1"))

# ── Neo4j Aura ─────────────────────────────────────────────────────────────────
NEO4J_URI: str = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 1000        # characters per chunk
CHUNK_OVERLAP: int = 200      # overlap between consecutive chunks

# ── Retrieval ──────────────────────────────────────────────────────────────────
VECTOR_TOP_K: int = 5         # number of similar chunks from vector search
GRAPH_TRAVERSAL_DEPTH: int = 2  # max hops for graph context expansion

# ── Ingestion rate-limiting (Groq free tier) ───────────────────────────────────
EXTRACTION_DELAY_SECONDS: float = 2.0  # delay between Groq entity-extraction calls
EXTRACTION_BATCH_SIZE: int = 5         # chunks processed per batch

# ── Neo4j index names ─────────────────────────────────────────────────────────
VECTOR_INDEX_NAME: str = "chunk_embeddings"
FULLTEXT_INDEX_NAME: str = "entity_fulltext"

# ── PDF path ───────────────────────────────────────────────────────────────────
DEFAULT_PDF_PATH: str = os.getenv(
    "PDF_PATH",
    os.path.join(os.path.expanduser("~"), "Desktop", "KDIGO-2024-CKD-Guideline.pdf"),
)

# ── ChromaDB ───────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_data"),
)
CHROMA_COLLECTION_NAME: str = "kdigo_chunks"
