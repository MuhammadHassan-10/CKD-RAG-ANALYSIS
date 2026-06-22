"""
Pydantic request / response models for the API.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Chat ───────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    session_id: str = Field(default="default", description="Conversation session ID")
    rag_type: str = Field(
        default="graph",
        description="RAG pipeline to use: 'traditional', 'agentic', or 'graph'",
    )


class SourceChunk(BaseModel):
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    score: Optional[float] = None


class GraphTriple(BaseModel):
    source: str
    relationship: str
    target: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = []
    graph_context: list[GraphTriple] = []
    response_time_ms: float = 0
    steps_taken: list[str] = []
    rag_type: str = "graph"


# ── Compare ────────────────────────────────────────────────────────────────
class CompareRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to compare across RAG pipelines")
    session_id: str = Field(default="default", description="Conversation session ID")


class EvaluationMetrics(BaseModel):
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    response_time_ms: float = 0.0
    token_efficiency: float = 0.0
    source_coverage: int = 0


class RAGResult(BaseModel):
    answer: str
    sources: list[SourceChunk] = []
    graph_context: list[GraphTriple] = []
    response_time_ms: float = 0
    steps_taken: list[str] = []
    rag_type: str = "traditional"


class CompareResponse(BaseModel):
    question: str
    traditional: RAGResult
    agentic: RAGResult
    traditional_metrics: EvaluationMetrics
    agentic_metrics: EvaluationMetrics
    winner: str = "tie"
    summary: str = ""


# ── Ingestion ──────────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    file_path: Optional[str] = None  # uses DEFAULT_PDF_PATH when None


class IngestStatus(BaseModel):
    status: str  # "running" | "completed" | "failed" | "idle"
    progress: float = 0.0  # 0-100
    entities_count: int = 0
    relationships_count: int = 0
    chunks_count: int = 0
    message: str = ""


# ── Graph stats ────────────────────────────────────────────────────────────────
class GraphStats(BaseModel):
    total_nodes: int = 0
    total_relationships: int = 0
    node_labels: dict[str, int] = {}
    relationship_types: dict[str, int] = {}
