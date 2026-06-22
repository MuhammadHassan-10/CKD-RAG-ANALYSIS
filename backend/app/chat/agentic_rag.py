"""
Agentic RAG pipeline — LangGraph state machine with intelligent routing,
document grading, hallucination checking, and query rewriting.

Uses LangChain for LLM interaction and LangGraph for orchestration.
Combines ChromaDB vector search and Neo4j graph search with agent decisions.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Annotated, TypedDict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from app.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, VECTOR_TOP_K
from app.ingestion.embedder import embed_single
from app.vectorstore.chroma_store import similarity_search as chroma_search
from app.retrieval.graph_search import graph_search
from app.models import ChatResponse, SourceChunk, GraphTriple

log = logging.getLogger(__name__)


# ── Agent State ────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    rewritten_question: str
    documents: list[dict[str, Any]]
    graph_triples: list[dict[str, str]]
    graph_chunks: list[dict[str, Any]]
    generation: str
    route: str
    relevance_scores: list[float]
    is_grounded: bool
    retry_count: int
    max_retries: int
    steps_taken: list[str]
    timing: dict[str, float]


# ── LLM helper ─────────────────────────────────────────────────────────────
def _get_llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=temperature,
        max_tokens=4096,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  GRAPH NODES
# ═══════════════════════════════════════════════════════════════════════════

async def route_query(state: AgentState) -> AgentState:
    """Classify the query to decide retrieval strategy."""
    t0 = time.perf_counter()

    llm = _get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query router for a medical knowledge system about Chronic Kidney Disease (CKD).
Classify the user's question into ONE of these categories:
- "simple": Factual lookup, definitions, simple questions (use vector search only)
- "complex": Requires understanding relationships between entities, multi-hop reasoning (use graph search only)
- "both": Benefits from both vector similarity and graph relationships

Respond with ONLY one word: simple, complex, or both"""),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    try:
        route = await chain.ainvoke({"question": state["question"]})
        route = route.strip().lower()
        if route not in ("simple", "complex", "both"):
            route = "both"
    except Exception as exc:
        log.warning("Route classification failed: %s, defaulting to 'both'", exc)
        route = "both"

    state["route"] = route
    state["steps_taken"] = state.get("steps_taken", []) + [f"route_query→{route}"]
    state["timing"]["route"] = time.perf_counter() - t0
    log.info("Query routed as: %s", route)
    return state


async def vector_retrieve(state: AgentState) -> AgentState:
    """Retrieve documents from ChromaDB via vector similarity."""
    t0 = time.perf_counter()

    query = state.get("rewritten_question") or state["question"]
    query_embedding = embed_single(query)
    results = chroma_search(query_embedding, top_k=VECTOR_TOP_K)

    # Merge with existing documents (avoid duplicates)
    existing_indices = {d.get("chunk_index") for d in state.get("documents", [])}
    new_docs = [r for r in results if r.get("chunk_index") not in existing_indices]

    state["documents"] = state.get("documents", []) + new_docs
    state["steps_taken"] = state.get("steps_taken", []) + [f"vector_retrieve({len(new_docs)} docs)"]
    state["timing"]["vector_retrieve"] = time.perf_counter() - t0
    return state


async def graph_retrieve(state: AgentState) -> AgentState:
    """Retrieve context from the Neo4j knowledge graph."""
    t0 = time.perf_counter()

    query = state.get("rewritten_question") or state["question"]
    graph_results = await graph_search(query)

    # Store triples
    state["graph_triples"] = graph_results.get("triples", [])

    # Merge graph chunks with existing docs
    existing_indices = {d.get("chunk_index") for d in state.get("documents", [])}
    graph_chunks = graph_results.get("related_chunks", [])
    new_chunks = []
    for c in graph_chunks:
        if c.get("chunk_index") not in existing_indices:
            c["score"] = 0.0
            c["retrieval_method"] = "graph"
            new_chunks.append(c)
            existing_indices.add(c["chunk_index"])

    state["documents"] = state.get("documents", []) + new_chunks
    state["graph_chunks"] = new_chunks
    state["steps_taken"] = state.get("steps_taken", []) + [
        f"graph_retrieve({len(state['graph_triples'])} triples, {len(new_chunks)} chunks)"
    ]
    state["timing"]["graph_retrieve"] = time.perf_counter() - t0
    return state


async def grade_documents(state: AgentState) -> AgentState:
    """LLM grades each retrieved document for relevance to the question."""
    t0 = time.perf_counter()

    if not state.get("documents"):
        state["relevance_scores"] = []
        state["steps_taken"] = state.get("steps_taken", []) + ["grade_documents(0 docs)"]
        state["timing"]["grade"] = time.perf_counter() - t0
        return state

    llm = _get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a relevance grader. Given a question and a document, determine if the document 
contains information relevant to answering the question.
Respond with ONLY 'yes' or 'no'."""),
        ("human", "Question: {question}\n\nDocument: {document}\n\nIs this document relevant?"),
    ])
    chain = prompt | llm | StrOutputParser()

    question = state.get("rewritten_question") or state["question"]
    relevant_docs = []
    scores = []

    for doc in state["documents"][:8]:  # Cap grading at 8 docs to manage API calls
        try:
            result = await chain.ainvoke({
                "question": question,
                "document": doc["text"][:500],
            })
            is_relevant = "yes" in result.strip().lower()
            scores.append(1.0 if is_relevant else 0.0)
            if is_relevant:
                relevant_docs.append(doc)
        except Exception as exc:
            log.warning("Document grading failed: %s", exc)
            relevant_docs.append(doc)  # Keep on failure
            scores.append(0.5)

    state["documents"] = relevant_docs
    state["relevance_scores"] = scores
    state["steps_taken"] = state.get("steps_taken", []) + [
        f"grade_documents({len(relevant_docs)}/{len(scores)} relevant)"
    ]
    state["timing"]["grade"] = time.perf_counter() - t0
    return state


async def rewrite_query(state: AgentState) -> AgentState:
    """Rewrite the query if no relevant documents were found."""
    t0 = time.perf_counter()

    llm = _get_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query rewriter for a medical knowledge system about CKD (Chronic Kidney Disease).
Rewrite the question to be more specific and likely to match relevant medical documents.
Use standard medical terminology. Return ONLY the rewritten question."""),
        ("human", "Original question: {question}\n\nRewrite this question:"),
    ])
    chain = prompt | llm | StrOutputParser()

    try:
        rewritten = await chain.ainvoke({"question": state["question"]})
        state["rewritten_question"] = rewritten.strip()
    except Exception as exc:
        log.warning("Query rewrite failed: %s", exc)
        state["rewritten_question"] = state["question"]

    state["retry_count"] = state.get("retry_count", 0) + 1
    state["steps_taken"] = state.get("steps_taken", []) + [
        f"rewrite_query→'{state['rewritten_question'][:60]}…'"
    ]
    state["timing"]["rewrite"] = time.perf_counter() - t0
    return state


async def generate(state: AgentState) -> AgentState:
    """Generate an answer from the filtered context."""
    t0 = time.perf_counter()

    # Build context
    context_parts = []
    for doc in state.get("documents", [])[:10]:
        source_info = f"[Page {doc.get('page_start', '?')}]"
        if doc.get("section"):
            source_info += f" [{doc['section']}]"
        context_parts.append(f"{source_info}\n{doc['text']}")

    context_text = "\n\n---\n\n".join(context_parts) if context_parts else "(No relevant documents found)"

    # Build graph context
    triple_lines = []
    for t in state.get("graph_triples", []):
        triple_lines.append(f"• {t['source']} —[{t['relationship']}]→ {t['target']}")
    graph_text = "\n".join(triple_lines) if triple_lines else "(No graph relationships found)"

    llm = _get_llm(temperature=GROQ_TEMPERATURE)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a clinical knowledge assistant specializing in Chronic Kidney Disease (CKD),
powered by the KDIGO 2024 Clinical Practice Guidelines.

INSTRUCTIONS:
1. Answer questions based ONLY on the provided context (document excerpts and knowledge graph).
2. Be precise and cite specific guideline sections, page numbers, or recommendations when available.
3. If the context doesn't contain enough information to answer, say so clearly.
4. Use medical terminology appropriately but explain complex concepts when helpful.
5. Structure your answers with clear headings and bullet points when appropriate.
6. When discussing recommendations, indicate the strength and evidence quality if mentioned.
7. Format your response using Markdown for better readability."""),
        ("human", """=== KNOWLEDGE GRAPH CONTEXT ===
{graph_context}

=== DOCUMENT CONTEXT ===
{context}

=== QUESTION ===
{question}

Please provide a detailed, accurate answer based on the context above."""),
    ])

    chain = prompt | llm | StrOutputParser()
    try:
        answer = await chain.ainvoke({
            "context": context_text,
            "graph_context": graph_text,
            "question": state.get("rewritten_question") or state["question"],
        })
        state["generation"] = answer
    except Exception as exc:
        log.error("Agentic RAG generation failed: %s", exc)
        state["generation"] = f"I encountered an error generating the response: {str(exc)}"

    state["steps_taken"] = state.get("steps_taken", []) + ["generate"]
    state["timing"]["generate"] = time.perf_counter() - t0
    return state


async def hallucination_check(state: AgentState) -> AgentState:
    """Check if the generated answer is grounded in the context."""
    t0 = time.perf_counter()

    if not state.get("documents"):
        state["is_grounded"] = True  # Can't check without docs
        state["steps_taken"] = state.get("steps_taken", []) + ["hallucination_check→skip"]
        state["timing"]["hallucination"] = time.perf_counter() - t0
        return state

    context_text = " ".join(d["text"][:300] for d in state["documents"][:5])

    llm = _get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a hallucination grader. Given a set of source documents and a generated answer,
determine if the answer is grounded in / supported by the documents.
Respond with ONLY 'yes' or 'no'."""),
        ("human", """Source documents: {context}

Generated answer: {answer}

Is this answer grounded in the source documents?"""),
    ])
    chain = prompt | llm | StrOutputParser()

    try:
        result = await chain.ainvoke({
            "context": context_text,
            "answer": state.get("generation", ""),
        })
        is_grounded = "yes" in result.strip().lower()
    except Exception as exc:
        log.warning("Hallucination check failed: %s", exc)
        is_grounded = True  # Assume grounded on failure

    state["is_grounded"] = is_grounded
    state["steps_taken"] = state.get("steps_taken", []) + [
        f"hallucination_check→{'grounded' if is_grounded else 'NOT grounded'}"
    ]
    state["timing"]["hallucination"] = time.perf_counter() - t0
    return state


# ═══════════════════════════════════════════════════════════════════════════
#  EDGE ROUTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def route_after_classification(state: AgentState) -> str:
    """Decide retrieval strategy based on query classification."""
    route = state.get("route", "both")
    if route == "simple":
        return "vector_retrieve"
    elif route == "complex":
        return "graph_retrieve"
    else:
        return "both_retrieve"


def route_after_grading(state: AgentState) -> str:
    """Decide whether to generate or rewrite based on doc relevance."""
    docs = state.get("documents", [])
    retry = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 1)

    if len(docs) > 0:
        return "generate"
    elif retry < max_retries:
        return "rewrite"
    else:
        return "generate"  # Generate anyway with what we have


def route_after_hallucination(state: AgentState) -> str:
    """Decide whether to return or re-generate."""
    is_grounded = state.get("is_grounded", True)
    retry = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 1)

    if is_grounded:
        return "end"
    elif retry < max_retries:
        return "regenerate"
    else:
        return "end"  # Return anyway after max retries


# ═══════════════════════════════════════════════════════════════════════════
#  BUILD THE GRAPH
# ═══════════════════════════════════════════════════════════════════════════

async def _both_retrieve(state: AgentState) -> AgentState:
    """Run both vector and graph retrieval."""
    state = await vector_retrieve(state)
    state = await graph_retrieve(state)
    return state


def build_agent_graph() -> StateGraph:
    """Construct the LangGraph agent workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("route_query", route_query)
    workflow.add_node("vector_retrieve", vector_retrieve)
    workflow.add_node("graph_retrieve", graph_retrieve)
    workflow.add_node("both_retrieve", _both_retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("hallucination_check", hallucination_check)

    # Set entry point
    workflow.set_entry_point("route_query")

    # Route after classification
    workflow.add_conditional_edges(
        "route_query",
        route_after_classification,
        {
            "vector_retrieve": "vector_retrieve",
            "graph_retrieve": "graph_retrieve",
            "both_retrieve": "both_retrieve",
        },
    )

    # All retrieval paths lead to grading
    workflow.add_edge("vector_retrieve", "grade_documents")
    workflow.add_edge("graph_retrieve", "grade_documents")
    workflow.add_edge("both_retrieve", "grade_documents")

    # After grading: generate or rewrite
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
        },
    )

    # Rewrite loops back to vector retrieve
    workflow.add_edge("rewrite_query", "vector_retrieve")

    # After generation: hallucination check
    workflow.add_edge("generate", "hallucination_check")

    # After hallucination check: end or regenerate
    workflow.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination,
        {
            "end": END,
            "regenerate": "generate",
        },
    )

    return workflow


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

# Compile the graph once at module load
_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        workflow = build_agent_graph()
        _compiled_graph = workflow.compile()
    return _compiled_graph


async def agentic_rag_response(question: str) -> ChatResponse:
    """
    Full Agentic RAG pipeline via LangGraph:
    1. Route query (simple/complex/both)
    2. Retrieve from appropriate sources
    3. Grade document relevance
    4. Generate answer (with possible query rewrite)
    5. Check for hallucination
    6. Return structured response with timing + decision trace
    """
    start_time = time.perf_counter()

    initial_state: AgentState = {
        "question": question,
        "rewritten_question": "",
        "documents": [],
        "graph_triples": [],
        "graph_chunks": [],
        "generation": "",
        "route": "",
        "relevance_scores": [],
        "is_grounded": True,
        "retry_count": 0,
        "max_retries": 1,
        "steps_taken": [],
        "timing": {},
    }

    # Run the graph
    graph = _get_graph()
    final_state = await graph.ainvoke(initial_state)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Package response
    sources = [
        SourceChunk(
            text=d["text"][:300],
            page=d.get("page_start"),
            section=d.get("section", ""),
            score=d.get("score", 0.0),
        )
        for d in final_state.get("documents", [])[:10]
    ]

    graph_context = [
        GraphTriple(
            source=t["source"],
            relationship=t["relationship"],
            target=t["target"],
        )
        for t in final_state.get("graph_triples", [])
    ]

    log.info(
        "Agentic RAG completed in %.0fms | Steps: %s",
        elapsed_ms,
        " → ".join(final_state.get("steps_taken", [])),
    )

    return ChatResponse(
        answer=final_state.get("generation", "No answer generated."),
        sources=sources,
        graph_context=graph_context,
        response_time_ms=round(elapsed_ms, 1),
        steps_taken=final_state.get("steps_taken", []),
        rag_type="agentic",
    )
