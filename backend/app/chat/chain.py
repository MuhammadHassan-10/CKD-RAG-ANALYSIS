"""
Chat chain router — dispatches to the correct RAG pipeline based on rag_type.

Also contains the original Graph RAG chain (renamed to graph_rag_response)
which uses hybrid retrieval (vector + graph via Neo4j).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from groq import AsyncGroq

from app.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE
from app.retrieval.hybrid_retriever import hybrid_retrieve
from app.models import ChatResponse, SourceChunk, GraphTriple

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical knowledge assistant specializing in Chronic Kidney Disease (CKD),
powered by the KDIGO 2024 Clinical Practice Guidelines.

INSTRUCTIONS:
1. Answer questions based ONLY on the provided context (document excerpts and knowledge graph).
2. Be precise and cite specific guideline sections, page numbers, or recommendations when available.
3. If the context doesn't contain enough information to answer, say so clearly.
4. Use medical terminology appropriately but explain complex concepts when helpful.
5. Structure your answers with clear headings and bullet points when appropriate.
6. When discussing recommendations, indicate the strength (Grade 1 = strong, Grade 2 = weak)
   and evidence quality (A-D) if mentioned in the context.
7. Format your response using Markdown for better readability."""

USER_PROMPT_TEMPLATE = """=== KNOWLEDGE GRAPH CONTEXT ===
The following entity-relationship triples were found in the medical knowledge graph:
{graph_triples}

=== DOCUMENT CONTEXT ===
The following excerpts are from the KDIGO 2024 CKD Guidelines:
{context_text}

=== QUESTION ===
{question}

Please provide a detailed, accurate answer based on the context above."""


async def graph_rag_response(question: str) -> ChatResponse:
    """
    Original Graph RAG pipeline:
    1. Hybrid retrieval (vector + graph)
    2. Prompt assembly
    3. Groq LLM generation
    4. Response packaging with timing
    """
    start_time = time.perf_counter()
    steps = ["hybrid_retrieve"]

    # Step 1: Retrieve context
    retrieval = await hybrid_retrieve(question)

    # Step 2: Build prompt
    steps.append("build_prompt")
    user_message = USER_PROMPT_TEMPLATE.format(
        graph_triples=retrieval["graph_triples"],
        context_text=retrieval["context_text"],
        question=question,
    )

    # Step 3: Call Groq LLM
    steps.append("llm_generate")
    client = AsyncGroq(api_key=GROQ_API_KEY)
    try:
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=GROQ_TEMPERATURE,
            max_tokens=4096,
        )
        answer = response.choices[0].message.content or "I could not generate a response."
    except Exception as exc:
        log.error("Groq LLM call failed: %s", exc)
        answer = f"I encountered an error generating the response: {str(exc)}"

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Step 4: Package response
    sources = [
        SourceChunk(
            text=s["text"],
            page=s.get("page"),
            section=s.get("section", ""),
            score=s.get("score", 0.0),
        )
        for s in retrieval["sources"]
    ]

    graph_context = [
        GraphTriple(
            source=t["source"],
            relationship=t["relationship"],
            target=t["target"],
        )
        for t in retrieval["raw_triples"]
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        graph_context=graph_context,
        response_time_ms=round(elapsed_ms, 1),
        steps_taken=steps,
        rag_type="graph",
    )


async def generate_response(question: str, rag_type: str = "graph") -> ChatResponse:
    """
    Router — dispatches to the correct RAG pipeline.

    rag_type:
        - "traditional": Simple vector-only RAG (ChromaDB)
        - "agentic": LangGraph agent with routing + grading
        - "graph": Original hybrid Graph RAG (Neo4j)
    """
    if rag_type == "traditional":
        from app.chat.traditional_rag import traditional_rag_response
        return await traditional_rag_response(question)
    elif rag_type == "agentic":
        from app.chat.agentic_rag import agentic_rag_response
        return await agentic_rag_response(question)
    else:
        return await graph_rag_response(question)
