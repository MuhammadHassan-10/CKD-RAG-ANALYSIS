"""
Traditional RAG pipeline — simple vector-only retrieval + generation.

Uses ChromaDB for similarity search and Groq LLM (via LangChain) for generation.
No graph traversal, no re-ranking, no routing — pure vanilla RAG.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, VECTOR_TOP_K
from app.ingestion.embedder import embed_single
from app.vectorstore.chroma_store import similarity_search
from app.models import ChatResponse, SourceChunk

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical knowledge assistant specializing in Chronic Kidney Disease (CKD),
powered by the KDIGO 2024 Clinical Practice Guidelines.

INSTRUCTIONS:
1. Answer questions based ONLY on the provided context (document excerpts).
2. Be precise and cite specific guideline sections, page numbers, or recommendations when available.
3. If the context doesn't contain enough information to answer, say so clearly.
4. Use medical terminology appropriately but explain complex concepts when helpful.
5. Structure your answers with clear headings and bullet points when appropriate.
6. When discussing recommendations, indicate the strength and evidence quality if mentioned.
7. Format your response using Markdown for better readability."""

USER_PROMPT = """=== DOCUMENT CONTEXT ===
The following excerpts are from the KDIGO 2024 CKD Guidelines:
{context}

=== QUESTION ===
{question}

Please provide a detailed, accurate answer based on the context above."""


def _build_chain():
    """Build the LangChain RAG chain."""
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=GROQ_TEMPERATURE,
        max_tokens=4096,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    chain = prompt | llm | StrOutputParser()
    return chain


async def traditional_rag_response(question: str) -> ChatResponse:
    """
    Full Traditional RAG pipeline:
    1. Embed query → ChromaDB vector search
    2. Format retrieved chunks as context
    3. LLM generation via LangChain
    4. Return structured response with timing
    """
    start_time = time.perf_counter()
    steps: list[str] = ["embed_query"]

    # Step 1: Vector retrieval from ChromaDB
    query_embedding = embed_single(question)
    steps.append("chroma_vector_search")

    retrieved_chunks = similarity_search(query_embedding, top_k=VECTOR_TOP_K)

    # Step 2: Build context text
    context_parts: list[str] = []
    for chunk in retrieved_chunks:
        source_info = f"[Page {chunk.get('page_start', '?')}]"
        if chunk.get("section"):
            source_info += f" [{chunk['section']}]"
        context_parts.append(f"{source_info}\n{chunk['text']}")

    context_text = "\n\n---\n\n".join(context_parts) if context_parts else "(No relevant documents found)"
    steps.append("build_context")

    # Step 3: Generate answer
    steps.append("llm_generate")
    chain = _build_chain()
    try:
        answer = await chain.ainvoke({
            "context": context_text,
            "question": question,
        })
    except Exception as exc:
        log.error("Traditional RAG LLM call failed: %s", exc)
        answer = f"I encountered an error generating the response: {str(exc)}"

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Step 4: Package response
    sources = [
        SourceChunk(
            text=c["text"][:300],
            page=c.get("page_start"),
            section=c.get("section", ""),
            score=c.get("score", 0.0),
        )
        for c in retrieved_chunks
    ]

    log.info("Traditional RAG completed in %.0fms (%d sources)", elapsed_ms, len(sources))

    return ChatResponse(
        answer=answer,
        sources=sources,
        graph_context=[],
        response_time_ms=round(elapsed_ms, 1),
        steps_taken=steps,
        rag_type="traditional",
    )
