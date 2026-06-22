"""
RAGAS-inspired evaluation engine for comparing RAG pipelines.

Computes metrics using the Groq LLM as a judge:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are top-ranked retrieved docs relevant?
- Context Recall: Does the context cover all answer claims?
- Response Time: End-to-end latency
- Token Efficiency: Answer conciseness relative to context
- Source Coverage: Unique source pages referenced
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.models import (
    ChatResponse,
    EvaluationMetrics,
    RAGResult,
    CompareResponse,
    SourceChunk,
    GraphTriple,
)

log = logging.getLogger(__name__)

# In-memory comparison history
_comparison_history: list[dict[str, Any]] = []


def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.0,
        max_tokens=1024,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

async def _score_faithfulness(answer: str, context: str) -> float:
    """
    Faithfulness: How well is the answer grounded in the context?
    LLM rates 0.0 to 1.0.
    """
    if not context.strip() or not answer.strip():
        return 0.0

    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an evaluation judge. Rate how faithfully the answer is grounded in the source context.
A faithful answer makes claims that are directly supported by the context.
An unfaithful answer adds claims not found in the context.

Rate from 0.0 (completely unfaithful) to 1.0 (perfectly faithful).
Respond with ONLY a decimal number between 0.0 and 1.0."""),
        ("human", """Context: {context}

Answer: {answer}

Faithfulness score:"""),
    ])
    chain = prompt | llm | StrOutputParser()

    try:
        result = await chain.ainvoke({
            "context": context[:3000],
            "answer": answer[:1500],
        })
        score = float(re.search(r"[\d.]+", result.strip()).group())
        return max(0.0, min(1.0, score))
    except Exception as exc:
        log.warning("Faithfulness scoring failed: %s", exc)
        return 0.5


async def _score_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy: Does the answer actually address the question?
    LLM rates 0.0 to 1.0.
    """
    if not answer.strip():
        return 0.0

    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an evaluation judge. Rate how relevant the answer is to the question.
A relevant answer directly addresses what was asked.
An irrelevant answer discusses unrelated topics or misses the point.

Rate from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).
Respond with ONLY a decimal number between 0.0 and 1.0."""),
        ("human", """Question: {question}

Answer: {answer}

Relevancy score:"""),
    ])
    chain = prompt | llm | StrOutputParser()

    try:
        result = await chain.ainvoke({
            "question": question,
            "answer": answer[:1500],
        })
        score = float(re.search(r"[\d.]+", result.strip()).group())
        return max(0.0, min(1.0, score))
    except Exception as exc:
        log.warning("Answer relevancy scoring failed: %s", exc)
        return 0.5


async def _score_context_precision(question: str, sources: list[SourceChunk]) -> float:
    """
    Context Precision: Are the top-ranked retrieved documents relevant?
    LLM grades each document, weighted by rank position.
    """
    if not sources:
        return 0.0

    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an evaluation judge. Given a question and a document excerpt,
determine if the document contains information relevant to answering the question.
Respond with ONLY 'yes' or 'no'."""),
        ("human", """Question: {question}

Document: {document}

Is this document relevant?"""),
    ])
    chain = prompt | llm | StrOutputParser()

    relevant_count = 0
    weighted_sum = 0.0
    total_weight = 0.0

    for i, source in enumerate(sources[:5]):  # Top 5
        weight = 1.0 / (i + 1)  # Higher weight for top-ranked docs
        total_weight += weight

        try:
            result = await chain.ainvoke({
                "question": question,
                "document": source.text[:400],
            })
            is_relevant = "yes" in result.strip().lower()
            if is_relevant:
                relevant_count += 1
                weighted_sum += weight
        except Exception:
            weighted_sum += weight * 0.5  # Assume partial relevance on failure

    return round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0


async def _score_context_recall(question: str, answer: str, context: str) -> float:
    """
    Context Recall: Does the retrieved context cover the claims in the answer?
    LLM checks what fraction of answer claims are supported by context.
    """
    if not answer.strip() or not context.strip():
        return 0.0

    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an evaluation judge. Determine what fraction of the claims
in the answer can be attributed to the provided context.

Rate from 0.0 (no claims supported) to 1.0 (all claims supported).
Respond with ONLY a decimal number between 0.0 and 1.0."""),
        ("human", """Context: {context}

Answer: {answer}

What fraction of the answer's claims are supported by the context?"""),
    ])
    chain = prompt | llm | StrOutputParser()

    try:
        result = await chain.ainvoke({
            "context": context[:3000],
            "answer": answer[:1500],
        })
        score = float(re.search(r"[\d.]+", result.strip()).group())
        return max(0.0, min(1.0, score))
    except Exception as exc:
        log.warning("Context recall scoring failed: %s", exc)
        return 0.5


def _compute_token_efficiency(answer: str, context: str) -> float:
    """
    Token Efficiency: How concise is the answer relative to the context?
    Lower is better (less repetition). Normalized to 0-1 scale.
    """
    if not context:
        return 0.0
    ratio = len(answer) / max(len(context), 1)
    # Ideal ratio is ~0.1-0.3 (concise summary)
    # Score: 1.0 at ratio=0.2, decreasing as ratio moves away
    if ratio <= 0.3:
        return min(1.0, ratio / 0.3)
    else:
        return max(0.0, 1.0 - (ratio - 0.3) / 0.7)


def _compute_source_coverage(sources: list[SourceChunk]) -> int:
    """Count unique source pages referenced."""
    pages = set()
    for s in sources:
        if s.page is not None:
            pages.add(s.page)
    return len(pages)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

async def evaluate_response(
    question: str,
    response: ChatResponse,
) -> EvaluationMetrics:
    """
    Evaluate a single RAG response across all metrics.
    Returns EvaluationMetrics.
    """
    # Build context text from sources
    context = "\n\n".join(s.text for s in response.sources)

    # Run LLM-based evaluations concurrently
    faithfulness_task = asyncio.create_task(
        _score_faithfulness(response.answer, context)
    )
    relevancy_task = asyncio.create_task(
        _score_answer_relevancy(question, response.answer)
    )
    precision_task = asyncio.create_task(
        _score_context_precision(question, response.sources)
    )
    recall_task = asyncio.create_task(
        _score_context_recall(question, response.answer, context)
    )

    # Await all LLM evaluations
    faithfulness = await faithfulness_task
    relevancy = await relevancy_task
    precision = await precision_task
    recall = await recall_task

    # Compute non-LLM metrics
    token_eff = _compute_token_efficiency(response.answer, context)
    source_cov = _compute_source_coverage(response.sources)

    metrics = EvaluationMetrics(
        faithfulness=round(faithfulness, 3),
        answer_relevancy=round(relevancy, 3),
        context_precision=round(precision, 3),
        context_recall=round(recall, 3),
        response_time_ms=response.response_time_ms,
        token_efficiency=round(token_eff, 3),
        source_coverage=source_cov,
    )

    log.info(
        "Evaluation [%s]: faith=%.2f rel=%.2f prec=%.2f rec=%.2f time=%.0fms",
        response.rag_type,
        metrics.faithfulness,
        metrics.answer_relevancy,
        metrics.context_precision,
        metrics.context_recall,
        metrics.response_time_ms,
    )

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  COMPARISON FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def _determine_winner(
    trad_metrics: EvaluationMetrics,
    agent_metrics: EvaluationMetrics,
) -> str:
    """Determine the overall winner based on weighted metric scores."""
    weights = {
        "faithfulness": 0.25,
        "answer_relevancy": 0.25,
        "context_precision": 0.15,
        "context_recall": 0.15,
        "token_efficiency": 0.10,
        "response_time": 0.10,
    }

    # Normalize response time (lower is better) — cap at 30s
    trad_time_score = max(0, 1.0 - trad_metrics.response_time_ms / 30000)
    agent_time_score = max(0, 1.0 - agent_metrics.response_time_ms / 30000)

    trad_score = (
        trad_metrics.faithfulness * weights["faithfulness"]
        + trad_metrics.answer_relevancy * weights["answer_relevancy"]
        + trad_metrics.context_precision * weights["context_precision"]
        + trad_metrics.context_recall * weights["context_recall"]
        + trad_metrics.token_efficiency * weights["token_efficiency"]
        + trad_time_score * weights["response_time"]
    )

    agent_score = (
        agent_metrics.faithfulness * weights["faithfulness"]
        + agent_metrics.answer_relevancy * weights["answer_relevancy"]
        + agent_metrics.context_precision * weights["context_precision"]
        + agent_metrics.context_recall * weights["context_recall"]
        + agent_metrics.token_efficiency * weights["token_efficiency"]
        + agent_time_score * weights["response_time"]
    )

    diff = agent_score - trad_score
    if abs(diff) < 0.05:
        return "tie"
    return "agentic" if diff > 0 else "traditional"


async def compare_pipelines(question: str) -> CompareResponse:
    """
    Run both Traditional and Agentic RAG on the same question,
    evaluate both, and return a comparison.
    """
    from app.chat.traditional_rag import traditional_rag_response
    from app.chat.agentic_rag import agentic_rag_response

    # Run both pipelines concurrently
    trad_task = asyncio.create_task(traditional_rag_response(question))
    agent_task = asyncio.create_task(agentic_rag_response(question))

    trad_response = await trad_task
    agent_response = await agent_task

    # Evaluate both responses concurrently
    trad_eval_task = asyncio.create_task(evaluate_response(question, trad_response))
    agent_eval_task = asyncio.create_task(evaluate_response(question, agent_response))

    trad_metrics = await trad_eval_task
    agent_metrics = await agent_eval_task

    # Determine winner
    winner = _determine_winner(trad_metrics, agent_metrics)

    # Build summary
    summary = (
        f"{'Agentic RAG' if winner == 'agentic' else 'Traditional RAG' if winner == 'traditional' else 'Both pipelines'} "
        f"{'performed better' if winner != 'tie' else 'performed equally'} for this query. "
        f"Traditional: {trad_metrics.response_time_ms:.0f}ms, "
        f"Agentic: {agent_metrics.response_time_ms:.0f}ms."
    )

    # Convert to RAGResult
    trad_result = RAGResult(
        answer=trad_response.answer,
        sources=trad_response.sources,
        graph_context=trad_response.graph_context,
        response_time_ms=trad_response.response_time_ms,
        steps_taken=trad_response.steps_taken,
        rag_type="traditional",
    )

    agent_result = RAGResult(
        answer=agent_response.answer,
        sources=agent_response.sources,
        graph_context=agent_response.graph_context,
        response_time_ms=agent_response.response_time_ms,
        steps_taken=agent_response.steps_taken,
        rag_type="agentic",
    )

    comparison = CompareResponse(
        question=question,
        traditional=trad_result,
        agentic=agent_result,
        traditional_metrics=trad_metrics,
        agentic_metrics=agent_metrics,
        winner=winner,
        summary=summary,
    )

    # Store in history
    _comparison_history.append(comparison.model_dump())

    log.info("Comparison complete: winner=%s", winner)
    return comparison


def get_comparison_history() -> list[dict[str, Any]]:
    """Return all past comparisons."""
    return list(_comparison_history)


def clear_comparison_history() -> None:
    """Clear the comparison history."""
    _comparison_history.clear()
