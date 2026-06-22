"""
Embedding module using ChromaDB's built-in ONNX model.
This replaces sentence-transformers (PyTorch) to drastically reduce memory usage,
allowing the app to run on Render's 512MB Free Tier.
"""
from __future__ import annotations

import logging

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

log = logging.getLogger(__name__)

_ef = None

def _get_ef():
    global _ef
    if _ef is None:
        log.info("Loading lightweight ONNX embedding model...")
        _ef = DefaultEmbeddingFunction()
        log.info("ONNX Embedding model loaded ✓")
    return _ef


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Encode a list of texts into dense vectors using Chroma's default ONNX model.
    """
    if not texts:
        return []
    ef = _get_ef()
    # The DefaultEmbeddingFunction takes a list of strings and returns a list of float-lists
    embeddings = ef(texts)
    # Ensure they are standard python floats
    return [[float(x) for x in emb] for emb in embeddings]


def embed_single(text: str) -> list[float]:
    """Encode a single text string."""
    return embed_texts([text])[0]
