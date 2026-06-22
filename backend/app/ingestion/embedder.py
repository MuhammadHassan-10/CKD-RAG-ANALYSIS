"""
Sentence-transformers embedding module.
Provides a singleton model instance and batch-encoding helpers.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the sentence-transformer model once and cache it."""
    log.info("Loading embedding model: %s …", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    log.info("Embedding model loaded ✓  (dim=%d)", model.get_sentence_embedding_dimension())
    return model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Encode a list of texts into dense vectors.

    Returns a list of float-lists, each of length EMBEDDING_DIMENSION (384).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,
    )
    return [emb.tolist() for emb in embeddings]


def embed_single(text: str) -> list[float]:
    """Encode a single text string."""
    return embed_texts([text])[0]
