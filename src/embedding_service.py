# File: src/embedding_service.py
"""Text embedding generation service."""

from __future__ import annotations

import os

from fastembed import TextEmbedding


DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text list using configured open-source model."""
    model_name = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    embedder = TextEmbedding(model_name=model_name)
    return [list(vec) for vec in embedder.embed(texts)]
