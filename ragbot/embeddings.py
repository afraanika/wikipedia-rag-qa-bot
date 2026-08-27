"""This module turns text into vectors.

This module loads the embedding model once, and it reuses that model for
every call. Ingestion and query both use this module, so a chunk and a
question end up in the same vector space.
"""

from sentence_transformers import SentenceTransformer

from ragbot import config

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """This function returns one vector for each text in the list."""
    vectors = _get_model().encode(texts, convert_to_numpy=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """This function returns one vector for a single query text."""
    return embed_texts([text])[0]
