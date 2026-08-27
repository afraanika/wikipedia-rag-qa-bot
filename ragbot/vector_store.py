"""This module stores and retrieves chunk vectors with Chroma."""

import re
from dataclasses import dataclass

import chromadb

from ragbot import config

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    return _client


def get_collection() -> chromadb.Collection:
    """This function returns the collection that stores article chunks."""
    return _get_client().get_or_create_collection(config.CHROMA_COLLECTION_NAME)


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def replace_article_chunks(
    title: str, url: str, chunks: list[str], vectors: list[list[float]]
) -> None:
    """This function replaces all stored chunks for one article title.

    This function deletes existing chunks for the title first, then it
    adds the new chunks. This keeps re-ingestion of one article correct
    even when the chunk count changes between runs.
    """
    collection = get_collection()
    collection.delete(where={"title": title})

    if not chunks:
        return

    slug = _slug(title)
    ids = [f"{slug}-{index}" for index in range(len(chunks))]
    metadatas = [
        {"title": title, "url": url, "chunk_index": index} for index in range(len(chunks))
    ]
    collection.add(ids=ids, embeddings=vectors, documents=chunks, metadatas=metadatas)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    title: str
    url: str
    chunk_index: int


def query_top_k(query_vector: list[float], top_k: int = config.TOP_K) -> list[RetrievedChunk]:
    """This function returns the top_k stored chunks closest to the query vector."""
    collection = get_collection()
    result = collection.query(query_embeddings=[query_vector], n_results=top_k)

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    return [
        RetrievedChunk(
            text=document,
            title=metadata["title"],
            url=metadata["url"],
            chunk_index=metadata["chunk_index"],
        )
        for document, metadata in zip(documents, metadatas)
    ]


def is_empty() -> bool:
    """This function returns true when the collection has no stored chunks."""
    return get_collection().count() == 0
