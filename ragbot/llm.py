"""This module builds prompts and calls a local Ollama model."""

import ollama

from ragbot import config
from ragbot.errors import OllamaUnavailableError
from ragbot.vector_store import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the question only from "
    "the context below. If the context does not have enough information, "
    "say so. Do not use knowledge outside the context. Name the source "
    "title for each fact you use."
)

_client: ollama.Client | None = None


def _get_client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=config.OLLAMA_HOST)
    return _client


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """This function builds the user prompt from retrieved context chunks.

    This function labels each chunk with its source title and chunk index.
    A reader can trace an answer back to a chunk with this label.
    """
    context_blocks = [
        f"[Source: {chunk.title}, chunk {chunk.chunk_index}]\n{chunk.text}" for chunk in chunks
    ]
    context = "\n\n".join(context_blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """This function asks the local model to answer a question from context."""
    client = _get_client()
    prompt = build_prompt(question, chunks)

    try:
        response = client.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"num_predict": config.MAX_TOKENS},
        )
    except ollama.ResponseError as error:
        raise OllamaUnavailableError(
            f"Ollama returned an error for model '{config.OLLAMA_MODEL}'. "
            f"Run 'ollama pull {config.OLLAMA_MODEL}' to download it."
        ) from error
    except ConnectionError as error:
        raise OllamaUnavailableError(
            "The bot could not connect to Ollama. Start it with 'ollama serve'."
        ) from error

    return response["message"]["content"]
