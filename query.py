"""This script answers a question from the stored Wikipedia chunks.

Usage: python query.py "What is quantum computing?"
Usage with no argument: python query.py, then type questions at the prompt.
"""

import sys

from ragbot import config, embeddings, llm, vector_store
from ragbot.errors import EmptyKnowledgeBaseError, OllamaUnavailableError


def answer_question(question: str) -> None:
    if vector_store.is_empty():
        raise EmptyKnowledgeBaseError(
            "The knowledge base is empty. Run ingest.py with a Wikipedia title first."
        )

    query_vector = embeddings.embed_query(question)
    chunks = vector_store.query_top_k(query_vector, top_k=config.TOP_K)

    try:
        answer = llm.generate_answer(question, chunks)
    except OllamaUnavailableError as error:
        print(str(error))
        return

    print(answer)

    sources = []
    for chunk in chunks:
        source = (chunk.title, chunk.url)
        if source not in sources:
            sources.append(source)
    print("\nSources:")
    for title, url in sources:
        print(f"- {title} ({url})")


def main() -> None:
    args = sys.argv[1:]
    if args:
        try:
            answer_question(" ".join(args))
        except EmptyKnowledgeBaseError as error:
            print(str(error))
        return

    print("Type a question, or type 'exit' to stop.")
    while True:
        question = input("> ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        try:
            answer_question(question)
        except EmptyKnowledgeBaseError as error:
            print(str(error))
            break


if __name__ == "__main__":
    main()
