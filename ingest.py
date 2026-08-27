"""This script fetches Wikipedia articles and stores them for the bot.

Usage: python ingest.py "Topic A" "Topic B"
"""

import sys

from ragbot import chunking, embeddings, vector_store, wikipedia_source
from ragbot.errors import AmbiguousTitleError, ArticleNotFoundError


def ingest_title(title: str) -> bool:
    """This function ingests one Wikipedia title. It returns true on success."""
    try:
        article = wikipedia_source.fetch_article(title)
    except ArticleNotFoundError:
        print(f"No Wikipedia article matches '{title}'. Skip this title.")
        return False
    except AmbiguousTitleError as error:
        options = ", ".join(error.options[:5])
        print(f"The title '{title}' is ambiguous. Options: {options}.")
        print("Run again with a more specific title.")
        return False

    chunks = chunking.chunk_text(article.content)
    if not chunks:
        print(f"The article '{article.title}' has no text to store. Skip this title.")
        return False

    vectors = embeddings.embed_texts(chunks)
    vector_store.replace_article_chunks(article.title, article.url, chunks, vectors)
    print(f"Stored {len(chunks)} chunks for '{article.title}'.")
    return True


def main() -> None:
    titles = sys.argv[1:]
    if not titles:
        print('Give at least one Wikipedia title. Example: python ingest.py "Quantum computing"')
        sys.exit(1)

    results = [ingest_title(title) for title in titles]
    stored = sum(results)
    print(f"Stored {stored} of {len(titles)} articles.")


if __name__ == "__main__":
    main()
