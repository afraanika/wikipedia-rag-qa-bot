"""This module splits article text into overlapping chunks."""


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """This function splits text into overlapping word-based chunks.

    This function returns a list of chunk strings in order. Each chunk has
    up to `chunk_size` words. Adjacent chunks share `overlap` words, so the
    bot does not lose context at a chunk boundary. This function returns an
    empty list when the text has no words.
    """
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks
