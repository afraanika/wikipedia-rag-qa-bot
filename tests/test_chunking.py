from ragbot.chunking import chunk_text


def test_empty_string_returns_no_chunks():
    assert chunk_text("") == []


def test_text_shorter_than_chunk_size_returns_one_chunk():
    text = "one two three"
    assert chunk_text(text, chunk_size=200, overlap=40) == [text]


def test_exact_multiple_length_splits_cleanly():
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=10, overlap=0)
    assert chunks == [" ".join(words[0:10]), " ".join(words[10:20])]


def test_overlap_repeats_words_between_adjacent_chunks():
    words = [f"w{i}" for i in range(15)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=10, overlap=4)
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-4:] == second_words[:4]


def test_no_chunk_splits_a_word():
    text = "alpha beta gamma delta epsilon"
    chunks = chunk_text(text, chunk_size=2, overlap=0)
    for chunk in chunks:
        for word in chunk.split():
            assert word in text.split()
