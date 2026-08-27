from ragbot.llm import build_prompt
from ragbot.vector_store import RetrievedChunk


def test_prompt_includes_each_chunk_text_and_title():
    chunks = [
        RetrievedChunk(text="Quantum bits store more than one state.", title="Quantum computing", url="https://example.org/qc", chunk_index=0),
        RetrievedChunk(text="Qubits can become entangled.", title="Quantum computing", url="https://example.org/qc", chunk_index=1),
    ]
    prompt = build_prompt("What is a qubit?", chunks)

    for chunk in chunks:
        assert chunk.text in prompt
        assert chunk.title in prompt


def test_prompt_includes_question_once():
    chunks = [RetrievedChunk(text="Some fact.", title="Topic", url="https://example.org/t", chunk_index=0)]
    question = "What is the topic about?"
    prompt = build_prompt(question, chunks)

    assert prompt.count(question) == 1
