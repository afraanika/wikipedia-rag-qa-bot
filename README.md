# domain-rag-qa-bot

This project is a question-answering bot for a specific domain. The bot
uses retrieval-augmented generation (RAG). It finds relevant text from a
knowledge source, then it gives that text to a language model to form an
answer.

The bot ingests Wikipedia articles that you name, then it answers
questions from that stored text only.

## Setup

1. Create a virtual environment: `python -m venv .venv`.
2. Activate it: `source .venv/bin/activate`.
3. Install the dependencies: `pip install -r requirements.txt`. This step
   downloads `torch`, which is large. The first install can take several
   minutes.
4. Install Ollama from `ollama.com`, then pull a model: `ollama pull
   llama3`. This runs the language model on your machine, at no cost.

## Usage

1. Ingest one or more Wikipedia articles: `python ingest.py "Quantum
   computing"`.
2. Ask a question: `python query.py "What is quantum computing?"`.
3. Run `python query.py` with no argument for an interactive prompt.
   Type `exit` to stop.

## Documentation style

This project writes documentation with Simplified Technical English
(STE). The word choices and sentence rules are in `CONTRIBUTING.md`.
