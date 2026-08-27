"""This module holds the configuration values for the bot.

Every value here has a default. You can set an environment variable to
change a value without you having to edit this file.
"""

import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
CHROMA_COLLECTION_NAME = "wikipedia_articles"
TOP_K = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "200"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "40"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
