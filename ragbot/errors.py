"""This module defines the error types for the bot."""


class RagBotError(Exception):
    """This is the base error type for the bot."""


class ArticleNotFoundError(RagBotError):
    """This error means no Wikipedia article matches the title."""


class AmbiguousTitleError(RagBotError):
    """This error means the Wikipedia title matches more than one article."""

    def __init__(self, title: str, options: list[str]):
        self.title = title
        self.options = options
        super().__init__(f"The title '{title}' matches more than one article.")


class OllamaUnavailableError(RagBotError):
    """This error means the bot could not reach the local Ollama server."""


class EmptyKnowledgeBaseError(RagBotError):
    """This error means the vector store has no stored chunks yet."""
