"""This module fetches article text from Wikipedia."""

from dataclasses import dataclass

import wikipedia

from ragbot.errors import AmbiguousTitleError, ArticleNotFoundError

wikipedia.set_rate_limiting(True)
wikipedia.set_user_agent(
    "domain-rag-qa-bot/0.1 (local RAG QA bot project; run for personal, non-commercial use)"
)


@dataclass(frozen=True)
class WikipediaArticle:
    title: str
    url: str
    content: str


def fetch_article(title: str) -> WikipediaArticle:
    """This function retrieves one Wikipedia article by title.

    This function raises ArticleNotFoundError when no article matches the
    title. This function raises AmbiguousTitleError when the title matches
    more than one article.
    """
    try:
        page = wikipedia.page(title, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as error:
        raise AmbiguousTitleError(title, list(error.options)) from error
    except wikipedia.exceptions.PageError as error:
        raise ArticleNotFoundError(f"No Wikipedia article matches '{title}'.") from error

    return WikipediaArticle(title=page.title, url=page.url, content=page.content)
