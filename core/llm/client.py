"""Provider-agnostic LLM client.

Every LLM call in the framework goes through ``LLMClient`` so that:

* prompts and completions are cached on disk for reproducibility,
* retries and timeouts are handled centrally,
* swapping OpenAI → Azure → a local model is a one-line config change.
"""
from __future__ import annotations

import abc
from collections.abc import Iterable
from typing import Any


class LLMClient(abc.ABC):
    """Common interface for chat-style LLM providers."""

    @abc.abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Return a single completion string."""
        raise NotImplementedError

    @abc.abstractmethod
    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a parsed JSON object. Implementations must validate
        the response against the requested schema if one is provided.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Return one dense vector per input text."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete reference implementations (stubs)
# ---------------------------------------------------------------------------

class OpenAIClient(LLMClient):
    """Thin wrapper over the official ``openai`` SDK ≥ 1.30."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        request_timeout_s: int = 60,
        max_retries: int = 3,
        cache_dir: str | None = None,
    ) -> None:
        self._model = model
        self._embedding_model = embedding_model
        self._request_timeout_s = request_timeout_s
        self._max_retries = max_retries
        self._cache_dir = cache_dir

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        raise NotImplementedError


class CachedLLMClient(LLMClient):
    """Decorator: caches ``complete`` / ``complete_json`` results on disk."""

    def __init__(self, inner: LLMClient, *, cache_dir: str) -> None:
        self._inner = inner
        self._cache_dir = cache_dir

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        # Embedding caching delegates to the inner client — caching here
        # would be a different key strategy and is left to the embedder.
        return self._inner.embed(texts, model=model)
