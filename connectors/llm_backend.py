"""Provider-agnostic LLM abstraction with structured-output support.

The framework talks to *one* interface — :class:`LLMBackend` — regardless
of whether the model is served by OpenAI, Azure-OpenAI, an OpenAI-compatible
proxy, or a local Ollama daemon. Every concrete backend must implement:

* ``chat(messages, *, model, temperature, max_tokens, response_format)``
* ``embed(texts, *, model)`` (optional — may raise ``NotImplementedError``)

Structured outputs
------------------

Two strategies are supported, picked per-request via the ``response_format``
keyword:

* ``{"type": "json_schema", "json_schema": {...}}`` — OpenAI's *Structured
  Outputs* mode, which guarantees the response parses against a JSON
  Schema. This is the recommended path for Pydantic-validated outputs.
* ``{"type": "json_object"}`` — the older JSON Mode. The model is
  instructed (in the system message) to emit JSON only; the caller
  validates against a Pydantic model.

Backends declare which modes they support. OpenAI supports both;
Ollama (via ``format="json"``) supports JSON Mode and *partial* JSON
Schema since 0.5.x. The :class:`LLMBackend` wrapper chooses the
highest-fidelity mode the backend advertises and downgrades on
``NotImplementedError``.

Caching
-------

A :class:`DiskCache` is wired in by default. Identical ``(model, messages,
temperature, max_tokens)`` tuples return the cached completion, which
makes prompt iteration and CI deterministic.
"""
from __future__ import annotations

import abc
import hashlib
import json
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

T = TypeVar("T", bound=BaseModel)

Role = Literal["system", "user", "assistant"]


# ---------------------------------------------------------------------------
# Message / request types
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """A single chat message — intentionally tiny."""

    model_config = ConfigDict(extra="forbid")
    role: Role
    content: str


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LLMError(RuntimeError):
    """Base class for backend-level failures."""


class SchemaValidationError(LLMError):
    """Raised when a structured-output completion fails Pydantic validation."""


class BackendUnavailable(LLMError):
    """Raised when no backend can serve the request (e.g. Ollama is down)."""


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class LLMBackend(abc.ABC):
    """Abstract base class for any chat-completion backend."""

    name: str

    @abc.abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
        stop: Iterable[str] | None = None,
        timeout_s: float = 60.0,
    ) -> str:
        """Return the assistant message content as a plain string."""
        raise NotImplementedError

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        """Default: backends are not required to provide embeddings."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement embeddings"
        )

    def supports_json_schema(self) -> bool:
        """Whether the backend honours ``response_format={"type":"json_schema"}``."""
        return False

    def supports_json_mode(self) -> bool:
        """Whether the backend honours ``response_format={"type":"json_object"}``."""
        return False


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible backend
# ---------------------------------------------------------------------------

class OpenAIBackend(LLMBackend):
    """Backend for any service speaking the OpenAI Chat Completions API.

    Tested against:
        * OpenAI (api.openai.com)
        * Azure OpenAI (set ``base_url`` to your deployment URL)
        * vLLM, LM-Studio, OpenRouter (set ``base_url`` + ``api_key``)
    """

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        default_model: str = "gpt-4o-mini",
        default_embedding_model: str = "text-embedding-3-small",
        request_timeout_s: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        # Import lazily so the rest of the module works without `openai`
        # installed (useful for docs builds and for Ollama-only users).
        from openai import OpenAI  # type: ignore

        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
            organization=organization or os.environ.get("OPENAI_ORG") or None,
            timeout=request_timeout_s,
            max_retries=max_retries,
        )
        self._default_model = default_model
        self._default_embedding_model = default_embedding_model
        self._timeout = request_timeout_s

    # --- Capabilities -------------------------------------------------------

    def supports_json_schema(self) -> bool:  # noqa: D401
        return True

    def supports_json_mode(self) -> bool:
        return True

    # --- Chat ---------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
        stop: Iterable[str] | None = None,
        timeout_s: float | None = None,
    ) -> str:
        model = model or self._default_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if stop is not None:
            kwargs["stop"] = list(stop)
        if timeout_s is not None:
            kwargs["timeout"] = timeout_s

        # The OpenAI Python client raises ``openai.APIError`` subclasses on
        # failure; we let those propagate. ``max_retries`` is set on the
        # client so transient 429/5xx are retried automatically.
        resp = self._client.chat.completions.create(**kwargs)
        if not resp.choices:
            raise LLMError(f"OpenAI returned no choices (model={model})")
        return resp.choices[0].message.content or ""

    # --- Embeddings ---------------------------------------------------------

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        model = model or self._default_embedding_model
        # Batching is handled by the SDK; we just pass the list through.
        resp = self._client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]


# ---------------------------------------------------------------------------
# Ollama backend (local models)
# ---------------------------------------------------------------------------

class OllamaBackend(LLMBackend):
    """Backend for a local Ollama daemon.

    Talks directly to ``http://localhost:11434`` via ``httpx`` so we don't
    pin to the (sometimes stale) ``ollama`` PyPI package. Two Ollama
    features are used:

    * ``/api/chat`` with ``format="json"`` for JSON Mode
    * ``/api/embed`` for embeddings
    """

    name = "ollama"

    def __init__(
        self,
        *,
        host: str = "http://localhost:11434",
        default_model: str = "llama3.1:8b-instruct-q5_K_M",
        default_embedding_model: str = "nomic-embed-text",
        request_timeout_s: float = 120.0,
    ) -> None:
        self._host = host.rstrip("/")
        self._default_model = default_model
        self._default_embedding_model = default_embedding_model
        self._timeout = request_timeout_s

    # --- Capabilities -------------------------------------------------------

    def supports_json_schema(self) -> bool:
        return False  # Ollama's schema support is partial; stay safe.

    def supports_json_mode(self) -> bool:
        return True

    # --- Chat ---------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
        stop: Iterable[str] | None = None,
        timeout_s: float | None = None,
    ) -> str:
        model = model or self._default_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        if stop is not None:
            payload["options"]["stop"] = list(stop)
        if response_format is not None:
            # Map OpenAI's response_format to Ollama's ``format`` field.
            # We only handle the JSON shapes; everything else is ignored.
            fmt_type = response_format.get("type")
            if fmt_type in ("json_object", "json_schema"):
                payload["format"] = "json"
            elif fmt_type is not None:
                # Caller asked for something exotic — pass it through.
                payload["format"] = response_format

        try:
            with httpx.Client(timeout=timeout_s or self._timeout) as client:
                r = client.post(f"{self._host}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as e:
            raise BackendUnavailable(f"Ollama chat failed: {e}") from e

        try:
            return data["message"]["content"]
        except KeyError as e:  # pragma: no cover - defensive
            raise LLMError(f"Unexpected Ollama response shape: {data!r}") from e

    # --- Embeddings ---------------------------------------------------------

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        model = model or self._default_embedding_model
        out: list[list[float]] = []
        try:
            with httpx.Client(timeout=self._timeout) as client:
                for t in texts:
                    r = client.post(
                        f"{self._host}/api/embed",
                        json={"model": model, "input": t},
                    )
                    r.raise_for_status()
                    body = r.json()
                    # Ollama returns ``{"embeddings": [[...]]}`` (v0.3+);
                    # older versions return ``{"embedding": [...]}`` per
                    # text. Handle both.
                    if "embeddings" in body and body["embeddings"]:
                        out.append(body["embeddings"][0])
                    elif "embedding" in body:
                        out.append(body["embedding"])
                    else:  # pragma: no cover - defensive
                        raise LLMError(f"Unexpected Ollama embed response: {body!r}")
        except httpx.HTTPError as e:
            raise BackendUnavailable(f"Ollama embed failed: {e}") from e
        return out

    # --- Health check -------------------------------------------------------

    def ping(self) -> bool:
        """Return ``True`` if the Ollama daemon responds to ``/api/tags``."""
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self._host}/api/tags")
                return r.status_code == 200
        except httpx.HTTPError:
            return False


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

class DiskCache:
    """JSON-file cache keyed by SHA-256 of the request fingerprint.

    * Files live under ``root``.
    * Safe to share across processes (filenames include a hash, so writes
      never collide).
    """

    def __init__(self, root: str | Path, *, enabled: bool = True) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._enabled = enabled

    @staticmethod
    def fingerprint(payload: dict[str, Any]) -> str:
        """Stable hash of a JSON-serialisable payload."""
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def get(self, key: str) -> str | None:
        if not self._enabled:
            return None
        p = self._root / f"{key}.json"
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except OSError:
                return None
        return None

    def put(self, key: str, value: str) -> None:
        if not self._enabled:
            return
        p = self._root / f"{key}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(value, encoding="utf-8")
        tmp.replace(p)


# ---------------------------------------------------------------------------
# High-level wrapper: structured output + cache + retry
# ---------------------------------------------------------------------------

class StructuredLLM:
    """Façade that combines a backend, an on-disk cache, and Pydantic binding.

    Usage
    -----

    .. code-block:: python

        backend = OpenAIBackend()                      # or OllamaBackend()
        llm = StructuredLLM(backend, cache_dir="data/cache/llm")

        class Answer(BaseModel):
            concept: str
            definition: str

        out: Answer = llm.chat_struct(
            messages=[ChatMessage(role="user", content="Define 'cache'.")],
            model="gpt-4o-mini",
            schema_model=Answer,
        )
    """

    def __init__(
        self,
        backend: LLMBackend,
        *,
        cache_dir: str | Path | None = "data/cache/llm",
        cache_enabled: bool = True,
    ) -> None:
        self._backend = backend
        self._cache = DiskCache(cache_dir, enabled=cache_enabled) if cache_dir else None

    # --- Properties ---------------------------------------------------------

    @property
    def backend(self) -> LLMBackend:
        return self._backend

    @property
    def supports_json_schema(self) -> bool:
        return self._backend.supports_json_schema()

    # --- Plain chat ---------------------------------------------------------

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
        stop: Iterable[str] | None = None,
        timeout_s: float | None = None,
    ) -> str:
        key = self._cache_key(
            model=model or self._default_model(model),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            stop=list(stop) if stop is not None else None,
        )
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        text = self._backend.chat(
            messages,
            model=model or self._default_model(model),
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            stop=stop,
            timeout_s=timeout_s,
        )
        if self._cache is not None:
            self._cache.put(key, text)
        return text

    # --- Structured (Pydantic-bound) ----------------------------------------

    def chat_struct(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema_model: type[T],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_repair_attempts: int = 1,
    ) -> T:
        """Chat and parse the result into ``schema_model``.

        The backend is asked for the highest-fidelity structured format it
        supports (``json_schema`` > ``json_object`` > plain). If the
        response still fails Pydantic validation, up to
        ``max_repair_attempts`` repair rounds are attempted, each
        appending a corrective user message.
        """
        from core.models import OntologySchema  # local import to avoid cycle

        # Dispatch on the *actual* schema model — we only need the JSON
        # Schema when the backend advertises ``json_schema`` support.
        if schema_model is OntologySchema:
            schema_for_response = _ontology_schema_json_schema()
        else:
            schema_for_response = schema_model.model_json_schema()

        response_format = self._best_response_format(schema_for_response)
        last_error: Exception | None = None
        msgs: list[ChatMessage] = list(messages)
        for attempt in range(max_repair_attempts + 1):
            text = self.chat(
                msgs,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            try:
                return _parse_structured(text, schema_model)
            except (ValidationError, ValueError) as e:
                last_error = e
                if attempt >= max_repair_attempts:
                    break
                # Append a corrective message and try again.
                msgs = msgs + [
                    ChatMessage(
                        role="user",
                        content=(
                            "Your previous reply failed JSON validation. "
                            f"Error: {e}. Respond with a single JSON object "
                            "that strictly matches the schema — no prose, no "
                            "markdown fences."
                        ),
                    )
                ]
        raise SchemaValidationError(
            f"Failed to parse LLM output into {schema_model.__name__}: {last_error}"
        )

    # --- Embeddings ---------------------------------------------------------

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        return self._backend.embed(texts, model=model or self._default_embedding_model(model))

    # --- Internals ----------------------------------------------------------

    def _best_response_format(self, json_schema: dict[str, Any]) -> dict[str, Any] | None:
        """Pick the most capable response format the backend advertises."""
        if self._backend.supports_json_schema():
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "structured_output"),
                    "schema": json_schema,
                    "strict": True,
                },
            }
        if self._backend.supports_json_mode():
            return {"type": "json_object"}
        return None  # best effort

    def _default_model(self, override: str | None) -> str:  # pragma: no cover - trivial
        if override is not None:
            return override
        # Backends may store a default — best-effort duck-typing.
        for attr in ("_default_model", "default_model"):
            if hasattr(self._backend, attr):
                return getattr(self._backend, attr)
        raise ValueError("No model specified and backend has no default")

    def _default_embedding_model(self, override: str | None) -> str:
        if override is not None:
            return override
        for attr in ("_default_embedding_model", "default_embedding_model"):
            if hasattr(self._backend, attr):
                return getattr(self._backend, attr)
        raise ValueError("No embedding model specified and backend has no default")

    def _cache_key(self, **parts: Any) -> str:
        return DiskCache.fingerprint(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_structured(text: str, schema_model: type[T]) -> T:
    """Parse ``text`` into ``schema_model`` after stripping code-fence noise."""
    cleaned = text.strip()
    # Some models wrap the JSON in ```json ... ``` even when asked not to.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON: {e}\n---\n{text}\n---") from e
    # Pydantic v2 — strict model_validate catches type mismatches the
    # JSON Schema layer may have missed.
    return schema_model.model_validate(obj)


# ---------------------------------------------------------------------------
# JSON Schema for OntologySchema (special-cased)
# ---------------------------------------------------------------------------

def _ontology_schema_json_schema() -> dict[str, Any]:
    """A JSON Schema description of ``OntologySchema`` suitable for
    OpenAI's ``response_format.json_schema`` field.

    This is hand-written because (a) Pydantic's ``model_json_schema()``
    emits a few fields that the OpenAI schema strict-mode rejects, and
    (b) we want to keep the prompt in this module independent of the
    ``core.models`` import cycle.
    """
    return {
        "title": "OntologySchema",
        "type": "object",
        "additionalProperties": False,
        "required": ["entity_types", "relations"],
        "properties": {
            "entity_types": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "description"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "description": {"type": "string"},
                        "parent_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Names of parent classes (inheritance).",
                        },
                        "aliases": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "description"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "description": {"type": "string"},
                        "domain": {"type": ["string", "null"], "description": "Class name or null."},
                        "range": {"type": ["string", "null"], "description": "Class name or null."},
                        "direction": {
                            "type": "string",
                            "enum": ["directed", "undirected", "bidirectional"],
                        },
                        "symmetric": {"type": "boolean"},
                        "transitive": {"type": "boolean"},
                    },
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def default_backend(
    *,
    prefer: Literal["openai", "ollama", "auto"] = "auto",
) -> LLMBackend:
    """Return a sensible default backend.

    Resolution order when ``prefer='auto'``:
        1. ``OPENAI_API_KEY`` present  → :class:`OpenAIBackend`
        2. Ollama daemon responding    → :class:`OllamaBackend`
        3. raise :class:`BackendUnavailable`
    """
    if prefer == "openai" or (prefer == "auto" and os.environ.get("OPENAI_API_KEY")):
        return OpenAIBackend()
    if prefer == "ollama":
        return OllamaBackend()
    # auto fallback
    ollama = OllamaBackend()
    if ollama.ping():
        return ollama
    raise BackendUnavailable(
        "No LLM backend available. Set OPENAI_API_KEY or start an Ollama daemon."
    )


__all__ = [
    "ChatMessage",
    "LLMBackend",
    "OpenAIBackend",
    "OllamaBackend",
    "StructuredLLM",
    "DiskCache",
    "LLMError",
    "SchemaValidationError",
    "BackendUnavailable",
    "default_backend",
]
