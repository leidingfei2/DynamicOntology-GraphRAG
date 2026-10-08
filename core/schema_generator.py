"""Dynamic ontology (T-Box) construction from unstructured text.

Pipeline
--------

    Documents
        │
        ▼  chunk()                  split into LLM-sized windows
    list[str] (chunks)
        │
        ▼  build()  (per chunk)     one LLM call →  OntologyChunkProposal
        │                           (classes + relations as NAMES, not IDs)
        ▼  merge()                  union across chunks; resolve parent links
        ▼  finalise()               assign stable IDs, build OntologySchema
    OntologySchema

The heavy lifting is delegated to :class:`SchemaGenerator`, which is
constructable with any :class:`StructuredLLM` (OpenAI, Ollama, mock, …).
A :class:`MockLLM` is exported for unit tests so the whole module can be
exercised without a network.
"""
from __future__ import annotations

import abc
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from connectors.llm_backend import (
    ChatMessage,
    LLMError,
    SchemaValidationError,
    StructuredLLM,
)
from core.models import (
    EntityType,
    OntologyRule,
    OntologySchema,
    Relation,
    RelationDirection,
)


# ---------------------------------------------------------------------------
# Interchange models — what the LLM is asked to emit
# ---------------------------------------------------------------------------

class ProposedClass(BaseModel):
    """One class as the LLM names it. No IDs — IDs are assigned post-hoc."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    description: str = Field(default="")
    parent_names: tuple[str, ...] = Field(default_factory=tuple)
    aliases: tuple[str, ...] = Field(default_factory=tuple)


class ProposedRelation(BaseModel):
    """One predicate as the LLM names it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    description: str = Field(default="")
    domain: str | None = None
    range: str | None = None
    direction: RelationDirection = RelationDirection.DIRECTED
    symmetric: bool = False
    transitive: bool = False


class OntologyChunkProposal(BaseModel):
    """A single chunk's contribution to the T-Box."""

    model_config = ConfigDict(extra="forbid")

    entity_types: list[ProposedClass] = Field(default_factory=list)
    relations: list[ProposedRelation] = Field(default_factory=list)
    notes: str = Field(default="", description="Free-form rationale.")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextChunk:
    text: str
    index: int
    source_doc_id: str | None = None


def chunk_text(
    text: str,
    *,
    chunk_size: int = 4000,
    chunk_overlap: int = 400,
    doc_id: str | None = None,
) -> list[TextChunk]:
    """Naive word-based chunking with overlap.

    We avoid heavy NLP dependencies at the schema stage — sentence-aware
    splitting is an easy swap-in but rarely changes the resulting schema.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    words = text.split()
    if not words:
        return []

    step = chunk_size - chunk_overlap
    out: list[TextChunk] = []
    for i, start in enumerate(range(0, len(words), step)):
        piece = words[start : start + chunk_size]
        if not piece:
            break
        out.append(TextChunk(text=" ".join(piece), index=i, source_doc_id=doc_id))
    return out


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior knowledge engineer. Your task is to read a chunk of
domain text and propose a SMALL, PRECISE ontology fragment (T-Box) that
captures the domain concepts and relations present in the chunk.

Strict rules
------------
1. Propose at most {max_classes} classes and at most {max_relations}
   relations for this chunk.
2. Use CamelCase for class names and lowerCamelCase for relation names.
3. Every relation MUST have a clear domain and range when possible; use
   null only if genuinely unknown.
4. Prefer concrete subclasses over generic roots (e.g. "LoadBalancer"
   over "NetworkDevice" when the text is about load balancers).
5. Avoid duplicating the same fact under two different names — if two
   classes look equivalent, keep the more specific one and merge.
6. Output ONLY a JSON object that matches the provided schema — no
   prose, no markdown fences, no comments.
"""


_USER_PROMPT = """\
Domain text (chunk {chunk_index}/{chunk_total}):

\"\"\"
{text}
\"\"\"

Propose the ontology fragment for this chunk. Remember:
- Class names in CamelCase, relation names in lowerCamelCase.
- Inheritance: list parent class NAMES in ``parent_names``.
- Use ``aliases`` for synonyms / abbreviations in the text.
"""


def _build_messages(chunk: TextChunk, *, total: int, max_classes: int, max_relations: int) -> list[ChatMessage]:
    sys_content = _SYSTEM_PROMPT.format(max_classes=max_classes, max_relations=max_relations)
    user_content = _USER_PROMPT.format(
        chunk_index=chunk.index + 1,
        chunk_total=total,
        text=chunk.text,
    )
    return [
        ChatMessage(role="system", content=sys_content),
        ChatMessage(role="user", content=user_content),
    ]


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

@dataclass
class _ClassAcc:
    name: str
    description: str = ""
    parent_names: set[str] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)
    support: int = 0
    # Tentative ID assigned by _stable_id once name is canonical.
    id: str | None = None


@dataclass
class _RelationAcc:
    name: str
    description: str = ""
    domain: str | None = None
    range: str | None = None
    direction: RelationDirection = RelationDirection.DIRECTED
    symmetric: bool = False
    transitive: bool = False
    support: int = 0


def _normalise(name: str) -> str:
    """Lowercase, strip, collapse whitespace — used for class/relation keys."""
    return re.sub(r"\s+", " ", name).strip()


def _stable_id(prefix: str, name: str) -> str:
    """Deterministic ID derived from the canonical name.

    Deriving IDs from names (rather than random UUIDs) means re-running
    the generator on the same text yields the same IDs, which keeps
    downstream caches and GraphML exports stable.
    """
    h = uuid.uuid5(uuid.NAMESPACE_DNS, f"{prefix}:{name.lower()}")
    return f"{prefix}_{h.hex[:12]}"


# ---------------------------------------------------------------------------
# Public façade
# ---------------------------------------------------------------------------

class SchemaGenerator:
    """End-to-end T-Box construction.

    Parameters
    ----------
    llm:
        A :class:`StructuredLLM` (or anything with the same ``chat_struct``
        method). Pass a :class:`MockLLM` for unit tests.
    max_classes, max_relations:
        Per-chunk limits. The final schema may exceed these because
        chunks can propose disjoint subsets that get unioned.
    min_class_support, min_relation_support:
        Threshold on the number of distinct chunks that mention a class /
        relation. Used as a quality filter; lower = more recall, higher =
        more precision.
    chunk_size, chunk_overlap:
        Word-based chunking parameters.
    """

    def __init__(
        self,
        llm: StructuredLLM | Any,
        *,
        max_classes: int = 25,
        max_relations: int = 30,
        min_class_support: int = 1,
        min_relation_support: int = 1,
        chunk_size: int = 4000,
        chunk_overlap: int = 400,
        model: str | None = None,
    ) -> None:
        self._llm = llm
        self._max_classes = max_classes
        self._max_relations = max_relations
        self._min_class_support = max(1, min_class_support)
        self._min_relation_support = max(1, min_relation_support)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._model = model

    # --- Public API ---------------------------------------------------------

    def build(self, documents: Iterable[str | tuple[str, str]]) -> OntologySchema:
        """Build an ``OntologySchema`` from one or many raw documents.

        Each input may be either:

        * a bare string (text), or
        * a ``(doc_id, text)`` tuple.

        Documents are concatenated then chunked; the chunk origin is
        tracked but not currently used in the merge step (it could be
        folded into support counts in a future iteration).
        """
        chunks = self._collect_chunks(documents)
        if not chunks:
            return OntologySchema()
        proposals = [self._propose_for_chunk(c, total=len(chunks)) for c in chunks]
        return self._finalise(proposals)

    def extend(self, schema: OntologySchema, documents: Iterable[str | tuple[str, str]]) -> OntologySchema:
        """Add new classes/relations to an existing schema."""
        new_schema = self.build(documents)
        return _merge_schemas(schema, new_schema)

    # --- Internals ----------------------------------------------------------

    def _collect_chunks(self, documents: Iterable[str | tuple[str, str]]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for i, doc in enumerate(documents):
            if isinstance(doc, tuple):
                doc_id, text = doc
            else:
                doc_id, text = f"doc_{i}", doc
            chunks.extend(
                chunk_text(
                    text,
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._chunk_overlap,
                    doc_id=doc_id,
                )
            )
        return chunks

    def _propose_for_chunk(self, chunk: TextChunk, *, total: int) -> OntologyChunkProposal:
        messages = _build_messages(
            chunk,
            total=total,
            max_classes=self._max_classes,
            max_relations=self._max_relations,
        )
        try:
            return self._llm.chat_struct(
                messages,
                schema_model=OntologyChunkProposal,
                model=self._model,
            )
        except SchemaValidationError as e:
            # Re-raise with chunk context; the orchestrator can decide to
            # skip-and-continue. We deliberately do NOT swallow it here.
            raise SchemaValidationError(
                f"LLM output for chunk {chunk.index} failed validation: {e}"
            ) from e

    def _finalise(self, proposals: list[OntologyChunkProposal]) -> OntologySchema:
        # --- Aggregate classes by canonical name -------------------------
        classes: dict[str, _ClassAcc] = {}
        for prop in proposals:
            for c in prop.entity_types:
                key = _normalise(c.name)
                if not key:
                    continue
                acc = classes.get(key)
                if acc is None:
                    acc = _ClassAcc(name=c.name.strip())
                    classes[key] = acc
                if c.description and not acc.description:
                    acc.description = c.description
                acc.parent_names.update(p.strip() for p in c.parent_names if p.strip())
                acc.aliases.update(a.strip() for a in c.aliases if a.strip())
                acc.support += 1

        # --- Aggregate relations by canonical name -----------------------
        relations: dict[str, _RelationAcc] = {}
        for prop in proposals:
            for r in prop.relations:
                key = _normalise(r.name)
                if not key:
                    continue
                acc = relations.get(key)
                if acc is None:
                    acc = _RelationAcc(name=r.name.strip())
                    relations[key] = acc
                if r.description and not acc.description:
                    acc.description = r.description
                if r.domain is not None and acc.domain is None:
                    acc.domain = r.domain.strip() or None
                if r.range is not None and acc.range is None:
                    acc.range = r.range.strip() or None
                if r.direction != RelationDirection.DIRECTED and acc.direction == RelationDirection.DIRECTED:
                    acc.direction = r.direction
                acc.symmetric = acc.symmetric or r.symmetric
                acc.transitive = acc.transitive or r.transitive
                acc.support += 1

        # --- Quality filter: drop low-support entries --------------------
        kept_classes = {
            k: v
            for k, v in classes.items()
            if v.support >= self._min_class_support
        }
        kept_relations = {
            k: v
            for k, v in relations.items()
            if v.support >= self._min_relation_support
        }

        # --- Assign stable IDs to every class first ---------------------
        for acc in kept_classes.values():
            acc.id = _stable_id("cls", acc.name)

        # --- Map parent names → parent IDs (drop unresolved) -----------
        for acc in kept_classes.values():
            parent_ids: list[str] = []
            for pname in acc.parent_names:
                pkey = _normalise(pname)
                if pkey in kept_classes and kept_classes[pkey].id != acc.id:
                    parent_ids.append(kept_classes[pkey].id)  # type: ignore[arg-type]
            acc.parent_names = set(parent_ids)  # reuse the field for IDs

        # --- Map relation domain/range names → class IDs ---------------
        for acc in kept_relations.values():
            if acc.domain:
                dk = _normalise(acc.domain)
                d = kept_classes.get(dk)
                acc.domain = d.id if d else None  # type: ignore[assignment]
            if acc.range:
                rk = _normalise(acc.range)
                r = kept_classes.get(rk)
                acc.range = r.id if r else None  # type: ignore[assignment]

        # --- Materialise into core.models -------------------------------
        schema = OntologySchema()
        for acc in kept_classes.values():
            schema.add_entity_type(
                EntityType(
                    id=acc.id,  # type: ignore[arg-type]
                    name=acc.name,
                    description=acc.description,
                    parent_ids=tuple(sorted(acc.parent_names)),
                    aliases=tuple(sorted(acc.aliases)),
                )
            )
        for acc in kept_relations.values():
            schema.add_relation(
                Relation(
                    id=_stable_id("rel", acc.name),
                    name=acc.name,
                    description=acc.description,
                    domain=acc.domain,  # type: ignore[arg-type]
                    range=acc.range,  # type: ignore[arg-type]
                    direction=acc.direction,
                    symmetric=acc.symmetric,
                    transitive=acc.transitive,
                )
            )
        return schema


def _merge_schemas(a: OntologySchema, b: OntologySchema) -> OntologySchema:
    """Combine two schemas; ``a`` wins on conflicts (it is the base)."""
    out = OntologySchema(
        entity_types=dict(a.entity_types),
        relations=dict(a.relations),
        rules=dict(a.rules),
        version=a.version,
        source_corpus_id=a.source_corpus_id,
    )
    for et in b.entity_types.values():
        if et.id not in out.entity_types:
            out.add_entity_type(et)
    for r in b.relations.values():
        if r.id not in out.relations:
            out.add_relation(r)
    for rule in b.rules.values():
        if rule.id not in out.rules:
            out.add_rule(rule)
    return out


# ---------------------------------------------------------------------------
# Mock LLM — deterministic in-memory backend for tests
# ---------------------------------------------------------------------------

class MockLLM(StructuredLLM):
    """An in-memory LLM stand-in.

    The default response is a fixed :class:`OntologyChunkProposal`. Tests
    that need varied responses can subclass and override :meth:`_reply`,
    or set :attr:`next_reply` directly between calls.
    """

    def __init__(self) -> None:
        # Bypass the real StructuredLLM constructor — we never touch a backend.
        self._calls: list[list[ChatMessage]] = []
        self._next_reply: OntologyChunkProposal = OntologyChunkProposal(
            entity_types=[
                ProposedClass(
                    name="Server",
                    description="A machine that provides services.",
                ),
                ProposedClass(
                    name="LoadBalancer",
                    description="Distributes traffic across servers.",
                    parent_names=("Server",),
                ),
            ],
            relations=[
                ProposedRelation(
                    name="routesTo",
                    description="Forwards traffic to a server.",
                    domain="LoadBalancer",
                    range="Server",
                ),
            ],
        )

    @property
    def calls(self) -> list[list[ChatMessage]]:
        return self._calls

    def set_next_reply(self, proposal: OntologyChunkProposal) -> None:
        self._next_reply = proposal

    # StructuredLLM surface we need -----------------------------------------

    @property
    def supports_json_schema(self) -> bool:  # noqa: D401
        return True

    def chat_struct(
        self,
        messages,
        *,
        schema_model,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_repair_attempts: int = 1,
    ):
        self._calls.append(list(messages))
        if schema_model is OntologyChunkProposal:
            return self._next_reply
        # For any other model, just validate the JSON dict if present.
        from connectors.llm_backend import _parse_structured

        return _parse_structured(self._next_reply.model_dump_json(), schema_model)

    def chat(
        self,
        messages,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format=None,
        stop=None,
        timeout_s=None,
    ) -> str:
        self._calls.append(list(messages))
        return self._next_reply.model_dump_json()


__all__ = [
    "ProposedClass",
    "ProposedRelation",
    "OntologyChunkProposal",
    "TextChunk",
    "chunk_text",
    "SchemaGenerator",
    "MockLLM",
]
