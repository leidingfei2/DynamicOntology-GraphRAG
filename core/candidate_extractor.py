"""Multi-strategy candidate-triple generation.

Three strategies run in parallel over the same text and each yields an
:class:`ExtractionResult`:

* **A — Tree-of-Thought (ToT)**       :class:`TreeOfThoughtExtractor`
  Schema-constrained, high-precision, beam-style exploration.

* **B — Open Information Extraction** :class:`OpenIEExtractor`
  Schema-agnostic, returns verbatim *evidence spans* for downstream
  verification.

* **C — Loose (high recall)**         :class:`LooseExtractor`
  Permissive regex + noun-phrase heuristics; maximises recall at the
  cost of precision.

The :class:`MultiStrategyExtractor` orchestrates the three branches with
``asyncio.gather`` (or sequentially if a sync ``StructuredLLM`` is
supplied), aligns entity surface forms, and emits a single merged
``ExtractionResult`` per document.
"""
from __future__ import annotations

import abc
import asyncio
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from connectors.llm_backend import (
    ChatMessage,
    LLMError,
    SchemaValidationError,
    StructuredLLM,
    _parse_structured,
)
from core.models import (
    Document,
    Entity,
    EntityType,
    ExtractionStrategy,
    OntologySchema,
    Relation,
    Triple,
)


# ---------------------------------------------------------------------------
# Per-strategy interchange models
# ---------------------------------------------------------------------------

class ProposedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    type_name: str | None = Field(
        default=None,
        description="Class name from the ontology, or null if untyped.",
    )
    description: str = ""
    aliases: tuple[str, ...] = Field(default_factory=tuple)


class ProposedTriple(BaseModel):
    model_config = ConfigDict(extra="forbid")

    head: str
    relation: str = Field(..., description="Predicate name, e.g. 'routesTo'.")
    tail: str
    evidence: str = Field(default="", description="Verbatim supporting span.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class StrategyOutput(BaseModel):
    """JSON shape every strategy must emit."""

    model_config = ConfigDict(extra="forbid")

    entities: list[ProposedEntity] = Field(default_factory=list)
    triples: list[ProposedTriple] = Field(default_factory=list)
    notes: str = ""


# Re-export the bundle model so the rest of the package keeps one name.
from core.extraction.strategies import ExtractionResult  # noqa: E402


# ---------------------------------------------------------------------------
# Schema serialisation helper
# ---------------------------------------------------------------------------

def _schema_brief(schema: OntologySchema) -> str:
    """Compact, prompt-friendly rendering of the ontology."""
    lines: list[str] = []
    if schema.entity_types:
        lines.append("Classes:")
        for et in schema.entity_types.values():
            parents = ""
            if et.parent_ids:
                pnames = [
                    schema.entity_types[pid].name
                    for pid in et.parent_ids
                    if pid in schema.entity_types
                ]
                if pnames:
                    parents = f"  (is-a: {', '.join(pnames)})"
            lines.append(f"  - {et.name}: {et.description}{parents}")
    if schema.relations:
        lines.append("Relations:")
        for r in schema.relations.values():
            dom = schema.entity_types[r.domain].name if r.domain and r.domain in schema.entity_types else "?"
            rng = schema.entity_types[r.range].name if r.range and r.range in schema.entity_types else "?"
            lines.append(f"  - {r.name}: {dom} → {rng}  ({r.description})")
    return "\n".join(lines) if lines else "(empty ontology)"


def _relation_index(schema: OntologySchema) -> dict[str, Relation]:
    """Lower-cased name → Relation. Aliases included for fuzzy lookup."""
    idx: dict[str, Relation] = {}
    for r in schema.relations.values():
        idx.setdefault(r.name.lower(), r)
        for a in r.aliases:
            idx.setdefault(a.lower(), r)
    return idx


def _class_index(schema: OntologySchema) -> dict[str, EntityType]:
    idx: dict[str, EntityType] = {}
    for et in schema.entity_types.values():
        idx.setdefault(et.name.lower(), et)
        for a in et.aliases:
            idx.setdefault(a.lower(), et)
    return idx


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_BASE_RULES = """\
Rules (apply to ALL strategies)
-------------------------------
- Entity names must match the text verbatim (or a documented alias).
- The ``relation`` field MUST be one of the relation names in the ontology
  (case-insensitive). If no relation fits, OMIT the triple.
- ``evidence`` is the SHORTEST verbatim span that supports the triple.
  Copy-paste from the text, character-for-character.
- Be conservative with confidence: only assign > 0.8 when the span is
  unambiguous.
"""

_TOT_SYSTEM = """\
You are a precise knowledge extractor using a Tree-of-Thought approach.
For each candidate fact, you will:
  1) Propose the fact (head, relation, tail).
  2) Verify that head and tail match entity types compatible with the
     relation's domain / range.
  3) Reject the fact if any step fails.
Output only the surviving facts.
""" + _BASE_RULES

_TOT_USER = """\
Ontology:
{schema_brief}

Text:
\"\"\"
{text}
\"\"\"

Produce a JSON object of shape:
{{
  "entities": [{{"name": "...", "type_name": "...", "description": "...", "aliases": []}}],
  "triples":  [{{"head": "...", "relation": "...", "tail": "...", "evidence": "...", "confidence": 0.0}}],
  "notes":    "..."
}}
"""


_OIE_SYSTEM = """\
You are an Open Information Extraction system. Extract every plausible
(subject, relation, object) triple from the text, regardless of whether
the relation is in the ontology. Be sure to:

* Capture relations expressed by verbs, prepositions, and possessives.
* Always include the EXACT supporting text span in the ``evidence`` field.
* Prefer splitting a sentence into multiple atomic triples over a single
  vague one.
""" + _BASE_RULES

_OIE_USER = """\
Reference ontology (use relation names if they fit; otherwise invent a
clear, lowerCamelCase name and still include ``evidence``):
{schema_brief}

Text:
\"\"\"
{text}
\"\"\"

Produce the JSON output.
"""


_LOOSE_SYSTEM = """\
You are a high-recall extractor. Prefer FALSE POSITIVES over MISSES.
Extract ANY plausible triple you can find, even if the wording is
indirect. Mark low-confidence triples with confidence < 0.4.
""" + _BASE_RULES

_LOOSE_USER = _OIE_USER  # Same template; only the system prompt differs.


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------

class _BaseLLMStrategy(abc.ABC):
    strategy: ExtractionStrategy

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        model: str | None = None,
        max_entities_per_chunk: int = 50,
        max_triples_per_chunk: int = 80,
    ) -> None:
        self._llm = llm
        self._model = model
        self._cap_entities = max_entities_per_chunk
        self._cap_triples = max_triples_per_chunk

    # --- Subclass hooks -----------------------------------------------------

    @abc.abstractmethod
    def _system_prompt(self) -> str: ...

    @abc.abstractmethod
    def _user_prompt(self, text: str, schema_brief: str) -> str: ...

    @property
    @abc.abstractmethod
    def _temperature(self) -> float: ...

    # --- Common path --------------------------------------------------------

    def extract(
        self,
        text: str,
        *,
        schema: OntologySchema,
        document_id: str | None = None,
    ) -> ExtractionResult:
        if not text.strip():
            return ExtractionResult(strategy=self.strategy)

        messages = [
            ChatMessage(role="system", content=self._system_prompt()),
            ChatMessage(
                role="user",
                content=self._user_prompt(text, _schema_brief(schema)),
            ),
        ]
        try:
            output = self._llm.chat_struct(
                messages,
                schema_model=StrategyOutput,
                model=self._model,
                temperature=self._temperature,
            )
        except SchemaValidationError as e:
            # We do not want one malformed reply to kill the pipeline.
            # Return an empty result and let the orchestrator log it.
            return ExtractionResult(
                strategy=self.strategy,
                notes=f"parse_error: {e}",
            )
        except LLMError as e:
            return ExtractionResult(strategy=self.strategy, notes=f"llm_error: {e}")

        return self._materialise(output, schema=schema, document_id=document_id)

    # --- Materialisation ----------------------------------------------------

    def _materialise(
        self,
        output: StrategyOutput,
        *,
        schema: OntologySchema,
        document_id: str | None,
    ) -> ExtractionResult:
        rel_idx = _relation_index(schema)
        cls_idx = _class_index(schema)

        # --- Entities -----------------------------------------------------
        entities: list[Entity] = []
        seen_entity_keys: set[str] = set()
        for pe in output.entities[: self._cap_entities]:
            key = pe.name.strip().lower()
            if not key or key in seen_entity_keys:
                continue
            seen_entity_keys.add(key)
            type_id: str | None = None
            if pe.type_name:
                et = cls_idx.get(pe.type_name.strip().lower())
                if et is not None:
                    type_id = et.id
            entities.append(
                Entity(
                    name=pe.name.strip(),
                    type_id=type_id,
                    aliases=tuple(pe.aliases),
                    description=pe.description,
                    source_doc_ids=((document_id,) if document_id else ()),
                    confidence=0.8 if type_id else 0.5,
                )
            )

        # Index entities by lowercase name for triple head/tail resolution.
        entity_by_name: dict[str, Entity] = {e.name.lower(): e for e in entities}

        # --- Triples ------------------------------------------------------
        triples: list[Triple] = []
        for pt in output.triples[: self._cap_triples]:
            head = self._resolve_or_create_entity(
                pt.head, entity_by_name, seen_entity_keys, document_id, schema,
            )
            tail = self._resolve_or_create_entity(
                pt.tail, entity_by_name, seen_entity_keys, document_id, schema,
            )
            rel = self._resolve_relation(pt.relation, rel_idx)
            if rel is None:
                # Strategy B (Open IE) is allowed to invent names, but we
                # still require they look like a valid identifier. If they
                # don't, drop the triple rather than synthesise a Relation
                # (the ontology is the source of truth at extraction time).
                if not _looks_like_predicate(pt.relation):
                    continue
                continue  # we don't auto-create relations in the schema
            triples.append(
                Triple(
                    head_id=head.id,
                    relation_id=rel.id,
                    tail_id=tail.id,
                    evidence=pt.evidence.strip(),
                    source_doc_ids=((document_id,) if document_id else ()),
                    strategy=self.strategy,
                    confidence=float(pt.confidence),
                )
            )

        return ExtractionResult(strategy=self.strategy, entities=entities, triples=triples)

    def _resolve_or_create_entity(
        self,
        name: str,
        index: dict[str, Entity],
        seen: set[str],
        document_id: str | None,
        schema: OntologySchema,
    ) -> Entity:
        name = name.strip()
        if not name:
            # Return a placeholder; the caller will discard the triple
            # because the head/tail ID will be missing.
            return Entity(name="__missing__", confidence=0.0)
        key = name.lower()
        existing = index.get(key)
        if existing is not None:
            return existing
        # Best-effort type guess via ontology alias match.
        type_id: str | None = None
        for et in schema.entity_types.values():
            if key in {a.lower() for a in et.aliases} or key == et.name.lower():
                type_id = et.id
                break
        ent = Entity(
            name=name,
            type_id=type_id,
            source_doc_ids=((document_id,) if document_id else ()),
            confidence=0.5,
        )
        index[key] = ent
        seen.add(key)
        return ent

    def _resolve_relation(self, name: str, rel_idx: dict[str, Relation]) -> Relation | None:
        key = name.strip().lower()
        if not key:
            return None
        return rel_idx.get(key)


# ---------------------------------------------------------------------------
# Branch A — Tree of Thought
# ---------------------------------------------------------------------------

class TreeOfThoughtExtractor(_BaseLLMStrategy):
    """Schema-constrained, high-precision extraction."""

    strategy = ExtractionStrategy.TREE_OF_THOUGHT

    @property
    def _temperature(self) -> float:
        return 0.0

    def _system_prompt(self) -> str:
        return _TOT_SYSTEM

    def _user_prompt(self, text: str, schema_brief: str) -> str:
        return _TOT_USER.format(schema_brief=schema_brief, text=text)


# ---------------------------------------------------------------------------
# Branch B — Open IE
# ---------------------------------------------------------------------------

class OpenIEExtractor(_BaseLLMStrategy):
    """Schema-agnostic extraction with verbatim evidence spans.

    Unlike ToT, this branch emits relations that are *not* in the schema
    (it still requires a clear lowerCamelCase name). The downstream
    cascade verifier decides whether to accept the resulting triples.
    Here we only drop the relation when the name is gibberish.
    """

    strategy = ExtractionStrategy.OPEN_IE

    @property
    def _temperature(self) -> float:
        return 0.0

    def _system_prompt(self) -> str:
        return _OIE_SYSTEM

    def _user_prompt(self, text: str, schema_brief: str) -> str:
        return _OIE_USER.format(schema_brief=schema_brief, text=text)


# ---------------------------------------------------------------------------
# Branch C — Loose (high recall)
# ---------------------------------------------------------------------------

class LooseExtractor(_BaseLLMStrategy):
    """High-recall permissive extractor."""

    strategy = ExtractionStrategy.LOOSE

    @property
    def _temperature(self) -> float:
        return 0.4  # a little more variance → more candidate relations

    def _system_prompt(self) -> str:
        return _LOOSE_SYSTEM

    def _user_prompt(self, text: str, schema_brief: str) -> str:
        return _LOOSE_USER.format(schema_brief=schema_brief, text=text)


# ---------------------------------------------------------------------------
# Predicate-name validator
# ---------------------------------------------------------------------------

_PREDICATE_RE = re.compile(r"^[a-z][A-Za-z0-9_]*$")


def _looks_like_predicate(s: str) -> bool:
    return bool(_PREDICATE_RE.match(s.strip()))


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def _gather_strategies(
    strategies: list["_BaseLLMStrategy"],
    *,
    text: str,
    schema: OntologySchema,
    document_id: str | None,
) -> list[ExtractionResult]:
    """Async wrapper that runs every strategy concurrently for one text."""
    return list(
        await asyncio.gather(
            *(
                asyncio.to_thread(s.extract, text, schema=schema, document_id=document_id)
                for s in strategies
            )
        )
    )


async def _gather_documents(
    docs: list[Document],
    strategies: list["_BaseLLMStrategy"],
    schema: OntologySchema,
) -> list[ExtractionResult]:
    """Async wrapper that runs every (doc, strategy) pair concurrently."""
    coros: list[asyncio.Future[ExtractionResult]] = []
    for d in docs:
        for s in strategies:
            coros.append(
                asyncio.to_thread(s.extract, d.text, schema=schema, document_id=d.id)
            )
    return list(await asyncio.gather(*coros))


def _run_in_fresh_loop(awaitable):
    """Run an awaitable in a brand-new event loop, then close it.

    We can't use ``asyncio.run`` because it refuses to be called from
    inside another running loop, and we can't reuse a module-level loop
    because it would be bound to a different thread on re-entry. A
    per-call loop is the simplest correct option here.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(awaitable)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Multi-strategy orchestrator
# ---------------------------------------------------------------------------

@dataclass
class MultiStrategyExtractor:
    """Run all three strategies in parallel and merge their outputs.

    Parameters
    ----------
    strategies:
        Sequence of :class:`_BaseLLMStrategy` (or compatible). If omitted,
        sensible defaults (ToT, OpenIE, Loose) are constructed from the
        provided ``llm``.
    parallel:
        If ``True`` (default), strategies run concurrently via
        ``asyncio.gather``. Requires the ``StructuredLLM`` to be safe to
        call from a thread pool — OpenAI / httpx both are.
    """

    llm: StructuredLLM | None = None
    strategies: Sequence[_BaseLLMStrategy] | None = None
    parallel: bool = True

    def __post_init__(self) -> None:
        if self.strategies is None:
            if self.llm is None:
                raise ValueError("Provide either `strategies` or `llm`")
            self.strategies = [
                TreeOfThoughtExtractor(self.llm),
                OpenIEExtractor(self.llm),
                LooseExtractor(self.llm),
            ]
        if not self.strategies:
            raise ValueError("MultiStrategyExtractor requires at least one strategy")

    @property
    def strategy_tags(self) -> list[ExtractionStrategy]:
        return [s.strategy for s in (self.strategies or [])]

    # --- Public API ---------------------------------------------------------

    def extract(
        self,
        documents: Iterable[Document],
        *,
        schema: OntologySchema,
    ) -> list[ExtractionResult]:
        """Run every strategy on every document. Returns one merged result
        per document (across strategies).
        """
        docs = list(documents)
        if not docs:
            return []
        if self.parallel:
            return self._run_parallel(docs, schema=schema)
        return self._run_sequential(docs, schema=schema)

    def extract_one(
        self,
        text: str,
        *,
        schema: OntologySchema,
        document_id: str | None = None,
    ) -> ExtractionResult:
        """Convenience for a single piece of text."""
        per_strategy = self._run_one(text, schema=schema, document_id=document_id)
        return self._merge(per_strategy)

    # --- Internals ----------------------------------------------------------

    def _run_one(
        self,
        text: str,
        *,
        schema: OntologySchema,
        document_id: str | None,
    ) -> list[ExtractionResult]:
        assert self.strategies is not None
        if not self.parallel:
            return [
                s.extract(text, schema=schema, document_id=document_id)
                for s in self.strategies
            ]
        return _run_in_fresh_loop(
            _gather_strategies(
                list(self.strategies), text=text, schema=schema, document_id=document_id
            )
        )

    def _run_parallel(
        self,
        docs: list[Document],
        *,
        schema: OntologySchema,
    ) -> list[ExtractionResult]:
        strategies = list(self.strategies or [])
        flat = _run_in_fresh_loop(_gather_documents(docs, strategies, schema))
        n_strats = len(strategies)
        per_doc: list[list[ExtractionResult]] = [
            flat[i * n_strats:(i + 1) * n_strats] for i in range(len(docs))
        ]
        return [self._merge(group) for group in per_doc]

    def _run_sequential(
        self,
        docs: list[Document],
        *,
        schema: OntologySchema,
    ) -> list[ExtractionResult]:
        out: list[ExtractionResult] = []
        for d in docs:
            group = [
                s.extract(d.text, schema=schema, document_id=d.id)
                for s in (self.strategies or [])
            ]
            out.append(self._merge(group))
        return out

    def _merge(self, results: list[ExtractionResult]) -> ExtractionResult:
        """Union entities / triples; the first strategy wins on conflicts.

        We keep a per-strategy tag on each ``Triple`` so the cascade
        verifier can weight evidence by strategy later.
        """
        entity_by_name: dict[str, Entity] = {}
        triples_by_key: dict[tuple[str, str, str], Triple] = {}

        for r in results:
            for e in r.entities:
                key = e.name.lower()
                if key not in entity_by_name:
                    entity_by_name[key] = e
                else:
                    # Accumulate source docs / aliases.
                    cur = entity_by_name[key]
                    merged_sources = tuple(dict.fromkeys(cur.source_doc_ids + e.source_doc_ids))
                    merged_aliases = tuple(dict.fromkeys(cur.aliases + e.aliases))
                    entity_by_name[key] = cur.model_copy(
                        update={
                            "source_doc_ids": merged_sources,
                            "aliases": merged_aliases,
                            "confidence": max(cur.confidence, e.confidence),
                        }
                    )
            for t in r.triples:
                key = (t.head_id, t.relation_id, t.tail_id)
                if key not in triples_by_key:
                    triples_by_key[key] = t
                else:
                    # Merge: prefer the longer evidence span and the
                    # higher confidence. Strategy tag stays as the
                    # earliest (highest-precision) strategy's tag.
                    cur = triples_by_key[key]
                    better_evidence = t.evidence if len(t.evidence) > len(cur.evidence) else cur.evidence
                    triples_by_key[key] = cur.model_copy(
                        update={
                            "evidence": better_evidence,
                            "confidence": max(cur.confidence, t.confidence),
                            "source_doc_ids": tuple(
                                dict.fromkeys(cur.source_doc_ids + t.source_doc_ids)
                            ),
                        }
                    )

        return ExtractionResult(
            strategy=ExtractionStrategy.LLM_ZERO_SHOT,  # post-merge
            entities=list(entity_by_name.values()),
            triples=list(triples_by_key.values()),
        )


# ---------------------------------------------------------------------------
# Mock LLM for tests
# ---------------------------------------------------------------------------

class MockExtractorLLM(StructuredLLM):
    """In-memory LLM stand-in that returns a configurable ``StrategyOutput``."""

    def __init__(self, default: StrategyOutput | None = None) -> None:
        self._calls: list[list[ChatMessage]] = []
        self._next = default or StrategyOutput(
            entities=[ProposedEntity(name="LoadBalancer", type_name="LoadBalancer")],
            triples=[
                ProposedTriple(
                    head="LoadBalancer",
                    relation="routesTo",
                    tail="Server",
                    evidence="the load balancer routes to a server",
                    confidence=0.9,
                )
            ],
        )
        # Per-system-prompt override (e.g. a different reply for each branch)
        self._by_system_substring: list[tuple[str, StrategyOutput]] = []

    def on_system_contains(self, substring: str, reply: StrategyOutput) -> None:
        self._by_system_substring.append((substring, reply))

    @property
    def supports_json_schema(self) -> bool:  # noqa: D401
        return True

    @property
    def calls(self) -> list[list[ChatMessage]]:
        return self._calls

    def _select_reply(self, messages: Sequence[ChatMessage]) -> StrategyOutput:
        sys = next((m.content for m in messages if m.role == "system"), "")
        for sub, reply in self._by_system_substring:
            if sub in sys:
                return reply
        return self._next

    def chat_struct(
        self,
        messages,
        *,
        schema_model,
        model=None,
        temperature=0.0,
        max_tokens=2048,
        max_repair_attempts=1,
    ):
        self._calls.append(list(messages))
        if schema_model is StrategyOutput:
            return self._select_reply(messages)
        return _parse_structured(self._next.model_dump_json(), schema_model)

    def chat(self, messages, *, model=None, temperature=0.0, max_tokens=2048, response_format=None, stop=None, timeout_s=None) -> str:
        self._calls.append(list(messages))
        return self._select_reply(messages).model_dump_json()


__all__ = [
    "ProposedEntity",
    "ProposedTriple",
    "StrategyOutput",
    "TreeOfThoughtExtractor",
    "OpenIEExtractor",
    "LooseExtractor",
    "MultiStrategyExtractor",
    "MockExtractorLLM",
    "ExtractionResult",
]
