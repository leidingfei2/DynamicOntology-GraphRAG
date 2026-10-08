"""Core data models for DynamicOntology-GraphRAG.

All inter-module contracts live here. Every subsystem (ontology generation,
candidate extraction, evidence verification, hierarchical retrieval) imports
from this module — never from each other's internals.

Conventions
-----------
* All models are Pydantic ``BaseModel`` v2 — strict-by-default, immutable
  where it makes sense (``frozen=True``), and JSON-serialisable.
* Confidence / evidence scores are floats in ``[0.0, 1.0]``.
* Identifiers are opaque strings (UUIDv4 by default). The system never
  assumes human-readable names are unique.
* ``TBox`` / ``ABox`` terminology follows Description Logic: ``TBox`` is the
  ontology (classes, relations, rules); ``ABox`` is the instance-level graph
  of entities and triples.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Identifier helpers
# ---------------------------------------------------------------------------

def _new_id(prefix: str) -> str:
    """Generate a prefixed UUIDv4 identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RelationDirection(str, Enum):
    """Edge directionality for a ``Relation`` schema entry."""

    DIRECTED = "directed"
    UNDIRECTED = "undirected"
    BIDIRECTIONAL = "bidirectional"


class ExtractionStrategy(str, Enum):
    """Candidate-generation strategy tag.

    Used to route a ``Triple`` back to its provenance and to weight evidence
    during cascade verification.
    """

    TREE_OF_THOUGHT = "tot"            # ToT-style structured reasoning
    OPEN_IE = "open_ie"                # Open Information Extraction
    LOOSE = "loose"                    # Permissive regex / heuristic
    LLM_ZERO_SHOT = "llm_zero_shot"    # Raw LLM extraction baseline


class VerificationLevel(str, Enum):
    """The three tiers of the evidence-driven cascade."""

    SYNTACTIC = "syntactic"        # Tier-1: surface / format checks
    SEMANTIC = "semantic"          # Tier-2: ontology compatibility
    EVIDENTIAL = "evidential"      # Tier-3: corpus-grounded E_B / E_C scoring


class CommunityAlgorithm(str, Enum):
    """Community-detection backends for hierarchical clustering."""

    LOUVAIN = "louvain"
    LEIDEN = "leiden"
    LABEL_PROPAGATION = "label_propagation"
    HIERARCHICAL = "hierarchical"


# ---------------------------------------------------------------------------
# T-Box: Ontology schema
# ---------------------------------------------------------------------------

class EntityType(BaseModel):
    """A class / concept in the dynamic T-Box.

    Equivalent to an OWL ``owl:Class``. May carry a textual description to
    ground LLM prompts during verification and retrieval.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: _new_id("cls"))
    name: str = Field(..., min_length=1, description="Canonical class name, e.g. ``Person``.")
    description: str = Field(default="", description="Natural-language definition.")
    parent_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="IDs of parent ``EntityType``s for taxonomic ``is-a`` links.",
    )
    aliases: tuple[str, ...] = Field(default_factory=tuple)
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Open-ended key/value metadata (e.g. ``color: blue``).",
    )

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("EntityType.name must be non-empty after strip")
        return v


class Relation(BaseModel):
    """A property / predicate in the dynamic T-Box.

    Equivalent to an OWL ``owl:ObjectProperty``. Constrains what ``Triple``s
    are legal in the ABox.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    id: str = Field(default_factory=lambda: _new_id("rel"))
    name: str = Field(..., min_length=1, description="Predicate name, e.g. ``worksFor``.")
    description: str = Field(default="")
    domain: str | None = Field(
        default=None,
        description="Required ``EntityType.id`` of the head entity, or ``None`` for unconstrained.",
    )
    range: str | None = Field(
        default=None,
        description="Required ``EntityType.id`` of the tail entity, or ``None`` for unconstrained. "
        "Stored as ``range_`` internally because ``range`` shadows ``builtins.range``.",
    )
    direction: RelationDirection = Field(default=RelationDirection.DIRECTED)
    symmetric: bool = Field(default=False)
    transitive: bool = Field(default=False)
    aliases: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def range_id(self) -> str | None:
        """Backwards-compatible accessor for the range ``EntityType.id``."""
        return self.range

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Relation.name must be non-empty after strip")
        return v


class OntologyRule(BaseModel):
    """A first-order-style integrity rule over the T-Box.

    The rule body is an opaque expression string; the engine is responsible
    for interpretation (e.g. via a rule DSL or LLM-mediated checking).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: _new_id("rule"))
    name: str = Field(..., min_length=1)
    expression: str = Field(..., min_length=1, description="Rule body in the engine's DSL.")
    description: str = Field(default="")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class OntologySchema(BaseModel):
    """The full T-Box: classes, relations, rules.

    This is the output of dynamic ontology construction and the input to
    every downstream stage.
    """

    model_config = ConfigDict(extra="forbid")

    entity_types: dict[str, EntityType] = Field(default_factory=dict)
    relations: dict[str, Relation] = Field(default_factory=dict)
    rules: dict[str, OntologyRule] = Field(default_factory=dict)
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_corpus_id: str | None = Field(default=None)

    # --- Convenience helpers ------------------------------------------------

    def add_entity_type(self, et: EntityType) -> None:
        self.entity_types[et.id] = et

    def add_relation(self, rel: Relation) -> None:
        self.relations[rel.id] = rel

    def add_rule(self, rule: OntologyRule) -> None:
        self.rules[rule.id] = rule

    def to_relation_index(self) -> dict[str, Relation]:
        """Return ``alias -> Relation`` for fast LLM-prompt lookup."""
        idx: dict[str, Relation] = {}
        for rel in self.relations.values():
            idx[rel.name.lower()] = rel
            for a in rel.aliases:
                idx[a.lower()] = rel
        return idx


# ---------------------------------------------------------------------------
# A-Box: Instance graph
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """An instance node in the ABox.

    Bound to a T-Box ``EntityType.id``. ``name`` is the surface form; the
    canonical entity is keyed by ``id``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: _new_id("ent"))
    name: str = Field(..., min_length=1)
    type_id: str | None = Field(
        default=None,
        description="Foreign key into ``OntologySchema.entity_types``. ``None`` means untyped.",
    )
    aliases: tuple[str, ...] = Field(default_factory=tuple)
    description: str = Field(default="")
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_doc_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Document IDs from which this entity was extracted.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    embedding: list[float] | None = Field(
        default=None,
        description="Optional dense vector for retrieval. Length is model-dependent.",
    )

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Entity.name must be non-empty after strip")
        return v


class Triple(BaseModel):
    """A candidate or verified fact ``(head, relation, tail)``.

    A ``Triple`` is the atomic unit flowing between the candidate-generation
    and verification stages. Whether it is *accepted* is tracked via
    ``is_verified`` and the ``verification`` payload.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: _new_id("tri"))
    head_id: str = Field(..., description="``Entity.id`` of the subject.")
    relation_id: str = Field(..., description="``Relation.id`` of the predicate.")
    tail_id: str = Field(..., description="``Entity.id`` of the object.")
    evidence: str = Field(default="", description="Quoted supporting text span.")
    source_doc_ids: tuple[str, ...] = Field(default_factory=tuple)
    strategy: ExtractionStrategy = Field(default=ExtractionStrategy.LLM_ZERO_SHOT)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_verified: bool = Field(default=False)
    verification: "VerificationResult | None" = Field(default=None)

    def to_tuple(self) -> tuple[str, str, str]:
        """Return the canonical ``(head, relation, tail)`` ID triple."""
        return (self.head_id, self.relation_id, self.tail_id)


# ---------------------------------------------------------------------------
# Evidence-driven cascade verification
# ---------------------------------------------------------------------------

class EvidenceScore(BaseModel):
    """A pair of corpus-grounded evidence scores.

    The cascade uses these two orthogonal signals:

    * ``E_B`` — *Believability*: how much the supporting passage is itself
      trustworthy (source reliability, internal consistency, no
      contradictions).
    * ``E_C`` — *Corroboration*: how widely the fact is supported across
      the corpus (number of independent spans, diversity of documents).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    E_B: float = Field(..., ge=0.0, le=1.0, description="Believability of the evidence.")
    E_C: float = Field(..., ge=0.0, le=1.0, description="Corroboration across the corpus.")
    supporting_spans: tuple[str, ...] = Field(default_factory=tuple)
    contradicting_spans: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def combined(self) -> float:
        """Geometric-mean aggregation; penalises imbalance between E_B and E_C."""
        return (self.E_B * self.E_C) ** 0.5


class VerificationResult(BaseModel):
    """Outcome of running a ``Triple`` through the three-tier cascade."""

    model_config = ConfigDict(extra="forbid")

    triple_id: str
    accepted: bool
    level_reached: VerificationLevel
    tier_scores: dict[VerificationLevel, float] = Field(default_factory=dict)
    evidence: EvidenceScore | None = None
    rationale: str = Field(default="")
    rejected_reason: str | None = None
    verified_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Hierarchical community structure
# ---------------------------------------------------------------------------

class CommunityNode(BaseModel):
    """A cluster of entities at a particular hierarchy level.

    Communities form the retrieval units in the Graph Beam Search. Each
    community has a textual summary used for coarse pre-filtering.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: _new_id("com"))
    level: int = Field(..., ge=0, description="0 = finest grain; larger = coarser.")
    entity_ids: tuple[str, ...] = Field(default_factory=tuple)
    parent_id: str | None = Field(default=None, description="Parent community at level+1.")
    child_ids: tuple[str, ...] = Field(default_factory=tuple)
    algorithm: CommunityAlgorithm = Field(default=CommunityAlgorithm.LOUVAIN)
    modularity: float = Field(default=0.0, ge=-1.0, le=1.0)
    summary: str = Field(default="", description="LLM-generated description of the cluster.")
    centroid_embedding: list[float] | None = None
    triple_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Triples whose both endpoints lie in this community.",
    )

    @property
    def size(self) -> int:
        return len(self.entity_ids)


# ---------------------------------------------------------------------------
# Retrieval payloads
# ---------------------------------------------------------------------------

T = TypeVar("T")


class ScoredItem(BaseModel, Generic[T]):
    """A retrieval hit with its score and a reference to the underlying object."""

    model_config = ConfigDict(extra="forbid")

    item: T
    score: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(default="")
    path: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Trail of community / hop IDs used to reach this item.",
    )


class QueryContext(BaseModel):
    """The resolved query after expansion / rewriting."""

    model_config = ConfigDict(extra="forbid")

    raw_query: str
    expanded_queries: tuple[str, ...] = Field(default_factory=tuple)
    target_entity_ids: tuple[str, ...] = Field(default_factory=tuple)
    target_community_ids: tuple[str, ...] = Field(default_factory=tuple)
    embedding: list[float] | None = None


class RetrievalResponse(BaseModel):
    """Final Graph Beam Search result bundle."""

    model_config = ConfigDict(extra="forbid")

    query: QueryContext
    communities: list[ScoredItem[CommunityNode]] = Field(default_factory=list)
    entities: list[ScoredItem[Entity]] = Field(default_factory=list)
    triples: list[ScoredItem[Triple]] = Field(default_factory=list)
    beam_trace: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Diagnostic trace of the beam search frontier at each step.",
    )


# ---------------------------------------------------------------------------
# Corpus-level containers
# ---------------------------------------------------------------------------

class Document(BaseModel):
    """An input document from the corpus."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: _new_id("doc"))
    title: str = ""
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class GraphSnapshot(BaseModel):
    """A point-in-time dump of the verified ABox + TBox.

    The pipeline produces snapshots so retrieval and visualisation can be
    decoupled from ingestion.

    Note
    ----
    The T-Box is stored under the field name ``schema`` (not ``tbox``) to
    keep the public API short. Pydantic v2 still warns about the parent
    ``BaseModel.schema()`` shadowing; this is silenced via ``model_config``.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    schema_: OntologySchema = Field(
        alias="schema",
        description="The T-Box at the time of snapshot creation.",
    )
    entities: dict[str, Entity] = Field(default_factory=dict)
    triples: dict[str, Triple] = Field(default_factory=dict)
    communities: dict[str, CommunityNode] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1, ge=1)

    @property
    def schema(self) -> OntologySchema:
        return self.schema_

    def verified_triples(self) -> list[Triple]:
        return [t for t in self.triples.values() if t.is_verified]

    def triples_for_entity(self, entity_id: str) -> list[Triple]:
        return [
            t for t in self.triples.values()
            if (t.head_id == entity_id or t.tail_id == entity_id) and t.is_verified
        ]


# Resolve forward references (Triple.verification: VerificationResult)
Triple.model_rebuild()
