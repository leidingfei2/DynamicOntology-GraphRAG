"""Dynamic T-Box ontology construction.

The ``OntologyBuilder`` takes a corpus of ``Document`` objects (or any
text-bearing iterable) and emits a fully-populated ``OntologySchema``
containing ``EntityType``s, ``Relation``s, and ``OntologyRule``s.

The default pipeline is LLM-driven, but the construction is split into
pluggable phases — each phase is its own small class so callers can
swap in non-LLM strategies (taxonomy induction, Hearst patterns, etc.).
"""
from __future__ import annotations

import abc
from collections.abc import Iterable

from core.models import (
    Document,
    EntityType,
    OntologyRule,
    OntologySchema,
    Relation,
)


# ---------------------------------------------------------------------------
# Phase-level interfaces
# ---------------------------------------------------------------------------

class ClassInducer(abc.ABC):
    """Phase 1: propose candidate ``EntityType``s from the corpus."""

    @abc.abstractmethod
    def induce(
        self,
        documents: Iterable[Document],
        *,
        max_classes: int = 200,
        min_support: int = 3,
    ) -> list[EntityType]:
        """Return candidate entity types, ranked by support/confidence."""
        raise NotImplementedError

    @abc.abstractmethod
    def refine(
        self,
        candidates: list[EntityType],
        documents: Iterable[Document],
    ) -> list[EntityType]:
        """Merge near-duplicates and prune low-support classes."""
        raise NotImplementedError


class RelationInducer(abc.ABC):
    """Phase 2: propose candidate ``Relation``s given the induced classes."""

    @abc.abstractmethod
    def induce(
        self,
        classes: list[EntityType],
        documents: Iterable[Document],
        *,
        max_relations: int = 300,
        min_support: int = 3,
    ) -> list[Relation]:
        """Return candidate predicates, optionally typed by ``classes``."""
        raise NotImplementedError


class RuleMiner(abc.ABC):
    """Phase 3: mine ``OntologyRule``s over the induced T-Box."""

    @abc.abstractmethod
    def mine(
        self,
        classes: list[EntityType],
        relations: list[Relation],
        documents: Iterable[Document],
    ) -> list[OntologyRule]:
        """Return a list of integrity / inference rules."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

class OntologyBuilder:
    """Façade that wires the three phases into a single ``build`` call.

    Parameters
    ----------
    class_inducer:
        Pluggable class-induction strategy. Defaults to ``LLMClassInducer``.
    relation_inducer:
        Pluggable relation-induction strategy.
    rule_miner:
        Pluggable rule-mining strategy. May be ``None`` to skip rule mining.
    """

    def __init__(
        self,
        class_inducer: ClassInducer,
        relation_inducer: RelationInducer,
        rule_miner: RuleMiner | None = None,
    ) -> None:
        self._class_inducer = class_inducer
        self._relation_inducer = relation_inducer
        self._rule_miner = rule_miner

    # --- Public API ---------------------------------------------------------

    def build(
        self,
        documents: Iterable[Document],
        *,
        max_classes: int = 200,
        max_relations: int = 300,
        min_class_support: int = 3,
        min_relation_support: int = 3,
        source_corpus_id: str | None = None,
    ) -> OntologySchema:
        """Run the full T-Box construction pipeline and return a schema.

        Steps
        -----
        1. ``ClassInducer.induce``  → refine
        2. ``RelationInducer.induce`` against the refined classes
        3. ``RuleMiner.mine`` (optional)
        4. Assemble into a single ``OntologySchema``.
        """
        raise NotImplementedError

    def extend(
        self,
        schema: OntologySchema,
        documents: Iterable[Document],
    ) -> OntologySchema:
        """Incrementally grow an existing schema with new documents."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Reference LLM-backed implementations (stubs)
# ---------------------------------------------------------------------------

class LLMClassInducer(ClassInducer):
    """LLM-backed class inducer. Uses a hierarchical prompt + self-consistency."""

    def __init__(self, llm_client: object, prompt_template: str) -> None:
        self._llm = llm_client
        self._prompt_template = prompt_template

    def induce(
        self,
        documents: Iterable[Document],
        *,
        max_classes: int = 200,
        min_support: int = 3,
    ) -> list[EntityType]:
        raise NotImplementedError

    def refine(
        self,
        candidates: list[EntityType],
        documents: Iterable[Document],
    ) -> list[EntityType]:
        raise NotImplementedError


class LLMRelationInducer(RelationInducer):
    """LLM-backed relation inducer. Conditions on the induced class set."""

    def __init__(self, llm_client: object, prompt_template: str) -> None:
        self._llm = llm_client
        self._prompt_template = prompt_template

    def induce(
        self,
        classes: list[EntityType],
        documents: Iterable[Document],
        *,
        max_relations: int = 300,
        min_support: int = 3,
    ) -> list[Relation]:
        raise NotImplementedError


class LLMRuleMiner(RuleMiner):
    """LLM-backed rule miner. Emits simple Horn-style rules."""

    def __init__(self, llm_client: object, prompt_template: str) -> None:
        self._llm = llm_client
        self._prompt_template = prompt_template

    def mine(
        self,
        classes: list[EntityType],
        relations: list[Relation],
        documents: Iterable[Document],
    ) -> list[OntologyRule]:
        raise NotImplementedError
