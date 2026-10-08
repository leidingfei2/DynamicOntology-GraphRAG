"""Multi-strategy candidate-triple generation.

Three strategies live here as separate small classes, all behind the
``CandidateExtractor`` ABC so the orchestrator can swap them:

* ``TreeOfThoughtExtractor``       — ToT-style structured exploration
* ``OpenIEExtractor``              — Open Information Extraction (spaCy / rules)
* ``LooseExtractor``               — Permissive regex / heuristic

Each strategy operates on a chunk of text and returns ``ExtractionResult``s.
The orchestrator (``MultiStrategyExtractor``) is responsible for merging /
deduplicating across strategies.
"""
from __future__ import annotations

import abc
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from core.models import (
    Document,
    Entity,
    ExtractionStrategy,
    OntologySchema,
    Triple,
)


# ---------------------------------------------------------------------------
# Per-strategy result
# ---------------------------------------------------------------------------

class ExtractionResult(BaseModel):
    """A bundle of ``Entity`` / ``Triple`` candidates from one strategy."""

    model_config = ConfigDict(extra="forbid")

    strategy: ExtractionStrategy
    entities: list[Entity] = Field(default_factory=list)
    triples: list[Triple] = Field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Per-strategy abstract base
# ---------------------------------------------------------------------------

class CandidateExtractor(abc.ABC):
    """Base class for any single-strategy candidate extractor."""

    strategy: ExtractionStrategy  # set by subclass

    @abc.abstractmethod
    def extract(
        self,
        text: str,
        *,
        schema: OntologySchema,
        document_id: str | None = None,
    ) -> ExtractionResult:
        """Extract candidate entities and triples from a single text chunk."""
        raise NotImplementedError

    def extract_batch(
        self,
        documents: Iterable[Document],
        *,
        schema: OntologySchema,
    ) -> list[ExtractionResult]:
        """Convenience: apply ``extract`` to every document."""
        return [
            self.extract(doc.text, schema=schema, document_id=doc.id)
            for doc in documents
        ]


# ---------------------------------------------------------------------------
# Concrete strategies (stubs)
# ---------------------------------------------------------------------------

class TreeOfThoughtExtractor(CandidateExtractor):
    """Tree-of-Thought structured reasoning over a schema-constrained prompt.

    The model is asked to enumerate alternative parsing branches and pick
    the most consistent one. Implementation lands in milestone 2.
    """

    strategy = ExtractionStrategy.TREE_OF_THOUGHT

    def __init__(self, llm_client: object, *, branch_factor: int = 3, depth: int = 2) -> None:
        self._llm = llm_client
        self._branch_factor = branch_factor
        self._depth = depth

    def extract(
        self,
        text: str,
        *,
        schema: OntologySchema,
        document_id: str | None = None,
    ) -> ExtractionResult:
        raise NotImplementedError


class OpenIEExtractor(CandidateExtractor):
    """Open Information Extraction.

    Default backend is spaCy + a pattern-based relation extractor. Plug
    in any other OIE system by subclassing and overriding ``extract``.
    """

    strategy = ExtractionStrategy.OPEN_IE

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self._spacy_model_name = spacy_model
        self._nlp: object | None = None  # lazy-loaded

    def extract(
        self,
        text: str,
        *,
        schema: OntologySchema,
        document_id: str | None = None,
    ) -> ExtractionResult:
        raise NotImplementedError


class LooseExtractor(CandidateExtractor):
    """Permissive regex / heuristic extractor — high recall, low precision.

    Used as a recall floor; most of its output will be filtered out by the
    cascade verifier.
    """

    strategy = ExtractionStrategy.LOOSE

    def __init__(self, max_triples_per_chunk: int = 50) -> None:
        self._cap = max_triples_per_chunk

    def extract(
        self,
        text: str,
        *,
        schema: OntologySchema,
        document_id: str | None = None,
    ) -> ExtractionResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Multi-strategy orchestrator
# ---------------------------------------------------------------------------

class MultiStrategyExtractor:
    """Run a list of ``CandidateExtractor``s and merge their outputs.

    Merge rules are intentionally simple at the stub stage:

    1. Concatenate every strategy's ``ExtractionResult``.
    2. Align entity surface forms via case-insensitive alias matching
       (delegated to ``EntityAligner``).
    3. Drop near-duplicate triples whose (head, relation, tail) sets match
       above ``merge_overlap_threshold``.
    """

    def __init__(
        self,
        extractors: Sequence[CandidateExtractor],
        *,
        merge_overlap_threshold: float = 0.7,
    ) -> None:
        if not extractors:
            raise ValueError("MultiStrategyExtractor requires at least one strategy")
        self._extractors = list(extractors)
        self._merge_overlap_threshold = merge_overlap_threshold

    @property
    def strategies(self) -> list[ExtractionStrategy]:
        return [e.strategy for e in self._extractors]

    def extract(
        self,
        documents: Iterable[Document],
        *,
        schema: OntologySchema,
    ) -> list[ExtractionResult]:
        """Run every strategy on every document and merge per-strategy results."""
        raise NotImplementedError

    def merge(self, results: Iterable[ExtractionResult]) -> ExtractionResult:
        """Merge per-document, per-strategy results into a single bundle."""
        raise NotImplementedError
