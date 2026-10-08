"""Evidence-driven three-tier cascade verification.

The cascade is the project's *zero-hallucination* gate. Every candidate
``Triple`` must clear all three tiers in order:

    Tier 1 — Syntactic     surface / format checks (cheap)
    Tier 2 — Semantic      ontology compatibility (cheap-ish)
    Tier 3 — Evidential    corpus-grounded E_B / E_C scoring (expensive)

A failure at any tier short-circuits the cascade; the ``Triple`` is
rejected with a ``rejected_reason`` recorded in its ``VerificationResult``.

Each tier is its own class implementing the ``Tier`` ABC, so an
experimenter can drop in a custom tier (e.g. a fine-tuned NLI model)
without touching the others.
"""
from __future__ import annotations

import abc

from core.models import (
    Document,
    Entity,
    EvidenceScore,
    OntologySchema,
    Triple,
    VerificationLevel,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Tier interface
# ---------------------------------------------------------------------------

class Tier(abc.ABC):
    """One filter in the cascade. Pure: no mutation of inputs."""

    level: VerificationLevel

    @abc.abstractmethod
    def check(
        self,
        triple: Triple,
        *,
        schema: OntologySchema,
        entities: dict[str, Entity],
        documents: dict[str, Document],
    ) -> "TierOutcome":
        """Return a ``TierOutcome`` capturing pass/fail and a score in [0, 1]."""
        raise NotImplementedError


class TierOutcome:
    """Lightweight result object for a single tier check.

    Using a plain dataclass here (instead of a Pydantic model) because it
    is created and consumed entirely inside the cascade — it never
    crosses a network boundary.
    """

    __slots__ = ("passed", "score", "rationale", "rejected_reason")

    def __init__(
        self,
        passed: bool,
        score: float,
        rationale: str = "",
        rejected_reason: str | None = None,
    ) -> None:
        self.passed = passed
        self.score = score
        self.rationale = rationale
        self.rejected_reason = rejected_reason

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"TierOutcome(passed={self.passed}, score={self.score:.3f}, "
            f"reason={self.rejected_reason!r})"
        )


# ---------------------------------------------------------------------------
# Tier implementations (stubs)
# ---------------------------------------------------------------------------

class SyntacticTier(Tier):
    """Tier 1: well-formedness, non-empty evidence, no self-loops, etc."""

    level = VerificationLevel.SYNTACTIC

    def __init__(self, *, min_evidence_chars: int = 5, reject_self_loops: bool = True) -> None:
        self._min_evidence_chars = min_evidence_chars
        self._reject_self_loops = reject_self_loops

    def check(
        self,
        triple: Triple,
        *,
        schema: OntologySchema,
        entities: dict[str, Entity],
        documents: dict[str, Document],
    ) -> TierOutcome:
        raise NotImplementedError


class SemanticTier(Tier):
    """Tier 2: ontology compatibility (domain/range, type constraints)."""

    level = VerificationLevel.SEMANTIC

    def __init__(
        self,
        *,
        require_typed_entities: bool = True,
        reject_out_of_schema: bool = True,
    ) -> None:
        self._require_typed = require_typed_entities
        self._reject_oor = reject_out_of_schema

    def check(
        self,
        triple: Triple,
        *,
        schema: OntologySchema,
        entities: dict[str, Entity],
        documents: dict[str, Document],
    ) -> TierOutcome:
        raise NotImplementedError


class EvidentialTier(Tier):
    """Tier 3: corpus-grounded ``E_B`` (believability) and ``E_C`` (corroboration)."""

    level = VerificationLevel.EVIDENTIAL

    def __init__(
        self,
        *,
        E_B_min: float = 0.55,
        E_C_min: float = 0.30,
        combined_min: float = 0.45,
        max_supporting_spans: int = 5,
        llm_client: object | None = None,
    ) -> None:
        self._E_B_min = E_B_min
        self._E_C_min = E_C_min
        self._combined_min = combined_min
        self._max_spans = max_supporting_spans
        self._llm = llm_client

    def check(
        self,
        triple: Triple,
        *,
        schema: OntologySchema,
        entities: dict[str, Entity],
        documents: dict[str, Document],
    ) -> TierOutcome:
        raise NotImplementedError

    # --- Sub-tasks (public for unit-testing) --------------------------------

    def compute_evidence_score(
        self,
        triple: Triple,
        documents: dict[str, Document],
    ) -> EvidenceScore:
        """Compute ``(E_B, E_C)`` for a triple against the corpus."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Cascade orchestrator
# ---------------------------------------------------------------------------

class CascadeVerifier:
    """Apply tiers in order; return a single ``VerificationResult`` per triple.

    Usage
    -----

    .. code-block:: python

        verifier = CascadeVerifier([SyntacticTier(), SemanticTier(), EvidentialTier()])
        results = verifier.verify_batch(candidates, schema=schema, entities=ents, documents=docs)
    """

    def __init__(self, tiers: list[Tier]) -> None:
        if not tiers:
            raise ValueError("CascadeVerifier requires at least one tier")
        levels = [t.level for t in tiers]
        if levels != sorted(levels, key=lambda lv: lv.value):
            # Soft warning at stub stage; the orchestrator still runs them in
            # the order they were given.
            pass
        self._tiers = tiers

    @property
    def tier_levels(self) -> list[VerificationLevel]:
        return [t.level for t in self._tiers]

    def verify(
        self,
        triple: Triple,
        *,
        schema: OntologySchema,
        entities: dict[str, Entity],
        documents: dict[str, Document],
    ) -> VerificationResult:
        """Verify a single triple. Mutates ``triple.is_verified`` and
        ``triple.verification`` as a side-effect so the caller can index
        accepted triples directly.
        """
        raise NotImplementedError

    def verify_batch(
        self,
        triples: list[Triple],
        *,
        schema: OntologySchema,
        entities: dict[str, Entity],
        documents: dict[str, Document],
    ) -> list[VerificationResult]:
        """Verify many triples. Returns one ``VerificationResult`` per input."""
        raise NotImplementedError
