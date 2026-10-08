"""Evidence-driven three-tier cascade verification.

This module is the project's *zero-hallucination gate*. Every candidate
``Triple`` from :mod:`core.candidate_extractor` must clear three filters
in order:

    Filter 1 — Cross-prompt consensus + lexical anchoring
                (high-precision shortcut, no E_B/E_C math required)

    Filter 2 — Explicit-evidence weighted score ``E_B``
                (uses a verbatim ``evidence`` span)

    Filter 3 — Soft-grounding fallback score ``E_C``
                (recovers residual candidates via fuzzy matching)

A triple is *accepted* if any filter approves it. The output is a
``GraphSnapshot`` populated with only the verified triples and the
entities they reference.

Mathematical specification
--------------------------

Filter 2 (Believability, ``E_B``)::

    E_B = α_B · CompleteMatch
        + β_B · SubjectMatch
        + γ_B · ObjectMatch
        + δ_B · SpanConciseness

with default weights ``α_B=0.35``, ``β_B=0.25``, ``γ_B=0.25``,
``δ_B=0.15`` and acceptance threshold ``E_B ≥ 0.9``.

Filter 3 (Corroboration, ``E_C``)::

    E_C = α_C · SubjectGrounding
        + β_C · ObjectGrounding
        + γ_C · TripleCoherence

with default weights ``α_C=0.45``, ``β_C=0.45``, ``γ_C=0.10``. The spec
does not pin an acceptance threshold; we default to ``E_C ≥ 0.75`` (a
stricter value than the (0,1)-normalised scores' geometric mean) and
expose it as a constructor argument.

Persistence
-----------

Verified triples are written to a :class:`networkx.MultiDiGraph` and
materialised into a :class:`core.models.GraphSnapshot`. The graph is
also returned directly so callers can dump GraphML, query neighbours, or
serialise via :mod:`pickle` / :mod:`json`.
"""
from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]
    _HAS_NX = False

from core.models import (
    Document,
    Entity,
    GraphSnapshot,
    OntologySchema,
    Relation,
    Triple,
    VerificationLevel,
    VerificationResult,
)


# ===========================================================================
# Weights and thresholds
# ===========================================================================

@dataclass(frozen=True)
class EBWeights:
    """Filter 2 (Believability) weights — must sum to 1.0."""

    alpha_complete: float = 0.35
    beta_subject: float = 0.25
    gamma_object: float = 0.25
    delta_conciseness: float = 0.15

    def __post_init__(self) -> None:
        total = self.alpha_complete + self.beta_subject + self.gamma_object + self.delta_conciseness
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"EBWeights must sum to 1.0, got {total:.4f}")


@dataclass(frozen=True)
class ECWeights:
    """Filter 3 (Corroboration) weights — must sum to 1.0."""

    alpha_subject: float = 0.45
    beta_object: float = 0.45
    gamma_coherence: float = 0.10

    def __post_init__(self) -> None:
        total = self.alpha_subject + self.beta_object + self.gamma_coherence
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ECWeights must sum to 1.0, got {total:.4f}")


@dataclass(frozen=True)
class CascadeThresholds:
    """All numeric cut-offs in one place.

    ``consensus_min_strategies``:  Filter 1 — how many distinct strategies
                                    must agree.
    ``consensus_anchor_threshold``:Filter 1 — minimum head/tail surface
                                    similarity to the source text.
    ``EB_accept``:                 Filter 2 — E_B above this ⇒ accept.
    ``EC_accept``:                 Filter 3 — E_C above this ⇒ accept.
    """

    consensus_min_strategies: int = 2
    consensus_anchor_threshold: float = 0.90
    EB_accept: float = 0.90
    EC_accept: float = 0.75


# ===========================================================================
# Per-filter decisions
# ===========================================================================

class FilterTag(str, Enum):
    """Which filter accepted (or rejected) a triple."""

    F1_CONSENSUS = "F1_consensus"
    F2_EB = "F2_EB"
    F3_EC = "F3_EC"
    REJECTED = "rejected"


@dataclass
class EBComponents:
    """Per-component scores for the ``E_B`` formula."""

    complete_match: float
    subject_match: float
    object_match: float
    span_conciseness: float
    total: float = 0.0

    def __post_init__(self) -> None:
        self.total = (
            self.complete_match + self.subject_match
            + self.object_match + self.span_conciseness
        )


@dataclass
class ECComponents:
    """Per-component scores for the ``E_C`` formula."""

    subject_grounding: float
    object_grounding: float
    triple_coherence: float
    total: float = 0.0

    def __post_init__(self) -> None:
        self.total = (
            self.subject_grounding + self.object_grounding + self.triple_coherence
        )


@dataclass
class TripleDecision:
    """The outcome of running a single triple through the cascade."""

    triple: Triple
    accepted: bool
    filter_tag: FilterTag
    EB: EBComponents | None = None
    EC: ECComponents | None = None
    consensus_strategies: tuple[str, ...] = field(default_factory=tuple)
    rejected_reason: str | None = None


# ===========================================================================
# Text similarity utilities
# ===========================================================================

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Punctuation and whitespace stripped."""
    return _WORD_RE.findall(text.lower())


def jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity in ``[0, 1]``.

    Defined as ``|A ∩ B| / |A ∪ B|`` over the multisets of word tokens.
    Two empty inputs return ``1.0`` (perfectly similar vacuously).
    """
    ta, tb = Counter(tokenize(a)), Counter(tokenize(b))
    if not ta and not tb:
        return 1.0
    intersection = sum((ta & tb).values())
    union = sum((ta | tb).values())
    if union == 0:
        return 1.0
    return intersection / union


def character_jaccard(a: str, b: str) -> float:
    """Character-level Jaccard. Used by SpanConciseness where word
    granularity is too coarse for short evidence spans.
    """
    if not a and not b:
        return 1.0
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 1.0
    return inter / union


def normalized_levenshtein_ratio(a: str, b: str) -> float:
    """Normalised Levenshtein ratio in ``[0, 1]``.

    Uses the standard ``1 - distance / max(len(a), len(b))`` formula,
    which equals ``1.0`` for two identical strings and ``0.0`` for
    two completely disjoint strings of equal length.

    Implementation is a classic two-row Wagner–Fischer DP — O(len(a)·
    len(b)) time, O(min(len(a), len(b))) memory. We swap inputs so the
    *shorter* string drives the row width.
    """
    a, b = a or "", b or ""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    n, m = len(a), len(b)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for j in range(1, m + 1):
        curr[0] = j
        for i in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(
                prev[i] + 1,        # deletion
                curr[i - 1] + 1,    # insertion
                prev[i - 1] + cost, # substitution
            )
        prev, curr = curr, prev
    distance = prev[n]
    return 1.0 - distance / m


# ===========================================================================
# Helpers used by Filter 1 and Filter 2
# ===========================================================================

def _best_lev_in_text(needle: str, haystack: str) -> float:
    """Highest normalised Levenshtein ratio between ``needle`` and any
    *contiguous* substring of ``haystack`` of similar length.

    A brute-force search would be O(|H|²); we restrict the window to
    ``len(needle) ± max(2, len(needle)//3)`` which is a reasonable
    trade-off for the document sizes we expect.
    """
    if not needle or not haystack:
        return 0.0
    n = len(needle)
    lo = max(1, n - max(2, n // 3))
    hi = n + max(2, n // 3)
    H = haystack
    best = 0.0
    if lo > len(H):
        return normalized_levenshtein_ratio(needle, H)
    for L in (lo, n, hi):
        if L <= 0 or L > len(H):
            continue
        for start in range(0, len(H) - L + 1):
            window = H[start : start + L]
            r = normalized_levenshtein_ratio(needle, window)
            if r > best:
                best = r
                if best >= 0.999:
                    return best
    return best


def _span_conciseness(span: str, source: str) -> float:
    """SpanConciseness = Jaccard(span, source) at the *character* level.

    Defined this way (per the spec) to penalise evidence spans that
    drag in large irrelevant chunks of source text.
    """
    if not span or not source:
        return 0.0
    return character_jaccard(span.lower(), source.lower())


# ===========================================================================
# Filter 1 — cross-prompt consensus + lexical anchoring
# ===========================================================================

def _entity_surface_similarity(entity_name: str, source_text: str) -> float:
    """How strongly an entity's name is anchored in the source text.

    Returns the *max* of:
      * exact-case-insensitive substring containment (1.0 if contained)
      * normalised Levenshtein ratio against the best matching window

    Both are clamped to ``[0, 1]``.
    """
    if not entity_name or not source_text:
        return 0.0
    name_l = entity_name.lower()
    text_l = source_text.lower()
    if name_l in text_l:
        return 1.0
    return _best_lev_in_text(name_l, text_l)


# (Filter 1's logic lives on ``EvidenceVerifier._run_filter1`` — it
# needs the full pool of triples and the entity index, which the
# orchestrator builds.)


# ===========================================================================
# Filter 2 — explicit-evidence weighted score E_B
# ===========================================================================

def compute_EB(
    *,
    head_name: str,
    relation_name: str,
    tail_name: str,
    evidence: str,
    source_text: str,
    weights: EBWeights,
) -> EBComponents:
    """Compute the four components of ``E_B`` and their weighted sum.

    Components
    ---------
    * ``CompleteMatch``  — 1.0 if subject, object, *and* relation name all
                            appear (case-insensitive) inside the evidence
                            span; else the average of the three binary
                            indicators.
    * ``SubjectMatch``   — 1.0 if subject appears in the evidence span
                            (case-insensitive substring); else normalised
                            Levenshtein ratio of subject vs. evidence.
    * ``ObjectMatch``    — same logic for the object.
    * ``SpanConciseness``— character-level Jaccard of evidence vs.
                            source text (1.0 = evidence is exactly the
                            source, 0.0 = disjoint).
    """
    ev = evidence or ""
    src = source_text or ""

    def _match(name: str) -> float:
        n = (name or "").strip().lower()
        e = ev.lower()
        if not n or not e:
            return 0.0
        if n in e:
            return 1.0
        return normalized_levenshtein_ratio(n, e)

    subject_match = _match(head_name)
    object_match = _match(tail_name)
    relation_match = _match(relation_name)

    # CompleteMatch: best-case all three present; otherwise the mean of
    # the three binary indicators (still bounded in [0, 1]).
    if subject_match == 1.0 and object_match == 1.0 and relation_match == 1.0:
        complete_match = 1.0
    else:
        complete_match = (subject_match + object_match + relation_match) / 3.0

    conciseness = _span_conciseness(ev, src)

    return EBComponents(
        complete_match=complete_match * weights.alpha_complete,
        subject_match=subject_match * weights.beta_subject,
        object_match=object_match * weights.gamma_object,
        span_conciseness=conciseness * weights.delta_conciseness,
    )


def filter2_EB(
    triple: Triple,
    *,
    head_name: str,
    relation_name: str,
    tail_name: str,
    source_text: str,
    weights: EBWeights,
    threshold: float,
) -> TripleDecision:
    """Run Filter 2 on a triple that *has* an evidence span.

    Returns a ``TripleDecision`` with ``accepted`` set according to
    ``E_B ≥ threshold``. The function does not raise if the evidence
    is empty — it simply scores 0 across the matching components.
    """
    eb = compute_EB(
        head_name=head_name,
        relation_name=relation_name,
        tail_name=tail_name,
        evidence=triple.evidence,
        source_text=source_text,
        weights=weights,
    )
    accepted = eb.total >= threshold
    return TripleDecision(
        triple=triple,
        accepted=accepted,
        filter_tag=FilterTag.F2_EB,
        EB=eb,
        rejected_reason=None if accepted else f"E_B={eb.total:.3f} < {threshold}",
    )


# ===========================================================================
# Filter 3 — soft-grounding fallback E_C
# ===========================================================================

def compute_EC(
    *,
    head_name: str,
    tail_name: str,
    relation_name: str,
    source_text: str,
    weights: ECWeights,
) -> ECComponents:
    """Compute the three components of ``E_C`` and their weighted sum.

    Components
    ---------
    * ``SubjectGrounding`` — normalised Levenshtein ratio between the
                             head's surface form and the best matching
                             window of the source text.
    * ``ObjectGrounding``  — same for the tail.
    * ``TripleCoherence``  — word-level Jaccard between the joined
                             triple text and the source text. Captures
                             the intuition that a believable triple
                             should re-use vocabulary from its source.
    """
    subj = _best_lev_in_text((head_name or "").lower(), (source_text or "").lower())
    obj = _best_lev_in_text((tail_name or "").lower(), (source_text or "").lower())

    triple_text = " ".join(
        x for x in (head_name, relation_name, tail_name) if x
    )
    coherence = jaccard(triple_text, source_text)

    return ECComponents(
        subject_grounding=subj * weights.alpha_subject,
        object_grounding=obj * weights.beta_object,
        triple_coherence=coherence * weights.gamma_coherence,
    )


def filter3_EC(
    triple: Triple,
    *,
    head_name: str,
    relation_name: str,
    tail_name: str,
    source_text: str,
    weights: ECWeights,
    threshold: float,
) -> TripleDecision:
    """Run Filter 3 (soft-grounding fallback) on a triple."""
    ec = compute_EC(
        head_name=head_name,
        relation_name=relation_name,
        tail_name=tail_name,
        source_text=source_text,
        weights=weights,
    )
    accepted = ec.total >= threshold
    return TripleDecision(
        triple=triple,
        accepted=accepted,
        filter_tag=FilterTag.F3_EC,
        EC=ec,
        rejected_reason=None if accepted else f"E_C={ec.total:.3f} < {threshold}",
    )


# ===========================================================================
# EvidenceVerifier — top-level orchestrator
# ===========================================================================

class EvidenceVerifier:
    """Run the full three-tier cascade over a pool of candidate triples.

    The verifier is *stateless across calls* — construct one with your
    desired weights / thresholds and reuse it.

    Parameters
    ----------
    EB_weights, EC_weights:
        The α/β/γ/δ coefficients for the two scoring formulas. The
        defaults match the spec.
    thresholds:
        Numeric cut-offs. ``EC_accept`` defaults to ``0.75`` because
        the spec is silent on it; the rest match the spec.
    source_text_resolver:
        Callable mapping a ``Triple`` to the source document text used
        for grounding. The default looks the text up in ``documents``
        using the triple's ``source_doc_ids``. If a triple references
        a doc the resolver cannot find, the source text is ``""`` and
        grounding components score 0.
    name_resolver:
        Callable mapping a ``Triple`` to ``(head_name, relation_name,
        tail_name)`` strings. Default uses ``documents`` metadata
        ``entity_index`` / ``relation_index`` when available.
    """

    def __init__(
        self,
        *,
        EB_weights: EBWeights | None = None,
        EC_weights: ECWeights | None = None,
        thresholds: CascadeThresholds | None = None,
        source_text_resolver: Any | None = None,
        name_resolver: Any | None = None,
    ) -> None:
        self.EB_weights = EB_weights or EBWeights()
        self.EC_weights = EC_weights or ECWeights()
        self.thresholds = thresholds or CascadeThresholds()
        self._source_resolver = source_text_resolver
        self._name_resolver = name_resolver

    # --- Public API --------------------------------------------------------

    def verify_batch(
        self,
        triples: list[Triple],
        *,
        schema: OntologySchema,
        documents: list[Document] | None = None,
    ) -> tuple[list[TripleDecision], GraphSnapshot]:
        """Verify every triple and return (decisions, verified snapshot).

        The returned ``GraphSnapshot`` contains only the accepted
        triples and the entities they reference.
        """
        # Build resolver scopes.
        docs = documents or []
        entity_index, relation_index = self._build_indexes(schema, triples, docs)

        def _source_text(t: Triple) -> str:
            if self._source_resolver is not None:
                return self._source_resolver(t) or ""
            for did in t.source_doc_ids:
                for d in docs:
                    if d.id == did:
                        return d.text
            return ""

        def _names(t: Triple) -> tuple[str, str, str]:
            if self._name_resolver is not None:
                return self._name_resolver(t)
            h = entity_index.get(t.head_id)
            r = relation_index.get(t.relation_id)
            te = entity_index.get(t.tail_id)
            return (
                h.name if h else "",
                r.name if r else "",
                te.name if te else "",
            )

        decisions: list[TripleDecision] = []
        for t in triples:
            head_name, relation_name, tail_name = _names(t)
            src = _source_text(t)

            # --- Filter 1 -------------------------------------------------
            f1 = self._run_filter1(
                t, all_triples=triples, entity_index=entity_index, source_text=src,
            )
            if f1 is not None and f1.accepted:
                decisions.append(f1)
                continue

            # --- Filter 2 -------------------------------------------------
            if t.evidence:
                eb_decision = filter2_EB(
                    t,
                    head_name=head_name,
                    relation_name=relation_name,
                    tail_name=tail_name,
                    source_text=src,
                    weights=self.EB_weights,
                    threshold=self.thresholds.EB_accept,
                )
                if eb_decision.accepted:
                    decisions.append(eb_decision)
                    continue
            else:
                eb_decision = None

            # --- Filter 3 -------------------------------------------------
            ec_decision = filter3_EC(
                t,
                head_name=head_name,
                relation_name=relation_name,
                tail_name=tail_name,
                source_text=src,
                weights=self.EC_weights,
                threshold=self.thresholds.EC_accept,
            )
            decisions.append(
                ec_decision
                if ec_decision.accepted
                else TripleDecision(
                    triple=t,
                    accepted=False,
                    filter_tag=FilterTag.REJECTED,
                    EB=eb_decision.EB if eb_decision else None,
                    EC=ec_decision.EC,
                    rejected_reason=(
                        eb_decision.rejected_reason
                        if (eb_decision and not eb_decision.accepted)
                        else ec_decision.rejected_reason
                    ),
                )
            )

        snapshot = self._build_snapshot(decisions, schema, entity_index)
        return decisions, snapshot

    def verify_batch_to_graph(
        self,
        triples: list[Triple],
        *,
        schema: OntologySchema,
        documents: list[Document] | None = None,
    ):
        """Like :meth:`verify_batch` but also returns the NetworkX graph."""
        if not _HAS_NX:
            raise ImportError(
                "networkx is required for verify_batch_to_graph(); "
                "install with `pip install networkx`."
            )
        decisions, snapshot = self.verify_batch(triples, schema=schema, documents=documents)
        graph = self.snapshot_to_graph(snapshot)
        return graph, decisions

    # --- Persistence helpers ----------------------------------------------

    @staticmethod
    def snapshot_to_graph(snapshot: GraphSnapshot):
        """Materialise a verified snapshot into a ``networkx.MultiDiGraph``.

        Node attributes: ``name``, ``type_id`` (for entities) or
        ``kind='schema_class'`` (for ontology classes).
        Edge attributes: ``relation_id``, ``strategy``, ``confidence``,
        ``filter_tag``, ``evidence``.

        Requires the optional ``networkx`` dependency. If it is not
        installed, :class:`ImportError` is re-raised with a hint.
        """
        if not _HAS_NX:
            raise ImportError(
                "networkx is required for snapshot_to_graph(); "
                "install with `pip install networkx`."
            )
        g: "nx.MultiDiGraph" = nx.MultiDiGraph()

        for et in snapshot.schema_.entity_types.values():
            g.add_node(
                f"class::{et.id}",
                kind="schema_class",
                name=et.name,
                description=et.description,
            )
        for r in snapshot.schema_.relations.values():
            g.add_node(
                f"relation::{r.id}",
                kind="schema_relation",
                name=r.name,
                description=r.description,
                domain=r.domain,
                range=r.range,
            )
        for e in snapshot.entities.values():
            g.add_node(
                e.id,
                kind="entity",
                name=e.name,
                type_id=e.type_id,
                confidence=e.confidence,
            )
            if e.type_id:
                g.add_edge(e.id, f"class::{e.type_id}", kind="instance_of")
        for t in snapshot.verified_triples():
            g.add_node(t.head_id, kind="entity")
            g.add_node(t.tail_id, kind="entity")
            g.add_edge(
                t.head_id,
                t.tail_id,
                kind="triple",
                relation_id=t.relation_id,
                strategy=t.strategy.value if hasattr(t.strategy, "value") else str(t.strategy),
                confidence=t.confidence,
                evidence=t.evidence,
                filter_tag=(
                    t.verification.rationale if t.verification else None
                ),
            )
        return g

    @staticmethod
    def save_graph(graph: nx.MultiDiGraph, path: str, *, fmt: str = "graphml") -> None:
        """Persist a graph. ``fmt`` is one of ``"graphml"``, ``"gexf"``,
        ``"json"``, ``"pickle"``.
        """
        if fmt == "graphml":
            nx.write_graphml(graph, path)
        elif fmt == "gexf":
            nx.write_gexf(graph, path)
        elif fmt == "json":
            data = nx.node_link_data(graph, edges="links")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif fmt == "pickle":
            with open(path, "wb") as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Unknown graph format: {fmt!r}")

    @staticmethod
    def load_graph(path: str, *, fmt: str = "graphml") -> nx.MultiDiGraph:
        if fmt == "graphml":
            return nx.read_graphml(path)
        if fmt == "gexf":
            return nx.read_gexf(path)
        if fmt == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return nx.node_link_graph(data, edges="links")
        if fmt == "pickle":
            with open(path, "rb") as f:
                return pickle.load(f)
        raise ValueError(f"Unknown graph format: {fmt!r}")

    # --- Internals ---------------------------------------------------------

    def _run_filter1(
        self,
        triple: Triple,
        *,
        all_triples: list[Triple],
        entity_index: dict[str, Entity],
        source_text: str,
    ) -> TripleDecision | None:
        # Filter 1 needs cross-strategy consensus, so we operate on the
        # full pool. We also need head/tail *names* — pull them from
        # the index built by ``_build_indexes``.
        key = (triple.head_id, triple.relation_id, triple.tail_id)
        strategies = sorted({
            t.strategy.value if hasattr(t.strategy, "value") else str(t.strategy)
            for t in all_triples
            if (t.head_id, t.relation_id, t.tail_id) == key
        })
        if len(strategies) < self.thresholds.consensus_min_strategies:
            return None

        head = entity_index.get(triple.head_id)
        tail = entity_index.get(triple.tail_id)
        if head is None or tail is None:
            return None

        head_sim = _entity_surface_similarity(head.name, source_text)
        tail_sim = _entity_surface_similarity(tail.name, source_text)
        if (
            head_sim < self.thresholds.consensus_anchor_threshold
            or tail_sim < self.thresholds.consensus_anchor_threshold
        ):
            return None

        return TripleDecision(
            triple=triple,
            accepted=True,
            filter_tag=FilterTag.F1_CONSENSUS,
            consensus_strategies=tuple(strategies),
        )

    def _build_indexes(
        self,
        schema: OntologySchema,
        triples: list[Triple],
        documents: list[Document],
    ) -> tuple[dict[str, Entity], dict[str, Relation]]:
        """Build best-effort entity / relation name indexes.

        Entities are inferred from the triples' head/tail names, by
        combining:
          * explicit ``Document.metadata['entity_index']`` if present,
          * any pre-existing ``Entity`` on a ``Triple`` if attached
            later by a higher-level pipeline.
        """
        entity_index: dict[str, Entity] = {}
        relation_index: dict[str, Relation] = dict(schema.relations)

        for d in documents:
            md = d.metadata or {}
            for ent in md.get("entity_index", []) or []:
                eid = ent.get("id")
                if eid:
                    entity_index[eid] = Entity(**ent)

        # If still empty, build stub entities from the triples' head/tail
        # names. This is a best-effort fallback so the verifier works
        # out-of-the-box with just ``Triple`` objects.
        if not entity_index:
            for t in triples:
                for eid in (t.head_id, t.tail_id):
                    if eid in entity_index:
                        continue
                    entity_index[eid] = Entity(id=eid, name=eid)
        return entity_index, relation_index

    def _build_snapshot(
        self,
        decisions: list[TripleDecision],
        schema: OntologySchema,
        entity_index: dict[str, Entity] | None = None,
    ) -> GraphSnapshot:
        """Materialise accepted decisions into a ``GraphSnapshot``.

        Entities referenced by accepted triples are written with their
        real surface names when ``entity_index`` carries them; otherwise
        a name=id stub keeps the snapshot self-consistent.
        """
        snap = GraphSnapshot(schema=schema)
        index = entity_index or {}
        now_used_ids: set[str] = set()
        for d in decisions:
            t = d.triple
            if not d.accepted:
                continue
            t.is_verified = True
            t.verification = VerificationResult(
                triple_id=t.id,
                accepted=True,
                level_reached=_level_for(d.filter_tag),
                tier_scores=_tier_scores(d),
                evidence=None,
                rationale=(
                    d.filter_tag.value
                    + (f" (strategies={','.join(d.consensus_strategies)})"
                       if d.consensus_strategies else "")
                ),
            )
            snap.triples[t.id] = t
            now_used_ids.add(t.head_id)
            now_used_ids.add(t.tail_id)

        for eid in now_used_ids:
            known = index.get(eid)
            snap.entities[eid] = known.model_copy(
                update={"source_doc_ids": tuple(dict.fromkeys(known.source_doc_ids))}
            ) if known is not None else Entity(id=eid, name=eid)
        return snap


def _level_for(tag: FilterTag) -> VerificationLevel:
    if tag == FilterTag.F1_CONSENSUS:
        return VerificationLevel.SEMANTIC
    if tag == FilterTag.F2_EB:
        return VerificationLevel.EVIDENTIAL
    if tag == FilterTag.F3_EC:
        return VerificationLevel.EVIDENTIAL
    return VerificationLevel.SYNTACTIC


def _tier_scores(d: TripleDecision) -> dict[VerificationLevel, float]:
    out: dict[VerificationLevel, float] = {}
    if d.EB is not None:
        out[VerificationLevel.EVIDENTIAL] = d.EB.total
    if d.EC is not None:
        # Use the average when both are present.
        existing = out.get(VerificationLevel.EVIDENTIAL, 0.0)
        out[VerificationLevel.EVIDENTIAL] = (existing + d.EC.total) / 2
    return out


__all__ = [
    "EBWeights",
    "ECWeights",
    "CascadeThresholds",
    "FilterTag",
    "EBComponents",
    "ECComponents",
    "TripleDecision",
    "tokenize",
    "jaccard",
    "character_jaccard",
    "normalized_levenshtein_ratio",
    "compute_EB",
    "compute_EC",
    "filter2_EB",
    "filter3_EC",
    "EvidenceVerifier",
]
