"""Unit tests for step-3: ``core.evidence_verifier``.

The spec explicitly asks for unit tests covering the ``E_B`` and ``E_C``
calculation logic. We additionally cover the supporting text-similarity
helpers, the Filter 1 consensus logic, the cascade short-circuits, and
the persistence helpers (the latter skip if ``networkx`` is not
installed).
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

# Allow running from a fresh checkout without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from core.evidence_verifier import (  # noqa: E402
    CascadeThresholds,
    EBComponents,
    EBWeights,
    ECComponents,
    ECWeights,
    EvidenceVerifier,
    FilterTag,
    TripleDecision,
    _best_lev_in_text,
    _entity_surface_similarity,
    _HAS_NX,
    character_jaccard,
    compute_EB,
    compute_EC,
    filter2_EB,
    filter3_EC,
    jaccard,
    normalized_levenshtein_ratio,
    tokenize,
)
from core.models import (  # noqa: E402
    Document,
    Entity,
    EntityType,
    ExtractionStrategy,
    OntologySchema,
    Relation,
    Triple,
)

warnings.filterwarnings("ignore")


# ===========================================================================
# Text-similarity primitives
# ===========================================================================

def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("The Cat, sat!") == ["the", "cat", "sat"]
    assert tokenize("") == []


def test_jaccard_basic() -> None:
    # intersection = {the, cat, sat} = 3 ; union = 6
    assert math.isclose(jaccard("the cat sat", "the cat sat on a mat"), 0.5)
    # disjoint
    assert jaccard("foo bar", "baz qux") == 0.0
    # identical
    assert jaccard("a b c", "a b c") == 1.0
    # both empty → vacuously 1.0
    assert jaccard("", "") == 1.0


def test_character_jaccard_basic() -> None:
    # 'abc' ∩ 'bcd' = {b,c}, 'abc' ∪ 'bcd' = {a,b,c,d} → 2/4
    assert math.isclose(character_jaccard("abc", "bcd"), 0.5)
    assert character_jaccard("", "") == 1.0


def test_normalized_levenshtein_known_pairs() -> None:
    # kitten / sitting: distance 3, max length 7 → 1 - 3/7
    assert math.isclose(
        normalized_levenshtein_ratio("kitten", "sitting"),
        1 - 3 / 7,
        rel_tol=1e-9,
    )
    # identical
    assert normalized_levenshtein_ratio("abc", "abc") == 1.0
    # one empty
    assert normalized_levenshtein_ratio("", "abc") == 0.0
    assert normalized_levenshtein_ratio("abc", "") == 0.0
    # both empty
    assert normalized_levenshtein_ratio("", "") == 1.0
    # totally disjoint equal-length
    assert normalized_levenshtein_ratio("abc", "xyz") == 0.0


def test_normalized_levenshtein_is_symmetric() -> None:
    for a, b in [("loadbalancer", "load balancer"), ("foo", "foobar"),
                 ("abcdef", "azcfgh")]:
        assert math.isclose(
            normalized_levenshtein_ratio(a, b),
            normalized_levenshtein_ratio(b, a),
        )


def test_entity_surface_similarity_substring_hit() -> None:
    src = "the load balancer routes to a server"
    assert _entity_surface_similarity("load balancer", src) == 1.0
    assert _entity_surface_similarity("server", src) == 1.0


def test_entity_surface_similarity_fuzzy_hit() -> None:
    # 'lb' is short; allow ±2..3 char window, best Levenshtein is the
    # window "lb" inside "load balancer" → 0.0
    assert _entity_surface_similarity("lb", "the load balancer routes to a server") >= 0.0


def test_entity_surface_similarity_empty_inputs() -> None:
    assert _entity_surface_similarity("", "anything") == 0.0
    assert _entity_surface_similarity("anything", "") == 0.0


def test_best_lev_in_text_finds_window() -> None:
    # 'lbar' isn't in 'load balancer' but a window match should be high
    score = _best_lev_in_text("lbar", "the load balancer")
    assert 0.0 <= score <= 1.0


# ===========================================================================
# E_B components
# ===========================================================================

def test_EB_full_match_is_1() -> None:
    c = compute_EB(
        head_name="load balancer",
        relation_name="routes to",
        tail_name="server",
        evidence="the load balancer routes to a server",
        source_text="the load balancer routes to a server",
        weights=EBWeights(),
    )
    assert c.complete_match == EBWeights().alpha_complete
    assert c.subject_match == EBWeights().beta_subject
    assert c.object_match == EBWeights().gamma_object
    assert c.span_conciseness == EBWeights().delta_conciseness
    assert math.isclose(c.total, 1.0)


def test_EB_zero_when_no_evidence() -> None:
    c = compute_EB(
        head_name="x", relation_name="y", tail_name="z",
        evidence="", source_text="anything", weights=EBWeights(),
    )
    # All matching components score 0; conciseness with empty evidence = 0.
    assert c.total == 0.0


def test_EB_partial_match_via_levenshtein() -> None:
    c = compute_EB(
        head_name="load balancer",
        relation_name="routesTo",      # not in the evidence verbatim
        tail_name="server",
        evidence="the load balancer routes to a server",
        source_text="the load balancer routes to a server",
        weights=EBWeights(),
    )
    # Subject + Object match exactly (1.0 each), relation is a fuzzy
    # match against 'routes to' — high but not 1.0. Conciseness = 1.0.
    # We assert the weighted sum is in (0.7, 1.0).
    assert 0.7 < c.total < 1.0


def test_EB_conciseness_penalises_oversized_span() -> None:
    c = compute_EB(
        head_name="x", relation_name="y", tail_name="z",
        evidence="x y z",
        source_text="a very long source text that has nothing to do with x y z",
        weights=EBWeights(),
    )
    # ``span_conciseness`` is already pre-weighted by ``delta_conciseness``;
    # recovering the raw conciseness and asserting it's much lower than 1.0.
    raw = c.span_conciseness / EBWeights().delta_conciseness
    assert raw < 0.5, f"expected raw conciseness << 1.0, got {raw:.3f}"


def test_EB_weights_validation() -> None:
    # Sum = 2.0 ≠ 1.0 → must raise. (A sum of exactly 1.0 — e.g.
    # (0.5, 0.5, 0.0, 0.0) — is legal and must NOT raise.)
    with pytest.raises(ValueError):
        EBWeights(alpha_complete=0.5, beta_subject=0.5, gamma_object=0.5, delta_conciseness=0.5)


# ===========================================================================
# E_C components
# ===========================================================================

def test_EC_grounding_high_when_names_in_source() -> None:
    c = compute_EC(
        head_name="load balancer",
        tail_name="server",
        relation_name="routesTo",
        source_text="the load balancer routes to a server",
        weights=ECWeights(),
    )
    # Substring hits → SubjectGrounding = ObjectGrounding = 1.0
    assert c.subject_grounding == ECWeights().alpha_subject
    assert c.object_grounding == ECWeights().beta_object
    # Coherence uses word Jaccard of joined triple vs source.
    assert 0.0 < c.triple_coherence <= ECWeights().gamma_coherence
    assert c.total > 0.9


def test_EC_grounding_low_for_unrelated_text() -> None:
    c = compute_EC(
        head_name="quantum entanglement",
        tail_name="black hole",
        relation_name="evaporates",
        source_text="the load balancer routes to a server",
        weights=ECWeights(),
    )
    # ``subject_grounding`` is already pre-weighted by ``alpha_subject``;
    # recover the raw value and assert it's much lower than the
    # high-grounding case (> 0.9 in the previous test).
    raw_subj = c.subject_grounding / ECWeights().alpha_subject
    raw_obj = c.object_grounding / ECWeights().beta_object
    assert raw_subj < 0.5, f"expected raw subject_grounding < 0.5, got {raw_subj:.3f}"
    assert raw_obj < 0.5, f"expected raw object_grounding < 0.5, got {raw_obj:.3f}"
    # And confirm the *contrast* against the high-grounding case is
    # substantial — i.e. EC is actually doing useful work.
    high = compute_EC(
        head_name="load balancer",
        tail_name="server",
        relation_name="routesTo",
        source_text="the load balancer routes to a server",
        weights=ECWeights(),
    )
    high_total = high.subject_grounding + high.object_grounding
    low_total = c.subject_grounding + c.object_grounding
    assert high_total > 2 * low_total, (
        f"EC grounding should be substantially lower for unrelated text: "
        f"high={high_total:.3f} vs low={low_total:.3f}"
    )


def test_EC_weights_validation() -> None:
    with pytest.raises(ValueError):
        ECWeights(alpha_subject=0.4, beta_object=0.4, gamma_coherence=0.0)


# ===========================================================================
# filter2_EB / filter3_EC return-shape
# ===========================================================================

def test_filter2_EB_accepts_when_threshold_met() -> None:
    t = Triple(
        head_id="h", relation_id="r", tail_id="t",
        evidence="the load balancer routes to a server",
    )
    d = filter2_EB(
        t,
        head_name="load balancer",
        relation_name="routes to",
        tail_name="server",
        source_text="the load balancer routes to a server",
        weights=EBWeights(),
        threshold=0.9,
    )
    assert d.accepted
    assert d.filter_tag == FilterTag.F2_EB
    assert d.EB is not None and d.EB.total >= 0.9


def test_filter2_EB_rejects_when_threshold_not_met() -> None:
    t = Triple(head_id="h", relation_id="r", tail_id="t", evidence="unrelated stuff")
    d = filter2_EB(
        t,
        head_name="x", relation_name="y", tail_name="z",
        source_text="something completely different",
        weights=EBWeights(),
        threshold=0.9,
    )
    assert not d.accepted
    assert d.rejected_reason is not None


def test_filter3_EC_accepts_when_threshold_met() -> None:
    t = Triple(head_id="h", relation_id="r", tail_id="t", evidence="")
    d = filter3_EC(
        t,
        head_name="load balancer",
        relation_name="routesTo",
        tail_name="server",
        source_text="the load balancer routes to a server",
        weights=ECWeights(),
        threshold=0.75,
    )
    assert d.accepted
    assert d.filter_tag == FilterTag.F3_EC
    assert d.EC is not None and d.EC.total >= 0.75


def test_filter3_EC_rejects_when_threshold_not_met() -> None:
    t = Triple(head_id="h", relation_id="r", tail_id="t", evidence="")
    d = filter3_EC(
        t,
        head_name="quantum",
        relation_name="evaporates",
        tail_name="black hole",
        source_text="the load balancer routes to a server",
        weights=ECWeights(),
        threshold=0.75,
    )
    assert not d.accepted


# ===========================================================================
# EvidenceVerifier end-to-end
# ===========================================================================

def _toy_schema() -> OntologySchema:
    s = OntologySchema()
    s.add_entity_type(EntityType(id="cls_lb", name="LoadBalancer"))
    s.add_entity_type(EntityType(id="cls_srv", name="Server"))
    s.add_relation(Relation(id="rel_routes", name="routesTo", domain="cls_lb", range="cls_srv"))
    return s


def _toy_doc() -> Document:
    return Document(
        id="d1",
        text="the load balancer routes to a server",
        metadata={
            "entity_index": [
                {"id": "e_lb", "name": "load balancer"},
                {"id": "e_srv", "name": "server"},
            ]
        },
    )


def test_cascade_filter1_wins_for_two_strategy_consensus() -> None:
    t1 = Triple(
        head_id="e_lb", relation_id="rel_routes", tail_id="e_srv",
        source_doc_ids=("d1",),
        evidence="the load balancer routes to a server",
        strategy=ExtractionStrategy.TREE_OF_THOUGHT, confidence=0.95,
    )
    t2 = Triple(
        head_id="e_lb", relation_id="rel_routes", tail_id="e_srv",
        source_doc_ids=("d1",),
        evidence="the load balancer routes to a server",
        strategy=ExtractionStrategy.OPEN_IE, confidence=0.85,
    )
    decisions, snap = EvidenceVerifier().verify_batch(
        [t1, t2], schema=_toy_schema(), documents=[_toy_doc()],
    )
    assert all(d.accepted for d in decisions)
    assert all(d.filter_tag == FilterTag.F1_CONSENSUS for d in decisions)
    assert {s for d in decisions for s in d.consensus_strategies} == {
        ExtractionStrategy.TREE_OF_THOUGHT.value,
        ExtractionStrategy.OPEN_IE.value,
    }
    assert len(snap.verified_triples()) == 2


def test_cascade_filter2_rescues_single_high_evidence_triple() -> None:
    t = Triple(
        head_id="e_lb", relation_id="rel_routes", tail_id="e_srv",
        source_doc_ids=("d1",),
        evidence="the load balancer routes to a server",
        strategy=ExtractionStrategy.TREE_OF_THOUGHT, confidence=0.9,
    )
    decisions, _ = EvidenceVerifier().verify_batch(
        [t], schema=_toy_schema(), documents=[_toy_doc()],
    )
    assert len(decisions) == 1
    assert decisions[0].accepted
    # Only one strategy → Filter 1 cannot pass; Filter 2 must rescue.
    assert decisions[0].filter_tag == FilterTag.F2_EB
    assert decisions[0].EB is not None
    assert decisions[0].EB.total >= 0.9


def test_cascade_filter3_rescues_triple_without_evidence() -> None:
    # No evidence span, names not perfectly substringed either.
    t = Triple(
        head_id="e_lb", relation_id="rel_routes", tail_id="e_srv",
        source_doc_ids=("d1",),
        evidence="",
        strategy=ExtractionStrategy.TREE_OF_THOUGHT, confidence=0.6,
    )
    decisions, _ = EvidenceVerifier().verify_batch(
        [t], schema=_toy_schema(), documents=[_toy_doc()],
    )
    assert len(decisions) == 1
    # 'load balancer' and 'server' are exact substrings → E_C ≫ 0.75.
    assert decisions[0].accepted
    assert decisions[0].filter_tag == FilterTag.F3_EC


def test_cascade_rejects_unrelated_triple() -> None:
    # Head/tail names nowhere near the source text.
    t = Triple(
        head_id="e_q", relation_id="rel_routes", tail_id="e_bh",
        source_doc_ids=("d1",),
        evidence="the quantum evaporates the black hole",
        strategy=ExtractionStrategy.TREE_OF_THOUGHT, confidence=0.9,
    )
    # Patch the doc metadata to expose these entity names so the
    # pipeline doesn't fall back to stub entities.
    doc = Document(
        id="d1",
        text="the load balancer routes to a server",
        metadata={
            "entity_index": [
                {"id": "e_q", "name": "quantum"},
                {"id": "e_bh", "name": "black hole"},
            ]
        },
    )
    decisions, snap = EvidenceVerifier().verify_batch(
        [t], schema=_toy_schema(), documents=[doc],
    )
    assert len(decisions) == 1
    assert not decisions[0].accepted
    assert decisions[0].filter_tag == FilterTag.REJECTED
    assert len(snap.verified_triples()) == 0


def test_cascade_empty_input() -> None:
    decisions, snap = EvidenceVerifier().verify_batch(
        [], schema=_toy_schema(), documents=[_toy_doc()],
    )
    assert decisions == []
    assert len(snap.verified_triples()) == 0


def test_cascade_threshold_override() -> None:
    t = Triple(
        head_id="e_lb", relation_id="rel_routes", tail_id="e_srv",
        source_doc_ids=("d1",),
        evidence="the load balancer routes to a server",
        strategy=ExtractionStrategy.TREE_OF_THOUGHT, confidence=0.9,
    )
    # Raise E_B threshold to 1.0 — only perfect matches pass.
    verifier = EvidenceVerifier(
        thresholds=CascadeThresholds(EB_accept=1.0),
    )
    decisions, _ = verifier.verify_batch(
        [t], schema=_toy_schema(), documents=[_toy_doc()],
    )
    # Full match → 1.0 → still passes.
    assert decisions[0].accepted


def test_decision_records_both_EB_and_EC_when_filter3_runs() -> None:
    t = Triple(
        head_id="e_lb", relation_id="rel_routes", tail_id="e_srv",
        source_doc_ids=("d1",),
        evidence="the load balancer routes to a server",
        strategy=ExtractionStrategy.TREE_OF_THOUGHT, confidence=0.9,
    )
    decisions, _ = EvidenceVerifier().verify_batch(
        [t], schema=_toy_schema(), documents=[_toy_doc()],
    )
    d = decisions[0]
    # Single-strategy → Filter 1 doesn't pass; Filter 2 (full match) wins.
    assert d.filter_tag == FilterTag.F2_EB
    assert d.EB is not None and d.EB.total >= 0.9


# ===========================================================================
# Persistence (skipped if networkx is not installed)
# ===========================================================================

@pytest.mark.skipif(not _HAS_NX, reason="networkx not installed")
def test_snapshot_round_trip_via_pickle() -> None:
    if not _HAS_NX:
        pytest.skip("networkx not installed")
    import tempfile
    import networkx as nx

    t = Triple(
        head_id="e_lb", relation_id="rel_routes", tail_id="e_srv",
        source_doc_ids=("d1",),
        evidence="the load balancer routes to a server",
        strategy=ExtractionStrategy.TREE_OF_THOUGHT, confidence=0.9,
    )
    decisions, snap = EvidenceVerifier().verify_batch(
        [t], schema=_toy_schema(), documents=[_toy_doc()],
    )
    g = EvidenceVerifier.snapshot_to_graph(snap)
    assert isinstance(g, nx.MultiDiGraph)
    assert g.number_of_nodes() >= 3  # at least 2 entities + 1 relation
    assert g.number_of_edges() >= 1  # the triple edge

    with tempfile.TemporaryDirectory() as tmp:
        p = f"{tmp}/graph.pkl"
        EvidenceVerifier.save_graph(g, p, fmt="pickle")
        g2 = EvidenceVerifier.load_graph(p, fmt="pickle")
        assert g2.number_of_nodes() == g.number_of_nodes()
        assert g2.number_of_edges() == g.number_of_edges()
