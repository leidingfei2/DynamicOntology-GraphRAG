"""End-to-end integration test: graph clustering → beam search → output.

Demonstrates the full step-4 pipeline on a simulated knowledge graph:

    1. Build a toy ``GraphSnapshot`` (verified entities + triples)
       representing an infrastructure domain with three natural
       clusters (web tier, data tier, caching tier).
    2. ``HierarchicalIndex.build`` — bottom-up community clustering,
       summaries, parent-blended context vectors, navigation tree.
    3. ``GraphBeamSearch.search`` — three-stage coarse→medium→fine
       retrieval with beam width k=3.
    4. ``RetrievalPipeline`` / ``retrieve`` — the pipeline entry, plus
       ``format_response`` for the explainable report.

Run with::

    pytest tests/integration/test_retrieval_pipeline_e2e.py -v
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from core.hierarchical_index import (  # noqa: E402
    _HAS_NX,
    HashingEmbedder,
    HierarchicalIndex,
)
from core.models import (  # noqa: E402
    Entity,
    EntityType,
    GraphSnapshot,
    OntologySchema,
    Relation,
    Triple,
)
from retriever.graph_beam_search import GraphBeamSearch  # noqa: E402
from retriever.pipeline import RetrievalPipeline, format_response, retrieve  # noqa: E402

warnings.filterwarnings("ignore")

pytestmark = pytest.mark.skipif(not _HAS_NX, reason="networkx not installed")


# ===========================================================================
# Fixture: a simulated infrastructure knowledge graph
# ===========================================================================

def build_toy_snapshot() -> GraphSnapshot:
    """Three natural clusters sharing a hub — mimics a verified ABox."""
    schema = OntologySchema()
    schema.add_entity_type(EntityType(id="cls_lb", name="LoadBalancer"))
    schema.add_entity_type(EntityType(id="cls_srv", name="Server"))
    schema.add_entity_type(EntityType(id="cls_db", name="Database"))
    schema.add_entity_type(EntityType(id="cls_cache", name="Cache"))
    schema.add_relation(Relation(id="rel_conn", name="connectsTo"))

    names = {
        "lb1": "load balancer 1", "lb2": "load balancer 2",
        "web1": "web server 1", "web2": "web server 2", "web3": "web server 3",
        "db1": "database 1", "db2": "database 2",
        "cache1": "cache node 1", "cache2": "cache node 2",
    }
    entities = {f"e_{k}": Entity(id=f"e_{k}", name=v) for k, v in names.items()}

    edges = [
        ("lb1", "web1", "load balancer 1 routes to web server 1"),
        ("lb1", "web2", "load balancer 1 routes to web server 2"),
        ("lb2", "web3", "load balancer 2 routes to web server 3"),
        ("web1", "db1", "web server 1 reads database 1"),
        ("web2", "db2", "web server 2 reads database 2"),
        ("web3", "db1", "web server 3 reads database 1"),
        ("web1", "cache1", "web server 1 uses cache node 1"),
        ("web2", "cache2", "web server 2 uses cache node 2"),
    ]
    triples = {
        f"t_{i}": Triple(
            id=f"t_{i}",
            head_id=f"e_{h}", relation_id="rel_conn", tail_id=f"e_{t}",
            evidence=ev, is_verified=True,
        )
        for i, (h, t, ev) in enumerate(edges)
    }
    return GraphSnapshot(schema=schema, entities=entities, triples=triples)


@pytest.fixture(scope="module")
def snapshot() -> GraphSnapshot:
    return build_toy_snapshot()


@pytest.fixture(scope="module")
def index(snapshot: GraphSnapshot) -> HierarchicalIndex:
    return HierarchicalIndex(
        embedder=HashingEmbedder(),
        hierarchy_levels=3,
        min_community_size=2,
    ).build(snapshot)


# ===========================================================================
# Stage A — hierarchical clustering
# ===========================================================================

def test_clusters_form_multiple_bottom_communities(index: HierarchicalIndex) -> None:
    bottom = index._levels_data[0]
    assert len(bottom) >= 2, "expected ≥ 2 level-0 communities"
    # Every entity is covered exactly once at level 0.
    covered = [e for c in bottom for e in c.entity_ids]
    assert len(covered) == len(set(covered)) == 9


def test_coarser_level_references_children(index: HierarchicalIndex) -> None:
    levels = index._levels_data
    assert len(levels) >= 2, "expected at least two hierarchy levels"
    for coarser in levels[1:]:
        for c in coarser:
            assert c.child_ids, "coarse community must reference children"


def test_every_community_has_summary_and_vector(index: HierarchicalIndex) -> None:
    for node in index.walk():
        assert node.summary, f"{node.community_id} lacks a summary"
        assert node.centroid and any(x != 0.0 for x in node.centroid)


def test_navigation_tree_roots_exist(index: HierarchicalIndex) -> None:
    roots = index.root_nodes()
    assert roots, "no navigation root"
    # Walking from the roots visits every community exactly once.
    seen = [n.community_id for n in index.walk()]
    assert len(seen) == len(set(seen))


def test_index_writes_communities_back(snapshot: GraphSnapshot) -> None:
    assert snapshot.communities, "build() should populate snapshot.communities"


# ===========================================================================
# Stage B — three-stage beam search
# ===========================================================================

def test_beam_search_returns_structured_response(
    index: HierarchicalIndex, snapshot: GraphSnapshot
) -> None:
    resp = GraphBeamSearch(k=3).search(
        "which load balancer routes to web server",
        index=index, snapshot=snapshot,
    )
    assert resp.query.raw_query == "which load balancer routes to web server"
    assert resp.query.embedding is not None
    # Three trace rows: macro, medium, fine.
    assert [row["stage"] for row in resp.beam_trace] == [1, 2, 3]
    assert resp.beam_trace[0]["level"] == "macro"
    assert resp.beam_trace[1]["level"] == "medium"
    assert resp.beam_trace[2]["level"] == "fine"
    # Beam width respected.
    assert len(resp.beam_trace[0]["kept"]) <= 3
    assert len(resp.beam_trace[1]["kept"]) <= 3
    assert len(resp.beam_trace[2]["kept_entities"]) <= 3


def test_beam_search_finds_relevant_triples(
    index: HierarchicalIndex, snapshot: GraphSnapshot
) -> None:
    resp = GraphBeamSearch(k=3).search(
        "which load balancer routes to web server",
        index=index, snapshot=snapshot,
    )
    assert resp.triples, "expected triple hits"
    # Every hit carries provenance evidence mentioning the topic.
    for hit in resp.triples:
        ev = hit.item.evidence.lower()
        assert "routes to" in ev or "load balancer" in ev


def test_beam_search_separates_topics(
    index: HierarchicalIndex, snapshot: GraphSnapshot
) -> None:
    """A cache-oriented query must surface cache evidence, not routing."""
    resp = GraphBeamSearch(k=3).search(
        "cache node usage for web server",
        index=index, snapshot=snapshot,
    )
    assert resp.triples
    top_evidence = resp.triples[0].item.evidence.lower()
    assert "cache" in top_evidence


def test_scores_are_normalised(
    index: HierarchicalIndex, snapshot: GraphSnapshot
) -> None:
    resp = GraphBeamSearch(k=3).search(
        "database reads",
        index=index, snapshot=snapshot,
    )
    for group in (resp.communities, resp.entities, resp.triples):
        for hit in group:
            assert 0.0 <= hit.score <= 1.0


# ===========================================================================
# Stage C — pipeline entry + formatted output
# ===========================================================================

def test_pipeline_class_end_to_end(snapshot: GraphSnapshot) -> None:
    pipe = RetrievalPipeline(hierarchy_levels=3, min_community_size=2, beam_width=3)
    pipe.build_index(snapshot)
    resp = pipe.retrieve("load balancer routing")
    assert resp.communities or resp.entities or resp.triples
    # Repeated retrieval reuses the built index (no rebuild).
    resp2 = pipe.retrieve("database reads")
    assert resp2 is not resp


def test_free_function_retrieve_reuses_index(snapshot: GraphSnapshot) -> None:
    idx = HierarchicalIndex(hierarchy_levels=2, min_community_size=2).build(snapshot)
    a = retrieve("cache", snapshot=snapshot, index=idx, beam_width=3)
    b = retrieve("database", snapshot=snapshot, index=idx, beam_width=3)
    assert a.query.raw_query == "cache"
    assert b.query.raw_query == "database"


def test_format_response_is_explainable(snapshot: GraphSnapshot) -> None:
    pipe = RetrievalPipeline().build_index(snapshot)
    text = format_response(pipe.retrieve("load balancer routes to web server"))
    assert "Query:" in text
    assert "Beam trace" in text
    assert "Communities" in text and "Entities" in text and "Triples" in text
    assert "evidence:" in text  # provenance spans are rendered


# ===========================================================================
# Runnable demo — `python tests/integration/test_retrieval_pipeline_e2e.py`
# ===========================================================================

if __name__ == "__main__":
    snap = build_toy_snapshot()
    pipe = RetrievalPipeline(hierarchy_levels=3, min_community_size=2, beam_width=3)
    pipe.build_index(snap)

    print("=" * 72)
    print("1) HIERARCHICAL CLUSTERING")
    print("=" * 72)
    for layer_i, layer in enumerate(pipe.index._levels_data):
        print(f" level {layer_i}: {len(layer)} communities")
    for n in pipe.index.walk():
        print(f"  L{n.level} {n.community_id[:20]} entities={len(n.entity_ids)}"
              f" children={len(n.children)} summary={n.summary[:58]!r}")

    for query in (
        "which load balancer routes to web server",
        "cache node usage for web server",
        "database reads",
    ):
        print()
        print("=" * 72)
        print(f"2) BEAM SEARCH — {query!r}")
        print("=" * 72)
        resp = pipe.retrieve(query)
        for row in resp.beam_trace:
            print(f"  stage {row['stage']} ({row['level']}): "
                  f"kept={row.get('kept') or row.get('kept_entities', [])}")
        print(format_response(resp))

    print()
    print("demo complete.")
