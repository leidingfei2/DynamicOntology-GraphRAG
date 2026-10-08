#!/usr/bin/env python3
"""DynamicOntology-GraphRAG — end-to-end demo.

Runs the FULL five-stage pipeline on a simulated ops-incident document:

    ① dynamic ontology generation   (T-Box: classes + relations)
    ② multi-strategy extraction     (ToT / Open IE / Loose, in parallel)
    ③ evidence-driven verification  (F1 consensus → F2 E_B → F3 E_C)
    ④ hierarchical clustering       (multi-level communities + vectors)
    ⑤ graph beam search             (coarse → medium → fine, k=3)

Zero-dependency out of the box: the LLM calls in stages ①② are served
by deterministic in-process mocks, so the script runs with no API key
and no Ollama daemon. Pass ``--live`` to switch stages ①② onto a real
backend (needs ``OPENAI_API_KEY`` or a local Ollama on :11434).

Usage::

    python examples/demo.py            # deterministic mock run
    python examples/demo.py --live     # real LLM (OpenAI / Ollama)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo importable when run from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connectors.llm_backend import StructuredLLM, default_backend  # noqa: E402
from core.candidate_extractor import (  # noqa: E402
    LooseExtractor,
    OpenIEExtractor,
    TreeOfThoughtExtractor,
)
from core.evidence_verifier import (  # noqa: E402
    EvidenceVerifier,
    FilterTag,
)
from core.hierarchical_index import HierarchicalIndex  # noqa: E402
from core.models import (  # noqa: E402
    Document,
    Entity,
    ExtractionStrategy,
    GraphSnapshot,
    Triple,
)
from core.schema_generator import (  # noqa: E402
    OntologyChunkProposal,
    ProposedClass,
    ProposedRelation,
    SchemaGenerator,
)
from retriever.graph_beam_search import GraphBeamSearch  # noqa: E402
from retriever.pipeline import format_response  # noqa: E402


# ===========================================================================
# Simulated unstructured corpus — an ops incident write-up
# ===========================================================================

DEMO_TEXT = """\
Production incident postmortem, 2026-09-30. At 03:12 UTC the nginx load
balancer in front of the checkout service began returning 502 responses.
The load balancer routes requests to api server 1 and api server 2, both
of which run the checkout service v2.4.1. Api server 1 caches session
state in redis cache to keep p99 reads under 5 ms. Api server 2 connects
to postgres primary for transactional writes; replication lag on the
postgres primary spiked to 40 seconds during the incident. Prometheus
scrapes metrics from both api servers every 15 seconds. Alertmanager
receives alerts from prometheus and pages the on-call engineer. Root
cause: a misconfigured health check on the load balancer marked both api
servers unhealthy after a rolling restart. Mitigation: the health check
threshold was raised and traffic rebalanced across api server 1 and
api server 2 within nine minutes.
"""

QUERIES = [
    "which load balancer routes to api server",
    "redis cache session state",
    "postgres primary replication lag",
]


# ===========================================================================
# Stage ① — dynamic ontology generation
# ===========================================================================

def build_ontology(llm: StructuredLLM, doc_id: str) -> tuple[object, Document]:
    """Run the T-Box generator over the demo corpus."""
    print_stage(1, "Dynamic ontology generation (T-Box)")
    doc = Document(id=doc_id, text=DEMO_TEXT)

    generator = SchemaGenerator(llm, max_classes=30, max_relations=30,
                                min_class_support=1)
    schema = generator.build([(doc.id, DEMO_TEXT)])

    classes = sorted(schema.entity_types.values(), key=lambda c: c.name)
    rels = sorted(schema.relations.values(), key=lambda r: r.name)
    print(f"    induced {len(classes)} classes, {len(rels)} relations")
    for c in classes:
        print(f"      class  {c.name:<16} {c.description[:46]}")
    for r in rels:
        print(f"      rel    {r.name:<16} {r.description[:46]}")
    return schema, doc


# ===========================================================================
# Stage ② — multi-strategy candidate extraction
# ===========================================================================

def extract_candidates(llm: StructuredLLM, schema, doc: Document):
    """Run the three extraction branches independently.

    NOTE: we deliberately do NOT merge the per-strategy outputs —
    Filter 1 of the cascade counts *cross-strategy consensus* on the
    raw candidate pool, so duplicates across strategies are the
    signal, not noise. Merging happens implicitly at graph-write time.
    """
    print_stage(2, "Multi-strategy candidate extraction")
    strategies = {
        "A: Tree-of-Thought": TreeOfThoughtExtractor(llm),
        "B: Open IE": OpenIEExtractor(llm),
        "C: Loose": LooseExtractor(llm),
    }

    results = []
    for label, extractor in strategies.items():
        result = extractor.extract(DEMO_TEXT, schema=schema, document_id=doc.id)
        results.append(result)
        print(f"    [{label:<17}] entities={len(result.entities):<2} "
              f"triples={len(result.triples)}")

    entities, triples = canonicalise(results, schema)
    print(f"    canonical pool: {len(entities)} entities, {len(triples)} candidate triples")
    return entities, triples


def canonicalise(results, schema) -> tuple[list[Entity], list[Triple]]:
    """Glue layer: align entity surface forms across strategies.

    Each strategy independently instantiates ``Entity`` objects, so the
    same real-world object appears under different auto-generated IDs.
    We canonicalise by lower-cased surface name (first occurrence wins)
    and rewrite every triple's head/tail to the canonical ID — this is
    what makes cross-strategy consensus detectable downstream.
    """
    by_name: dict[str, Entity] = {}
    entities: list[Entity] = []
    remap: dict[str, str] = {}

    for result in results:
        for e in result.entities:
            key = e.name.strip().lower()
            if key not in by_name:
                by_name[key] = e
                entities.append(e)
            remap[e.id] = by_name[key].id

    triples: list[Triple] = []
    for result in results:
        for t in result.triples:
            head, tail = remap.get(t.head_id, t.head_id), remap.get(t.tail_id, t.tail_id)
            if head == t.head_id and tail == t.tail_id:
                triples.append(t)
                continue
            triples.append(t.model_copy(update={"head_id": head, "tail_id": tail}))
    return entities, triples


# ===========================================================================
# Stage ③ — evidence-driven cascade verification
# ===========================================================================

def verify_candidates(entities, triples, schema, doc: Document) -> GraphSnapshot:
    """Push the candidate pool through the three-filter cascade."""
    print_stage(3, "Evidence-driven cascade verification (F1 → F2 → F3)")

    doc_with_index = doc.model_copy(update={"metadata": {
        "entity_index": [
            {"id": e.id, "name": e.name} for e in entities
        ]
    }})
    verifier = EvidenceVerifier()
    decisions, snapshot = verifier.verify_batch(
        triples, schema=schema, documents=[doc_with_index],
    )

    accepted = [d for d in decisions if d.accepted]
    counts = {tag: 0 for tag in FilterTag}
    for d in accepted:
        counts[d.filter_tag] += 1
    print(f"    accepted {len(accepted)}/{len(decisions)} candidates — "
          f"F1 consensus: {counts[FilterTag.F1_CONSENSUS]}, "
          f"F2 E_B≥0.9: {counts[FilterTag.F2_EB]}, "
          f"F3 E_C≥0.75: {counts[FilterTag.F3_EC]}, "
          f"rejected: {counts[FilterTag.REJECTED]}")
    print("    per-candidate decisions:")
    for d in decisions:
        name_of = {e.id: e.name for e in entities}
        rel_name = schema.relations[d.triple.relation_id].name \
            if d.triple.relation_id in schema.relations else d.triple.relation_id
        triple_repr = (f"{name_of.get(d.triple.head_id, '?')} "
                       f"--{rel_name}--> "
                       f"{name_of.get(d.triple.tail_id, '?')}")
        if d.filter_tag is FilterTag.F1_CONSENSUS:
            detail = f"consensus of {len(d.consensus_strategies)} strategies"
        elif d.EB is not None:
            detail = f"E_B={d.EB.total:.3f}"
        elif d.EC is not None:
            detail = f"E_C={d.EC.total:.3f}"
        else:
            detail = ""
        verdict = "ACCEPT" if d.accepted else "reject"
        print(f"      [{verdict}] {d.filter_tag.value:<12} {triple_repr:<48} {detail}")

    # Cross-strategy consensus accepts the SAME fact once per strategy;
    # collapse duplicates (same h/r/t) before the fact enters the graph.
    by_key: dict[tuple[str, str, str], Triple] = {}
    for t in list(snapshot.triples.values()):
        key = t.to_tuple()
        if key in by_key:
            # Keep the copy with the richer evidence span.
            if len(t.evidence) > len(by_key[key].evidence):
                by_key[key] = t
            del snapshot.triples[t.id]
        else:
            by_key[key] = t
    print(f"    graph holds {len(snapshot.triples)} deduplicated verified facts "
          f"over {len(snapshot.entities)} entities")
    return snapshot


# ===========================================================================
# Stage ④ — hierarchical community clustering
# ===========================================================================

def build_index(snapshot: GraphSnapshot) -> HierarchicalIndex:
    print_stage(4, "Hierarchical community clustering")
    index = HierarchicalIndex(
        hierarchy_levels=3,
        min_community_size=2,
    ).build(snapshot)
    for layer_i, layer in enumerate(index._levels_data):
        print(f"    level {layer_i}: {len(layer)} communities")
    for node in index.walk():
        print(f"      L{node.level} entities={len(node.entity_ids)} "
              f"children={len(node.children)}  {node.summary[:64]!r}")
    return index


# ===========================================================================
# Stage ⑤ — graph beam search retrieval
# ===========================================================================

def run_queries(index: HierarchicalIndex, snapshot: GraphSnapshot) -> None:
    print_stage(5, "Graph beam search retrieval (k=3)")
    searcher = GraphBeamSearch(k=3)
    for query in QUERIES:
        resp = searcher.search(query, index=index, snapshot=snapshot)
        print(f"\n    query: {query!r}")
        for row in resp.beam_trace:
            kept = row.get("kept") or (row.get("kept_entities", [])
                                       + row.get("kept_triples", []))
            print(f"      stage {row['stage']} ({row['level']:<6}) kept {len(kept)}")
        print("    top hits:")
        for hit in resp.triples[:3]:
            head = snapshot.entities.get(hit.item.head_id)
            tail = snapshot.entities.get(hit.item.tail_id)
            print(f"      [{hit.score:.3f}] "
                  f"{head.name if head else '?'} → {tail.name if tail else '?'}"
                  f"   evidence={hit.item.evidence[:52]!r}")


# ===========================================================================
# LLM wiring
# ===========================================================================

def make_mock_llm() -> StructuredLLM:
    """Deterministic in-process LLM with demo-tuned per-strategy replies."""
    from core.candidate_extractor import MockExtractorLLM
    from core.schema_generator import MockLLM as SchemaMockLLM

    schema_llm = SchemaMockLLM()
    schema_llm.set_next_reply(OntologyChunkProposal(
        entity_types=[
            ProposedClass(name="LoadBalancer", description="distributes traffic"),
            ProposedClass(name="Server", description="hosts a service"),
            ProposedClass(name="Database", description="persistent storage"),
            ProposedClass(name="CacheService", description="ephemeral KV store"),
        ],
        relations=[
            ProposedRelation(name="routesTo", description="forwards requests",
                             domain="LoadBalancer", range="Server"),
            ProposedRelation(name="caches", description="stores in cache",
                             domain="Server", range="CacheService"),
            ProposedRelation(name="connectsTo", description="persistent link",
                             domain="Server", range="Database"),
        ],
    ))

    extract_llm = MockExtractorLLM()
    from core.candidate_extractor import ProposedEntity, ProposedTriple, StrategyOutput

    # Branch A — high-precision ToT (verbatim evidence).
    extract_llm.on_system_contains("Tree-of-Thought", StrategyOutput(
        entities=[
            ProposedEntity(name="load balancer", type_name="LoadBalancer"),
            ProposedEntity(name="api server 1", type_name="Server"),
            ProposedEntity(name="api server 2", type_name="Server"),
        ],
        triples=[
            ProposedTriple(head="load balancer", relation="routesTo",
                           tail="api server 1",
                           evidence="The load balancer routes requests to api server 1",
                           confidence=0.95),
            ProposedTriple(head="load balancer", relation="routesTo",
                           tail="api server 2",
                           evidence="routes requests to api server 1 and api server 2",
                           confidence=0.93),
        ],
    ))
    # Branch B — Open IE (evidence spans for facts ToT missed).
    extract_llm.on_system_contains("Open Information Extraction", StrategyOutput(
        entities=[
            ProposedEntity(name="api server 1", type_name="Server"),
            ProposedEntity(name="redis cache", type_name="CacheService"),
            ProposedEntity(name="api server 2", type_name="Server"),
            ProposedEntity(name="postgres primary", type_name="Database"),
        ],
        triples=[
            ProposedTriple(head="api server 1", relation="caches",
                           tail="redis cache",
                           evidence="Api server 1 caches session state in redis cache",
                           confidence=0.9),
            ProposedTriple(head="api server 2", relation="connectsTo",
                           tail="postgres primary",
                           evidence="Api server 2 connects to postgres primary",
                           confidence=0.88),
        ],
    ))
    # Branch C — loose: repeats a ToT fact (→ F1 consensus) and adds one
    # hallucinated triple (→ rejected by the cascade).
    extract_llm.on_system_contains("high-recall", StrategyOutput(
        entities=[
            ProposedEntity(name="load balancer"),
            ProposedEntity(name="api server 1"),
            ProposedEntity(name="quantum gateway"),  # hallucination
        ],
        triples=[
            ProposedTriple(head="load balancer", relation="routesTo",
                           tail="api server 1",
                           evidence="The load balancer routes requests to api server 1",
                           confidence=0.4),
            ProposedTriple(head="quantum gateway", relation="routesTo",
                           tail="api server 1",
                           evidence="the mesh encrypts everything end to end",
                           confidence=0.3),
        ],
    ))

    # A single duck-typed ``StructuredLLM`` routing by target model:
    # stage ① asks for ``OntologyChunkProposal``, stage ② for
    # ``StrategyOutput`` — each gets its demo-tuned reply.
    class DemoLLM:
        def __init__(self) -> None:
            self._schema_llm = schema_llm
            self._extract_llm = extract_llm

        @property
        def supports_json_schema(self) -> bool:
            return True

        def chat_struct(self, messages, *, schema_model, model=None,
                        temperature=0.0, max_tokens=2048,
                        max_repair_attempts=1):
            if getattr(schema_model, "__name__", "") == "OntologyChunkProposal":
                return self._schema_llm.chat_struct(
                    messages, schema_model=schema_model, model=model)
            return self._extract_llm.chat_struct(
                messages, schema_model=schema_model, model=model)

    return DemoLLM()  # type: ignore[return-value]


# ===========================================================================
# Console helpers
# ===========================================================================

def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def print_stage(n: int, title: str) -> None:
    print(f"\n--- Stage {n}: {title} " + "-" * max(0, 44 - len(title)))


# ===========================================================================
# main
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="use a real LLM backend (OpenAI key or local Ollama)")
    parser.add_argument("--no-mock-flavour", dest="flavour", action="store_false",
                        help=argparse.SUPPRESS)  # reserved
    args = parser.parse_args()

    banner("DynamicOntology-GraphRAG — end-to-end demo")
    if args.live:
        llm = default_backend()
        print(f"live backend: {type(llm).__name__}")
    else:
        llm = make_mock_llm()  # type: ignore[assignment]
        print("deterministic mock backend (use --live for a real LLM)")

    doc_id = "demo_doc_1"

    schema, doc = build_ontology(llm, doc_id)            # ①
    entities, triples = extract_candidates(llm, schema, doc)  # ②
    snapshot = verify_candidates(entities, triples, schema, doc)  # ③
    index = build_index(snapshot)                        # ④
    run_queries(index, snapshot)                         # ⑤

    banner("Demo complete")
    print("Sample structured report (query 1):")
    resp = GraphBeamSearch(k=3).search(QUERIES[0], index=index, snapshot=snapshot)
    print(format_response(resp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
