"""Unit tests for step-2 modules: llm_backend, schema_generator,
candidate_extractor. Run with ``pytest tests/unit/``.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

# Allow running the tests from a fresh checkout without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connectors.llm_backend import (  # noqa: E402
    ChatMessage,
    DiskCache,
    OllamaBackend,
    OpenAIBackend,
    SchemaValidationError,
    StructuredLLM,
    _parse_structured,
    _ontology_schema_json_schema,
)
from core.candidate_extractor import (  # noqa: E402
    LooseExtractor,
    MockExtractorLLM,
    MultiStrategyExtractor,
    OpenIEExtractor,
    ProposedEntity,
    ProposedTriple,
    StrategyOutput,
    TreeOfThoughtExtractor,
)
from core.models import Document, ExtractionStrategy  # noqa: E402
from core.schema_generator import (  # noqa: E402
    MockLLM as SchemaMockLLM,
    OntologyChunkProposal,
    ProposedClass,
    ProposedRelation,
    SchemaGenerator,
    chunk_text,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

def test_disk_cache_fingerprint_is_order_independent() -> None:
    a = DiskCache.fingerprint({"b": [1, 2], "a": 1})
    b = DiskCache.fingerprint({"a": 1, "b": [1, 2]})
    assert a == b


def test_ollama_ping_returns_false_when_unreachable() -> None:
    o = OllamaBackend(host="http://127.0.0.1:1")
    assert o.ping() is False
    assert o.supports_json_mode() is True
    assert o.supports_json_schema() is False


def test_openai_response_format_is_json_schema() -> None:
    o = OpenAIBackend(api_key="sk-test", base_url="http://example.invalid")
    llm = StructuredLLM(o, cache_dir=None)
    fmt = llm._best_response_format(_ontology_schema_json_schema())
    assert fmt is not None
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True


def test_ollama_response_format_falls_back_to_json_object() -> None:
    o = OllamaBackend(host="http://127.0.0.1:1")
    llm = StructuredLLM(o, cache_dir=None)
    fmt = llm._best_response_format(_ontology_schema_json_schema())
    assert fmt == {"type": "json_object"}


def test_parse_structured_strips_code_fences() -> None:
    from pydantic import BaseModel

    class Out(BaseModel):
        x: int

    text = "```json\n{\"x\": 7}\n```"
    out = _parse_structured(text, Out)
    assert out.x == 7


# ---------------------------------------------------------------------------
# Schema generator
# ---------------------------------------------------------------------------

def test_chunk_text_respects_size_and_overlap() -> None:
    chunks = chunk_text("a " * 5000, chunk_size=1000, chunk_overlap=100)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 1000 for c in chunks)


def test_schema_generator_end_to_end_with_mock() -> None:
    gen = SchemaGenerator(SchemaMockLLM(), max_classes=50, max_relations=50)
    schema = gen.build(
        ["A load balancer distributes traffic to backend servers."]
    )
    names = {c.name for c in schema.entity_types.values()}
    assert "LoadBalancer" in names
    assert "Server" in names
    lb = next(c for c in schema.entity_types.values() if c.name == "LoadBalancer")
    assert len(lb.parent_ids) == 1
    routes = next(r for r in schema.relations.values() if r.name == "routesTo")
    assert routes.domain is not None and routes.range is not None


def test_schema_generator_is_deterministic() -> None:
    text = "A load balancer distributes traffic to backend servers."
    a = SchemaGenerator(SchemaMockLLM()).build([text])
    b = SchemaGenerator(SchemaMockLLM()).build([text])
    assert set(a.entity_types) == set(b.entity_types)
    assert set(a.relations) == set(b.relations)


def test_schema_generator_min_support_filters() -> None:
    gen = SchemaGenerator(
        SchemaMockLLM(), min_class_support=5, min_relation_support=5,
    )
    schema = gen.build(["short text"])
    assert len(schema.entity_types) == 0
    assert len(schema.relations) == 0


def test_schema_generator_empty_input_returns_empty_schema() -> None:
    gen = SchemaGenerator(SchemaMockLLM())
    schema = gen.build([])
    assert len(schema.entity_types) == 0
    assert len(schema.relations) == 0


# ---------------------------------------------------------------------------
# Candidate extractor
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_schema() -> object:
    from core.models import OntologySchema, EntityType, Relation

    s = OntologySchema()
    s.add_entity_type(EntityType(name="Server", description="m"))
    s.add_entity_type(EntityType(name="LoadBalancer", description="m"))
    s.add_relation(
        Relation(name="routesTo", description="m", domain="LoadBalancer", range="Server")
    )
    return s


def _setup_branched_llm() -> MockExtractorLLM:
    llm = MockExtractorLLM()
    llm.on_system_contains("Tree-of-Thought", StrategyOutput(
        entities=[ProposedEntity(name="LoadBalancer", type_name="LoadBalancer")],
        triples=[ProposedTriple(
            head="LoadBalancer", relation="routesTo", tail="Server",
            evidence="the load balancer routes to a server", confidence=0.95,
        )],
    ))
    llm.on_system_contains("Open Information Extraction", StrategyOutput(
        entities=[ProposedEntity(name="Cache")],
        triples=[ProposedTriple(
            head="Cache", relation="routesTo", tail="Server",
            evidence="the cache routes to a server", confidence=0.6,
        )],
    ))
    llm.on_system_contains("high-recall", StrategyOutput(
        triples=[ProposedTriple(
            head="LoadBalancer", relation="routesTo", tail="Server",
            evidence="load balancer routes to server", confidence=0.4,
        )],
    ))
    return llm


def test_multistrategy_runs_all_three_branches(toy_schema) -> None:
    llm = _setup_branched_llm()
    doc = Document(id="d1", text="...")
    mse = MultiStrategyExtractor(llm=llm, parallel=True)
    res = mse.extract([doc], schema=toy_schema)
    assert len(res) == 1
    assert len(llm.calls) == 3  # one LLM call per strategy


def test_multistrategy_merges_duplicate_triples(toy_schema) -> None:
    llm = _setup_branched_llm()
    doc = Document(id="d1", text="...")
    mse = MultiStrategyExtractor(llm=llm, parallel=False)
    res = mse.extract([doc], schema=toy_schema)
    # ToT and Loose both produce routesTo(LoadBalancer, Server) → merged.
    head_tail = [(t.head_id, t.tail_id) for t in res[0].triples]
    duplicates = [k for k in head_tail if head_tail.count(k) > 1]
    assert duplicates == [], f"merge failed: {duplicates}"


def test_multistrategy_sequential_matches_parallel(toy_schema) -> None:
    doc = Document(id="d1", text="...")
    par = MultiStrategyExtractor(llm=_setup_branched_llm(), parallel=True).extract([doc], schema=toy_schema)
    seq = MultiStrategyExtractor(llm=_setup_branched_llm(), parallel=False).extract([doc], schema=toy_schema)
    assert len(par[0].triples) == len(seq[0].triples)
    assert len(par[0].entities) == len(seq[0].entities)


def test_multistrategy_drops_unknown_relations(toy_schema) -> None:
    """OpenIE may invent a relation not in the ontology; we drop those."""
    llm = MockExtractorLLM()
    llm.on_system_contains("Open Information Extraction", StrategyOutput(
        triples=[ProposedTriple(
            head="LoadBalancer", relation="inventsNewVerb", tail="Server",
            evidence="...", confidence=0.9,
        )],
    ))
    doc = Document(id="d1", text="...")
    mse = MultiStrategyExtractor(llm=llm, parallel=False)
    res = mse.extract([doc], schema=toy_schema)
    # The unknown-relation triple must be dropped.
    rel_names = {toy_schema.relations[t.relation_id].name for t in res[0].triples}
    assert "inventsNewVerb" not in rel_names


def test_extract_one_convenience(toy_schema) -> None:
    llm = _setup_branched_llm()
    mse = MultiStrategyExtractor(llm=llm, parallel=True)
    res = mse.extract_one("...", schema=toy_schema, document_id="d2")
    assert isinstance(res.strategy, ExtractionStrategy)


def test_empty_inputs(toy_schema) -> None:
    mse = MultiStrategyExtractor(llm=_setup_branched_llm())
    assert mse.extract([], schema=toy_schema) == []
    empty = mse.extract_one("", schema=toy_schema)
    assert not empty.entities and not empty.triples
