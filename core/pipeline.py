"""Top-level pipeline orchestrator.

``GraphRAGPipeline`` is the single entry-point that the rest of the
project (CLI, examples, notebooks, FastAPI service) should use. It wires
together the four subsystems in their canonical order:

    Documents
       │
       ▼  1) ontology
    OntologySchema
       │
       ▼  2) extraction   (multi-strategy)
    Candidate triples
       │
       ▼  3) verification  (three-tier cascade)
    Verified ABox
       │
       ▼  4) retrieval     (community cluster + beam search)
    RetrievalResponse

The pipeline is **configurable** — every subsystem is injectable — but
sensible defaults are provided so ``GraphRAGPipeline()`` is enough to
get a working stack.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from core.extraction import MultiStrategyExtractor
from core.graph import GraphAdapter, GraphPersistence
from core.models import (
    Document,
    GraphSnapshot,
    OntologySchema,
    QueryContext,
    RetrievalResponse,
)
from core.ontology import OntologyBuilder
from core.retrieval import (
    GraphBeamSearchRetriever,
    HierarchicalClusterer,
)
from core.verification import CascadeVerifier


@dataclass
class PipelineConfig:
    """Hyperparameters for every pipeline stage.

    Field names mirror the keys in ``configs/default.yaml`` so a YAML
    loader can populate this dataclass directly.
    """

    # Ontology
    max_classes: int = 200
    max_relations: int = 300
    min_class_support: int = 3
    min_relation_support: int = 3

    # Extraction
    merge_overlap_threshold: float = 0.7

    # Verification
    E_B_min: float = 0.55
    E_C_min: float = 0.30
    combined_min: float = 0.45

    # Retrieval
    hierarchy_levels: int = 3
    min_community_size: int = 3
    beam_width: int = 8
    max_hops: int = 3
    community_top_k: int = 5
    entity_top_k: int = 20
    triple_top_k: int = 20

    extras: dict[str, object] = field(default_factory=dict)


class GraphRAGPipeline:
    """High-level orchestrator. All four subsystems are injected."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        ontology_builder: OntologyBuilder,
        extractor: MultiStrategyExtractor,
        verifier: CascadeVerifier,
        clusterer: HierarchicalClusterer,
        retriever: GraphBeamSearchRetriever,
    ) -> None:
        self.config = config
        self._builder = ontology_builder
        self._extractor = extractor
        self._verifier = verifier
        self._clusterer = clusterer
        self._retriever = retriever

    # --- Build (offline) ----------------------------------------------------

    def build(
        self,
        documents: Iterable[Document],
        *,
        existing_schema: OntologySchema | None = None,
    ) -> GraphSnapshot:
        """Run the offline build phase and return a populated snapshot."""
        raise NotImplementedError

    # --- Query (online) -----------------------------------------------------

    def query(
        self,
        query: QueryContext,
        snapshot: GraphSnapshot,
    ) -> RetrievalResponse:
        """Run the online retrieval phase on a previously built snapshot."""
        raise NotImplementedError

    # --- Persistence --------------------------------------------------------

    def save_snapshot(self, snapshot: GraphSnapshot, path: str) -> None:
        """Persist a snapshot to disk via ``GraphPersistence``."""
        GraphPersistence.to_json(snapshot, path)

    def load_snapshot(self, path: str) -> GraphSnapshot:
        """Load a snapshot from disk via ``GraphPersistence``."""
        return GraphPersistence.from_json(path)

    # --- Convenience --------------------------------------------------------

    def adapter(self, snapshot: GraphSnapshot) -> GraphAdapter:
        """Return a ``GraphAdapter`` view over ``snapshot``."""
        return GraphAdapter(snapshot)
