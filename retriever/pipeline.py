"""Retrieval pipeline entry point.

Wires :class:`~core.hierarchical_index.HierarchicalIndex` (offline:
cluster + summarise + vectorise) and
:class:`~retriever.graph_beam_search.GraphBeamSearch` (online:
three-stage coarse-to-fine search) into one callable surface:

    snapshot ──build_index()──▶ HierarchicalIndex   (offline, once)
    query ──retrieve()──▶ RetrievalResponse          (online, many)

``retrieve`` is the single entry function the spec asks for; it
returns a structured, explainable :class:`core.models.RetrievalResponse`
(beam trace, per-hit scores and routes, plus provenance evidence from
the verified triples).
"""
from __future__ import annotations

from collections.abc import Sequence

from core.hierarchical_index import (
    CommunityDetector,
    Embedder,
    GreedyModularityDetector,
    HashingEmbedder,
    HierarchicalIndex,
    LeadKSummariser,
    Summariser,
)
from core.models import GraphSnapshot, RetrievalResponse, ScoredItem, Triple
from retriever.graph_beam_search import (
    GraphBeamSearch,
    HybridReranker,
    QueryExpander,
    IdentityExpander,
    Reranker,
)


# ===========================================================================
# Default factories
# ===========================================================================

def default_embedder() -> HashingEmbedder:
    """Dependency-free deterministic embedder (swap for OpenAI/local)."""
    return HashingEmbedder()


def default_detector() -> GreedyModularityDetector:
    """Louvain-style greedy modularity clustering via ``networkx``."""
    return GreedyModularityDetector()


# ===========================================================================
# Pipeline
# ===========================================================================

class RetrievalPipeline:
    """Offline index build + online beam-search retrieval.

    Parameters mirror the two underlying components; ``None`` selects
    the deterministic defaults (HashingEmbedder, LeadKSummariser,
    GreedyModularityDetector, HybridReranker, IdentityExpander).
    """

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        summariser: Summariser | None = None,
        detector: CommunityDetector | None = None,
        reranker: Reranker | None = None,
        query_expander: QueryExpander | None = None,
        hierarchy_levels: int = 3,
        min_community_size: int = 2,
        beam_width: int = 3,
    ) -> None:
        self._embedder = embedder or default_embedder()
        self._index = HierarchicalIndex(
            embedder=self._embedder,
            summariser=summariser or LeadKSummariser(),
            detector=detector or default_detector(),
            hierarchy_levels=hierarchy_levels,
            min_community_size=min_community_size,
        )
        self._search = GraphBeamSearch(
            reranker=reranker or HybridReranker(embedder=self._embedder),
            query_expander=query_expander or IdentityExpander(),
            embedder=self._embedder,
            k=beam_width,
        )
        self._snapshot: GraphSnapshot | None = None

    # --- Offline -------------------------------------------------------------

    def build_index(self, snapshot: GraphSnapshot) -> RetrievalPipeline:
        """Cluster + summarise + vectorise the verified snapshot.

        Must be called once before :meth:`retrieve`. The pipeline keeps
        a handle to the snapshot so retrieval can resolve entity /
        triple payloads without the caller re-passing them.
        """
        self._index.build(snapshot)
        self._snapshot = snapshot
        return self

    @property
    def index(self) -> HierarchicalIndex:
        return self._index

    @property
    def snapshot(self) -> GraphSnapshot:
        if self._snapshot is None:
            raise RuntimeError("call build_index(snapshot) before retrieving")
        return self._snapshot

    # --- Online --------------------------------------------------------------

    def retrieve(self, query: str) -> RetrievalResponse:
        """Beam-search the built index for ``query``.

        Returns a structured response with:

        * ``communities`` — the macro / medium communities the beam kept;
        * ``entities``    — top fine-grained entity hits;
        * ``triples``     — top verified facts, each carrying its
                            ``evidence`` span for provenance;
        * ``beam_trace``  — per-stage kept IDs + scores for debugging.
        """
        return self._search.search(query, index=self._index, snapshot=self.snapshot)

    def retrieve_batch(self, queries: Sequence[str]) -> list[RetrievalResponse]:
        return [self.retrieve(q) for q in queries]


# ===========================================================================
# Free-function entry points (the spec's "管线入口函数")
# ===========================================================================

def retrieve(
    query: str,
    *,
    snapshot: GraphSnapshot,
    embedder: Embedder | None = None,
    hierarchy_levels: int = 3,
    min_community_size: int = 2,
    beam_width: int = 3,
    index: HierarchicalIndex | None = None,
) -> RetrievalResponse:
    """One-shot convenience: build (or reuse) an index and search.

    Pass a pre-built ``index`` to skip re-clustering across repeated
    queries; otherwise the index is built from ``snapshot`` on the fly.
    """
    if index is None:
        index = HierarchicalIndex(
            embedder=embedder,
            hierarchy_levels=hierarchy_levels,
            min_community_size=min_community_size,
        ).build(snapshot)
    searcher = GraphBeamSearch(
        embedder=embedder or index.embedder,  # same space as the centroids
        k=beam_width,
    )
    return searcher.search(query, index=index, snapshot=snapshot)


def format_response(resp: RetrievalResponse, *, max_items: int = 5) -> str:
    """Render a response as a human-readable, explainable report."""
    lines: list[str] = []
    lines.append(f"Query: {resp.query.raw_query!r}")
    if len(resp.query.expanded_queries) > 1:
        lines.append(f"Expansions: {list(resp.query.expanded_queries)}")

    lines.append("")
    lines.append("## Beam trace")
    for row in resp.beam_trace:
        kept = row.get("kept") or row.get("kept_entities", []) + row.get("kept_triples", [])
        lines.append(f"  stage {row.get('stage')} ({row.get('level')}): kept {len(kept)}")

    def _block(title: str, items: list[ScoredItem]) -> None:
        lines.append("")
        lines.append(f"## {title}")
        if not items:
            lines.append("  (none)")
        for hit in items[:max_items]:
            item = hit.item
            if isinstance(item, Triple):
                lines.append(
                    f"  [{hit.score:.3f}] {item.head_id} --{item.relation_id}--> {item.tail_id}"
                )
                if item.evidence:
                    lines.append(f"          evidence: {item.evidence!r}")
            else:
                name = getattr(item, "name", None) or getattr(item, "summary", "")[:60]
                lvl = getattr(item, "level", "")
                lines.append(f"  [{hit.score:.3f}] L{lvl} {name}")

    _block("Communities", resp.communities)
    _block("Entities", resp.entities)
    _block("Triples", resp.triples)
    return "\n".join(lines)


__all__ = [
    "RetrievalPipeline",
    "retrieve",
    "format_response",
    "default_embedder",
    "default_detector",
]
