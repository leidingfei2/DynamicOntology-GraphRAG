"""Graph Beam Search retrieval over the hierarchical community tree.

The retriever walks the community tree from coarse (root) to fine
(entities / triples), keeping a fixed-width beam of the most promising
nodes at every hop. The result is a ranked bundle of
``CommunityNode`` / ``Entity`` / ``Triple`` hits plus a diagnostic
``beam_trace``.

Key design points
-----------------

* The **frontier** is a priority queue of ``BeamState`` carrying an
  accumulated score, a path, and a reference to the node.
* A **scorer** is pluggable — default is a hybrid of dense embedding
  similarity and sparse BM25 over community summaries.
* The retriever is **stateless across queries** — re-build from a
  ``GraphSnapshot`` is cheap.
"""
from __future__ import annotations

import abc
from collections.abc import Iterable

from core.models import (
    CommunityNode,
    Entity,
    GraphSnapshot,
    QueryContext,
    RetrievalResponse,
    ScoredItem,
    Triple,
)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class Scorer(abc.ABC):
    """Score a single ``BeamState`` against the resolved query."""

    @abc.abstractmethod
    def score(
        self,
        query: QueryContext,
        *,
        community: CommunityNode | None = None,
        entity: Entity | None = None,
        triple: Triple | None = None,
    ) -> float:
        """Return a relevance score in ``[0.0, 1.0]``."""
        raise NotImplementedError


class HybridScorer(Scorer):
    """Dense (embedding cosine) + sparse (BM25 on summary) blend.

    Weights are configurable; defaults favour dense retrieval because
    community summaries are short and semantically dense.
    """

    def __init__(
        self,
        *,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> None:
        if abs(dense_weight + sparse_weight - 1.0) > 1e-6:
            raise ValueError("dense_weight + sparse_weight must equal 1.0")
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    def score(
        self,
        query: QueryContext,
        *,
        community: CommunityNode | None = None,
        entity: Entity | None = None,
        triple: Triple | None = None,
    ) -> float:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Beam state
# ---------------------------------------------------------------------------

class BeamState:
    """One candidate in the beam-search frontier.

    Internal data class — never crosses the package boundary, so a plain
    class is appropriate.
    """

    __slots__ = ("node_id", "level", "score", "path")

    def __init__(
        self,
        node_id: str,
        level: int,
        score: float,
        path: tuple[str, ...] = (),
    ) -> None:
        self.node_id = node_id
        self.level = level
        self.score = score
        self.path = path

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"BeamState(id={self.node_id!r}, level={self.level}, "
            f"score={self.score:.3f}, |path|={len(self.path)})"
        )


# ---------------------------------------------------------------------------
# Top-level retriever
# ---------------------------------------------------------------------------

class GraphBeamSearchRetriever:
    """Hierarchical Graph Beam Search retriever.

    Parameters
    ----------
    scorer:
        Pluggable relevance scorer. Defaults to ``HybridScorer``.
    beam_width:
        Number of states kept at every hop.
    max_hops:
        Maximum depth traversed from a community to its child entities /
        triples.
    community_top_k, entity_top_k, triple_top_k:
        Final cut-offs in the ``RetrievalResponse``.
    """

    def __init__(
        self,
        scorer: Scorer | None = None,
        *,
        beam_width: int = 8,
        max_hops: int = 3,
        community_top_k: int = 5,
        entity_top_k: int = 20,
        triple_top_k: int = 20,
    ) -> None:
        self._scorer = scorer or HybridScorer()
        self._beam_width = beam_width
        self._max_hops = max_hops
        self._community_top_k = community_top_k
        self._entity_top_k = entity_top_k
        self._triple_top_k = triple_top_k

    # --- Public API ---------------------------------------------------------

    def retrieve(
        self,
        query: QueryContext,
        snapshot: GraphSnapshot,
    ) -> RetrievalResponse:
        """Run Graph Beam Search and return a ranked result bundle."""
        raise NotImplementedError

    def explain(self, response: RetrievalResponse) -> str:
        """Render ``response.beam_trace`` as a human-readable summary."""
        raise NotImplementedError

    # --- Internals (exposed for unit tests) ---------------------------------

    def _initialise_frontier(
        self,
        query: QueryContext,
        snapshot: GraphSnapshot,
    ) -> list[BeamState]:
        """Seed the beam with the top-``beam_width`` root communities."""
        raise NotImplementedError

    def _expand(
        self,
        state: BeamState,
        snapshot: GraphSnapshot,
    ) -> list[BeamState]:
        """Return child states of ``state`` (community → entities/triples)."""
        raise NotImplementedError

    def _prune(
        self,
        frontier: Iterable[BeamState],
    ) -> list[BeamState]:
        """Keep the top-``beam_width`` states by score."""
        raise NotImplementedError

    def _collect_results(
        self,
        accepted_states: list[BeamState],
        snapshot: GraphSnapshot,
    ) -> tuple[list[ScoredItem[CommunityNode]], list[ScoredItem[Entity]], list[ScoredItem[Triple]]]:
        """Slice the accepted states into the three ranked lists."""
        raise NotImplementedError
