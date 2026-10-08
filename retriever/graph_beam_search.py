"""Graph Beam Search: coarse → medium → fine three-stage retrieval.

The retriever walks the :class:`~core.hierarchical_index.HierarchicalIndex`
from the root (coarsest) communities down to individual entities and
triples, keeping a fixed-width beam of the ``k`` most promising
candidates at every hop:

* **Stage 1 — macro filtering.** Score every top-level community
  against the query, prune to the top-``k``.
* **Stage 2 — medium refinement.** Expand the survivors' children
  (mid-level sub-communities) and re-score them *with the parent's
  score as a prior*, so the search stays anchored to coarse relevance.
* **Stage 3 — entity-level fine retrieval.** Descend to entities and
  verified triples, blending each candidate's own signal with its
  community's context, and emit the final ranked, explainable hits.

Every stage records what it kept (and the scores) into
``RetrievalResponse.beam_trace`` so callers can explain *why* a result
was returned.
"""
from __future__ import annotations

import abc
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.hierarchical_index import Embedder, HashingEmbedder, HierarchicalIndex, IndexNode
from core.models import (
    CommunityNode,
    Entity,
    GraphSnapshot,
    QueryContext,
    RetrievalResponse,
    ScoredItem,
    Triple,
)


# ===========================================================================
# Query expansion (pluggable)
# ===========================================================================

class QueryExpander(abc.ABC):
    """Turn a raw user query into a :class:`QueryContext`."""

    @abc.abstractmethod
    def expand(self, raw_query: str) -> QueryContext: ...


class IdentityExpander(QueryExpander):
    """Default: the query is its own expansion.

    Swap in an LLM-backed expander (synonyms, decomposition) in
    production.
    """

    def expand(self, raw_query: str) -> QueryContext:
        return QueryContext(raw_query=raw_query, expanded_queries=(raw_query,))


# ===========================================================================
# Reranker (pluggable)
# ===========================================================================

class Reranker(abc.ABC):
    """Score a candidate against the query; output MUST be in ``[0, 1]``.

    ``parent_score`` carries the accumulated relevance of the
    candidate's ancestor in the navigation tree so implementations can
    blend local match with global context.
    """

    @abc.abstractmethod
    def score_community(
        self, query: QueryContext, community: IndexNode, *, parent_score: float
    ) -> float: ...

    @abc.abstractmethod
    def score_entity(
        self, query: QueryContext, entity: Entity, *, parent_score: float
    ) -> float: ...

    @abc.abstractmethod
    def score_triple(
        self, query: QueryContext, triple: Triple, *, parent_score: float
    ) -> float: ...


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"\W+", s.lower()) if t}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n))) or 1.0
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n))) or 1.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _lex_overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _blend_with_parent(local: float, parent: float, w_parent: float) -> float:
    """ ``(1-w)·local + w·parent`` clamped to ``[0, 1]``. """
    return max(0.0, min(1.0, (1 - w_parent) * local + w_parent * parent))


class HybridReranker(Reranker):
    """Cosine similarity + lexical overlap, blended with a parent prior.

    The cosine term uses the same embedder the index was built with,
    so the query and the community centroids live in one space.
    """

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        embedder_weight: float = 0.6,
        lexical_weight: float = 0.4,
        parent_weight: float = 0.5,
    ) -> None:
        if abs(embedder_weight + lexical_weight - 1.0) > 1e-6:
            raise ValueError("embedder_weight + lexical_weight must equal 1.0")
        self._embedder = embedder or HashingEmbedder()
        self._w_emb = embedder_weight
        self._w_lex = lexical_weight
        self._w_par = parent_weight

    # --- Internals -----------------------------------------------------------

    def _query_vec(self, query: QueryContext) -> list[float]:
        if query.embedding:
            return list(query.embedding)
        return self._embedder.embed(query.raw_query)

    def _local(self, query_vec: Sequence[float], query_text: str, surface: str, cand_vec: Sequence[float]) -> float:
        cos = _cosine(query_vec, cand_vec)
        lex = _lex_overlap(query_text, surface)
        return self._w_emb * cos + self._w_lex * lex

    # --- Reranker API ----------------------------------------------------------

    def score_community(
        self, query: QueryContext, community: IndexNode, *, parent_score: float
    ) -> float:
        qv = self._query_vec(query)
        local = self._local(qv, query.raw_query, community.summary, community.centroid)
        return _blend_with_parent(local, parent_score, self._w_par)

    def score_entity(
        self, query: QueryContext, entity: Entity, *, parent_score: float
    ) -> float:
        qv = self._query_vec(query)
        ev = entity.embedding if entity.embedding is not None else self._embedder.embed(entity.name)
        local = self._local(qv, query.raw_query, entity.name, ev)
        return _blend_with_parent(local, parent_score, self._w_par)

    def score_triple(
        self, query: QueryContext, triple: Triple, *, parent_score: float
    ) -> float:
        qv = self._query_vec(query)
        surface = triple.evidence or f"{triple.head_id} {triple.relation_id} {triple.tail_id}"
        local = self._local(qv, query.raw_query, surface, self._embedder.embed(surface))
        return _blend_with_parent(local, parent_score, self._w_par)


# ===========================================================================
# Beam state
# ===========================================================================

@dataclass
class BeamState:
    """One candidate in the search frontier (internal)."""

    label: str                            # community_id / entity_id / triple_id
    kind: str = "community"               # "community" | "entity" | "triple"
    level: int = 0
    score: float = 0.0
    parent_label: str | None = None
    path: tuple[str, ...] = field(default_factory=tuple)
    payload: Any = None


# ===========================================================================
# Graph Beam Search
# ===========================================================================

class GraphBeamSearch:
    """Three-stage coarse-to-fine beam search (spec default: ``k=3``)."""

    def __init__(
        self,
        *,
        reranker: Reranker | None = None,
        query_expander: QueryExpander | None = None,
        embedder: Embedder | None = None,
        k: int = 3,
        max_entities_per_query: int = 200,
        max_triples_per_query: int = 300,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self._embedder = embedder or HashingEmbedder()
        self._reranker = reranker or HybridReranker(embedder=self._embedder)
        self._expander = query_expander or IdentityExpander()
        self._k = k
        self._max_entities = max_entities_per_query
        self._max_triples = max_triples_per_query

    @property
    def k(self) -> int:
        return self._k

    # --- Public API --------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        index: HierarchicalIndex,
        snapshot: GraphSnapshot,
    ) -> RetrievalResponse:
        ctx = self._expander.expand(query)
        if ctx.embedding is None:
            ctx.embedding = self._embedder.embed(query)

        trace: list[dict[str, Any]] = []

        # --- Stage 1: macro filtering on root communities -------------
        roots = index.root_nodes()
        coarse = self._top_k(
            [
                BeamState(label=n.community_id, kind="community", level=n.level, payload=n)
                for n in roots
            ],
            lambda s: self._reranker.score_community(ctx, s.payload, parent_score=0.5),
        )
        trace.append(self._trace_row(1, "macro", coarse))

        # --- Stage 2: medium refinement over children -----------------
        coarse_children: dict[str, list[IndexNode]] = {
            s.label: list(s.payload.children) for s in coarse
        }
        parent_score_of: dict[str, float] = {}
        medium_candidates: list[BeamState] = []
        seen: set[str] = set()
        for s in coarse:
            for child in coarse_children[s.label]:
                if child.community_id in seen:
                    continue
                seen.add(child.community_id)
                parent_score_of[child.community_id] = s.score
                medium_candidates.append(
                    BeamState(
                        label=child.community_id,
                        kind="community",
                        level=child.level,
                        parent_label=s.label,
                        payload=child,
                    )
                )
        if medium_candidates:
            medium = self._top_k(
                medium_candidates,
                lambda st: self._reranker.score_community(
                    ctx, st.payload, parent_score=parent_score_of[st.label]
                ),
            )
        else:
            # Degenerate single-level hierarchy: reuse the coarse layer.
            medium = coarse
        trace.append(self._trace_row(2, "medium", medium))

        # --- Stage 3: entity-level fine retrieval ---------------------
        entity_parent: dict[str, float] = {}
        fine_entities: list[Entity] = []
        seen_eids: set[str] = set()
        for ms in medium:
            for eid in index.entities_in(ms.label):
                if eid in seen_eids:
                    continue
                seen_eids.add(eid)
                entity_parent[eid] = max(entity_parent.get(eid, 0.0), ms.score)
                ent = snapshot.entities.get(eid)
                if ent is not None:
                    fine_entities.append(ent)
                if len(fine_entities) >= self._max_entities:
                    break
            if len(fine_entities) >= self._max_entities:
                break

        entity_states = [
            BeamState(
                label=e.id, kind="entity", level=-1,
                parent_label=None, payload=e,
            )
            for e in fine_entities
        ]
        entity_top = self._top_k(
            entity_states,
            lambda st: self._reranker.score_entity(
                ctx, st.payload, parent_score=entity_parent.get(st.label, 0.3)
            ),
        )

        fine_entity_ids = seen_eids
        fine_triples = [
            t
            for t in snapshot.verified_triples()
            if t.head_id in fine_entity_ids or t.tail_id in fine_entity_ids
        ][: self._max_triples]
        triple_states = [
            BeamState(label=t.id, kind="triple", level=-1, payload=t)
            for t in fine_triples
        ]
        triple_top = self._top_k(
            triple_states,
            lambda st: self._reranker.score_triple(
                ctx,
                st.payload,
                parent_score=max(
                    entity_parent.get(st.payload.head_id, 0.3),
                    entity_parent.get(st.payload.tail_id, 0.3),
                ),
            ),
        )

        trace.append(
            {
                "stage": 3,
                "level": "fine",
                "kept_entities": [s.label for s in entity_top],
                "entity_scores": [round(s.score, 3) for s in entity_top],
                "kept_triples": [s.label for s in triple_top],
                "triple_scores": [round(s.score, 3) for s in triple_top],
            }
        )

        # --- Assemble the explainable response -------------------------
        community_hits: list[ScoredItem[CommunityNode]] = []
        for s in coarse + medium:
            cn = snapshot.communities.get(s.label) or self._synthesise(s.payload)
            community_hits.append(
                ScoredItem(
                    item=cn,
                    score=s.score,
                    explanation=self._explain(s),
                    path=s.path,
                )
            )
        entity_hits = [
            ScoredItem(item=s.payload, score=s.score, explanation=self._explain(s), path=s.path)
            for s in entity_top
        ]
        triple_hits = [
            ScoredItem(item=s.payload, score=s.score, explanation=self._explain(s), path=s.path)
            for s in triple_top
        ]

        return RetrievalResponse(
            query=ctx,
            communities=community_hits,
            entities=entity_hits,
            triples=triple_hits,
            beam_trace=trace,
        )

    # --- Internals ---------------------------------------------------------

    def _top_k(
        self, states: list[BeamState], score_fn
    ) -> list[BeamState]:
        """Score every state in place, keep the top-k, and stamp paths."""
        for s in states:
            s.score = score_fn(s)
        ranked = sorted(states, key=lambda s: s.score, reverse=True)[: self._k]
        for s in ranked:
            s.path = (s.parent_label, s.label) if s.parent_label else (s.label,)
        return ranked

    @staticmethod
    def _trace_row(stage: int, level_name: str, states: Sequence[BeamState]) -> dict[str, Any]:
        return {
            "stage": stage,
            "level": level_name,
            "kept": [s.label for s in states],
            "scores": [round(s.score, 3) for s in states],
        }

    @staticmethod
    def _synthesise(node: IndexNode) -> CommunityNode:
        """Fallback when the snapshot lacks the annotated community."""
        return CommunityNode(
            id=node.community_id,
            level=node.level,
            entity_ids=node.entity_ids,
            summary=node.summary,
            centroid_embedding=node.centroid,
        )

    @staticmethod
    def _explain(s: BeamState) -> str:
        route = "/".join(s.path) or "-"
        return f"[{s.kind}] score={s.score:.3f} route={route}"


__all__ = [
    "QueryExpander",
    "IdentityExpander",
    "Reranker",
    "HybridReranker",
    "BeamState",
    "GraphBeamSearch",
]
