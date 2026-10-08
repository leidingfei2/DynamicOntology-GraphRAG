"""Hierarchical community clustering and index construction.

Given a verified :class:`core.models.GraphSnapshot`, this module:

1. Builds a ``networkx`` graph (entity + triple layer).
2. Runs community detection bottom-up: level 0 (finest, real
   communities) → level L-1 (coarsest, root communities).
3. Generates a textual summary and a context-aware vector for every
   community. The context vector is a weighted blend of:
      * the community's own centroid embedding,
      * the centroid of its parent (so children pull the parent
        semantic context "upward" into the navigation index).
4. Exposes a top-down navigation tree so the retriever can walk
   coarse-to-fine without re-computing anything.

All heavy components (embedder, summariser, community-detection
backend) are pluggable. The defaults are deterministic and
*dependency-free*:

* :class:`HashingEmbedder` — a 256-dim bag-of-character-trigrams
  pseudo-embedding. Same text → same vector; similar text → similar
  vector via shared n-grams. Good enough for cosine ranking and zero
  install.
* :class:`LeadKSummariser` — first-K-character summary. Replace with
  any LLM-backed implementation in production.
* :class:`NetworkXHierarchicalClusterer` — Louvain via
  ``networkx``'s greedy modularity (no ``python-louvain`` required).
  Swappable to Leiden when ``leidenalg`` is available.
"""
from __future__ import annotations

import abc
import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

# Optional networkx
try:
    import networkx as nx
    _HAS_NX = True
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]
    _HAS_NX = False

from core.models import (  # noqa: E402
    CommunityAlgorithm,
    CommunityNode,
    GraphSnapshot,
)


# ===========================================================================
# Embedder (pluggable)
# ===========================================================================

class Embedder(abc.ABC):
    """Map a string to a dense ``list[float]`` of fixed dimensionality."""

    @property
    @abc.abstractmethod
    def dim(self) -> int: ...

    @abc.abstractmethod
    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class HashingEmbedder(Embedder):
    """Deterministic character-trigram hash embedding.

    Why this exists
    ---------------
    We want a single embedder that works *with no model download* so
    the e2e pipeline can be exercised on a fresh machine. The trick:

    * Build a vocabulary of character trigrams (256 possible first
      chars × varying continuations).
    * Hash each trigram to one of ``dim`` buckets with a salted
      SHA-256.
    * L2-normalise so cosine = dot product.

    Properties
    ----------
    * Identical text → identical vector.
    * Texts sharing trigrams → vectors with non-zero dot product.
    * Fully deterministic across processes and Python versions.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim < 16:
            raise ValueError("dim must be at least 16")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        if not text:
            return vec
        # Padded trigrams so we capture leading / trailing context.
        padded = f"  {text.lower()}  "
        for i in range(len(padded) - 2):
            tri = padded[i : i + 3]
            h = hashlib.sha256(tri.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:4], "big") % self._dim
            # Sign ±1 from the next byte for symmetry.
            sign = 1.0 if h[4] & 1 else -1.0
            vec[bucket] += sign
        # L2 normalise
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


# ===========================================================================
# Summariser (pluggable)
# ===========================================================================

class Summariser(abc.ABC):
    """Produce a short natural-language summary of a community."""

    @abc.abstractmethod
    def summarise(self, names: Sequence[str], evidence: Sequence[str]) -> str: ...


class LeadKSummariser(Summariser):
    """Concatenate the top entity names + the first non-empty evidence
    spans, truncated to ``max_chars``. Always returns something useful.
    """

    def __init__(self, max_chars: int = 280) -> None:
        self._max = max_chars

    def summarise(self, names: Sequence[str], evidence: Sequence[str]) -> str:
        head = ", ".join(n for n in dict.fromkeys(names) if n)[: self._max]
        tail = ""
        for ev in evidence:
            ev = (ev or "").strip()
            if not ev:
                continue
            tail = ev[: self._max]
            break
        if not tail:
            return head
        return f"{head}. {tail}"[: self._max]


# ===========================================================================
# Community detection (pluggable)
# ===========================================================================

class CommunityDetector(abc.ABC):
    """Run a single layer of community detection on a graph."""

    algorithm: CommunityAlgorithm

    @abc.abstractmethod
    def detect(
        self,
        graph: "nx.Graph",
        *,
        level: int,
        min_community_size: int = 3,
    ) -> list[CommunityNode]:
        """Return a flat list of ``CommunityNode``s for this layer."""


class GreedyModularityDetector(CommunityDetector):
    """Louvain-equivalent via ``networkx.algorithms.community.greedy_modularity_communities``.

    This ships with ``networkx`` itself — no extra dependency. For
    higher-quality Leiden clusters, install ``leidenalg`` and use
    :class:`LeidenDetector` instead.
    """

    algorithm = CommunityAlgorithm.LOUVAIN

    def __init__(self, cutoff: int = 1) -> None:
        self._cutoff = cutoff

    def detect(
        self,
        graph: "nx.Graph",
        *,
        level: int,
        min_community_size: int = 3,
    ) -> list[CommunityNode]:
        if graph.number_of_nodes() == 0:
            return []

        # An edgeless graph cannot be split by modularity: fall back to
        # one community covering everything (subject to the size floor).
        if graph.number_of_edges() == 0:
            if graph.number_of_nodes() < min_community_size:
                return []
            members = sorted(graph.nodes())
            return [
                CommunityNode(
                    id=_cid("cm", level, 0, members),
                    level=level,
                    entity_ids=tuple(members),
                    algorithm=self.algorithm,
                    modularity=0.0,
                )
            ]

        from networkx.algorithms.community import greedy_modularity_communities

        comms = list(greedy_modularity_communities(graph, cutoff=self._cutoff))
        # Modularity is a property of the *partition*, not of a single
        # community — compute it once for the full result set.
        try:
            mod = float(nx.community.modularity(graph, comms))
        except Exception:  # pragma: no cover - defensive (e.g. singleton graphs)
            mod = 0.0

        out: list[CommunityNode] = []
        for cidx, members in enumerate(comms):
            members = sorted(members)
            if len(members) < min_community_size:
                continue
            out.append(
                CommunityNode(
                    id=_cid("cm", level, cidx, members),
                    level=level,
                    entity_ids=tuple(members),
                    algorithm=self.algorithm,
                    modularity=mod,
                )
            )
        return out


# ===========================================================================
# Hierarchical index
# ===========================================================================

@dataclass
class IndexNode:
    """A node in the top-down navigation tree.

    Plain dataclass (not Pydantic) because these are *internal* — they
    never cross a network boundary.
    """

    community_id: str
    level: int
    centroid: list[float] = field(default_factory=list)
    summary: str = ""
    entity_ids: tuple[str, ...] = field(default_factory=tuple)
    children: list["IndexNode"] = field(default_factory=list)


class HierarchicalIndex:
    """Multi-level community tree over a verified ``GraphSnapshot``.

    Usage
    -----

    .. code-block:: python

        index = HierarchicalIndex().build(snapshot)
        roots = index.root_nodes()       # start beam search from here
        for n in index.walk():
            ...                          # depth-first traversal
    """

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        summariser: Summariser | None = None,
        detector: CommunityDetector | None = None,
        hierarchy_levels: int = 3,
        min_community_size: int = 2,
        parent_blend: float = 0.4,
    ) -> None:
        if hierarchy_levels < 1:
            raise ValueError("hierarchy_levels must be >= 1")
        if not 0.0 <= parent_blend <= 1.0:
            raise ValueError("parent_blend must be in [0, 1]")
        if not _HAS_NX:
            raise ImportError(
                "HierarchicalIndex requires networkx; "
                "install with `pip install networkx`."
            )
        self._embedder = embedder or HashingEmbedder()
        self._summariser = summariser or LeadKSummariser()
        self._detector = detector or GreedyModularityDetector()
        self._levels = hierarchy_levels
        self._min_size = min_community_size
        self._parent_blend = parent_blend

        # Built state.
        self._snapshot: GraphSnapshot | None = None
        self._graph: "nx.Graph | None" = None
        self._levels_data: list[list[CommunityNode]] = []
        self._by_id: dict[str, CommunityNode] = {}
        self._roots: list[IndexNode] = []

    # --- Public API --------------------------------------------------------

    @property
    def levels(self) -> int:
        return self._levels

    @property
    def embedder(self) -> Embedder:
        """The embedder this index was built with (public so callers can
        build the reranker in the same vector space)."""
        return self._embedder

    def build(self, snapshot: GraphSnapshot) -> "HierarchicalIndex":
        """Build the full hierarchy from a verified snapshot.

        Idempotent: calling twice resets state and rebuilds.

        Side effect: the fully-annotated :class:`CommunityNode`s are
        written into ``snapshot.communities`` so downstream consumers
        (beam search, persistence) can look them up by ID.
        """
        self._reset()
        self._snapshot = snapshot
        self._graph = self._build_entity_graph(snapshot)
        self._levels_data = self._cluster_bottom_up(self._graph)
        self._by_id = {c.id: c for layer in self._levels_data for c in layer}
        self._annotate_with_summaries_and_vectors()
        self._roots = self._build_navigation_tree()
        # Write the annotated communities back into the snapshot.
        snapshot.communities.update(self._by_id)
        return self

    def root_nodes(self) -> list[IndexNode]:
        """Return the coarsest-level communities as ``IndexNode``s."""
        return list(self._roots)

    def get(self, community_id: str) -> IndexNode | None:
        """Look up any community in the tree by ID (DFS)."""
        return self._find(self._roots, community_id)

    def walk(self) -> Iterable[IndexNode]:
        """Depth-first iteration over all communities in the tree."""
        yield from self._walk(self._roots)

    def entities_in(self, community_id: str) -> list[str]:
        """Return the entity IDs that directly belong to ``community_id``."""
        c = self._by_id.get(community_id)
        return list(c.entity_ids) if c else []

    def children_of(self, community_id: str) -> list[IndexNode]:
        """Return the immediate child communities in the navigation tree."""
        node = self.get(community_id)
        return list(node.children) if node else []

    # --- Internals ----------------------------------------------------------

    def _reset(self) -> None:
        self._snapshot = None
        self._graph = None
        self._levels_data = []
        self._by_id = {}
        self._roots = []

    def _build_entity_graph(self, snapshot: GraphSnapshot) -> "nx.Graph":
        g: "nx.Graph" = nx.Graph()
        for e in snapshot.entities.values():
            g.add_node(e.id, name=e.name, type_id=e.type_id)
        for t in snapshot.verified_triples():
            h, _, te = t.to_tuple()
            if h in g and te in g:
                if g.has_edge(h, te):
                    # Multi-edge: bump weight.
                    g[h][te]["weight"] = g[h][te].get("weight", 1.0) + 1.0
                else:
                    g.add_edge(h, te, weight=1.0, relation_id=t.relation_id)
        return g

    def _cluster_bottom_up(self, graph: "nx.Graph") -> list[list[CommunityNode]]:
        """Cluster bottom-up: detect once per level over a *contracted*
        graph whose nodes are the previous level's communities.

        At level 0 the detector runs on the raw entity graph and the
        resulting communities contain **entity IDs**. At every level
        ℓ ≥ 1 the detector runs on a graph whose nodes are the level-ℓ-1
        **community IDs**, so each detected community's ``entity_ids``
        field initially holds community IDs — we immediately expand it
        to the union of the underlying entity IDs and wire the
        parent/child links in both directions.
        """
        levels: list[list[CommunityNode]] = []
        prev: list[CommunityNode] = []
        prev_sig: frozenset[frozenset[str]] | None = None

        for lvl in range(self._levels):
            if lvl == 0:
                target_graph = graph
            else:
                target_graph = self._contract_graph(graph, prev)

            comms = self._detector.detect(
                target_graph, level=lvl, min_community_size=self._min_size
            )

            if lvl > 0:
                id_to_comm = {c.id: c for c in prev}
                for c in comms:
                    expanded: set[str] = set()
                    child_ids: list[str] = []
                    for node_id in c.entity_ids:
                        child = id_to_comm.get(node_id)
                        if child is not None:
                            expanded.update(child.entity_ids)
                            child_ids.append(child.id)
                        else:
                            # Defensive: a raw entity ID that somehow
                            # survived contraction unmapped.
                            expanded.add(node_id)
                    c.entity_ids = tuple(sorted(expanded))
                    c.child_ids = tuple(child_ids)
                    for child_id in child_ids:
                        id_to_comm[child_id].parent_id = c.id

                # Early-stop conditions (the graph cannot get any
                # coarser): no community survived the size floor, or
                # the partition is identical to the level below.
                if not comms:
                    break
                sig = frozenset(frozenset(c.entity_ids) for c in comms)
                if sig == prev_sig:
                    break

            levels.append(comms)
            prev = comms
            prev_sig = frozenset(frozenset(c.entity_ids) for c in comms)

        return levels

    def _contract_graph(
        self,
        original: "nx.Graph",
        comms: list[CommunityNode],
    ) -> "nx.Graph":
        """Return a graph whose nodes are the given communities and whose
        edges are weighted sums of cross-community edges in the
        original graph.
        """
        g: "nx.Graph" = nx.Graph()
        # Map each original node to its community id.
        node_to_comm: dict[str, str] = {}
        for c in comms:
            for n in c.entity_ids:
                node_to_comm[n] = c.id
            g.add_node(c.id, size=len(c.entity_ids))
        # Aggregate cross-community edges.
        agg: dict[tuple[str, str], float] = defaultdict(float)
        for u, v, data in original.edges(data=True):
            cu, cv = node_to_comm.get(u), node_to_comm.get(v)
            if cu is None or cv is None or cu == cv:
                continue
            a, b = sorted((cu, cv))
            agg[(a, b)] += float(data.get("weight", 1.0))
        for (a, b), w in agg.items():
            g.add_edge(a, b, weight=w)
        return g

    def _annotate_with_summaries_and_vectors(self) -> None:
        """Attach a context-aware vector and a summary to every community."""
        snap = self._snapshot
        assert snap is not None
        # Pre-compute entity-name lookup.
        name_of: dict[str, str] = {e.id: e.name for e in snap.entities.values()}
        # Per-community, gather (a) entity names, (b) evidence strings.
        ev_by_entity: dict[str, list[str]] = defaultdict(list)
        for t in snap.verified_triples():
            for eid in (t.head_id, t.tail_id):
                if t.evidence:
                    ev_by_entity[eid].append(t.evidence)
        # Annotate bottom-up so parents blend in their children's info.
        for layer in reversed(self._levels_data):  # coarsest → finest
            for c in layer:
                names = [name_of.get(e, e) for e in c.entity_ids]
                evidence: list[str] = []
                for e in c.entity_ids:
                    evidence.extend(ev_by_entity.get(e, [])[:3])
                c.summary = self._summariser.summarise(names, evidence)
                own_text = c.summary or " ".join(names)
                own_vec = self._embedder.embed(own_text)
                # Blend with parent.
                if c.parent_id and c.parent_id in self._by_id:
                    parent_vec = self._by_id[c.parent_id].centroid_embedding
                    if parent_vec:
                        own_vec = _l2_normalise(
                            _lerp(parent_vec, own_vec, self._parent_blend)
                        )
                c.centroid_embedding = own_vec

    def _build_navigation_tree(self) -> list[IndexNode]:
        """Build the top-down ``IndexNode`` tree from ``CommunityNode``s."""
        # Build a dict of (id → IndexNode) so we can wire children.
        idx: dict[str, IndexNode] = {}
        for layer in self._levels_data:
            for c in layer:
                idx[c.id] = IndexNode(
                    community_id=c.id,
                    level=c.level,
                    centroid=list(c.centroid_embedding or []),
                    summary=c.summary,
                    entity_ids=c.entity_ids,
                )
        # Wire children.
        for c in [cc for layer in self._levels_data for cc in layer]:
            if c.parent_id and c.parent_id in idx:
                idx[c.parent_id].children.append(idx[c.id])
        # Roots = coarsest *non-empty* level's parentless communities.
        for layer in reversed(self._levels_data):
            if layer:
                return [idx[c.id] for c in layer if c.parent_id is None]
        return []

    # --- DFS helpers --------------------------------------------------------

    def _find(self, nodes: Sequence[IndexNode], cid: str) -> IndexNode | None:
        for n in nodes:
            if n.community_id == cid:
                return n
            hit = self._find(n.children, cid)
            if hit is not None:
                return hit
        return None

    def _walk(self, nodes: Sequence[IndexNode]) -> Iterable[IndexNode]:
        for n in nodes:
            yield n
            yield from self._walk(n.children)


# ===========================================================================
# Helpers
# ===========================================================================

def _cid(prefix: str, level: int, idx: int, members: Sequence[str]) -> str:
    """Deterministic community ID derived from level + member set."""
    h = hashlib.sha1(f"{level}:{idx}:{','.join(sorted(members))}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{level}_{h}"


def _l2_normalise(v: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _lerp(a: Sequence[float], b: Sequence[float], t: float) -> list[float]:
    """``(1-t) * a + t * b`` — element-wise linear blend."""
    if len(a) != len(b):
        # Pad / truncate the shorter to keep the contract.
        n = min(len(a), len(b))
        a = list(a[:n])
        b = list(b[:n])
    return [(1 - t) * x + t * y for x, y in zip(a, b)]


__all__ = [
    "Embedder",
    "HashingEmbedder",
    "Summariser",
    "LeadKSummariser",
    "CommunityDetector",
    "GreedyModularityDetector",
    "IndexNode",
    "HierarchicalIndex",
]
