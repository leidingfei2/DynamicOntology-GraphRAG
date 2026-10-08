"""NetworkX adapter for the verified ABox.

Centralises all NetworkX interactions so other modules never touch the
graph library directly. This makes it cheap to swap in ``igraph``,
``rustworkx``, or a persistent backend later.
"""
from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from core.models import (
    CommunityNode,
    Entity,
    GraphSnapshot,
    Relation,
    Triple,
)


class GraphAdapter:
    """Bi-directional adapter between a ``GraphSnapshot`` and ``networkx.Graph``.

    Two graphs are exposed:

    * ``entity_graph`` — undirected, nodes = entities, edges = triples.
      Edge weight = ``Triple.confidence``.
    * ``typed_graph``  — directed, nodes = entity IDs, edge attributes
      include ``relation_id`` and ``verified`` flag.

    Use the typed graph for routing / beam search; the entity graph for
    community detection.
    """

    def __init__(self, snapshot: GraphSnapshot) -> None:
        self._snapshot = snapshot
        self._entity_graph: nx.Graph | None = None
        self._typed_graph: nx.DiGraph | None = None

    # --- Build --------------------------------------------------------------

    def build_entity_graph(self) -> nx.Graph:
        """Build (or return cached) undirected entity graph."""
        raise NotImplementedError

    def build_typed_graph(self) -> nx.DiGraph:
        """Build (or return cached) directed typed graph with relation edges."""
        raise NotImplementedError

    # --- Lookups ------------------------------------------------------------

    def neighbors(self, entity_id: str, *, max_hops: int = 1) -> list[str]:
        """Return the entity IDs reachable within ``max_hops``."""
        raise NotImplementedError

    def triples_between(self, head_id: str, tail_id: str) -> list[Triple]:
        """Return all verified ``Triple``s whose endpoints are head/tail."""
        raise NotImplementedError

    def entities_in_community(self, community: CommunityNode) -> Iterable[Entity]:
        """Yield every entity member of ``community``."""
        raise NotImplementedError

    # --- Mutators (rare; usually snapshot is rebuilt) -----------------------

    def add_community(self, community: CommunityNode) -> None:
        """Attach a community to the snapshot (and to the entity graph)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class GraphPersistence:
    """Save / load ``GraphSnapshot`` in JSON and GraphML formats."""

    @staticmethod
    def to_json(snapshot: GraphSnapshot, path: str) -> None:
        raise NotImplementedError

    @staticmethod
    def from_json(path: str) -> GraphSnapshot:
        raise NotImplementedError

    @staticmethod
    def to_graphml(snapshot: GraphSnapshot, path: str) -> None:
        """Export the entity graph as GraphML for external visualisation."""
        raise NotImplementedError
