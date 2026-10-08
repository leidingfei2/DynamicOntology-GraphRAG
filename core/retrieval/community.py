"""Hierarchical community clustering over the verified ABox.

``HierarchicalClusterer`` builds a multi-level ``CommunityNode`` tree on
top of a ``GraphSnapshot`` so the retriever can navigate from coarse
communities down to fine-grained entities/triples.

Backends are pluggable via the ``CommunityDetector`` ABC:

* ``LouvainDetector``
* ``LeidenDetector``
* ``LabelPropagationDetector``
* ``HierarchicalAgglomerativeDetector`` (e.g. for very small graphs)
"""
from __future__ import annotations

import abc
from collections.abc import Iterable

import networkx as nx

from core.models import (
    CommunityAlgorithm,
    CommunityNode,
    GraphSnapshot,
)


class CommunityDetector(abc.ABC):
    """One community-detection backend, run at a single level."""

    algorithm: CommunityAlgorithm

    @abc.abstractmethod
    def detect(
        self,
        graph: nx.Graph,
        *,
        level: int,
        min_community_size: int = 3,
    ) -> list[CommunityNode]:
        """Return flat communities for the given graph at ``level``."""
        raise NotImplementedError


class LouvainDetector(CommunityDetector):
    algorithm = CommunityAlgorithm.LOUVAIN

    def __init__(self, resolution: float = 1.0, random_state: int = 42) -> None:
        self._resolution = resolution
        self._random_state = random_state

    def detect(
        self,
        graph: nx.Graph,
        *,
        level: int,
        min_community_size: int = 3,
    ) -> list[CommunityNode]:
        raise NotImplementedError


class LeidenDetector(CommunityDetector):
    algorithm = CommunityAlgorithm.LEIDEN

    def __init__(self, resolution: float = 1.0, random_state: int = 42) -> None:
        self._resolution = resolution
        self._random_state = random_state

    def detect(
        self,
        graph: nx.Graph,
        *,
        level: int,
        min_community_size: int = 3,
    ) -> list[CommunityNode]:
        raise NotImplementedError


class HierarchicalAgglomerativeDetector(CommunityDetector):
    algorithm = CommunityAlgorithm.HIERARCHICAL

    def __init__(self, distance_threshold: float = 0.5) -> None:
        self._threshold = distance_threshold

    def detect(
        self,
        graph: nx.Graph,
        *,
        level: int,
        min_community_size: int = 3,
    ) -> list[CommunityNode]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

class HierarchicalClusterer:
    """Build a multi-level community tree on top of a ``GraphSnapshot``.

    The top level (largest ``level`` index) is the coarsest — root of the
    beam-search traversal. The bottom level (level 0) is the finest.
    """

    def __init__(
        self,
        detector: CommunityDetector,
        *,
        hierarchy_levels: int = 3,
        min_community_size: int = 3,
    ) -> None:
        if hierarchy_levels < 1:
            raise ValueError("hierarchy_levels must be >= 1")
        self._detector = detector
        self._hierarchy_levels = hierarchy_levels
        self._min_community_size = min_community_size

    def cluster(self, snapshot: GraphSnapshot) -> list[CommunityNode]:
        """Return the flattened set of ``CommunityNode``s across all levels."""
        raise NotImplementedError

    def root_communities(self, communities: Iterable[CommunityNode]) -> list[CommunityNode]:
        """Filter to the coarsest level only (no ``parent_id``)."""
        raise NotImplementedError
