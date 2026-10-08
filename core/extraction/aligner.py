"""Entity alignment across extraction strategies.

Two ``Entity``s from different strategies may refer to the same real-world
object (e.g. ``"NYC"`` vs ``"New York City"``). ``EntityAligner`` is
responsible for canonicalising them to a single ``Entity.id``.

Implementations may be:

* rule-based (alias tables, case-folding, Levenshtein)
* embedding-based (cosine similarity on entity embeddings)
* hybrid (LLM-mediated only for the ambiguous tail)
"""
from __future__ import annotations

import abc

from core.models import Entity


class EntityAligner(abc.ABC):
    """Decide whether two ``Entity``s refer to the same real-world object."""

    @abc.abstractmethod
    def are_same(self, a: Entity, b: Entity) -> bool:
        """Return ``True`` if ``a`` and ``b`` should be merged."""
        raise NotImplementedError

    @abc.abstractmethod
    def canonical(self, entities: list[Entity]) -> list[Entity]:
        """Cluster and pick a canonical entity per cluster."""
        raise NotImplementedError


class RuleBasedAligner(EntityAligner):
    """Case-insensitive alias overlap + token Jaccard."""

    def __init__(self, *, jaccard_threshold: float = 0.6) -> None:
        self._threshold = jaccard_threshold

    def are_same(self, a: Entity, b: Entity) -> bool:
        raise NotImplementedError

    def canonical(self, entities: list[Entity]) -> list[Entity]:
        raise NotImplementedError


class EmbeddingAligner(EntityAligner):
    """Cosine similarity over pre-computed entity embeddings."""

    def __init__(self, *, cosine_threshold: float = 0.85) -> None:
        self._threshold = cosine_threshold

    def are_same(self, a: Entity, b: Entity) -> bool:
        raise NotImplementedError

    def canonical(self, entities: list[Entity]) -> list[Entity]:
        raise NotImplementedError
