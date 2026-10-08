"""Lightweight in-process metric collectors.

The framework intentionally does not depend on ``prometheus_client`` at
runtime; counters / histograms here are exposed via the ``MetricsRegistry``
and can be scraped by an external process (or just printed at exit).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Counter:
    name: str
    help: str = ""
    value: int = 0

    def inc(self, by: int = 1) -> None:
        self.value += by


@dataclass
class Histogram:
    name: str
    help: str = ""
    buckets: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    counts: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    total: int = 0
    sum_: float = 0.0

    def observe(self, value: float) -> None:
        self.total += 1
        self.sum_ += value
        for b in self.buckets:
            if value <= b:
                self.counts[b] += 1


class MetricsRegistry:
    """Process-wide registry of ``Counter`` and ``Histogram`` metrics."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}

    # --- Registration -------------------------------------------------------

    def counter(self, name: str, help: str = "") -> Counter:
        if name not in self._counters:
            self._counters[name] = Counter(name=name, help=help)
        return self._counters[name]

    def histogram(self, name: str, help: str = "", buckets: tuple[float, ...] | None = None) -> Histogram:
        if name not in self._histograms:
            h = Histogram(name=name, help=help)
            if buckets is not None:
                h.buckets = buckets
            self._histograms[name] = h
        return self._histograms[name]

    # --- Snapshot -----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "counters": {n: c.value for n, c in self._counters.items()},
            "histograms": {
                n: {
                    "total": h.total,
                    "sum": h.sum_,
                    "buckets": dict(h.counts),
                }
                for n, h in self._histograms.items()
            },
        }


# Module-level singleton, mirroring ``logging.getLogger`` ergonomics.
REGISTRY = MetricsRegistry()
