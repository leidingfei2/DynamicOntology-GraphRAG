"""Centralised logging setup.

All modules call ``get_logger(__name__)`` to get a logger that is
pre-configured by ``configure_logging`` (called once at process start).
"""
from __future__ import annotations

import logging
import sys


_CONFIGURED = False


def configure_logging(level: int = logging.INFO, *, json_logs: bool = False) -> None:
    """Idempotently install a single root handler."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler: logging.Handler
    if json_logs:
        try:
            from pythonjsonlogger import jsonlogger  # type: ignore

            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
            )
        except ImportError:  # pragma: no cover - graceful fallback
            handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger; lazily configure root logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
