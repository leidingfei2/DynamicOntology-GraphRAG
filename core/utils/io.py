"""Filesystem and JSON helpers.

Centralised so we can swap the underlying format (e.g. orjson → msgspec)
in one place.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson

from core.models import Document


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON-Lines file and return the list of decoded objects."""
    raise NotImplementedError


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write an iterable of objects as JSON-Lines."""
    raise NotImplementedError


def read_json(path: str | Path) -> Any:
    """Read a single JSON document, using orjson for speed."""
    return orjson.loads(Path(path).read_bytes())


def write_json(path: str | Path, obj: Any, *, indent: bool = False) -> None:
    """Write a single JSON document, using orjson for speed."""
    opts = orjson.OPT_INDENT_2 if indent else 0
    Path(path).write_bytes(orjson.dumps(obj, option=opts))


def load_documents(path: str | Path) -> list[Document]:
    """Read documents from a JSON-Lines file with one ``Document`` per line."""
    raise NotImplementedError


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file (typically under ``configs/``)."""
    import yaml  # local import to keep cold-start cheap

    return yaml.safe_load(Path(path).read_text())


# Avoid an unused-import warning in tools that only need the JSON helpers.
_ = json
