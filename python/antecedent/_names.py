"""Column names for a refusal, without reading the column values."""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Mapping
from typing import Any

# id(columns object) plus the names. A replaced ``.columns`` misses the cache.
_CACHE: weakref.WeakKeyDictionary[Any, tuple[int, tuple[str, ...]]] = weakref.WeakKeyDictionary()


def table_names(data: Any) -> tuple[str, ...]:
    """Names from ``.column_names``, ``.columns``, or a mapping's keys.

    This does not ingest values. Nested refusal wrappers used to call
    ``ingest_columns`` once per frame and turn a bad table into an empty name
    list. Only the attribute and key reads that can fail on a weird table are
    caught; anything else propagates.
    """
    token = _cache_token(data)
    cached: tuple[int, tuple[str, ...]] | None
    try:
        cached = _CACHE[data]
    except (KeyError, TypeError):
        cached = None
    if cached is not None and cached[0] == token:
        return cached[1]
    names = _read_names(data)
    with contextlib.suppress(TypeError):
        _CACHE[data] = (token, names)
    return names


def _cache_token(data: Any) -> int:
    columns = getattr(data, "column_names", None)
    if columns is None:
        columns = getattr(data, "columns", None)
    return id(data if columns is None else columns)


def _read_names(data: Any) -> tuple[str, ...]:
    if isinstance(data, (str, bytes)):
        return ()
    column_names = getattr(data, "column_names", None)
    if column_names is not None:
        try:
            return tuple(str(name) for name in column_names)
        except (TypeError, ValueError):
            return ()
    columns = getattr(data, "columns", None)
    if columns is not None and not isinstance(data, Mapping):
        try:
            return tuple(str(name) for name in columns)
        except (TypeError, ValueError):
            return ()
    if isinstance(data, Mapping):
        try:
            return tuple(str(key) for key in data)
        except (TypeError, ValueError):
            return ()
    return ()
