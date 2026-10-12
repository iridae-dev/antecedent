"""One recursive freeze for report payloads.

Call sites used to disagree: some recursed into every :class:`Mapping` and
every tuple, and some only into ``dict`` and ``list``. A tuple of dicts then
stayed mutable. JSON objects and arrays are ``dict`` and ``list``, so a
payload that came from ``json.loads`` freezes to the same proxies and tuples
as before. A tuple or a non-dict mapping is now frozen all the way down.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze(value: Any) -> Any:
    """Return an immutable copy of mappings, lists and tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value
