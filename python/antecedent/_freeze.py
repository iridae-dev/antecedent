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

import numpy as np

_ATOMIC_TYPES = frozenset((type(None), bool, int, float, complex, str, bytes))


def freeze(value: Any) -> Any:
    """Return an immutable copy of mappings, lists and tuples."""
    # Report schemas have many absent fields and scalar leaves. Exact types
    # avoid treating a scalar subclass with dataclass fields as immutable.
    if type(value) in _ATOMIC_TYPES:
        return value
    from dataclasses import fields, is_dataclass

    if is_dataclass(value) and not isinstance(value, type):
        import copy

        detached = copy.copy(value)
        for field in fields(value):
            object.__setattr__(detached, field.name, freeze(getattr(value, field.name)))
        return detached
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return tuple(freeze(item) for item in value.tolist())
        # An immutable bytes owner prevents callers from re-enabling writes.
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    return value


def thaw(value: Any) -> Any:
    """An independent JSON-compatible copy of a frozen read model."""
    from dataclasses import fields, is_dataclass
    from enum import Enum

    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw(item) for item in value]
    if isinstance(value, np.ndarray):
        return thaw(value.tolist())
    if isinstance(value, np.generic):
        return thaw(value.item())
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: thaw(getattr(value, f.name))
            for f in fields(value)
            if not f.name.startswith("_")
        }
    if isinstance(value, bytes):
        return value.hex()
    return value
