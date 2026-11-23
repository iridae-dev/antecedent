"""One recursive freeze for report payloads.

Call sites used to disagree: some recursed into every :class:`Mapping` and
every tuple, and some only into ``dict`` and ``list``. A tuple of dicts then
stayed mutable. JSON objects and arrays are ``dict`` and ``list``, so a
payload that came from ``json.loads`` freezes to the same proxies and tuples
as before. A tuple or a non-dict mapping is now frozen all the way down.
"""

from __future__ import annotations

import copy
import weakref
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

_ATOMIC_TYPES = frozenset((type(None), bool, int, float, complex, str, bytes))

#: Frozen containers retained by a live owner, keyed by ``id``: ``[value, owners]``.
#: Each entry holds a strong reference, so its ``id`` cannot be reused while it is
#: here; :func:`freeze` returns such a value as is instead of copying it again.
_RETAINED: dict[int, list[Any]] = {}


def freeze(value: Any) -> Any:
    """Return an immutable copy of mappings, lists and tuples."""
    # Report schemas have many absent fields and scalar leaves. Exact types
    # avoid treating a scalar subclass with dataclass fields as immutable.
    kind = type(value)
    if kind in _ATOMIC_TYPES:
        return value
    if is_retained_frozen(value):
        return value
    if kind is dict:
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if kind is list or kind is tuple:
        return tuple(freeze(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
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


def is_retained_frozen(value: Any) -> bool:
    """Whether ``value`` is a frozen container a live owner retains (see :func:`retain_frozen`)."""
    retained = _RETAINED.get(id(value))
    return retained is not None and retained[0] is value


def _frozen_containers(value: Any, found: list[Any]) -> None:
    # Only containers ``freeze`` produced are shared as is. A dataclass is
    # detached afresh on every freeze, so only its fields are recorded.
    kind = type(value)
    if (
        kind is MappingProxyType
        or kind is tuple
        or (kind is np.ndarray and not value.flags.writeable)
    ):
        found.append(value)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _frozen_containers(getattr(value, field.name), found)


def _release(keys: tuple[int, ...]) -> None:
    for key in keys:
        entry = _RETAINED.get(key)
        if entry is None:
            continue
        entry[1] -= 1
        if entry[1] <= 0:
            del _RETAINED[key]


def retain_frozen(owner: Any, *values: Any) -> None:
    """Record ``values``, already returned by :func:`freeze`, for ``owner``'s lifetime.

    A later freeze of the same object (a model copy, a report built from the
    owner's retained contract) then shares it instead of copying it again.
    An owner that cannot be weakly referenced records nothing.
    """
    found: list[Any] = []
    for value in values:
        _frozen_containers(value, found)
    if not found:
        return
    keys = tuple(id(value) for value in found)
    try:
        weakref.finalize(owner, _release, keys)
    except TypeError:
        return
    for key, value in zip(keys, found, strict=True):
        entry = _RETAINED.get(key)
        if entry is not None and entry[0] is value:
            entry[1] += 1
        else:
            _RETAINED[key] = [value, 1]


def thaw(value: Any) -> Any:
    """An independent JSON-compatible copy of a frozen read model."""
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
