"""Resource-checked column ingest shared by selective recalculation families."""

from collections.abc import Mapping, Sized
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from . import _native
from .errors import CausalTypeError, CausalValueError
from .recalc import _column

BYTES_PER_VALUE = 8
"""Bytes one ``float64`` value occupies in each copy the bridge makes."""


def _check_resources(
    memory_limit_bytes: int | None, cancel: _native.CancellationToken | None
) -> None:
    if memory_limit_bytes is not None:
        if isinstance(memory_limit_bytes, bool) or not isinstance(memory_limit_bytes, int):
            raise CausalTypeError("memory_limit_bytes must be an integer")
        if not 0 <= memory_limit_bytes <= 2**64 - 1:
            raise CausalValueError("memory_limit_bytes must be in 0..=18446744073709551615")
    if cancel is not None and not isinstance(cancel, _native.CancellationToken):
        raise CausalTypeError("cancel must be a CancellationToken")


def _admit(values: int, memory_limit_bytes: int | None) -> None:
    """Refuse a copy of ``values`` numbers above the memory limit, before it is made.

    Native code applies the same budget to its own copy. Cancellation is not decided here:
    the token travels to the run, which refuses with its family's ``recalc.cancelled`` result.
    """
    planned = values * BYTES_PER_VALUE
    if memory_limit_bytes is not None and planned > memory_limit_bytes:
        raise CausalValueError(
            f"recalc.memory_budget_exceeded: planned input copy of {planned} bytes "
            f"exceeds memory limit of {memory_limit_bytes} bytes",
            reason_code="invalid_argument",
        )


def _value_count(column: ArrayLike) -> int:
    size = getattr(column, "size", None)
    if isinstance(size, int):
        return size
    if isinstance(column, Sized):
        return len(column)
    return int(np.asarray(column).size)


def _columns(
    data: Mapping[str, ArrayLike],
    *,
    memory_limit_bytes: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> tuple[list[str], list[Any]]:
    """Validate and convert ``data`` to float64 columns within the caller's resource limits."""
    _check_resources(memory_limit_bytes, cancel)
    names = list(data)
    _admit(sum(_value_count(data[name]) for name in names), memory_limit_bytes)
    return names, [_column(name, data[name]) for name in names]
