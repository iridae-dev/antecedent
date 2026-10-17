"""Resource-checked column ingest shared by selective recalculation families."""

from collections.abc import Mapping, Sequence, Sized
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from . import _native
from .errors import CausalTypeError, CausalValueError
from .recalc import _column

BYTES_PER_VALUE = 8
"""Bytes one ``float64`` value occupies in each copy the bridge makes."""

MAX_PLANNED_BYTES = 2**64 - 1
"""Largest byte count native code can account for; a larger plan is an arithmetic overflow."""


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
    if planned > MAX_PLANNED_BYTES:
        raise CausalValueError(
            f"recalc.limits_exceeded: planned input of {values} values overflows the checked "
            f"byte count of {MAX_PLANNED_BYTES}",
            reason_code="invalid_argument",
        )
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


def _float_columns(
    named: Sequence[tuple[str, ArrayLike]],
    *,
    memory_limit_bytes: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> list[Any]:
    """Validate and convert named columns to float64 within the caller's resource limits.

    The size every column declares is admitted against the memory limit, with checked
    arithmetic, before any column is converted, so an input that claims more values than the
    limit allows (or than can be counted) is refused before a copy is made. A column whose
    converted length differs from the size it declared is refused.
    """
    _check_resources(memory_limit_bytes, cancel)
    declared = [_value_count(values) for _, values in named]
    _admit(sum(declared), memory_limit_bytes)
    converted = [_column(name, values) for name, values in named]
    for (name, _), count, column in zip(named, declared, converted, strict=True):
        if column.size != count:
            raise CausalValueError(
                f"recalc.declared_size_mismatch: column {name!r} declares {count} values "
                f"but holds {column.size}",
                reason_code="invalid_argument",
            )
    return converted


def _columns(
    data: Mapping[str, ArrayLike],
    *,
    memory_limit_bytes: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> tuple[list[str], list[Any]]:
    """Validate and convert ``data`` to float64 columns within the caller's resource limits."""
    names = list(data)
    columns = _float_columns(
        [(name, data[name]) for name in names],
        memory_limit_bytes=memory_limit_bytes,
        cancel=cancel,
    )
    return names, columns
