from __future__ import annotations

from ...errors import CausalTypeError, CausalValueError


def _non_negative(name: str, value: int) -> int:
    """A non-negative int, refused as a typed value error before it reaches native code."""
    if value is None:
        raise CausalValueError(f"{name} must be a non-negative integer")
    if isinstance(value, bool) or not isinstance(value, int):
        raise CausalTypeError(f"{name} must be an integer")
    if value < 0:
        raise CausalValueError(f"{name} must be a non-negative integer, got {value}")
    return value


def _optional_non_negative(name: str, value: int | None) -> int | None:
    """:func:`_non_negative`, with ``None`` meaning "no limit"."""
    return None if value is None else _non_negative(name, value)
