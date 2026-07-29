"""Numeric formatting shared by view ``__repr__`` and ``_repr_html_``.

One place decides how floats print — fixed decimals, explicit ``None``,
and ``nan``/``inf`` spelled out rather than left to fall through into a
mid-sentence ``nan`` that reads like a bug.
"""

from __future__ import annotations

import math

__all__ = ["fmt_float", "fmt_pct"]


def fmt_float(value: float | None, *, ndigits: int = 3) -> str:
    """Fixed-precision float formatting; ``None``/``nan``/``inf`` are explicit."""
    if value is None:
        return "None"
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return "None"
    if math.isnan(as_float):
        return "nan"
    if math.isinf(as_float):
        return "inf" if as_float > 0 else "-inf"
    return f"{as_float:.{ndigits}f}"


def fmt_pct(value: float | None, *, ndigits: int = 1) -> str:
    """Format a ``[0, 1]`` fraction as a percentage string; ``None``/``nan`` explicit."""
    if value is None:
        return "None"
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return "None"
    if math.isnan(as_float):
        return "nan"
    if math.isinf(as_float):
        return "inf" if as_float > 0 else "-inf"
    return f"{as_float * 100:.{ndigits}f}%"
