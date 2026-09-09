"""Numeric formatting shared by view ``__repr__`` and ``_repr_html_``.

One place decides how floats print — fixed decimals, explicit ``None``,
and ``nan``/``inf`` spelled out rather than left to fall through into a
mid-sentence ``nan`` that reads like a bug.
"""

from __future__ import annotations

import math

__all__ = ["fmt_float", "fmt_pct", "fmt_se"]


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


def fmt_se(value: float | None) -> str | None:
    """Return a formatted SE, or ``None`` when sampling uncertainty is unavailable.

    ``None``, non-numeric values, ``nan``, and ``±inf`` are not standard errors.
    Callers must print ``unavailable`` rather than interpolating those sentinels
    into ``±…`` so a withheld interval cannot look like a computed NaN.
    """
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(as_float) or math.isinf(as_float):
        return None
    return fmt_float(as_float)


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
