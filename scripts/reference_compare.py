"""Compare frozen reference documents without computing any reference values.

Numeric tolerance is supplied by the caller. Metadata, booleans, shapes and
integer coordinates compare exactly; non-finite values are never evidence.
"""

from __future__ import annotations

import math


def values_close(left: object, right: object, *, rtol: float, atol: float) -> bool:
    if not all(math.isfinite(t) and t >= 0 for t in (rtol, atol)):
        raise ValueError("tolerances must be finite and nonnegative")
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (
            math.isfinite(left)
            and math.isfinite(right)
            and abs(left - right) <= atol + rtol * abs(right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            values_close(left[k], right[k], rtol=rtol, atol=atol) for k in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            values_close(a, b, rtol=rtol, atol=atol)
            for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right
