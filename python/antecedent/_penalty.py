"""Bound penalty declarations before allocating a Python-owned copy."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from . import _native
from .errors import CausalUnsupportedError

_T = TypeVar("_T")


def bounded_penalty_list(values: Iterable[_T]) -> list[_T]:
    """Materialize at most the native context-free memory policy allows."""
    limit = _native._penalty_grid_max_items()
    try:
        declared = len(values)  # type: ignore[arg-type]
    except TypeError:
        declared = None
    except OverflowError as error:
        raise CausalUnsupportedError(
            "penalty grid exceeds the context-free memory policy",
            reason_code="execution_resource_limit",
        ) from error
    if declared is not None and declared > limit:
        raise CausalUnsupportedError(
            "penalty grid exceeds the context-free memory policy",
            reason_code="execution_resource_limit",
        )
    result: list[_T] = []
    for value in values:
        if len(result) >= limit:
            raise CausalUnsupportedError(
                "penalty grid exceeds the context-free memory policy",
                reason_code="execution_resource_limit",
            )
        result.append(value)
    return result
