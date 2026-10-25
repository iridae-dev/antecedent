"""Cohort-checked column projection and filter pushdown over a table or a table file.

An analysis reads a closure of columns: its target, validators, support and weight columns, split
columns and the columns its complete-case rule reads. The cohort it is entitled to is the rows
observed on **every** closure column. Reading fewer columns, or pushing a row filter into the
scan, is only legal when it delivers that same cohort:

* :func:`checked_projection` keeps a column list and checks that completing cases on the kept
  columns is the closure's cohort (a projection that drops a required column is refused, and one
  that keeps extra columns changes the cohort and is refused);
* :func:`checked_push_filter` pushes a row filter read off one column and checks it drops no row of
  the cohort (the refusal names the filter column, which still decides membership).

``source`` is any mapping of column name to array, so a :class:`~antecedent.data.TableFile`
works: only the closure columns, the kept columns and no others are read. A cell is observed when
it is not ``NaN`` / ``NaT`` / ``None``. The cohort, the legality rules and the check are
computed in Rust; this module reads the missingness and decodes the rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _native
from .errors import CausalTypeError, CausalValueError

__all__ = ["ColumnClosure", "checked_projection", "checked_push_filter"]


def _names(values: Sequence[str], what: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise CausalTypeError(f"{what} must be a sequence of column names, not one string")
    out = tuple(values)
    for name in out:
        if not isinstance(name, str) or not name:
            raise CausalValueError(f"{what} must be non-empty column names")
    return out


@dataclass(frozen=True)
class ColumnClosure:
    """The columns an analysis requires, by the duty that first requires each.

    A column named under several duties keeps the earliest: target, validators, support,
    weights, splits, then complete-case.
    """

    target: tuple[str, ...]
    validators: tuple[str, ...] = ()
    support: tuple[str, ...] = ()
    weights: tuple[str, ...] = ()
    splits: tuple[str, ...] = ()
    complete_case: tuple[str, ...] = ()

    def _wire(
        self,
    ) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
        return (
            list(_names(self.target, "target")),
            list(_names(self.validators, "validators")),
            list(_names(self.support, "support")),
            list(_names(self.weights, "weights")),
            list(_names(self.splits, "splits")),
            list(_names(self.complete_case, "complete_case")),
        )

    @property
    def columns(self) -> tuple[str, ...]:
        """Every required column, in closure order and without repeats."""
        seen: dict[str, None] = {}
        for group in (
            self.target,
            self.validators,
            self.support,
            self.weights,
            self.splits,
            self.complete_case,
        ):
            for name in group:
                seen.setdefault(name, None)
        return tuple(seen)


def _observed(values: Any) -> list[bool]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise CausalValueError("each column must be one-dimensional")
    if array.dtype.kind == "f":
        return [bool(v) for v in ~np.isnan(array)]
    if array.dtype.kind in "mM":
        return [bool(v) for v in ~np.isnat(array)]
    if array.dtype.kind == "O":
        return [
            not (
                v is None
                or (isinstance(v, float) and v != v)
                or (isinstance(v, np.datetime64) and bool(np.isnat(v)))
            )
            for v in array
        ]
    return [True] * int(array.shape[0])


def _frame(source: Mapping[str, Any], names: Sequence[str]) -> list[tuple[str, list[bool]]]:
    if not isinstance(source, Mapping):
        raise CausalTypeError("source must be a mapping of column name to array")
    frame: list[tuple[str, list[bool]]] = []
    for name in dict.fromkeys(names):
        if name not in source:
            raise CausalValueError(f"source has no column {name!r}")
        frame.append((name, _observed(source[name])))
    return frame


def checked_projection(
    source: Mapping[str, Any], closure: ColumnClosure, keep: Sequence[str]
) -> NDArray[np.int64]:
    """Rows of the closure's cohort, delivered by reading only ``keep``.

    Raises:
        CausalValueError: ``keep`` drops a required column, or keeps extra columns so that
            completing cases on them is not the cohort (the message names the column).
    """
    if not isinstance(closure, ColumnClosure):
        raise CausalTypeError("closure must be a ColumnClosure")
    kept = _names(keep, "keep")
    rows = _native.cohort_checked_projection(
        closure._wire(), _frame(source, (*closure.columns, *kept)), list(kept)
    )
    return np.asarray(rows, dtype=np.int64)


def checked_push_filter(
    source: Mapping[str, Any],
    closure: ColumnClosure,
    column: str,
    drop: Sequence[bool] | NDArray[np.bool_],
) -> NDArray[np.int64]:
    """Rows of the closure's cohort, delivered with the row filter ``drop`` pushed down.

    ``drop[i]`` drops row ``i``; ``column`` is the column the filter predicate reads.

    Raises:
        CausalValueError: the filter would drop a row of the cohort (the message names the
            filter column), or its mask is not one entry per row.
    """
    if not isinstance(closure, ColumnClosure):
        raise CausalTypeError("closure must be a ColumnClosure")
    if not isinstance(column, str) or not column:
        raise CausalValueError("column must be a non-empty column name")
    mask = [bool(v) for v in np.asarray(drop, dtype=np.bool_).reshape(-1)]
    rows = _native.cohort_checked_push_filter(
        closure._wire(), _frame(source, closure.columns), column, mask
    )
    return np.asarray(rows, dtype=np.int64)
