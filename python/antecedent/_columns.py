"""Resolve column-name inputs on design-carrying queries against the data.

A design-carrying query (randomized experiment, policy value, ...) may name a
row-aligned input by the data column that holds it instead of passing the array
inline: ``PolicyValue("y", assignment="arm", ...)`` rather than
``PolicyValue("y", assignment=[...], ...)``. This keeps the query readable and
makes ``refresh(new_data)`` re-read the design from the new table.

A class opts in by declaring ``_COLUMN_FIELDS`` (row-aligned field name → element
kind) and, in its ``__post_init__``, calling :func:`defer_columns` first: while any
declared field is a ``str`` the class stores its inputs unvalidated and sets
``_deferred_columns``; :func:`resolve_columns` (run at prepare, when the data is in
hand) resolves each name to its column and rebuilds the object, which re-runs the
ordinary array validation. A class that wraps another design-carrying object lists
it in ``_COLUMN_NESTED`` so resolution recurses. When no field is a ``str`` the
class validates exactly as before, so the inline-array path is untouched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, replace
from typing import Any, TypeVar, overload

import numpy as np

from .errors import CausalValueError

_T = TypeVar("_T")


def resolved(value: Sequence[_T] | str) -> Sequence[_T]:
    """Narrow a column-name field to its array once :func:`resolve_columns` has run.

    A design query accepts a row-aligned input either inline or as the name of a
    data column (a ``str``). :func:`resolve_columns` rebuilds the query with the
    named column's array before the native study is prepared, so by the time a
    prepared study reads the field the ``str`` branch is gone. This asserts that
    invariant and hands the caller (and the type checker) the sequence.
    """
    assert not isinstance(value, str), "column-name input must be resolved before prepare"
    return value


@overload
def resolved_opt(value: None) -> None: ...
@overload
def resolved_opt(value: Sequence[_T] | str) -> Sequence[_T]: ...
def resolved_opt(value: Sequence[_T] | str | None) -> Sequence[_T] | None:
    """Like :func:`resolved`, but pass ``None`` through for optional fields."""
    if value is None:
        return None
    assert not isinstance(value, str), "column-name input must be resolved before prepare"
    return value


def resolved_scalar(value: float | Sequence[float] | str) -> float | Sequence[float]:
    """Narrow a scalar-or-column field (e.g. a per-row propensity) once resolved.

    The field is either a scalar, an inline sequence, or the name of a data column
    that :func:`resolve_columns` has already turned into a sequence, so the ``str``
    branch is gone by prepare.
    """
    assert not isinstance(value, str), "column-name input must be resolved before prepare"
    return value


_KIND_CAST = {
    "bool": lambda arr: tuple(bool(v) for v in arr),
    "str": lambda arr: tuple(str(v) for v in arr),
    "float": lambda arr: tuple(float(v) for v in arr),
    "int": lambda arr: tuple(int(v) for v in arr),
}


def defer_columns(obj: Any) -> bool:
    """True when ``obj`` holds an unresolved column name, directly or in a nested design.

    Called at the top of an opting-in ``__post_init__``: a ``True`` return means the
    object (or one of its ``_COLUMN_NESTED`` designs) holds unresolved column names and
    must skip array validation until :func:`resolve_columns` rebuilds it. This lets a
    wrapper defer whenever a design it carries deferred, so its row-length checks do
    not run against a still-unresolved child.
    """
    cls = type(obj)
    spec = getattr(cls, "_COLUMN_FIELDS", None)
    if spec and any(isinstance(getattr(obj, name), str) for name in spec):
        return True
    for nested in getattr(cls, "_COLUMN_NESTED", ()):
        if getattr(getattr(obj, nested, None), "_deferred_columns", False):
            return True
    return False


def collect_column_names(obj: Any) -> set[str]:
    """Every data-column name a deferred query (and its nested designs) references."""
    if obj is None or not is_dataclass(obj) or isinstance(obj, type):
        return set()
    names: set[str] = set()
    spec = getattr(type(obj), "_COLUMN_FIELDS", None)
    if spec and getattr(obj, "_deferred_columns", False):
        names |= {getattr(obj, field) for field in spec if isinstance(getattr(obj, field), str)}
    for nested in getattr(type(obj), "_COLUMN_NESTED", ()):
        names |= collect_column_names(getattr(obj, nested, None))
    return names


def _data_column_map(data: Any, names: set[str]) -> dict[str, Any]:
    """Extract the requested columns from raw ``data`` (dict-like or DataFrame)."""
    if isinstance(data, Mapping):
        source: Any = data
    elif hasattr(data, "columns") and hasattr(data, "__getitem__"):
        source = data  # pandas / Arrow-like: column access by name
    else:
        raise CausalValueError(
            "column-name query inputs require dict or DataFrame data; pass the arrays "
            "inline instead, or supply the columns in a table"
        )
    out: dict[str, Any] = {}
    for name in names:
        try:
            column = source[name]
        except (KeyError, IndexError) as exc:
            raise CausalValueError(
                f"query names column {name!r}, which is not in the data"
            ) from exc
        out[name] = np.asarray(column.to_numpy() if hasattr(column, "to_numpy") else column)
    return out


def _drop_columns(data: Any, names: set[str]) -> Any:
    """Return ``data`` without the design columns, so the numeric ingest never sees them."""
    if not names:
        return data
    if isinstance(data, Mapping):
        return {key: value for key, value in data.items() if key not in names}
    if hasattr(data, "drop") and hasattr(data, "columns"):
        present = [name for name in names if name in getattr(data, "columns", ())]
        return data.drop(columns=present) if present else data
    return data


def resolve_query(query: Any, data: Any) -> tuple[Any, Any]:
    """Resolve a design query's column-name inputs from ``data``.

    Returns ``(resolved_query, data_without_design_columns)``. Design columns are
    read from the raw data (any dtype, so string unit / subject ids work) and then
    dropped, so the numeric analysis ingest only sees the outcome and covariates.
    A query that passed its inputs inline is returned unchanged with the data intact.
    """
    names = collect_column_names(query)
    if not names:
        return query, data
    columns = _data_column_map(data, names)
    return resolve_columns(query, columns), _drop_columns(data, names)


def _resolve_one(columns: Mapping[str, Any], name: str, kind: str, field: str) -> Any:
    if name not in columns:
        raise CausalValueError(
            f"{field}={name!r} names a column that is not in the data; "
            f"available columns: {sorted(columns)}"
        )
    arr = np.asarray(columns[name])
    if arr.ndim != 1:
        raise CausalValueError(f"{field}={name!r} must name a one-dimensional column")
    return _KIND_CAST.get(kind, lambda a: tuple(a.tolist()))(arr)


def resolve_columns(obj: Any, columns: Mapping[str, Any]) -> Any:
    """Return ``obj`` with any deferred column-name fields resolved from ``columns``.

    Objects that did not opt in, or that carry no unresolved column name, are
    returned unchanged (identity), so the inline-array path pays nothing.
    """
    if obj is None or not is_dataclass(obj) or isinstance(obj, type):
        return obj
    updates: dict[str, Any] = {}
    spec = getattr(type(obj), "_COLUMN_FIELDS", None)
    if spec and getattr(obj, "_deferred_columns", False):
        for name, kind in spec.items():
            value = getattr(obj, name)
            if isinstance(value, str):
                updates[name] = _resolve_one(columns, value, kind, name)
    for nested in getattr(type(obj), "_COLUMN_NESTED", ()):
        child = getattr(obj, nested, None)
        resolved = resolve_columns(child, columns)
        if resolved is not child:
            updates[nested] = resolved
    if not updates:
        return obj
    # replace() re-runs __post_init__; with the names now resolved to arrays the
    # class validates exactly as if the caller had passed the arrays inline.
    return replace(obj, **updates)
