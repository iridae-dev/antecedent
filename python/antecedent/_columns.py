"""Resolve column-name inputs on design-carrying queries against the data.

A design-carrying query (randomized experiment, policy value, ...) may name a
row-aligned input by the data column that holds it instead of passing the array
inline: ``PolicyValue("y", assignment="arm", ...)`` rather than
``PolicyValue("y", assignment=[...], ...)``. This keeps the query readable. The
design is read once, at prepare, and frozen into the prepared study: a later
``refresh(new_data)`` checks any supplied named design columns against the frozen
bindings before numeric ingest. Changed values or order require explicit repreparation.

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
from typing import Any, TypeVar, cast, overload

import numpy as np

from ._validate import count as _count
from .errors import CausalValueError

_T = TypeVar("_T")


def coerce_fields(obj: Any, /, **fields: Any) -> None:
    """Assign fields on a frozen dataclass.

    Frozen ``__post_init__`` methods cannot write ``self.x = ...``. This is the
    one helper for that write. The values are whatever the caller already
    validated; this function does not coerce them further.
    """
    for name, value in fields.items():
        object.__setattr__(obj, name, value)


def hold_unresolved(obj: Any, /, **fields: Any) -> bool:
    """Skip validation while a column name (or a nested design) is still unresolved.

    Returns True when ``__post_init__`` should return. Sets ``_deferred_columns``
    so :func:`resolve_columns` rebuilds the object against the table at prepare.
    """
    if not defer_columns(obj):
        return False
    coerce_fields(obj, _deferred_columns=True, **fields)
    return True


def _require_resolved(value: object) -> None:
    """Refuse a column name that reached prepare still unresolved.

    ``assert`` is stripped by ``python -O``, which would then hand the name to
    the estimator as if it were a one-row column.
    """
    if isinstance(value, str):
        raise CausalValueError(
            "column-name input must be resolved before prepare",
            reason_code="invalid_argument",
        )


def resolved(value: Sequence[_T] | str) -> Sequence[_T]:
    """Narrow a column-name field to its array once :func:`resolve_columns` has run.

    A design query accepts a row-aligned input either inline or as the name of a
    data column (a ``str``). :func:`resolve_columns` rebuilds the query with the
    named column's array before the native study is prepared, so by the time a
    prepared study reads the field the ``str`` branch is gone. This checks that
    invariant and hands the caller (and the type checker) the sequence.
    """
    _require_resolved(value)
    return cast(Sequence[_T], value)


@overload
def resolved_opt(value: None) -> None: ...
@overload
def resolved_opt(value: Sequence[_T] | str) -> Sequence[_T]: ...
def resolved_opt(value: Sequence[_T] | str | None) -> Sequence[_T] | None:
    """Like :func:`resolved`, but pass ``None`` through for optional fields."""
    if value is None:
        return None
    _require_resolved(value)
    return cast(Sequence[_T], value)


def resolved_scalar(value: float | Sequence[float] | str) -> float | Sequence[float]:
    """Narrow a scalar-or-column field (e.g. a per-row propensity) once resolved.

    The field is either a scalar, an inline sequence, or the name of a data column
    that :func:`resolve_columns` has already turned into a sequence, so the ``str``
    branch is gone by prepare.
    """
    _require_resolved(value)
    return cast(float | Sequence[float], value)


def _shown(value: Any) -> Any:
    """The value as ``tolist`` would have shown it in a refusal."""
    return value.item() if isinstance(value, np.generic) else value


def _first_bad(mask: np.ndarray, values: np.ndarray) -> Any:
    return _shown(values[np.flatnonzero(mask)[0]])


def _cast_bool(arr: np.ndarray, field: str, name: str) -> tuple[bool, ...]:
    """Booleans, or numbers that are exactly 0 or 1; anything else is refused.

    ``bool(v)`` would turn NaN, arm codes ``2``, and the string ``"False"`` into
    ``True``; the inline path refuses those, so the column path does too.
    """
    if arr.dtype == np.bool_ or arr.dtype == bool:
        return tuple(arr.tolist())
    if arr.dtype.kind in "iuf":
        ok = np.isfinite(arr) & ((arr == 0) | (arr == 1))
        if not np.all(ok):
            bad = _first_bad(~ok, arr)
            raise CausalValueError(
                f"{field}={name!r}: column values must be bool or encoded as 0/1; got {bad!r}"
            )
        return tuple(arr.astype(bool, copy=False).tolist())
    if arr.dtype == object:
        try:
            ok = np.isin(arr, (0, 1, False, True))
        except (TypeError, ValueError):
            ok = None
        if ok is not None:
            if not np.all(ok):
                bad = arr[np.flatnonzero(~ok)[0]]
                raise CausalValueError(
                    f"{field}={name!r}: column values must be bool or encoded as 0/1; "
                    f"got {_shown(bad)!r}"
                )
            return tuple(np.asarray(arr, dtype=bool).tolist())
    out: list[bool] = []
    for value in arr:
        if isinstance(value, (bool, np.bool_)) or (
            isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1)
        ):
            out.append(bool(value))
        else:
            raise CausalValueError(
                f"{field}={name!r}: column values must be bool or encoded as 0/1; got {value!r}"
            )
    return tuple(out)


def _cast_int(arr: np.ndarray, field: str, name: str) -> tuple[int, ...]:
    """Normalize each design integer exactly, with its location and native bounds."""
    out: list[int] = []
    for index, value in enumerate(arr):
        numeric = isinstance(value, (int, float, np.integer, np.floating))
        message = (
            "column values must be finite integers" if numeric else "column values must be integers"
        )
        out.append(
            _count(
                value,
                f"{field}={name!r}[{index}]",
                minimum=0,
                maximum=2**64 - 1,
                type_error=CausalValueError,
                type_message=f"{{what}}: {message}; got {{value!r}}",
            )
        )
    return tuple(out)


_KIND_CAST = {
    "bool": _cast_bool,
    "str": lambda arr, field, name: tuple(str(v) for v in arr),
    "float": lambda arr, field, name: tuple(float(v) for v in arr),
    "int": _cast_int,
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


def drop_columns(data: Any, names: set[str] | frozenset[str]) -> Any:
    """Return ``data`` without the design columns, so the numeric ingest never sees them."""
    if not names:
        return data
    if isinstance(data, Mapping):
        return {key: value for key, value in data.items() if key not in names}
    if hasattr(data, "column_names") and hasattr(data, "drop_columns"):
        # pyarrow.Table: ``.columns`` holds the arrays, the names are ``column_names``.
        present = [name for name in names if name in data.column_names]
        return data.drop_columns(present) if present else data
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
    return resolve_columns(query, columns), drop_columns(data, names)


def _resolve_one(columns: Mapping[str, Any], name: str, kind: str, field: str) -> Any:
    if name not in columns:
        raise CausalValueError(
            f"{field}={name!r} names a column that is not in the data; "
            f"available columns: {sorted(columns)}"
        )
    arr = np.asarray(columns[name])
    if arr.ndim != 1:
        raise CausalValueError(f"{field}={name!r} must name a one-dimensional column")
    cast = _KIND_CAST.get(kind)
    return tuple(arr.tolist()) if cast is None else cast(arr, field, name)


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
