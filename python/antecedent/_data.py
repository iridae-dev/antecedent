"""Shared data normalization for the Python facade."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


def as_columns(
    data: Mapping[str, Any] | Any,
) -> tuple[list[str], list[NDArray[np.float64]]]:
    """Normalize a mapping or pandas DataFrame to ``(names, float64 columns)``."""
    if isinstance(data, Mapping):
        names = list(data.keys())
        cols = [to_f64(data[n]) for n in names]
        return names, cols
    if hasattr(data, "columns") and hasattr(data, "to_numpy"):
        names = [str(c) for c in data.columns]
        cols = [to_f64(data[c].to_numpy()) for c in data.columns]
        return names, cols
    raise TypeError(
        "data must be a mapping of name→array or a pandas DataFrame; "
        f"got {type(data)!r}"
    )


def try_as_arrow_c_columns(
    data: Any,
) -> tuple[list[str], list[Any]] | None:
    """If ``data`` exports Arrow C Data Interface columns, return ``(names, cols)``.

    Accepts:
    - a mapping of name → object with ``__arrow_c_array__``
    - a table-like with ``column_names`` / ``column(i)`` (PyArrow Table)
    - a table-like with ``schema.names`` and ``column(i)``

    Returns ``None`` when the object is not an Arrow CDI exporter (caller should
    fall back to [`as_columns`]).
    """
    if isinstance(data, Mapping):
        names = list(data.keys())
        cols = [data[n] for n in names]
        if names and all(hasattr(c, "__arrow_c_array__") for c in cols):
            return names, cols
        return None

    # PyArrow Table / RecordBatch style
    names_attr = getattr(data, "column_names", None)
    if names_attr is None:
        schema = getattr(data, "schema", None)
        names_attr = getattr(schema, "names", None) if schema is not None else None
    if names_attr is not None and hasattr(data, "column"):
        names = [str(n) for n in list(names_attr)]
        cols = [data.column(i) for i in range(len(names))]
        flat: list[Any] = []
        for c in cols:
            if hasattr(c, "combine_chunks"):
                c = c.combine_chunks()
            flat.append(c)
        if flat and all(hasattr(c, "__arrow_c_array__") for c in flat):
            return names, flat
        return None

    # Frame with columns that each export CDI (e.g. Polars column export)
    if hasattr(data, "columns") and not hasattr(data, "to_numpy"):
        try:
            names = [str(c) for c in data.columns]
            cols = [data[c] for c in data.columns]
            if names and all(hasattr(c, "__arrow_c_array__") for c in cols):
                return names, cols
        except Exception:  # noqa: BLE001 — fall through to None
            return None
    return None


def to_f64(arr: Any) -> NDArray[np.float64]:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"expected 1-d column, got shape {a.shape}")
    if a.dtype == object:
        raise TypeError("object-dtype columns are not supported")
    return a


def as_multi_env_columns(
    data: Sequence[Mapping[str, Any] | Any],
) -> tuple[list[str], list[list[NDArray[np.float64]]]]:
    if not data:
        raise ValueError("expected a non-empty sequence of environment frames")
    names, first = as_columns(data[0])
    env_columns = [first]
    for i, env in enumerate(data[1:], start=1):
        n, cols = as_columns(env)
        if n != names:
            raise ValueError(
                f"environment {i} column names {n!r} do not match environment 0 {names!r}"
            )
        env_columns.append(cols)
    return names, env_columns


def coerce_data_args(
    data: Mapping[str, Any] | Any | None = None,
    *,
    names: list[str] | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
) -> tuple[list[str], list[NDArray[np.float64]]]:
    """Accept either ``data=`` (DataFrame/mapping) or ``names=`` + ``columns=``."""
    if data is not None:
        return as_columns(data)
    if names is None or columns is None:
        raise TypeError("provide data=… or both names= and columns=")
    return list(names), [to_f64(c) for c in columns]
