"""Shared data normalization for the Python facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def ingest_columns(
    data: Mapping[str, Any] | Any,
) -> tuple[list[str], list[Any]]:
    """Normalize to ``(names, columns)`` preferring Arrow CDI exporters."""
    arrow = try_as_arrow_c_columns(data)
    if arrow is not None:
        return arrow
    return as_columns(data)


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
        f"data must be a mapping of name→array or a pandas DataFrame; got {type(data)!r}"
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

    # Frame columns that export CDI, but not pandas (it has to_numpy and
    # Series.__arrow_c_array__; sending those skips the float64 cast).
    if hasattr(data, "columns") and not hasattr(data, "to_numpy"):
        try:
            names = [str(c) for c in data.columns]
            cols = [data[c] for c in data.columns]
            if names and all(hasattr(c, "__arrow_c_array__") for c in cols):
                return names, cols
        except Exception:  # noqa: BLE001 — try a table-level stream next
            pass

    # Table-level Arrow PyCapsule (Polars, DuckDB). Cast to float64 so
    # integer treatments match the dict / pandas to_f64 ingest.
    if hasattr(data, "__arrow_c_stream__"):
        try:
            import pyarrow as pa

            table = pa.table(data)
            names = [str(n) for n in table.column_names]
            cols = []
            for i in range(len(names)):
                col = table.column(i).combine_chunks()
                if not pa.types.is_float64(col.type):
                    col = col.cast(pa.float64())
                cols.append(col)
            if names and all(hasattr(c, "__arrow_c_array__") for c in cols):
                return names, cols
        except Exception:  # noqa: BLE001 — fall through to numpy ingest
            return None
    return None


def to_f64(arr: Any) -> NDArray[np.float64]:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"expected 1-d column, got shape {a.shape}")
    if a.dtype == object:
        raise TypeError("object-dtype columns are not supported")
    return a


def _materialize_f64(column: Any) -> NDArray[np.float64]:
    """Numpy view of an ingested column (Arrow CDI or array-like)."""
    if hasattr(column, "__arrow_c_array__"):
        try:
            import pyarrow as pa

            return to_f64(pa.array(column).to_numpy(zero_copy_only=False))
        except Exception:  # noqa: BLE001 — try array protocols
            pass
    if hasattr(column, "to_numpy"):
        return to_f64(column.to_numpy())
    return to_f64(column)


def as_multi_env_columns(
    data: Sequence[Mapping[str, Any] | Any],
) -> tuple[list[str], list[list[NDArray[np.float64]]]]:
    if not data:
        raise ValueError("expected a non-empty sequence of environment frames")
    names, first = ingest_columns(data[0])
    env_columns = [[_materialize_f64(col) for col in first]]
    for i, env in enumerate(data[1:], start=1):
        n, cols = ingest_columns(env)
        if n != names:
            raise ValueError(
                f"environment {i} column names {n!r} do not match environment 0 {names!r}"
            )
        env_columns.append([_materialize_f64(col) for col in cols])
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
