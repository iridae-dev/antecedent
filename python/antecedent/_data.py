"""Shared data normalization for the Python facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import CausalTypeError, CausalValueError

# Integers beyond this are not exactly representable as float64.
_EXACT_INT_LIMIT = 2**53


def _refuse_column(message: str) -> CausalTypeError:
    """Typed refusal for non-numeric or unsafe column coercions."""
    return CausalTypeError(message, reason_code="invalid_argument")


def _is_missing_scalar(value: Any) -> bool:
    """True for ``None`` and pandas-style NA sentinels (not NumPy NaN floats)."""
    if value is None:
        return True
    name = type(value).__name__
    return name in {"NAType", "NaTType", "_NaT"}


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
        cols = [to_f64(data[n], name=str(n)) for n in names]
        _check_column_lengths([str(n) for n in names], cols)
        return names, cols
    if hasattr(data, "columns") and hasattr(data, "to_numpy"):
        names = [str(c) for c in data.columns]
        duplicated = sorted({n for n in names if names.count(n) > 1})
        if duplicated:
            raise CausalValueError(f"data has duplicate column names: {duplicated}")
        cols = [to_f64(data[c].to_numpy(), name=str(c)) for c in data.columns]
        _check_column_lengths(names, cols)
        return names, cols
    arrow = try_as_arrow_c_columns(data)
    if arrow is not None:
        names, cols = arrow
        return names, [_materialize_f64(col) for col in cols]
    raise TypeError(
        "data must be a mapping of name→array, a pandas DataFrame, or an Arrow-exporting "
        f"table (PyArrow, Polars); got {type(data)!r}"
    )


def _arrow_float64(column: Any) -> Any:
    """An Arrow-exporting column as float64, so every Arrow route casts alike.

    Integer, boolean and floating columns cast (Arrow refuses an integer beyond
    2**53); any other type is refused, as strings and dates are on the numpy
    route. Without PyArrow the column passes through for the native reader.
    """
    try:
        import pyarrow as pa
    except ImportError:
        return column
    try:
        col = column if isinstance(column, pa.Array) else pa.array(column)
    except (TypeError, ValueError, pa.ArrowException):
        return column  # not a readable Arrow export; the native reader decides
    kind = col.type
    if pa.types.is_float64(kind):
        return col
    if pa.types.is_integer(kind) or pa.types.is_boolean(kind) or pa.types.is_floating(kind):
        return col.cast(pa.float64())
    raise _refuse_column(
        f"column type {kind} is not numeric; only float, integer and boolean columns are "
        "accepted (encode categories and parse dates explicitly)"
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
            return names, [_arrow_float64(c) for c in cols]
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
            return names, [_arrow_float64(c) for c in flat]
        return None

    # Frame columns that export CDI, but not pandas (it has to_numpy and
    # Series.__arrow_c_array__; sending those skips the float64 cast).
    if hasattr(data, "columns") and not hasattr(data, "to_numpy"):
        try:
            names = [str(c) for c in data.columns]
            cols = [data[c] for c in data.columns]
            if names and all(hasattr(c, "__arrow_c_array__") for c in cols):
                return names, [_arrow_float64(c) for c in cols]
        except (AttributeError, KeyError, TypeError):
            pass  # not a column-indexable frame; try a table-level stream next

    # Table-level Arrow PyCapsule (Polars, DuckDB); the same float64 cast as above.
    if hasattr(data, "__arrow_c_stream__"):
        try:
            import pyarrow as pa
        except ImportError:
            return None
        table = pa.table(data)
        names = [str(n) for n in table.column_names]
        cols = [_arrow_float64(table.column(i).combine_chunks()) for i in range(len(names))]
        if names:
            return names, cols
    return None


def to_f64(arr: Any, *, name: str | None = None) -> NDArray[np.float64]:
    """One numeric column as float64, naming the column in errors when known.

    Only numeric inputs are accepted: floats, integers within 2**53 (larger ones
    do not survive the cast), and booleans (as 0/1). Strings, datetimes,
    timedeltas and complex numbers are refused rather than parsed or reinterpreted
    (digit strings are labels, and a datetime is not a number of nanoseconds).
    A nullable pandas column's ``NA`` and ``None`` become NaN, which is missing.
    """
    label = f" {name!r}" if name is not None else ""
    raw = np.asarray(arr)
    if raw.ndim != 1:
        raise CausalValueError(f"expected 1-d column{label}, got shape {raw.shape}")
    kind = raw.dtype.kind
    if kind == "O":
        values: list[float] = []
        for value in raw:
            if _is_missing_scalar(value):
                values.append(float("nan"))
            elif isinstance(value, (bool, np.bool_)):
                values.append(float(value))
            elif isinstance(value, (int, np.integer)):
                if abs(int(value)) > _EXACT_INT_LIMIT:
                    raise _refuse_column(
                        f"integer {int(value)} exceeds 2**53 and would lose precision as float64"
                        + (f" (column{label})" if name is not None else "")
                    )
                values.append(float(value))
            elif isinstance(value, (float, np.floating)):
                values.append(float(value))
            else:
                raise _refuse_column(
                    f"column{label} holds {type(value).__name__} values; only numeric values are "
                    "accepted (encode categories / one-hot, or exclude the column; "
                    "typed numeric Arrow columns can use try_as_arrow_c_columns for zero-copy)"
                )
        return np.asarray(values, dtype=np.float64)
    if kind not in "fiub":
        raise _refuse_column(
            f"column{label} dtype {raw.dtype} is not numeric; only float, integer and boolean "
            "columns are accepted (encode categories / one-hot, or exclude the column; "
            "typed numeric Arrow columns can use try_as_arrow_c_columns for zero-copy)"
        )
    if kind in "iu" and raw.size:
        lowest, highest = int(raw.min()), int(raw.max())
        if highest > _EXACT_INT_LIMIT or lowest < -_EXACT_INT_LIMIT:
            raise _refuse_column(
                f"integer column{label} spans [{lowest}, {highest}], beyond 2**53, and would lose "
                "precision as float64"
            )
    return np.asarray(raw, dtype=np.float64)


def _check_column_lengths(names: list[str], cols: list[NDArray[np.float64]]) -> None:
    if len(cols) < 2:
        return
    lengths = [int(c.shape[0]) for c in cols]
    n0 = lengths[0]
    if all(n == n0 for n in lengths):
        return
    detail = ", ".join(f"{name}={n}" for name, n in zip(names, lengths, strict=True))
    raise ValueError(f"column length mismatch ({detail}); align rows or subset to a common index")


def _materialize_f64(column: Any) -> NDArray[np.float64]:
    """Numpy view of an ingested column (Arrow CDI or array-like)."""
    if hasattr(column, "__arrow_c_array__"):
        try:
            import pyarrow as pa
        except ImportError:
            pass  # no PyArrow: read the column through its array protocols
        else:
            return to_f64(pa.array(column).to_numpy(zero_copy_only=False))
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
    out_names = list(names)
    cols = [to_f64(c, name=str(n)) for n, c in zip(out_names, columns, strict=True)]
    _check_column_lengths(out_names, cols)
    return out_names, cols
