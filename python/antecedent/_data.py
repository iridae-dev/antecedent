"""Shared data normalization for the Python facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import CausalTypeError, CausalValueError

# Integers beyond this are not exactly representable as float64.
_EXACT_INT_LIMIT = 2**53
_PANDAS_MISSING = frozenset({"NAType", "NaTType", "_NaT"})


def _refuse_column(message: str) -> CausalTypeError:
    """Typed refusal for non-numeric or unsafe column coercions."""
    return CausalTypeError(message, reason_code="invalid_argument")


def _is_missing_scalar(value: Any) -> bool:
    """True for ``None`` and pandas-style NA sentinels (not NumPy NaN floats)."""
    if value is None:
        return True
    return type(value).__name__ in _PANDAS_MISSING


def table_column_names(data: Any) -> list[str] | None:
    """Column names (as strings) of a mapping, pandas / Polars frame or Arrow table.

    ``None`` for anything else (frames of the ``antecedent.data`` module, series,
    Arrow stream exporters without column access): those are not projected.
    """
    if isinstance(data, Mapping):
        return [str(key) for key in data]
    names = getattr(data, "column_names", None)  # pyarrow Table / RecordBatch
    if names is None and hasattr(data, "__getitem__"):
        names = getattr(data, "columns", None)  # pandas, Polars
    if names is None:
        return None
    return [str(name) for name in names]


def table_row_count(data: Any) -> int | None:
    """Row count of a mapping, frame or Arrow table; ``None`` for anything else."""
    names = table_column_names(data)
    if names is None:
        return None
    if hasattr(data, "num_rows"):
        return int(data.num_rows)
    if not names:
        return 0
    return len(read_columns(data, names[:1], what="row count")[names[0]])


def project_columns(
    data: Any,
    names: Sequence[str],
    *,
    subset: bool = False,
    what: str = "analysis",
) -> Any:
    """``data`` restricted to the named columns, before any column is converted.

    Columns the analysis does not declare (strings, dates, identifiers) are never
    converted, so they cannot refuse a numeric ingest. Columns are chosen by name,
    never by position, and a pandas index is not a column.

    With ``subset=False`` every name is required and the result follows the order of
    ``names`` (a typed graph or a prepared schema fixes the order). With
    ``subset=True`` the names that are present are kept in the data's own order and
    the rest of the check is left to the consumer. A selected name that occurs twice
    in the data is refused either way. Inputs that are not tables pass through.
    """
    available = table_column_names(data)
    if available is None:
        return data
    wanted = list(dict.fromkeys(str(name) for name in names))
    present = set(available)
    if subset:
        keep = set(wanted)
        chosen = [name for name in dict.fromkeys(available) if name in keep]
    else:
        missing = [name for name in wanted if name not in present]
        if missing:
            raise CausalValueError(
                f"required {what} columns are missing: {missing}; available: {available}",
                reason_code="invalid_argument",
            )
        chosen = wanted
    duplicated = sorted({name for name in chosen if available.count(name) > 1})
    if duplicated:
        raise CausalValueError(
            f"data has duplicate column names: {duplicated}", reason_code="invalid_argument"
        )
    if chosen == available:
        return data
    positions = [available.index(name) for name in chosen]
    if isinstance(data, Mapping):
        keys = {str(key): key for key in data}
        return {name: data[keys[name]] for name in chosen}
    if hasattr(data, "iloc"):
        return data.iloc[:, positions]
    if hasattr(data, "select") and hasattr(data, "column_names"):
        return data.select(positions)
    if hasattr(data, "select"):
        return data.select(chosen)
    return data


def read_columns(data: Any, names: Sequence[str], *, what: str) -> dict[str, NDArray[Any]]:
    """The named columns as raw arrays of any dtype, in the order of ``names``.

    Identifier and design columns (strings, dates, labels) are read this way and kept
    out of the numeric ingest. Missing or duplicated names, and unequal lengths, are
    refused.
    """
    projected = project_columns(data, names, what=what)
    available = table_column_names(projected)
    if available is None:
        raise CausalTypeError(
            f"{what} columns require a mapping of name→array, a pandas or Polars frame, or an "
            f"Arrow table; got {type(data)!r}",
            reason_code="invalid_argument",
        )
    out: dict[str, NDArray[Any]] = {}
    for position, name in enumerate(available):
        if name in out:
            continue
        out[name] = _raw_array(projected, position, name)
    ordered = {name: out[str(name)] for name in dict.fromkeys(str(n) for n in names)}
    if len({len(column) for column in ordered.values()}) > 1:
        detail = ", ".join(f"{name}={len(column)}" for name, column in ordered.items())
        raise CausalValueError(
            f"{what} columns must have equal row counts ({detail})",
            reason_code="invalid_argument",
        )
    return ordered


def _raw_array(table: Any, position: int, name: str) -> NDArray[Any]:
    """One column of a projected table as a 1-d array, without a numeric cast."""
    if isinstance(table, Mapping):
        column: Any = table[next(key for key in table if str(key) == name)]
    elif hasattr(table, "iloc"):
        column = table.iloc[:, position].to_numpy()
    elif hasattr(table, "column"):
        column = table.column(position)
        column = (
            column.to_numpy()
            if hasattr(column, "chunks")
            else column.to_numpy(zero_copy_only=False)
        )
    else:
        column = table.get_column(name).to_numpy()
    array = np.asarray(column)
    if array.ndim != 1:
        raise CausalValueError(
            f"expected 1-d column {name!r}, got shape {array.shape}", reason_code="invalid_argument"
        )
    return array


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
    raise CausalTypeError(
        "data must be a mapping of name→array, a pandas DataFrame, or an Arrow-exporting "
        f"table (PyArrow, Polars); got {type(data)!r}",
        reason_code="invalid_argument",
    )


def _arrow_float64(column: Any, name: str | None = None) -> Any:
    """An Arrow-exporting column as float64, so every Arrow route casts alike.

    Integer, boolean and floating columns cast (Arrow refuses an integer beyond
    2**53); any other type is refused naming the column and its type, as strings and
    dates are on the numpy route. Without PyArrow the column passes through for the
    native reader.
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
    label = f" {name!r}" if name is not None else ""
    raise _refuse_column(
        f"column{label} has type {kind}, which is not numeric; only float, integer and boolean "
        "columns are accepted (encode categories and parse dates explicitly)"
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
            return names, [_arrow_float64(c, n) for n, c in zip(names, cols, strict=True)]
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
            return names, [_arrow_float64(c, n) for n, c in zip(names, flat, strict=True)]
        return None

    # Frame columns that export CDI, but not pandas (it has to_numpy and
    # Series.__arrow_c_array__; sending those skips the float64 cast).
    if hasattr(data, "columns") and not hasattr(data, "to_numpy"):
        try:
            names = [str(c) for c in data.columns]
            cols = [data[c] for c in data.columns]
            if names and all(hasattr(c, "__arrow_c_array__") for c in cols):
                return names, [_arrow_float64(c, n) for n, c in zip(names, cols, strict=True)]
        except (AttributeError, KeyError, TypeError):
            pass  # not a column-indexable frame; try a table-level stream next

    # Table-level Arrow PyCapsule (Polars, DuckDB); the same float64 cast as above.
    if hasattr(data, "__arrow_c_stream__"):
        try:
            import pyarrow as pa
        except ImportError:
            return None
        # pandas would export a non-default index as an extra column; the index is
        # not data, so it is never exported.
        table = (
            pa.Table.from_pandas(data, preserve_index=False)
            if hasattr(data, "iloc")
            else pa.table(data)
        )
        names = [str(n) for n in table.column_names]
        cols = [
            _arrow_float64(table.column(i).combine_chunks(), names[i]) for i in range(len(names))
        ]
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
        return _object_column_f64(raw, name=name, label=label)
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


class _ObjectColumnFallback(Exception):
    """The vectorized object cast cannot classify this column."""


def _none_mask(raw: NDArray[Any]) -> NDArray[np.bool_]:
    """True where an object column holds ``None``.

    Walking the flat buffer uses ``is None``, so pandas ``NA`` stays out of
    this mask and falls through to the typed element loop.
    """
    flat = raw.reshape(-1)
    mask = np.fromiter((item is None for item in flat), dtype=np.bool_, count=flat.size)
    return mask.reshape(raw.shape)


def _object_column_f64(raw: NDArray[Any], *, name: str | None, label: str) -> NDArray[np.float64]:
    """Object column to float64.

    ``None`` and ordinary numbers cast in NumPy. ``astype(float64)`` would also
    parse digit strings and round integers past 2**53, so those, and pandas
    ``NA`` / ``NaT``, fall through to the typed element loop that names the value.
    """
    if raw.size == 0:
        return np.empty(0, dtype=np.float64)
    missing = _none_mask(raw)
    try:
        return _cast_object_numbers(raw, missing, name=name, label=label, na_checked=False)
    except _ObjectColumnFallback:
        return _object_column_f64_each(raw, name=name, label=label)


def _cast_object_numbers(
    values: NDArray[Any],
    missing: NDArray[np.bool_],
    *,
    name: str | None,
    label: str,
    na_checked: bool,
) -> NDArray[np.float64]:
    work = np.array(values, dtype=object, copy=True)
    if missing.any():
        work[missing] = 0.0
    try:
        too_wide = np.logical_or(
            np.greater(work, _EXACT_INT_LIMIT), np.less(work, -_EXACT_INT_LIMIT)
        )
    except TypeError:
        return _retry_object_na(values, missing, name=name, label=label, na_checked=na_checked)
    if np.any(too_wide):
        for value in work[np.flatnonzero(too_wide)]:
            if isinstance(value, (bool, np.bool_, float, np.floating)):
                continue
            if isinstance(value, (int, np.integer)):
                where = f" (column{label})" if name is not None else ""
                raise _refuse_column(
                    f"integer {int(value)} exceeds 2**53 and would lose precision as float64{where}"
                )
            return _retry_object_na(values, missing, name=name, label=label, na_checked=na_checked)
    try:
        out = work.astype(np.float64)
    except (TypeError, ValueError):
        return _retry_object_na(values, missing, name=name, label=label, na_checked=na_checked)
    if missing.any():
        out = np.array(out, dtype=np.float64, copy=True)
        out[missing] = np.nan
    return out


def _retry_object_na(
    values: NDArray[Any],
    missing: NDArray[np.bool_],
    *,
    name: str | None,
    label: str,
    na_checked: bool,
) -> NDArray[np.float64]:
    if na_checked:
        raise _ObjectColumnFallback
    extra = np.fromiter(
        (type(value).__name__ in _PANDAS_MISSING for value in values),
        dtype=bool,
        count=values.size,
    )
    if not extra.any():
        raise _ObjectColumnFallback
    return _cast_object_numbers(
        values, np.logical_or(missing, extra), name=name, label=label, na_checked=True
    )


def _object_column_f64_each(
    raw: NDArray[Any], *, name: str | None, label: str
) -> NDArray[np.float64]:
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


def _check_column_lengths(names: list[str], cols: list[NDArray[np.float64]]) -> None:
    if len(cols) < 2:
        return
    lengths = [int(c.shape[0]) for c in cols]
    n0 = lengths[0]
    if all(n == n0 for n in lengths):
        return
    detail = ", ".join(f"{name}={n}" for name, n in zip(names, lengths, strict=True))
    raise CausalValueError(
        f"column length mismatch ({detail}); align rows or subset to a common index",
        reason_code="invalid_argument",
    )


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
        raise CausalValueError(
            "expected a non-empty sequence of environment frames",
            reason_code="invalid_argument",
        )
    names, first = ingest_columns(data[0])
    env_columns = [[_materialize_f64(col) for col in first]]
    for i, env in enumerate(data[1:], start=1):
        n, cols = ingest_columns(env)
        if n != names:
            raise CausalValueError(
                f"environment {i} column names {n!r} do not match environment 0 {names!r}",
                reason_code="invalid_argument",
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
        raise CausalTypeError(
            "provide data=… or both names= and columns=",
            reason_code="invalid_argument",
        )
    out_names = list(names)
    cols = [to_f64(c, name=str(n)) for n, c in zip(out_names, columns, strict=True)]
    _check_column_lengths(out_names, cols)
    return out_names, cols
