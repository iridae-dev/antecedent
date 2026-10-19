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


class ArrowChunkColumn:
    """One logical column kept as Arrow chunks.

    ``__arrow_c_array__`` concatenates. Until a consumer calls it, ``chunks`` is
    the stream's batches, including an empty chunk.
    """

    def __init__(self, chunks: Sequence[Any], name: str) -> None:
        if not chunks:
            raise CausalValueError(
                f"column {name!r} has no Arrow chunks",
                reason_code="invalid_argument",
            )
        self.name = name
        self.chunks = tuple(chunks)

    def __len__(self) -> int:
        return sum(len(chunk) for chunk in self.chunks)

    def __arrow_c_array__(self, requested_schema: Any = None) -> Any:
        import pyarrow as pa

        if len(self.chunks) == 1:
            column = self.chunks[0]
        else:
            column = pa.chunked_array(list(self.chunks)).combine_chunks()
        return column.__arrow_c_array__(requested_schema)

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        import pyarrow as pa

        return pa.chunked_array(list(self.chunks)).__arrow_c_stream__(requested_schema)


def _is_capsule(obj: Any) -> bool:
    """True for an Arrow PyCapsule, including the ``arrow_array_stream`` capsule."""
    name = type(obj).__name__
    return name in {"PyCapsule", "Capsule"} or name.endswith("Capsule")


def _is_capsule_pair(obj: Any) -> bool:
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and _is_capsule(obj[0])
        and _is_capsule(obj[1])
    )


def _arrow_chunks(column: Any) -> list[Any] | None:
    """Chunk list when the column already exposes one. Does not combine."""
    chunks = getattr(column, "chunks", None)
    if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes)) and len(chunks) > 0:
        return list(chunks)
    return None


def _export_numeric(column: Any, name: str) -> Any:
    """Float64 view of a column. Several chunks stay several chunks."""
    chunks = _arrow_chunks(column)
    if chunks is not None:
        exported = [_arrow_float64(chunk, name) for chunk in chunks]
        if len(exported) == 1:
            return exported[0]
        return ArrowChunkColumn(exported, name)
    if hasattr(column, "combine_chunks"):
        column = column.combine_chunks()
    return _arrow_float64(column, name)


def _preserve_exact(column: Any, name: str | None = None) -> Any:
    """Keep integer ids, timestamps, dictionary categories, and float64 as themselves."""
    try:
        import pyarrow as pa
    except ImportError:
        return column
    try:
        col = column if isinstance(column, (pa.Array, pa.ChunkedArray)) else pa.array(column)
    except (TypeError, ValueError, pa.ArrowException):
        return column
    kind = col.type
    if (
        pa.types.is_integer(kind)
        or pa.types.is_timestamp(kind)
        or pa.types.is_dictionary(kind)
        or pa.types.is_float64(kind)
    ):
        return col
    label = f" {name!r}" if name is not None else ""
    raise _refuse_column(
        f"column{label} has type {kind}; integer ids, timestamps, dictionary categories, and "
        "float64 values are kept exactly (encode other types explicitly)"
    )


def _export_exact(column: Any, name: str) -> ArrowChunkColumn:
    """Exact typed chunks. A chunked column is not combined."""
    preserved = _preserve_exact(column, name)
    chunks = _arrow_chunks(preserved)
    if chunks is None:
        chunks = [preserved]
    else:
        chunks = [_preserve_exact(chunk, name) for chunk in chunks]
    return ArrowChunkColumn(chunks, name)


def _array_from_capsules(schema_capsule: Any, array_capsule: Any) -> Any:
    import pyarrow as pa

    importer = getattr(pa.Array, "_import_from_c_capsule", None)
    if importer is None:
        raise CausalTypeError(
            "this PyArrow build cannot import an Arrow C Data capsule pair",
            reason_code="invalid_argument",
        )
    return importer(schema_capsule, array_capsule)


def _table_from_arrow_export(data: Any) -> Any:
    """Record batches from a stream, a stream capsule, or a schema/array capsule pair.

    ``read_all`` keeps each batch as a chunk. Pandas indexes are not columns.
    """
    import pyarrow as pa

    if _is_capsule_pair(data):
        return pa.table({"value": _array_from_capsules(data[0], data[1])})
    if hasattr(data, "iloc"):
        return pa.Table.from_pandas(data, preserve_index=False)
    return pa.RecordBatchReader.from_stream(data).read_all()


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
            return names, [_export_numeric(c, str(n)) for n, c in zip(names, cols, strict=True)]
        return None

    # PyArrow Table / RecordBatch style
    names_attr = getattr(data, "column_names", None)
    if names_attr is None:
        schema = getattr(data, "schema", None)
        names_attr = getattr(schema, "names", None) if schema is not None else None
    if names_attr is not None and hasattr(data, "column"):
        names = [str(n) for n in list(names_attr)]
        cols = [_export_numeric(data.column(i), names[i]) for i in range(len(names))]
        if cols and all(_exports_arrow_array(c) for c in cols):
            return names, cols
        return None

    # Frame columns that export CDI, but not pandas (it has to_numpy and
    # Series.__arrow_c_array__; sending those skips the float64 cast).
    if hasattr(data, "columns") and not hasattr(data, "to_numpy"):
        try:
            names = [str(c) for c in data.columns]
            cols = [data[c] for c in data.columns]
            if names and all(hasattr(c, "__arrow_c_array__") for c in cols):
                return names, [_export_numeric(c, n) for n, c in zip(names, cols, strict=True)]
        except (AttributeError, KeyError, TypeError):
            pass  # not a column-indexable frame; try a table-level stream next

    # Table-level Arrow PyCapsule (Polars, DuckDB) or a bare stream/array capsule.
    # Batches stay chunks. The numeric route still casts each chunk to float64.
    if _is_capsule(data) or _is_capsule_pair(data) or hasattr(data, "__arrow_c_stream__"):
        try:
            table = _table_from_arrow_export(data)
        except ImportError:
            return None
        names = [str(n) for n in table.column_names]
        cols = [_export_numeric(table.column(i), names[i]) for i in range(len(names))]
        if names:
            return names, cols
    return None


def ingest_arrow_frame(data: Any) -> tuple[list[str], list[ArrowChunkColumn]] | None:
    """Chunk-preserving frame of exact ids, timestamps, categories, and float64.

    Integer columns stay integers (a unit id above 2**53 is not rounded).
    Timestamps stay timestamps. Dictionary columns stay dictionaries, with the
    dictionary order as the category domain. Chunks are not combined. Returns
    None when ``data`` is not an Arrow exporter. A mutable alias of the
    exporter is not a frozen study; freezing copies the bytes.
    """
    if isinstance(data, Mapping):
        names = [str(key) for key in data]
        cols = [data[key] for key in data]
        if not names or not all(
            hasattr(col, "__arrow_c_array__") or hasattr(col, "__arrow_c_stream__") for col in cols
        ):
            return None
        return names, [_export_exact(col, name) for name, col in zip(names, cols, strict=True)]

    names_attr = getattr(data, "column_names", None)
    if names_attr is None:
        schema = getattr(data, "schema", None)
        names_attr = getattr(schema, "names", None) if schema is not None else None
    if names_attr is not None and hasattr(data, "column"):
        names = [str(name) for name in list(names_attr)]
        if not names:
            return None
        return names, [_export_exact(data.column(i), names[i]) for i in range(len(names))]

    if hasattr(data, "columns") and not hasattr(data, "to_numpy"):
        try:
            names = [str(name) for name in data.columns]
            cols = [data[name] for name in data.columns]
        except (AttributeError, KeyError, TypeError):
            names, cols = [], []
        if names and all(
            hasattr(col, "__arrow_c_array__") or hasattr(col, "__arrow_c_stream__") for col in cols
        ):
            return names, [_export_exact(col, name) for name, col in zip(names, cols, strict=True)]

    if _is_capsule(data) or _is_capsule_pair(data) or hasattr(data, "__arrow_c_stream__"):
        try:
            table = _table_from_arrow_export(data)
        except ImportError:
            return None
        names = [str(name) for name in table.column_names]
        if not names:
            return None
        return names, [_export_exact(table.column(i), names[i]) for i in range(len(names))]
    return None


def _exports_arrow_array(column: Any) -> bool:
    return hasattr(column, "__arrow_c_array__") or hasattr(column, "__arrow_c_stream__")


def test_arrow_chunk_column_keeps_pieces() -> None:
    """Chunks stay addressable, including an empty piece, until export."""

    class _Piece:
        def __init__(self, n: int) -> None:
            self._n = n

        def __len__(self) -> int:
            return self._n

    class _Chunked:
        chunks = (_Piece(2), _Piece(0), _Piece(4))

    class _CombinedOnly:
        def combine_chunks(self) -> _Piece:
            return _Piece(6)

    pieces = _arrow_chunks(_Chunked())
    assert pieces is not None
    assert [len(piece) for piece in pieces] == [2, 0, 4]
    assert _arrow_chunks(_CombinedOnly()) is None
    column = ArrowChunkColumn(pieces, "unit_id")
    assert column.chunks[0] is pieces[0]
    assert len(column) == 6
    assert column.name == "unit_id"


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
