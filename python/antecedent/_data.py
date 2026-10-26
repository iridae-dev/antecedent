"""Shared data normalization for the Python facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._table_file import TableFile
from ._table_snapshot import Table
from .errors import CausalTypeError, CausalValueError

# Integers beyond this are not exactly representable as float64.
_EXACT_INT_LIMIT = 2**53
_PANDAS_MISSING = frozenset({"NAType", "NaTType", "_NaT"})


def _refuse_column(message: str) -> CausalTypeError:
    """Typed refusal for non-numeric or unsafe column coercions."""
    return CausalTypeError(message, reason_code="invalid_argument")


@dataclass(frozen=True)
class ColumnCopy:
    """One array allocation ingestion made in Python, before the native table took ownership.

    ``kind`` names why the allocation could not be avoided: ``dtype_conversion`` (integer,
    boolean or narrower-float column widened to float64), ``object_cast`` (an object column
    cast to float64), ``extension_to_float64`` (a pandas nullable column filled with NaN),
    ``arrow_cast`` (an Arrow integer, boolean or float column cast to float64),
    ``chunk_concatenation`` (several Arrow chunks joined into the one contiguous buffer the
    native reader borrows), ``arrow_to_numpy`` (an Arrow column with nulls filled with NaN) or
    ``pandas_to_arrow`` (a pandas column the Arrow conversion could not share).
    """

    column: str | None
    kind: str
    nbytes: int


@dataclass
class _CopyLedger:
    """Collects the copies one ingestion makes."""

    copies: list[ColumnCopy] = field(default_factory=list)

    def record(self, column: str | None, kind: str, nbytes: int) -> None:
        self.copies.append(ColumnCopy(column, kind, int(nbytes)))


def _note(ledger: _CopyLedger | None, column: str | None, kind: str, nbytes: int) -> None:
    if ledger is not None:
        ledger.record(column, kind, nbytes)


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
    if isinstance(data, TableFile):
        if data.num_rows is not None:
            return data.num_rows
        return len(data[names[0]]) if names else 0
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
    if isinstance(data, TableFile):
        # One planned read of exactly these columns; the rest of the file is never decoded.
        return data.read(chosen)
    if chosen == available:
        return data
    positions = [available.index(name) for name in chosen]
    if isinstance(data, Table):
        return data.select(chosen)
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
    names, columns, _ = ingest_columns_with_receipt(data)
    return names, columns


def ingest_columns_with_receipt(
    data: Mapping[str, Any] | Any,
    *,
    native_dtypes: bool = False,
) -> tuple[list[str], list[Any], tuple[ColumnCopy, ...]]:
    """:func:`ingest_columns`, with every array allocation the Python side made.

    The returned copies are the allocations ingestion needed before the native table took its
    own ownership copy (or borrowed an Arrow buffer): each is a conversion the column's type
    required, named by its ``kind``. A column that needed none appears nowhere, so an empty
    tuple means the columns reached the native reader as they were supplied.
    """
    ledger = _CopyLedger()
    if isinstance(data, TableFile):
        data = data.read()
    arrow = try_as_arrow_c_columns(data, receipt=ledger)
    if arrow is not None:
        return arrow[0], arrow[1], tuple(ledger.copies)
    names, columns = as_columns(data, receipt=ledger, native_dtypes=native_dtypes)
    return names, columns, tuple(ledger.copies)


@dataclass(frozen=True)
class IngestionReceipt:
    """What loading one table copied, end to end: Python conversions plus native ownership.

    ``python_copies`` are the allocations Python made before the native reader (see
    :class:`ColumnCopy`). ``native`` is the native reader's own account: ``bytes_copied`` are
    the ownership copies it retained (one per NumPy column, gathered in the same pass when the
    column is strided), ``bytes_borrowed`` the Arrow buffers it borrowed without copying.
    """

    column_names: tuple[str, ...]
    row_count: int
    python_copies: tuple[ColumnCopy, ...]
    native: Any

    @property
    def python_copy_count(self) -> int:
        """Allocations Python made before the native reader."""
        return len(self.python_copies)

    @property
    def python_bytes_copied(self) -> int:
        """Bytes of those allocations."""
        return sum(copy.nbytes for copy in self.python_copies)

    @property
    def ownership_bytes_copied(self) -> int:
        """Bytes the native table copied into its own storage."""
        return int(self.native.bytes_copied)

    @property
    def bytes_borrowed(self) -> int:
        """Bytes the native table borrowed from Arrow buffers."""
        return int(self.native.bytes_borrowed)

    @property
    def total_bytes_copied(self) -> int:
        """Every byte copied between the caller's table and the native table."""
        return self.python_bytes_copied + self.ownership_bytes_copied

    def copies_of(self, kind: str) -> tuple[ColumnCopy, ...]:
        """The Python allocations of one ``kind``."""
        return tuple(copy for copy in self.python_copies if copy.kind == kind)


def ingestion_receipt(data: Mapping[str, Any] | Any) -> IngestionReceipt:
    """Load ``data`` through the analysis ingestion and report every copy it made.

    The native reader is the one analysis uses: Arrow C Data exporters when every column
    exports one (borrowed when the buffer is contiguous float64), NumPy float64 columns
    otherwise (one owned copy per column). The loaded table is not retained.
    """
    from ._native import load_float64_arrow_c_columns, load_float64_columns

    names, columns, copies = ingest_columns_with_receipt(data, native_dtypes=True)
    if columns and all(hasattr(column, "__arrow_c_array__") for column in columns):
        native = load_float64_arrow_c_columns(names, columns)
    else:
        native = load_float64_columns(names, columns)
    return IngestionReceipt(
        column_names=tuple(names),
        row_count=int(native.row_count),
        python_copies=copies,
        native=native,
    )


def as_columns(
    data: Mapping[str, Any] | Any,
    *,
    receipt: _CopyLedger | None = None,
    native_dtypes: bool = False,
) -> tuple[list[str], list[NDArray[np.float64]]]:
    """Normalize a mapping or pandas DataFrame to ``(names, float64 columns)``."""
    if isinstance(data, TableFile):
        data = data.read()
    if isinstance(data, Mapping):
        names = list(data.keys())
        cols = [
            to_f64(data[n], name=str(n), receipt=receipt, native_dtypes=native_dtypes)
            for n in names
        ]
        _check_column_lengths([str(n) for n in names], cols)
        return names, cols
    if hasattr(data, "columns") and hasattr(data, "to_numpy"):
        names = [str(c) for c in data.columns]
        duplicated = sorted({n for n in names if names.count(n) > 1})
        if duplicated:
            raise CausalValueError(f"data has duplicate column names: {duplicated}")
        cols = [
            to_f64(
                _frame_column_values(data[c], name=str(c), receipt=receipt),
                name=str(c),
                receipt=receipt,
                native_dtypes=native_dtypes,
            )
            for c in data.columns
        ]
        _check_column_lengths(names, cols)
        return names, cols
    arrow = try_as_arrow_c_columns(data, receipt=receipt)
    if arrow is not None:
        names, cols = arrow
        return names, [
            _materialize_f64(col, name=name, receipt=receipt)
            for name, col in zip(names, cols, strict=True)
        ]
    raise CausalTypeError(
        "data must be a mapping of name→array, a pandas DataFrame, or an Arrow-exporting "
        f"table (PyArrow, Polars); got {type(data)!r}",
        reason_code="invalid_argument",
    )


def _frame_column_values(series: Any, *, name: str, receipt: _CopyLedger | None) -> Any:
    """One frame column as an array, with the fewest allocations its dtype allows.

    A NumPy-backed column is read as a view. A pandas nullable integer, float or boolean
    column is filled with NaN straight into float64: the detour through an object array (a
    Python object per row) that ``to_numpy()`` would take is never made. Integers past 2**53
    are refused first, as on every other route. Any other column is read as ``to_numpy()``
    gives it and refused or cast by :func:`to_f64`.
    """
    dtype = getattr(series, "dtype", None)
    if isinstance(dtype, np.dtype):
        return series.to_numpy()
    kind = getattr(dtype, "kind", None)
    if kind in ("i", "u", "b", "f") and hasattr(series, "min") and hasattr(series, "max"):
        if kind in ("i", "u") and len(series):
            lowest, highest = series.min(), series.max()
            if not (_is_missing_scalar(lowest) or _is_missing_scalar(highest)) and (
                int(highest) > _EXACT_INT_LIMIT or int(lowest) < -_EXACT_INT_LIMIT
            ):
                raise _refuse_column(
                    f"integer column {name!r} spans [{int(lowest)}, {int(highest)}], beyond "
                    "2**53, and would lose precision as float64"
                )
        values = series.to_numpy(dtype=np.float64, na_value=np.nan)
        _note(receipt, name, "extension_to_float64", values.nbytes)
        return values
    return series.to_numpy()


def _arrow_float64(column: Any, name: str | None = None, receipt: _CopyLedger | None = None) -> Any:
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
        _note(receipt, name, "arrow_cast", len(col) * 8)
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


def _chunk_layout(column: Any) -> tuple[int, ...]:
    """Row counts of the chunks a column reaches the native reader in."""
    if isinstance(column, ArrowChunkColumn):
        return tuple(len(chunk) for chunk in column.chunks)
    return (len(column),)


def _settle_chunks(columns: list[Any], receipt: _CopyLedger | None) -> list[Any]:
    """Keep chunked columns chunked when every column shares one chunk layout.

    The native table aligns chunks across columns, so columns of different layouts (a chunked
    column beside a plain one, or chunked differently) cannot be loaded chunk by chunk. In that
    case each chunked column is joined here, and the join is recorded.
    """
    if not any(isinstance(column, ArrowChunkColumn) for column in columns):
        return columns
    if len({_chunk_layout(column) for column in columns}) == 1:
        return columns
    import pyarrow as pa

    settled: list[Any] = []
    for column in columns:
        if isinstance(column, ArrowChunkColumn):
            _note(receipt, column.name, "chunk_concatenation", len(column) * 8)
            column = pa.chunked_array(list(column.chunks)).combine_chunks()
        settled.append(column)
    return settled


def _is_capsule(obj: Any) -> bool:
    """True for an Arrow PyCapsule, including the ``arrow_array_stream`` capsule."""
    name = type(obj).__name__
    return name in {"PyCapsule", "Capsule"} or name.endswith("Capsule")


def _is_capsule_pair(obj: Any) -> bool:
    return isinstance(obj, tuple) and len(obj) == 2 and _is_capsule(obj[0]) and _is_capsule(obj[1])


def _arrow_chunks(column: Any) -> list[Any] | None:
    """Chunk list when the column already exposes one. Does not combine."""
    chunks = getattr(column, "chunks", None)
    if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes)) and len(chunks) > 0:
        return list(chunks)
    return None


def _export_numeric(column: Any, name: str, receipt: _CopyLedger | None = None) -> Any:
    """Float64 view of a column. Several chunks stay several chunks.

    A column of several chunks stays chunked: the native reader takes the chunks as they are
    and joins them into the table's own buffer in the one ownership copy it makes, so Python
    allocates nothing for it (:func:`_settle_chunks` joins only when columns disagree on their
    chunking, which the native table cannot align).
    """
    chunks = _arrow_chunks(column)
    if chunks is not None:
        exported = [_arrow_float64(chunk, name, receipt) for chunk in chunks]
        if len(exported) == 1:
            return exported[0]
        return ArrowChunkColumn(exported, name)
    if hasattr(column, "combine_chunks"):
        _note(receipt, name, "chunk_concatenation", len(column) * 8)
        column = column.combine_chunks()
    return _arrow_float64(column, name, receipt)


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


def _table_from_arrow_export(data: Any, receipt: _CopyLedger | None = None) -> Any:
    """Record batches from a stream, a stream capsule, or a schema/array capsule pair.

    ``read_all`` keeps each batch as a chunk. Pandas indexes are not columns. A table the
    producer already holds in memory (a PyArrow table, a Polars frame) is used with its own
    chunks: re-slicing it into engine-sized batches and joining them again would add a copy of
    every multi-chunk column and bound nothing, since the producer has allocated it already.
    """
    import pyarrow as pa

    if _is_capsule_pair(data):
        return pa.table({"value": _array_from_capsules(data[0], data[1])})
    if hasattr(data, "iloc"):
        table = pa.Table.from_pandas(data, preserve_index=False)
        _record_pandas_conversion(data, table, receipt)
        return table
    if isinstance(data, pa.Table):
        return data
    if hasattr(data, "to_arrow") and hasattr(data, "iter_slices"):  # Polars DataFrame
        converted = data.to_arrow()
        if isinstance(converted, pa.Table):
            return converted
    return bounded_arrow_reader(data, ARROW_STREAM_CHUNK_ROWS).read_all()


def _record_pandas_conversion(frame: Any, table: Any, receipt: _CopyLedger | None) -> None:
    """Record the pandas columns the Arrow conversion could not share with the frame.

    A NumPy-backed column whose Arrow value buffer sits at the frame column's own address was
    shared; every other column (extension dtypes, NaN-bearing floats, converted types) was
    copied into its own buffer.
    """
    if receipt is None:
        return
    for position, name in enumerate(table.column_names):
        column = table.column(position)
        series = frame.iloc[:, position]
        shared = False
        if isinstance(getattr(series, "dtype", None), np.dtype) and column.num_chunks == 1:
            buffers = column.chunk(0).buffers()
            values = series.to_numpy()
            shared = (
                len(buffers) > 1
                and buffers[1] is not None
                and values.size > 0
                and buffers[1].address == values.ctypes.data
            )
        if not shared:
            receipt.record(str(name), "pandas_to_arrow", column.nbytes)


#: Rows a producer is asked to put in one record batch; the engine's default chunk size.
ARROW_STREAM_CHUNK_ROWS = 65_536


def bounded_arrow_reader(data: Any, max_rows: int) -> Any:
    """A ``pyarrow.RecordBatchReader`` over ``data`` with batches of at most ``max_rows`` rows.

    The batch size is requested from the producer before export, so an oversized batch is
    never built: pyarrow tables and datasets use ``to_reader(max_chunksize=...)``, Polars
    frames are sliced with ``iter_slices``, DuckDB relations and results use
    ``fetch_record_batch(rows_per_batch=...)``. A producer with no such request is read through
    its Arrow stream and each batch is split into zero-copy slices as it arrives; that bounds
    what the engine holds, not what the producer allocated.
    """
    import pyarrow as pa

    if max_rows < 1:
        raise CausalValueError("max_rows must be at least one row", reason_code="invalid_argument")
    if isinstance(data, pa.RecordBatchReader):
        return _split_reader(data, max_rows)
    if isinstance(data, pa.Table):
        return data.to_reader(max_chunksize=max_rows)
    if hasattr(data, "to_reader") and hasattr(data, "schema"):  # pyarrow dataset/scanner
        try:
            return data.to_reader(max_chunksize=max_rows)
        except TypeError:
            return _split_reader(data.to_reader(), max_rows)
    if hasattr(data, "fetch_record_batch"):  # DuckDB relation or result
        if hasattr(data, "to_arrow_reader"):
            return data.to_arrow_reader(batch_size=max_rows)
        return data.fetch_record_batch(rows_per_batch=max_rows)
    if hasattr(data, "iter_slices") and hasattr(data, "to_arrow"):  # Polars DataFrame
        first = data.head(0).to_arrow()
        slices = (frame.to_arrow().to_batches() for frame in data.iter_slices(max_rows))
        return pa.RecordBatchReader.from_batches(
            first.schema, (batch for group in slices for batch in group)
        )
    return _split_reader(pa.RecordBatchReader.from_stream(data), max_rows)


def _split_reader(reader: Any, max_rows: int) -> Any:
    """Re-slice each batch of ``reader`` to at most ``max_rows`` rows, without copying."""
    import pyarrow as pa

    def batches() -> Any:
        for batch in reader:
            for start in range(0, batch.num_rows, max_rows):
                yield batch.slice(start, min(max_rows, batch.num_rows - start))
            if batch.num_rows == 0:
                yield batch

    return pa.RecordBatchReader.from_batches(reader.schema, batches())


def try_as_arrow_c_columns(
    data: Any,
    *,
    receipt: _CopyLedger | None = None,
) -> tuple[list[str], list[Any]] | None:
    """If ``data`` exports Arrow C Data Interface columns, return ``(names, cols)``.

    Accepts:
    - a mapping of name → object with ``__arrow_c_array__``
    - a table-like with ``column_names`` / ``column(i)`` (PyArrow Table)
    - a table-like with ``schema.names`` and ``column(i)``

    Returns ``None`` when the object is not an Arrow CDI exporter (caller should
    fall back to [`as_columns`]).
    """
    if isinstance(data, TableFile):
        return None
    if isinstance(data, Mapping):
        names = list(data.keys())
        cols = [data[n] for n in names]
        if names and all(hasattr(c, "__arrow_c_array__") for c in cols):
            return names, _settle_chunks(
                [_export_numeric(c, str(n), receipt) for n, c in zip(names, cols, strict=True)],
                receipt,
            )
        return None

    # PyArrow Table / RecordBatch style
    names_attr = getattr(data, "column_names", None)
    if names_attr is None:
        schema = getattr(data, "schema", None)
        names_attr = getattr(schema, "names", None) if schema is not None else None
    if names_attr is not None and hasattr(data, "column"):
        names = [str(n) for n in list(names_attr)]
        cols = [_export_numeric(data.column(i), names[i], receipt) for i in range(len(names))]
        if cols and all(_exports_arrow_array(c) for c in cols):
            return names, _settle_chunks(cols, receipt)
        return None

    # Frame columns that export CDI, but not pandas (it has to_numpy and
    # Series.__arrow_c_array__; sending those skips the float64 cast).
    if hasattr(data, "columns") and not hasattr(data, "to_numpy"):
        try:
            names = [str(c) for c in data.columns]
            cols = [data[c] for c in data.columns]
            if names and all(hasattr(c, "__arrow_c_array__") for c in cols):
                return names, _settle_chunks(
                    [_export_numeric(c, n, receipt) for n, c in zip(names, cols, strict=True)],
                    receipt,
                )
        except (AttributeError, KeyError, TypeError):
            pass  # not a column-indexable frame; try a table-level stream next

    # Table-level Arrow PyCapsule (Polars, DuckDB) or a bare stream/array capsule.
    # Batches stay chunks. The numeric route still casts each chunk to float64.
    if _is_capsule(data) or _is_capsule_pair(data) or hasattr(data, "__arrow_c_stream__"):
        try:
            table = _table_from_arrow_export(data, receipt)
        except ImportError:
            return None
        names = [str(n) for n in table.column_names]
        cols = [_export_numeric(table.column(i), names[i], receipt) for i in range(len(names))]
        if names:
            return names, _settle_chunks(cols, receipt)
    return None


def ingest_arrow_frame(data: Any) -> tuple[list[str], list[ArrowChunkColumn]] | None:
    """Chunk-preserving frame of exact ids, timestamps, categories, and float64.

    Integer columns stay integers (a unit id above 2**53 is not rounded).
    Timestamps stay timestamps. Dictionary columns stay dictionaries, with the
    dictionary order as the category domain. Chunks are not combined. Returns
    None when ``data`` is not an Arrow exporter. A mutable alias of the
    exporter is not a frozen study; freezing copies the bytes.
    """
    if isinstance(data, TableFile):
        return None
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


def to_f64(
    arr: Any,
    *,
    name: str | None = None,
    receipt: _CopyLedger | None = None,
    native_dtypes: bool = False,
) -> NDArray[np.float64]:
    """One numeric column as float64, naming the column in errors when known.

    Only numeric inputs are accepted: floats, integers within 2**53 (larger ones
    do not survive the cast), and booleans (as 0/1). Strings, datetimes,
    timedeltas and complex numbers are refused rather than parsed or reinterpreted
    (digit strings are labels, and a datetime is not a number of nanoseconds).
    A nullable pandas column's ``NA`` and ``None`` become NaN, which is missing.

    With ``native_dtypes`` an integer, boolean or float32 NumPy column is range-checked and
    returned as it is: the native reader widens it to float64 while it makes its one ownership
    copy, so no float64 temporary is allocated here. Only for callers whose native entry accepts
    those dtypes (``tabular_from_py_columns``, ``load_float64_columns``).
    """
    label = f" {name!r}" if name is not None else ""
    raw = np.asarray(arr)
    if raw.ndim != 1:
        raise CausalValueError(f"expected 1-d column{label}, got shape {raw.shape}")
    if isinstance(arr, (list, tuple, range)):
        _note(receipt, name, "dtype_conversion", raw.nbytes)
    kind = raw.dtype.kind
    if native_dtypes and isinstance(arr, np.ndarray) and raw.dtype != np.float64:
        # The dtypes the native reader widens while it copies.
        if kind in "iub" or raw.dtype == np.float32:
            if kind in "iu" and raw.size:
                lowest, highest = int(raw.min()), int(raw.max())
                if highest > _EXACT_INT_LIMIT or lowest < -_EXACT_INT_LIMIT:
                    raise _refuse_column(
                        f"integer column{label} spans [{lowest}, {highest}], beyond 2**53, and "
                        "would lose precision as float64"
                    )
            return raw  # type: ignore[return-value]
    if kind == "O":
        return _object_column_f64(raw, name=name, label=label, receipt=receipt)
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
    out = np.asarray(raw, dtype=np.float64)
    if out is not raw and not np.shares_memory(out, raw):
        _note(receipt, name, "dtype_conversion", out.nbytes)
    return out


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


def _object_column_f64(
    raw: NDArray[Any], *, name: str | None, label: str, receipt: _CopyLedger | None = None
) -> NDArray[np.float64]:
    """Object column to float64.

    ``None`` and ordinary numbers cast in NumPy. ``astype(float64)`` would also
    parse digit strings and round integers past 2**53, so those, and pandas
    ``NA`` / ``NaT``, fall through to the typed element loop that names the value.
    """
    if raw.size == 0:
        return np.empty(0, dtype=np.float64)
    missing = _none_mask(raw)
    try:
        return _cast_object_numbers(
            raw, missing, name=name, label=label, na_checked=False, receipt=receipt
        )
    except _ObjectColumnFallback:
        out = _object_column_f64_each(raw, name=name, label=label)
        _note(receipt, name, "object_cast", out.nbytes)
        return out


def _cast_object_numbers(
    values: NDArray[Any],
    missing: NDArray[np.bool_],
    *,
    name: str | None,
    label: str,
    na_checked: bool,
    receipt: _CopyLedger | None = None,
) -> NDArray[np.float64]:
    # The input is only read, so it is copied just to fill its missing entries; the one
    # allocation every column needs is the float64 result.
    has_missing = bool(missing.any())
    if has_missing:
        work = np.array(values, dtype=object, copy=True)
        work[missing] = 0.0
        _note(receipt, name, "object_cast", work.nbytes)
    else:
        work = values
    try:
        too_wide = np.logical_or(
            np.greater(work, _EXACT_INT_LIMIT), np.less(work, -_EXACT_INT_LIMIT)
        )
    except TypeError:
        return _retry_object_na(
            values, missing, name=name, label=label, na_checked=na_checked, receipt=receipt
        )
    if np.any(too_wide):
        for value in work[np.flatnonzero(too_wide)]:
            if isinstance(value, (bool, np.bool_, float, np.floating)):
                continue
            if isinstance(value, (int, np.integer)):
                where = f" (column{label})" if name is not None else ""
                raise _refuse_column(
                    f"integer {int(value)} exceeds 2**53 and would lose precision as float64{where}"
                )
            return _retry_object_na(
                values, missing, name=name, label=label, na_checked=na_checked, receipt=receipt
            )
    try:
        out = work.astype(np.float64)
    except (TypeError, ValueError):
        return _retry_object_na(
            values, missing, name=name, label=label, na_checked=na_checked, receipt=receipt
        )
    _note(receipt, name, "object_cast", out.nbytes)
    if has_missing:
        # `astype` returned a fresh array, so the missing entries are written in place.
        out[missing] = np.nan
    return out


def _retry_object_na(
    values: NDArray[Any],
    missing: NDArray[np.bool_],
    *,
    name: str | None,
    label: str,
    na_checked: bool,
    receipt: _CopyLedger | None = None,
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
        values,
        np.logical_or(missing, extra),
        name=name,
        label=label,
        na_checked=True,
        receipt=receipt,
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


def _materialize_f64(
    column: Any, *, name: str | None = None, receipt: _CopyLedger | None = None
) -> NDArray[np.float64]:
    """Numpy view of an ingested column (Arrow CDI or array-like).

    An Arrow column is read in place when it has no nulls; only a column with nulls is
    filled with NaN into a new buffer, and that allocation is recorded.
    """
    if hasattr(column, "__arrow_c_array__"):
        try:
            import pyarrow as pa
        except ImportError:
            pass  # no PyArrow: read the column through its array protocols
        else:
            if isinstance(column, ArrowChunkColumn) and len(column.chunks) > 1:
                _note(receipt, name, "chunk_concatenation", len(column) * 8)
            array = pa.array(column)
            try:
                values = array.to_numpy(zero_copy_only=True)
            except pa.ArrowException:
                values = array.to_numpy(zero_copy_only=False)
                _note(receipt, name, "arrow_to_numpy", values.nbytes)
            return to_f64(values, name=name, receipt=receipt)
    if hasattr(column, "to_numpy"):
        return to_f64(column.to_numpy(), name=name, receipt=receipt)
    return to_f64(column, name=name, receipt=receipt)


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
