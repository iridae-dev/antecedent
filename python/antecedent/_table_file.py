"""Lazy handles on Parquet and Arrow IPC tables (``antecedent.data.from_parquet``).

A :class:`TableFile` reads only the schema and footer when it is opened. It is a
read-only mapping of column name to array, so ``ant.analyze``, ``ant.prepare`` and
a prepared study's ``estimate`` / ``refresh`` accept it wherever they accept a
mapping of columns: the analysis projects the columns it needs, and only those are
decoded, in bounded chunks charged to the admission pool, with the full-content
identity computed as they stream.

Reads keep types exactly. Floats are float64 with ``NaN`` in missing cells;
integers are int64 (an object array with ``None`` when a column has missing cells);
booleans are bool (object with ``None`` when missing); timestamps and dates are
``datetime64[ns]`` with ``NaT`` when missing; dictionaries and strings are object
arrays of labels with ``None`` when missing. Nested, decimal and binary columns are
refused with the next attempt, never cast.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import CausalTypeError, CausalValueError
from .execution import Execution

_PARQUET_SUFFIXES = (".parquet", ".parq", ".pq")
_IPC_FILE_SUFFIXES = (".arrow", ".feather", ".ipc")
_IPC_STREAM_SUFFIXES = (".arrows",)
_FORMATS = ("parquet", "arrow_ipc_file", "arrow_ipc_stream")


@dataclass(frozen=True)
class TableColumn:
    """One column of a table file's schema."""

    name: str
    kind: str
    """``float64``, ``int64``, ``boolean``, ``timestamp``, ``categorical`` or ``unsupported``."""
    nullable: bool
    arrow_type: str
    timezone: str | None = None
    ordered: bool = False


@dataclass(frozen=True)
class MemoryQuote:
    """Bytes a read needs, quoted before any byte is read.

    ``total_bytes`` / ``resident_bytes`` are ``None`` when the file records no row
    count (Arrow IPC); ``chunk_bytes`` is what one chunk of ``chunk_rows`` rows holds.
    """

    total_bytes: int | None
    resident_bytes: int | None
    chunk_bytes: int
    chunk_rows: int


@dataclass(frozen=True)
class GaussianSummary:
    """A table streamed through the bounded weighted Gaussian / OLS capacity route."""

    design_columns: list[str]
    coefficients: list[float]
    robust_covariance: NDArray[np.float64]
    """``HC0`` covariance of the coefficients, ``len(design_columns)`` square."""
    n_eff: float
    xtwx: NDArray[np.float64]
    xtwy: list[float]
    ytwy: float
    weighted_rss: float
    max_abs_residual: float
    leverage_max: float
    rows_read: int
    rows_used: int
    rows_dropped: int
    batches: int
    source_passes: int
    file_identity: str
    route_identity: str
    ceiling_total_bytes: int


def _native_class() -> Any:
    from ._native import NativeTableFile

    return NativeTableFile


def _format_of(path: str, format: str | None) -> str | None:
    if format is not None:
        if format not in _FORMATS and format not in ("ipc", "arrow", "feather", "arrows"):
            raise CausalValueError(
                f"format must be one of {list(_FORMATS)}; got {format!r}",
                reason_code="invalid_argument",
            )
        return format
    if os.path.isdir(path):
        return "parquet"
    return None


class TableFile(Mapping[str, NDArray[Any]]):
    """A Parquet file, a directory of Parquet files, or an Arrow IPC file or stream.

    Build one with :func:`antecedent.data.from_parquet` or
    :func:`antecedent.data.from_arrow_ipc`. Columns are read on demand: indexing one
    name reads that column; :meth:`read` reads several in one pass. Each read is
    planned from the requested columns only, streams in chunks of ``chunk_rows``
    rows charged to the admission pool (``execution.limits``), is cancelled by
    ``execution.cancel``, and is refused
    with the exact quote and the next attempt when the pool cannot hold it.
    """

    def __init__(
        self,
        native: Any,
        *,
        columns: Sequence[str] | None = None,
        chunk_rows: int | None = None,
        execution: Execution | None = None,
    ) -> None:
        if execution is not None and not isinstance(execution, Execution):
            raise CausalTypeError("execution must be an antecedent.Execution")
        memory_limit_bytes = None if execution is None else execution.limits.memory_bytes
        self._native = native
        self._execution = execution
        schema = [TableColumn(*row) for row in native.columns()]
        names = [column.name for column in schema]
        if columns is not None:
            wanted = [str(name) for name in columns]
            missing = [name for name in wanted if name not in names]
            if missing:
                raise CausalValueError(
                    f"columns {missing} are not in the table; available: {names}",
                    reason_code="invalid_argument",
                )
            if len(set(wanted)) != len(wanted):
                raise CausalValueError(
                    f"columns repeat a name: {wanted}", reason_code="invalid_argument"
                )
            by_name = {column.name: column for column in schema}
            schema = [by_name[name] for name in wanted]
        if chunk_rows is not None and int(chunk_rows) <= 0:
            raise CausalValueError("chunk_rows must be positive", reason_code="invalid_argument")
        if memory_limit_bytes is not None and int(memory_limit_bytes) <= 0:
            raise CausalValueError(
                "memory_limit_bytes must be positive", reason_code="invalid_argument"
            )
        self._schema = schema
        self._chunk_rows = None if chunk_rows is None else int(chunk_rows)
        self._memory_limit_bytes = None if memory_limit_bytes is None else int(memory_limit_bytes)
        self._cache: dict[str, NDArray[Any]] = {}

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str],
        *,
        format: str | None = None,
        columns: Sequence[str] | None = None,
        chunk_rows: int | None = None,
        execution: Execution | None = None,
    ) -> TableFile:
        """Open ``path``; the format follows the extension unless ``format`` names it."""
        if not isinstance(path, (str, os.PathLike)):
            raise CausalTypeError(
                f"path must be a str or os.PathLike; got {type(path)!r}",
                reason_code="invalid_argument",
            )
        text = os.fspath(path)
        native = _native_class().open(text, _format_of(text, format))
        return cls(native, columns=columns, chunk_rows=chunk_rows, execution=execution)

    # -- schema -----------------------------------------------------------------

    @property
    def format(self) -> str:
        """``parquet``, ``arrow_ipc_file`` or ``arrow_ipc_stream``."""
        return str(self._native.format)

    @property
    def paths(self) -> list[str]:
        """Files read, in order (one per Parquet file of a dataset)."""
        return list(self._native.paths)

    @property
    def schema(self) -> list[TableColumn]:
        """The visible columns, readable or not."""
        return list(self._schema)

    @property
    def names(self) -> list[str]:
        """Visible column names, in order."""
        return [column.name for column in self._schema]

    @property
    def num_rows(self) -> int | None:
        """Row count the footers record; ``None`` for Arrow IPC."""
        rows = self._native.declared_rows
        return None if rows is None else int(rows)

    # -- Mapping ----------------------------------------------------------------

    def __getitem__(self, name: str) -> NDArray[Any]:
        key = str(name)
        if key not in self.names:
            raise KeyError(name)
        if key not in self._cache:
            self._cache.update(self._read([key]))
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self._schema)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.names

    def __repr__(self) -> str:
        rows = self.num_rows
        return (
            f"TableFile(format={self.format!r}, files={len(self.paths)}, "
            f"columns={self.names!r}, rows={'unrecorded' if rows is None else rows})"
        )

    def without(self, names: set[str] | frozenset[str] | Sequence[str]) -> TableFile:
        """The same file with ``names`` hidden; nothing is read."""
        dropped = {str(name) for name in names}
        view = object.__new__(TableFile)
        view._native = self._native
        view._schema = [column for column in self._schema if column.name not in dropped]
        view._chunk_rows = self._chunk_rows
        view._memory_limit_bytes = self._memory_limit_bytes
        view._execution = self._execution
        view._cache = self._cache
        return view

    # -- reads ------------------------------------------------------------------

    def read(self, columns: Sequence[str] | None = None) -> dict[str, NDArray[Any]]:
        """``columns`` (every visible readable column when omitted) in one pass, in order."""
        wanted = (
            [column.name for column in self._schema if column.kind != "unsupported"]
            if columns is None
            else [str(name) for name in columns]
        )
        unknown = [name for name in wanted if name not in self.names]
        if unknown:
            raise CausalValueError(
                f"columns {unknown} are not in the table; available: {self.names}",
                reason_code="invalid_argument",
            )
        todo = [name for name in dict.fromkeys(wanted) if name not in self._cache]
        if todo:
            self._cache.update(self._read(todo))
        return {name: self._cache[name] for name in dict.fromkeys(wanted)}

    def _controls(self) -> dict[str, Any]:
        execution = self._execution
        return {
            "chunk_rows": self._chunk_rows,
            "memory_limit_bytes": self._memory_limit_bytes,
            "cancel": None if execution is None else execution.cancel,
        }

    def _run(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        if self._execution is None:
            return operation(*args, **kwargs)
        return self._execution._call(operation, *args, **kwargs)  # noqa: SLF001

    def _read(self, names: list[str]) -> dict[str, NDArray[Any]]:
        raw = self._run(self._native.read, names, **self._controls())
        return {str(entry["name"]): _to_numpy(entry) for entry in raw["columns"]}

    def _visible_readable(self) -> list[str]:
        return [column.name for column in self._schema if column.kind != "unsupported"]

    def content_identity(self, columns: Sequence[str] | None = None) -> str:
        """Full-content identity (hex) of ``columns``, streamed once; nothing is kept.

        The identity covers the row count, names, kinds, category domains and every
        value. It does not change with chunk size, row-group layout or container,
        and it changes when any value does.
        """
        names = self._visible_readable() if columns is None else [str(n) for n in columns]
        identity, _rows, _chunks = self._run(
            self._native.content_identity, names, **self._controls()
        )
        return str(identity)

    def quote(self, columns: Sequence[str] | None = None) -> MemoryQuote:
        """Bytes a resident read of ``columns`` needs, before reading."""
        names = self._visible_readable() if columns is None else [str(n) for n in columns]
        total, resident, chunk, chunk_rows = self._native.quote(names, chunk_rows=self._chunk_rows)
        return MemoryQuote(
            total_bytes=total, resident_bytes=resident, chunk_bytes=chunk, chunk_rows=chunk_rows
        )

    def summarize_weighted_gaussian(
        self,
        covariates: Sequence[str],
        response: str,
        *,
        weight: str | None = None,
        intercept: bool = True,
        missing: str = "refuse",
        batch_rows: int = 65_536,
        ceiling_bytes: int | None = None,
        control: str = "glm",
    ) -> GaussianSummary:
        """Stream the weighted Gaussian / OLS summary of ``response`` on ``covariates``.

        Only one chunk of ``batch_rows`` rows is resident at a time, so a table larger
        than memory is summarized under ``ceiling_bytes``. Returns coefficients (the
        intercept first when ``intercept``), the ``HC0`` covariance, the cross-products,
        and residual and leverage checks. ``missing="complete_case"`` leaves out rows
        missing a covariate, the response or the weight; ``"refuse"`` (the default)
        refuses the first one.
        """
        raw = self._run(
            self._native.summarize_weighted_gaussian,
            [str(name) for name in covariates],
            str(response),
            weight=None if weight is None else str(weight),
            intercept=bool(intercept),
            missing=str(missing),
            batch_rows=int(batch_rows),
            ceiling_bytes=None if ceiling_bytes is None else int(ceiling_bytes),
            control=str(control),
            memory_limit_bytes=self._memory_limit_bytes,
            cancel=self._controls()["cancel"],
        )
        width = len(raw["design_columns"])
        return GaussianSummary(
            design_columns=list(raw["design_columns"]),
            coefficients=[float(v) for v in raw["coefficients"]],
            robust_covariance=np.asarray(raw["robust_covariance"], dtype=np.float64).reshape(
                width, width
            ),
            n_eff=float(raw["n_eff"]),
            xtwx=np.asarray(raw["xtwx"], dtype=np.float64).reshape(width, width),
            xtwy=[float(v) for v in raw["xtwy"]],
            ytwy=float(raw["ytwy"]),
            weighted_rss=float(raw["weighted_rss"]),
            max_abs_residual=float(raw["max_abs_residual"]),
            leverage_max=float(raw["leverage_max"]),
            rows_read=int(raw["rows_read"]),
            rows_used=int(raw["rows_used"]),
            rows_dropped=int(raw["rows_dropped"]),
            batches=int(raw["batches"]),
            source_passes=int(raw["source_passes"]),
            file_identity=str(raw["file_identity"]),
            route_identity=str(raw["route_identity"]),
            ceiling_total_bytes=int(raw["ceiling_total_bytes"]),
        )


def _to_numpy(entry: Mapping[str, Any]) -> NDArray[Any]:
    """One native column as a NumPy array, missing cells as NaN / NaT / None."""
    kind = str(entry["kind"])
    values = np.asarray(entry["values"])
    valid = entry.get("valid")
    mask = None if valid is None else np.asarray(valid, dtype=bool)
    if kind == "float64":
        return np.asarray(values, dtype=np.float64)
    if kind == "timestamp":
        stamps = values.astype("datetime64[ns]")
        if mask is not None:
            stamps = stamps.copy()
            stamps[~mask] = np.datetime64("NaT")
        return stamps
    if kind == "categorical":
        labels = np.asarray(list(entry["labels"]), dtype=object)
        out = labels[values.astype(np.intp)] if len(labels) else np.full(len(values), None)
        out = np.asarray(out, dtype=object)
        if mask is not None:
            out = out.copy()
            out[~mask] = None
        return out
    if kind in ("int64", "boolean"):
        if mask is None:
            return values.astype(np.int64 if kind == "int64" else bool)
        out = values.astype(object)
        out[~mask] = None
        return out
    raise CausalTypeError(f"unexpected native column kind {kind!r}", reason_code="invalid_argument")


def from_parquet(
    path: str | os.PathLike[str],
    columns: Sequence[str] | None = None,
    *,
    chunk_rows: int | None = None,
    execution: Execution | None = None,
) -> TableFile:
    """Lazy handle on a Parquet file, or on a directory of Parquet files with one schema.

    Nothing but the schema and footer is read here. ``columns`` restricts the visible
    columns; an analysis then decodes only the ones it needs. ``chunk_rows`` bounds
    each chunk (65,536 rows by default); ``execution`` bounds the admission pool the
    chunks are charged to and cancels a read.

    Raises :class:`antecedent.errors.CausalValueError` for an unknown column and the
    native data error for a file that is not a readable Parquet table.
    """
    return TableFile.open(
        path,
        format="parquet",
        columns=columns,
        chunk_rows=chunk_rows,
        execution=execution,
    )


def from_arrow_ipc(
    path: str | os.PathLike[str],
    columns: Sequence[str] | None = None,
    *,
    stream: bool | None = None,
    chunk_rows: int | None = None,
    execution: Execution | None = None,
) -> TableFile:
    """Lazy handle on an Arrow IPC file (Feather v2) or, with ``stream=True``, an IPC stream.

    ``stream=None`` follows the extension (``.arrows`` is a stream). Record batches
    keep the size their writer chose; a batch larger than the chunk quote is charged
    when it is decoded and refused when the pool cannot hold it.
    """
    text = os.fspath(path)
    if stream is None:
        stream = text.lower().endswith(_IPC_STREAM_SUFFIXES)
    return TableFile.open(
        text,
        format="arrow_ipc_stream" if stream else "arrow_ipc_file",
        columns=columns,
        chunk_rows=chunk_rows,
        execution=execution,
    )


def is_table_path(data: Any) -> bool:
    """Whether ``data`` is a path to a table file or a Parquet dataset directory."""
    if not isinstance(data, (str, os.PathLike)):
        return False
    text = os.fspath(data)
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    if lowered.endswith(_PARQUET_SUFFIXES + _IPC_FILE_SUFFIXES + _IPC_STREAM_SUFFIXES):
        return True
    return os.path.isdir(text)


def coerce_table_source(data: Any) -> Any:
    """``data`` as a :class:`TableFile` when it is a path to a table; otherwise unchanged."""
    if isinstance(data, TableFile):
        return data
    if isinstance(data, (str, os.PathLike)):
        if not is_table_path(data):
            raise CausalValueError(
                f"data path {os.fspath(data)!r} is not a .parquet / .arrow / .feather / .arrows "
                "file or a directory of Parquet files",
                reason_code="invalid_argument",
            )
        return TableFile.open(data)
    return data
