"""Owned tables with native numeric buffers and immutable design labels."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _native
from .errors import CausalValueError


@dataclass(frozen=True, slots=True, repr=False)
class Table(Mapping[str, Any]):
    """An immutable snapshot, reusable across analyses without re-ingestion.

    Numeric column access returns a read-only zero-copy NumPy view which pins its
    native owner after this Table and its prepared handles are released. Exact
    labels and times stay in immutable Python storage for design adapters; they
    are never implicitly encoded as numeric analysis columns.
    """

    _handle: Any = field(repr=False)
    _names: tuple[str, ...]
    _design: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)
    _design_bytes: int = field(default=0, repr=False)
    _rows: int | None = field(default=None, repr=False)

    @property
    def column_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def num_rows(self) -> int:
        return int(self._handle.row_count) if self._handle is not None else int(self._rows or 0)

    @property
    def bytes_copied(self) -> int:
        """Ownership bytes copied, including retained categorical array/tuple storage.

        Immutable scalar strings in object columns may be shared; their character
        storage is not copied or included in this count.
        """
        return (
            int(self._handle.bytes_copied) if self._handle is not None else 0
        ) + self._design_bytes

    @property
    def bytes_borrowed(self) -> int:
        """Numeric bytes retained under an explicit borrowing contract."""
        return int(self._handle.bytes_borrowed) if self._handle is not None else 0

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(self, name: str) -> Any:
        if name not in self._names:
            raise KeyError(name)
        if name in self._design:
            return self._design[name]
        return np.asarray(self._handle.column(name))

    def select(self, names: Sequence[str]) -> Table:
        """Select or reorder columns by name, sharing their native allocations."""
        chosen = tuple(names)
        if not chosen or len(set(chosen)) != len(chosen):
            raise CausalValueError("select requires distinct, nonempty column names")
        missing = [name for name in chosen if name not in self._names]
        if missing:
            raise CausalValueError(f"snapshot columns are missing: {missing}")
        return (
            self
            if chosen == self._names
            else Table(self._handle, chosen, self._design, self._design_bytes, self.num_rows)
        )

    def _native_columns(self) -> tuple[list[str], list[Any]]:
        # NumPy views keep the native snapshot as their buffer owner. Both modern
        # snapshot-aware ingest and older ndarray-only consumers can use them;
        # native numpy_buffer recognizes the owner and shares the same allocation.
        return list(self._names), [self[name] for name in self._names]

    def __repr__(self) -> str:
        return f"Table(rows={self.num_rows}, columns={self._names!r})"


def table(data: Any, *, borrow: bool = False) -> Table:
    """Own mutable data once, or explicitly borrow compatible numeric Arrow buffers.

    With ``borrow=True``, callers must keep all retained foreign buffers unchanged,
    including writable aliases. Ordinary mutable arrays always transfer through
    one ownership copy; borrowing does not promise unsupported NumPy aliasing.
    """
    if isinstance(data, Table):
        return data
    from ._data import as_columns, try_as_arrow_c_columns

    raw = _raw_columns(data)
    design: dict[str, Any] = {}
    design_bytes = 0
    rows = None
    if raw is not None:
        names, columns = raw
        numeric = {}
        for name, column in zip(names, columns, strict=True):
            if not (hasattr(column, "__arrow_c_array__") or hasattr(column, "chunks")):
                column = np.asarray(column)
                if column.ndim != 1:
                    raise CausalValueError(
                        f"expected 1-d column {name!r}, got shape {column.shape}"
                    )
            # Numeric Arrow exporters stay on the original zero-copy conversion
            # path. Only declared label/time representations stay outside that
            # numeric schema; no category codes or scientific domains are inferred.
            array = _design_array(column)
            if array is not None:
                if array.ndim != 1:
                    raise CausalValueError(f"expected 1-d column {name!r}, got shape {array.shape}")
                owned, copied = _own_design_array(array, name)
                design[name] = owned
                design_bytes += copied
            else:
                numeric[name] = column
            count = len(column)
            if rows is not None and count != rows:
                raise CausalValueError("snapshot columns must have equal lengths")
            rows = count
        data = (
            data.select(list(numeric))
            if numeric and hasattr(data, "column_names") and hasattr(data, "select")
            else numeric
        )
    arrow = try_as_arrow_c_columns(data)
    if arrow is None:
        numeric_names, columns = as_columns(data, native_dtypes=True)
    else:
        numeric_names, columns = arrow
    handle = (
        _native._TableSnapshot(numeric_names, columns, borrow=borrow)
        if numeric_names or not design
        else None
    )
    if not design:
        return Table(handle, tuple(numeric_names))
    return Table(handle, tuple(names), MappingProxyType(design), design_bytes, rows)


def _raw_columns(data: Any) -> tuple[list[str], list[Any]] | None:
    if isinstance(data, Mapping):
        return list(data), list(data.values())
    if hasattr(data, "columns") and hasattr(data, "to_numpy"):
        names = [str(name) for name in data.columns]
        if len(set(names)) != len(names):
            raise CausalValueError("data has duplicate column names")
        return names, [data[name].to_numpy() for name in data.columns]
    # Keep numeric Arrow tables on their chunk-preserving native path. A table
    # with labels exposes its original columns without decoding numeric buffers.
    if hasattr(data, "column_names") and hasattr(data, "column"):
        names = list(data.column_names)
        return names, [data.column(i) for i in range(len(names))]
    return None


def _design_array(column: Any) -> NDArray[Any] | None:
    import datetime

    if hasattr(column, "__arrow_c_array__") or hasattr(column, "chunks"):
        import pyarrow as pa

        kind = column.type
        if not (
            pa.types.is_string(kind)
            or pa.types.is_large_string(kind)
            or pa.types.is_dictionary(kind)
            or pa.types.is_timestamp(kind)
            or pa.types.is_date(kind)
        ):
            return None
        return np.asarray(column)
    array = np.asarray(column)
    if array.dtype.kind in "USMm":
        return array
    if (
        array.dtype.kind in "iu"
        and array.size
        and (int(array.min()) < -(2**53) or int(array.max()) > 2**53)
    ):
        return array  # Exact design identifiers must never round through float64.
    if array.dtype.kind == "O" and any(
        isinstance(value, (str, bytes, datetime.date, datetime.timedelta)) for value in array.flat
    ):
        return array
    return None


def _own_design_array(array: NDArray[Any], name: str) -> tuple[Any, int]:
    if not array.dtype.hasobject:
        # An immutable bytes owner prevents callers from re-enabling writes.
        return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape), array.nbytes
    import datetime

    scalar_types = (
        str,
        bytes,
        int,
        float,
        bool,
        datetime.date,
        datetime.timedelta,
        np.number,
        np.str_,
        np.bytes_,
        np.bool_,
        np.datetime64,
        np.timedelta64,
    )
    if any(value is not None and not isinstance(value, scalar_types) for value in array):
        raise CausalValueError(f"design column {name!r} requires immutable scalar labels")
    return tuple(array), array.size * np.dtype(np.intp).itemsize
