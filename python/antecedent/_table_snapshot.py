"""Owned native tables with immutable, owner-pinned numeric views."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _native
from .errors import CausalValueError


@dataclass(frozen=True, slots=True, repr=False)
class Table(Mapping[str, NDArray[np.float64]]):
    """An immutable native snapshot, reusable across analyses without re-ingestion.

    Column access returns a read-only zero-copy NumPy view. The view pins its
    native owner even after this Table and its prepared handles are released.
    """

    _handle: Any = field(repr=False)
    _names: tuple[str, ...]

    @property
    def column_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def num_rows(self) -> int:
        return int(self._handle.row_count)

    @property
    def bytes_copied(self) -> int:
        """Numeric ownership bytes copied when this snapshot was created."""
        return int(self._handle.bytes_copied)

    @property
    def bytes_borrowed(self) -> int:
        """Numeric bytes retained under an explicit borrowing contract."""
        return int(self._handle.bytes_borrowed)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(self, name: str) -> NDArray[np.float64]:
        if name not in self._names:
            raise KeyError(name)
        return np.asarray(self._handle.column(name))

    def select(self, names: Sequence[str]) -> Table:
        """Select or reorder columns by name, sharing their native allocations."""
        chosen = tuple(names)
        if not chosen or len(set(chosen)) != len(chosen):
            raise CausalValueError("select requires distinct, nonempty column names")
        missing = [name for name in chosen if name not in self._names]
        if missing:
            raise CausalValueError(f"snapshot columns are missing: {missing}")
        return self if chosen == self._names else Table(self._handle, chosen)

    def _native_columns(self) -> tuple[list[str], list[Any]]:
        return list(self._names), [self._handle.column(name) for name in self._names]

    def __repr__(self) -> str:
        return f"Table(rows={self.num_rows}, columns={self._names!r})"


def table(data: Any, *, borrow: bool = False) -> Table:
    """Own mutable numeric data once, or explicitly borrow compatible Arrow buffers.

    With ``borrow=True``, callers must keep all retained foreign buffers unchanged,
    including writable aliases. Ordinary mutable arrays always transfer through
    one ownership copy; borrowing does not promise unsupported NumPy aliasing.
    """
    if isinstance(data, Table):
        return data
    from ._data import as_columns, try_as_arrow_c_columns

    arrow = try_as_arrow_c_columns(data)
    if arrow is None:
        names, columns = as_columns(data, native_dtypes=True)
    else:
        names, columns = arrow
    handle = _native._TableSnapshot(names, columns, borrow=borrow)
    return Table(handle, tuple(names))
