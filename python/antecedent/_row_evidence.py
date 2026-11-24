"""Where a prepared table's rows came from, and the units they observe.

Preparation keeps only the columns an analysis reads. Two studies prepared from the same
table but over different columns then hold different tables, yet observe the same rows: a
decision that joins their outcomes must count those rows once. A prepared static response
study carries the content identity of the whole table it was taken from (the identity the
engine gives a study of that table), whether preparation dropped columns or kept them all, and
its rows keep their positions in that table, so the overlap of such studies is judged on the
source rows.

Tables that are not the same table may still observe the same subjects. :class:`UnitKey`
names the column that identifies the subject of each row and the identifier space the
identifiers belong to; two studies keyed in one space are judged by the subjects they share.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._data import read_columns, table_column_names
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError

#: Personalisation of the table identity digest (at most 16 bytes).
_IDENTITY_PERSON = b"antecedent.rows"
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


@dataclass(frozen=True, slots=True)
class UnitKey:
    """The column identifying the unit each row observes, in a named identifier space.

    Pass as ``unit_key=`` to :func:`antecedent.prepare`. Studies of different tables whose
    outcomes a decision joins are judged for shared evidence by the units they share when both
    declare a key in the same ``namespace`` (for example ``"hospital_mrn"``); keys in different
    namespaces are never compared, and a key on one study only leaves the overlap with a study
    of another table unknown, which is refused. The column is read from the data as supplied,
    before the analysis columns are chosen, so it need not be an analysis variable.

    Identifiers keep their type: the text ``"1"`` and the integer ``1`` are different units, as
    are the integer ``1`` and the real ``1.0``. A missing identifier is refused.
    """

    column: str
    namespace: str

    def __post_init__(self) -> None:
        for name in ("column", "namespace"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise CausalTypeError(f"UnitKey.{name} must be a string")
            if not value.strip():
                raise CausalValueError(f"UnitKey.{name} must be non-empty")


def _label(value: Any, column: str) -> str:
    """The canonical, type-tagged label of one identifier."""
    if value is None or type(value).__name__ in ("NAType", "NaTType"):
        raise CausalValueError(
            f"unit key column {column!r} leaves a row without an identifier",
            reason_code="invalid_argument",
        )
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{int(bool(value))}"
    if isinstance(value, (int, np.integer)):
        number = int(value)
        if not _INT64_MIN <= number <= _INT64_MAX:
            raise CausalValueError(
                f"unit key column {column!r} holds an integer outside 64 bits",
                reason_code="invalid_argument",
            )
        return f"int64:{number}"
    if isinstance(value, (float, np.floating)):
        real = float(value)
        if not math.isfinite(real):
            raise CausalValueError(
                f"unit key column {column!r} leaves a row without a finite identifier",
                reason_code="invalid_argument",
            )
        return f"float64:{struct.unpack('<Q', struct.pack('<d', real))[0]:016x}"
    if isinstance(value, (str, np.str_)):
        text = str(value)
        if not text.strip():
            raise CausalValueError(
                f"unit key column {column!r} holds a blank identifier",
                reason_code="invalid_argument",
            )
        return f"text:{text}"
    raise CausalTypeError(
        f"unit key column {column!r} holds a {type(value).__name__}; identifiers are text, "
        "integers, reals or booleans",
        reason_code="invalid_argument",
    )


def _update_text(digest: Any, text: str) -> None:
    encoded = text.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _cell_text(value: Any) -> str:
    if value is None:
        return "none"
    return f"{type(value).__name__}:{value!r}"


def table_identity(data: Any, names: Sequence[str]) -> str:
    """The content identity of the whole table ``data``.

    It is the identity the engine gives a study of that table (the stored content digest of
    every column as ingested), so a study of a column projection and a study of the whole table
    name the same table. A table the engine cannot ingest whole (a column of a type no analysis
    reads) cannot be a study's table; its identity is then a digest of every column's name,
    type and values, in order. Neither fails on a column the analysis never reads: such a
    column is digested as it is, so preparing an analysis of the other columns never depends
    on it.
    """
    from ._data import ingest_columns
    from ._native import table_content_identity

    try:
        ingested, arrays = ingest_columns(data)
        return table_content_identity(list(ingested), list(arrays))
    except Exception:  # noqa: BLE001 - any column the engine cannot ingest is digested as it is
        return _column_digest(data, names)


def _digest_array(digest: Any, array: np.ndarray) -> None:
    _update_text(digest, array.dtype.str)
    if array.ndim == 1:
        digest.update(len(array).to_bytes(8, "little"))
    else:
        # A column that is not one value per row: its shape is part of its content.
        _update_text(digest, f"shape:{array.shape}")
    if array.dtype.kind in "biufcmM":
        digest.update(np.ascontiguousarray(array).tobytes())
    else:
        for value in array.ravel().tolist():
            _update_text(digest, _cell_text(value))


def _unread_column(data: Any, name: str) -> Any:
    """A column ``read_columns`` refuses (not one value per row, or another row count), as
    the table holds it."""
    if isinstance(data, Mapping):
        return next((data[key] for key in data if str(key) == name), None)
    return None


def _column_digest(data: Any, names: Sequence[str]) -> str:
    """A digest of every column's name, type and values, in order."""
    from .errors import CausalError

    digest = hashlib.blake2b(digest_size=32, person=_IDENTITY_PERSON)
    digest.update(len(names).to_bytes(8, "little"))
    for name in names:
        _update_text(digest, str(name))
        try:
            column = read_columns(data, [name], what="row provenance")[name]
        except (CausalError, TypeError, ValueError):
            raw = _unread_column(data, name)
            _update_text(digest, "unread")
            for dtype in (None, object):
                try:
                    array = np.asarray(raw, dtype=dtype)
                except (TypeError, ValueError):
                    continue
                _digest_array(digest, array)
                break
            else:
                _update_text(digest, _cell_text(raw))
            continue
        _digest_array(digest, np.asarray(column))
    return digest.hexdigest()


class RowSource:
    """What a prepared table keeps of the table it was taken from: the source's column names,
    its content identity, and the unit key's labels. It holds no
    data, so a prepared study that retains it retains nothing of the source table.
    """

    __slots__ = ("identity", "labels", "names", "unit_key")

    def __init__(
        self,
        names: tuple[str, ...] | None,
        identity: str | None,
        unit_key: UnitKey | None,
        labels: tuple[str, ...] | None,
    ) -> None:
        self.names = names
        self.identity = identity
        self.unit_key = unit_key
        self.labels = labels

    @classmethod
    def capture(
        cls,
        data: Any,
        *,
        unit_key: Any,
        provenance: bool,
    ) -> RowSource:
        """Read the unit key's labels from ``data`` and, when ``provenance`` applies, the
        content identity of the whole of ``data``."""
        if unit_key is not None and not isinstance(unit_key, UnitKey):
            raise CausalTypeError("unit_key must be an antecedent.UnitKey")
        available = table_column_names(data)
        names = tuple(available) if available is not None else None
        labels = None
        if unit_key is not None:
            if names is None or unit_key.column not in names:
                raise CausalValueError(
                    f"unit key column {unit_key.column!r} is not a column of the data",
                    reason_code="invalid_argument",
                )
            raw = read_columns(data, [unit_key.column], what="unit key")[unit_key.column]
            labels = tuple(_label(value, unit_key.column) for value in np.asarray(raw).tolist())
        identity = None
        if provenance and names is not None:
            identity = table_identity(data, names)
        return cls(names, identity, unit_key, labels)

    def native_options(self, *, applies: bool, route: str) -> dict[str, Any]:
        """The prepare options declaring the source table and the unit key; a unit key on a
        route that does not read it is refused."""
        options: dict[str, Any] = {}
        if self.unit_key is not None:
            if not applies:
                raise CausalUnsupportedError(
                    f"unit_key labels the rows of one table whose static ResponseCurve a "
                    f"decision may join; {route} does not read it",
                    reason_code="option_not_applicable",
                )
            options["unit_key"] = {
                "namespace": self.unit_key.namespace,
                "labels": list(self.labels or ()),
            }
        if applies and self.identity is not None:
            options["row_provenance"] = {"table": self.identity, "rows": None}
        return options


__all__ = ["UnitKey"]
