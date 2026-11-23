"""The one JSON digest every Python identity uses; Rust owns the encoding.

A host language never hashes a JSON document itself: ``json.dumps`` output
varies with key order, separators and number formatting, and two modules that
chose differently would disagree about the same value. The document goes to
Rust, which hashes a domain-separated BLAKE3 stream with sorted keys and typed
values (an integer ``1`` differs from a float ``1.0``).

Every durable identity names its domain, and changing what a domain hashes
means a new domain version, never a silent change under an old name.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

# Same text ``json.dumps(..., allow_nan=False)`` raises. Callers and tests match it.
_NON_FINITE = "Out of range float values are not JSON compliant"

#: Reserved mapping keys. A mapping that already carries one is refused, so a
#: caller's data can never pose as the digest of an array or table.
_ARRAY_KEY = "__antecedent_array__"
_TABLE_KEY = "__antecedent_table__"


def _reject_non_finite(value: float) -> float:
    """Refuse NaN and infinities before they can be hashed.

    ``allow_nan=False`` already rejects them in JSON text. Float arrays used to
    skip that check and hash the IEEE bytes, so a NaN changed the digest.
    Finite values are not rewritten.
    """
    if not math.isfinite(value):
        raise ValueError(_NON_FINITE)
    return value


def _array_form(array: np.ndarray) -> dict[str, Any]:
    """An array as its dtype, shape and the BLAKE3 of its little-endian bytes.

    The bytes are the data: nothing is rounded, truncated or ``repr``-ed, so two
    arrays share a form only when they hold the same values at the same dtype
    and shape. Object arrays are canonical only when every element is.
    """
    if array.dtype.kind in "fc" and array.size and not np.isfinite(array).all():
        raise ValueError(_NON_FINITE)
    if array.dtype.kind == "O":
        return {
            _ARRAY_KEY: {
                "dtype": "object",
                "shape": list(array.shape),
                "items": _plain(array.tolist()),
            }
        }
    if array.dtype.kind == "V":
        raise TypeError("a structured array has no canonical form; pass its columns")
    from . import _native

    ordered = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    return {
        _ARRAY_KEY: {
            "dtype": ordered.dtype.str,
            "shape": list(ordered.shape),
            "blake3": _native.blake3_digest_hex(ordered.tobytes()),
        }
    }


def _table_form(columns: list[tuple[str, np.ndarray]], index: np.ndarray | None) -> dict[str, Any]:
    return {
        _TABLE_KEY: {
            "columns": [[name, _array_form(values)] for name, values in columns],
            "index": None if index is None else _array_form(index),
        }
    }


def _foreign_table(item: Any) -> dict[str, Any] | None:
    """A pandas or Arrow table as ordered named columns, or ``None`` for anything else.

    Detected by interface, so neither library is imported here.
    """
    module = type(item).__module__.split(".")[0]
    if module == "pandas" and hasattr(item, "columns") and hasattr(item, "to_numpy"):
        if any(not isinstance(name, str) for name in item.columns):
            raise TypeError("table column names must be strings for canonical digests")
        names = list(item.columns)
        if len(set(names)) != len(names):
            raise TypeError("a table with repeated column names has no canonical form")
        index = item.index
        default = type(index).__name__ == "RangeIndex" and index.start == 0 and index.step == 1
        return _table_form(
            [
                (name, item[column].to_numpy())
                for name, column in zip(names, item.columns, strict=True)
            ],
            None if default else np.asarray(index),
        )
    if module == "pyarrow" and hasattr(item, "column_names") and hasattr(item, "to_pydict"):
        return _table_form(
            [
                (str(name), item.column(name).to_numpy(zero_copy_only=False))
                for name in item.column_names
            ],
            None,
        )
    return None


def _plain(item: Any) -> Any:
    """``item`` as JSON-native values; anything else is refused, never ``repr``-ed."""
    if isinstance(item, np.ndarray):
        return _array_form(item)
    if isinstance(item, np.generic):
        if item.dtype.kind == "V":
            raise TypeError("a structured scalar has no canonical form; pass its columns")
        value = item.item()
        # Some extended-precision scalars return another NumPy scalar from item().
        # Refuse instead of recursing forever or passing an unchecked object to JSON.
        if isinstance(value, np.generic):
            raise TypeError(f"{type(item).__name__} has no canonical scalar form")
        return _plain(value)
    if isinstance(item, Mapping):
        if any(not isinstance(key, str) for key in item):
            raise TypeError("mapping keys must be strings for canonical digests")
        reserved = {_ARRAY_KEY, _TABLE_KEY}.intersection(item)
        if reserved:
            raise TypeError(f"mapping key {sorted(reserved)[0]!r} is reserved for digests")
        return {key: _plain(value) for key, value in item.items()}
    if isinstance(item, (tuple, list)):
        return [_plain(value) for value in item]
    if isinstance(item, float):
        return _reject_non_finite(item)
    if isinstance(item, int) and not isinstance(item, bool):
        if item < -(1 << 63) or item > (1 << 64) - 1:
            raise ValueError("canonical digest integer is outside i64/u64 range")
        return item
    if isinstance(item, (str, bool)) or item is None:
        return item
    table = _foreign_table(item)
    if table is not None:
        return table
    raise TypeError(
        f"{type(item).__name__} has no canonical form; pass JSON values, NumPy arrays or columns"
    )


def canonical_digest(domain: str, value: Any) -> str:
    """Rust's canonical BLAKE3 digest (64 hex characters) of ``value`` under ``domain``.

    ``value`` is mappings, sequences, scalars, NumPy arrays and column tables
    (pandas, Arrow), string mapping keys and table column names, and finite numbers.
    Integers must fit
    signed or unsigned 64-bit range. Unsupported types raise :class:`TypeError`;
    non-finite or out-of-range numbers raise :class:`ValueError`.
    Arrays and tables enter by dtype, shape and the
    digest of their bytes, so a large dataset is hashed in full, never by its
    ``repr``.
    """
    from . import _native

    return str(
        _native.canonical_json_digest_hex(domain, json.dumps(_plain(value), allow_nan=False))
    )
