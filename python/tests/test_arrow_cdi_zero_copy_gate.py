"""gate: PyArrow CDI ingest prefers zero-copy borrow of float64 values.

Requires pyarrow. Skipped when unavailable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("antecedent")

import pyarrow as pa

import antecedent


def test_arrow_c_zero_copy_acceptance():
    names = ["x", "y"]
    columns = [
        pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float64()),
        pa.array([10.0, 20.0, 30.0, 40.0], type=pa.float64()),
    ]
    info = causal.load_float64_arrow_c_columns(names, columns)
    assert info.row_count == 4
    assert info.column_count == 2
    assert info.column_names == names
    assert info.bytes_borrowed > 0
    # Value buffers are borrowed; validity may still be copied (all-valid path).
    assert info.bytes_borrowed >= 4 * 2 * 8
