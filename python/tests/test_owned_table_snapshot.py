"""Native table ownership is paid once and shared by immutable views."""

from __future__ import annotations

import gc

import antecedent as ant
import numpy as np
import pytest


def test_table_owns_mutable_input_once_and_projects_without_copying():
    original = np.arange(24.0)
    table = ant.data.table({"x": original, "y": 2 * original})
    assert table.bytes_copied == 2 * 24 * 8
    assert table.bytes_borrowed == 0
    original[:] = -1
    np.testing.assert_array_equal(table["x"], np.arange(24.0))
    projected = table.select(["y", "x"])
    assert projected.column_names == ("y", "x")
    assert np.shares_memory(projected["x"], table["x"])
    assert ant.data.table(table) is table
    assert table.num_rows == 24


def test_table_column_view_is_read_only_and_pins_owner():
    table = ant.data.table({"x": np.arange(12.0)})
    view = table["x"]
    assert not view.flags.writeable
    with pytest.raises(ValueError, match="read.only"):
        view[0] = -1
    with pytest.raises(ValueError, match="WRITEABLE"):
        view.setflags(write=True)
    del table
    gc.collect()
    np.testing.assert_array_equal(view, np.arange(12.0))
    copied = np.array(view, copy=True)
    copied[0] = 99
    assert view[0] == 0


def test_table_widens_strided_integer_input_during_single_native_copy():
    values = np.arange(32, dtype=np.int32)[::2]
    assert not values.flags.c_contiguous
    table = ant.data.table({"x": values})
    assert table.bytes_copied == 16 * 8
    np.testing.assert_array_equal(table["x"], np.arange(32)[::2])
    assert table["x"].dtype == np.float64


def test_native_snapshot_projection_recognizes_owned_columns_without_ingestion():
    table = ant.data.table({"x": np.arange(10.0), "y": np.arange(10.0) + 4})
    projected = table.select(["y"])
    names, columns = projected._native_columns()
    native = ant._native._TableSnapshot(names, columns)
    assert native.bytes_copied == 0
    assert native.bytes_borrowed == 0
    assert np.shares_memory(np.asarray(native.column("y")), table["y"])


def test_arrow_default_snapshot_is_owned_and_borrowing_is_explicit():
    pa = pytest.importorskip("pyarrow")
    values = np.arange(10.0)
    arrow = pa.table({"x": pa.array(values)})
    owned = ant.data.table(arrow)
    borrowed = ant.data.table(arrow, borrow=True)
    assert owned.bytes_copied == 10 * 8
    assert owned.bytes_borrowed == 0
    assert borrowed.bytes_copied == 0
    assert borrowed.bytes_borrowed == 10 * 8
    assert np.shares_memory(values, borrowed["x"])
    assert not np.shares_memory(values, owned["x"])
    # Keep the borrowing contract here: the foreign buffer is never mutated.
