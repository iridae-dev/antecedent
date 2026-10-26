"""Native table ownership is paid once and shared by immutable views."""

from __future__ import annotations

import gc

import antecedent as ant
import numpy as np
import pytest
from antecedent.data import table


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


def test_numpy_snapshot_views_reenter_native_consumers_without_another_copy():
    from antecedent import _native

    data = table({"x": np.arange(8.0), "y": np.arange(8.0) + 1})
    info = _native.load_float64_columns(list(data), [data[name] for name in data])
    assert info.bytes_copied == 0
    assert info.diagnostic_count == 0
    assert info.row_count == 8
    # A sliced materialization owns exactly its selected observations.
    subset = _native.load_float64_columns(["x"], [data["x"][1:]])
    assert subset.bytes_copied == 7 * 8
    assert subset.diagnostic_count == 1


def test_snapshot_memory_admission_is_pinned_by_views_and_returns_on_last_owner_drop():
    from antecedent.execution import Execution, ResourceLimits

    policy = Execution(limits=ResourceLimits(memory_bytes=128))
    snapshot = policy._call(table, {"x": np.arange(8.0)})
    assert policy._handle.available_memory_bytes == 64
    view = snapshot["x"]
    del snapshot
    gc.collect()
    assert policy._handle.available_memory_bytes == 64
    np.testing.assert_array_equal(view, np.arange(8.0))
    del view
    gc.collect()
    assert policy._handle.available_memory_bytes == 128


def test_snapshot_ownership_refuses_insufficient_memory_and_honors_cancellation():
    from antecedent.errors import CausalCancelledError, CausalResourceError
    from antecedent.execution import Execution, ResourceLimits
    from antecedent.state import CancellationToken

    limited = Execution(limits=ResourceLimits(memory_bytes=1))
    with pytest.raises(CausalResourceError):
        limited._call(table, {"x": np.arange(8.0)})
    assert limited._handle.available_memory_bytes == 1
    token = CancellationToken()
    token.cancel()
    cancelled = Execution(cancel=token, limits=ResourceLimits(memory_bytes=128))
    with pytest.raises(CausalCancelledError):
        cancelled._call(table, {"x": np.arange(8.0)})
    assert cancelled._handle.available_memory_bytes == 128


def test_rewrapping_complete_snapshot_views_shares_buffer_and_original_admission():
    from antecedent.execution import Execution, ResourceLimits

    policy = Execution(limits=ResourceLimits(memory_bytes=64))
    original = policy._call(table, {"x": np.arange(8.0)})
    branch = policy._call(table, {"x": original["x"]})
    assert branch.bytes_copied == 0
    assert np.shares_memory(original["x"], branch["x"])
    assert policy._handle.available_memory_bytes == 0
    del original
    gc.collect()
    assert policy._handle.available_memory_bytes == 0
    del branch
    gc.collect()
    assert policy._handle.available_memory_bytes == 64
