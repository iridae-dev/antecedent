"""gate: Python loads the shared Arrow/float fixture with measured copy.

Fixture: conformance/gates/arrow_copy_fixture.json (shared with Rust).
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest

from _repo_text import REPO_ROOT, load_json

FIXTURE = REPO_ROOT / "conformance" / "gates" / "arrow_copy_fixture.json"


def test_arrow_load_reports_measured_copy():
    payload = load_json(FIXTURE)
    names = list(payload["column_names"])
    columns = [np.asarray(payload["columns"][name], dtype=np.float64) for name in names]
    info = antecedent.data.load_float64_columns(names, columns)
    assert info.row_count == payload["row_count"]
    assert info.column_count == len(names)
    assert info.column_names == names
    assert info.bytes_copied > 0
    assert info.diagnostic_count > 0


def test_numpy_ingestion_copies_each_column_exactly_once():
    rows = 1000
    columns = [np.arange(rows, dtype=np.float64) + k for k in range(3)]
    info = antecedent.data.load_float64_columns(["a", "b", "c"], columns)
    # One copy per column into the table's own buffer: no Arrow batch in between.
    assert info.bytes_copied == 3 * rows * 8
    assert info.bytes_borrowed == 0
    assert info.diagnostic_count == 3
    assert info.row_count == rows
    assert info.column_count == 3


def test_a_strided_numpy_column_is_gathered_in_the_same_single_pass():
    rows = 1000
    wide = np.arange(2 * rows, dtype=np.float64)
    strided = wide[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    info = antecedent.data.load_float64_columns(["a"], [strided])
    assert info.bytes_copied == rows * 8
    assert info.diagnostic_count == 1


# --- Copy counts per container type -------------------------------------------------------
#
# ``ingestion_receipt`` runs the analysis ingestion and reports every allocation it made: the
# Python conversions (``python_copies``) and the native reader's own account (``native``).

ROWS = 1000
COLUMN_BYTES = ROWS * 8


def _kinds(receipt) -> list[str]:
    return sorted(copy.kind for copy in receipt.python_copies)


def test_numpy_float64_columns_cost_one_ownership_copy_and_no_python_copy():
    data = {"a": np.arange(ROWS, dtype=np.float64), "b": np.arange(ROWS, dtype=np.float64) + 1}
    receipt = antecedent.data.ingestion_receipt(data)
    assert receipt.python_copy_count == 0
    assert receipt.ownership_bytes_copied == 2 * COLUMN_BYTES
    assert receipt.bytes_borrowed == 0
    assert receipt.total_bytes_copied == 2 * COLUMN_BYTES


def test_numpy_integer_column_is_widened_while_it_is_copied_once():
    receipt = antecedent.data.ingestion_receipt({"a": np.arange(ROWS, dtype=np.int64)})
    # No float64 temporary in Python: the native reader widens as it makes its one copy.
    assert receipt.python_copy_count == 0
    assert receipt.ownership_bytes_copied == COLUMN_BYTES
    assert receipt.total_bytes_copied == COLUMN_BYTES


def test_boolean_unsigned_and_float32_columns_are_widened_in_the_same_single_pass():
    columns = {
        "flag": np.arange(ROWS) % 2 == 0,
        "small": np.arange(ROWS, dtype=np.uint8),
        "single": np.arange(ROWS, dtype=np.float32),
    }
    receipt = antecedent.data.ingestion_receipt(columns)
    assert receipt.python_copy_count == 0
    assert receipt.ownership_bytes_copied == 3 * COLUMN_BYTES
    # The widened values are the exact ones.
    info = antecedent.data.load_float64_columns(list(columns), list(columns.values()))
    assert info.bytes_copied == 3 * COLUMN_BYTES and info.row_count == ROWS


def test_an_integer_beyond_2_53_is_refused_by_the_native_widening_too():
    big = np.array([1, 2**60], dtype=np.int64)
    with pytest.raises(Exception, match="2\\*\\*53"):
        antecedent.data.load_float64_columns(["a"], [big])
    with pytest.raises(Exception, match="beyond 2"):
        antecedent.data.ingestion_receipt({"a": big})


def test_a_strided_numpy_column_has_no_python_copy_and_one_gathering_copy():
    strided = np.arange(2 * ROWS, dtype=np.float64)[::2]
    receipt = antecedent.data.ingestion_receipt({"a": strided})
    assert receipt.python_copy_count == 0
    assert receipt.ownership_bytes_copied == COLUMN_BYTES


def test_object_columns_are_cast_in_one_allocation_unless_values_are_missing():
    clean = np.array([1.0, 2.0, 3.5] * 10, dtype=object)
    with_none = np.array([1.0, None, 3.5] * 10, dtype=object)
    names, columns, copies = antecedent._data.ingest_columns_with_receipt(
        {"clean": clean, "gap": with_none}
    )
    by_column = {name: [c for c in copies if c.column == name] for name in names}
    # The float64 result is the one allocation a clean column needs; a column with missing
    # entries also copies the input once so the caller's array is not written.
    assert [c.kind for c in by_column["clean"]] == ["object_cast"]
    assert [c.kind for c in by_column["gap"]] == ["object_cast", "object_cast"]
    assert np.isnan(columns[1][1]) and columns[1][0] == 1.0
    assert clean.dtype == object and with_none[1] is None


def test_pandas_numpy_backed_frame_is_read_as_views():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {"a": np.arange(ROWS, dtype=np.float64), "b": np.arange(ROWS, dtype=np.float64)}
    )
    ledger = antecedent._data._CopyLedger()
    _, columns = antecedent._data.as_columns(frame, receipt=ledger)
    assert ledger.copies == []
    assert all(
        np.shares_memory(column, frame[name].to_numpy())
        for name, column in zip("ab", columns, strict=True)
    )


def test_pandas_nullable_integer_is_filled_into_float64_once():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": pd.array([1, None, 3] * 100, dtype="Int64")})
    ledger = antecedent._data._CopyLedger()
    _, columns = antecedent._data.as_columns(frame, receipt=ledger)
    # One float64 allocation; the object-array detour (a Python object per row, then a
    # second cast) is not taken.
    assert [copy.kind for copy in ledger.copies] == ["extension_to_float64"]
    assert ledger.copies[0].nbytes == 300 * 8
    assert np.isnan(columns[0][1]) and columns[0][0] == 1.0 and columns[0][2] == 3.0


def test_pandas_nullable_integer_beyond_2_53_is_still_refused():
    pd = pytest.importorskip("pandas")
    from antecedent.errors import CausalTypeError

    frame = pd.DataFrame({"a": pd.array([2**60, 1], dtype="Int64")})
    with pytest.raises(CausalTypeError, match="beyond 2"):
        antecedent._data.as_columns(frame)


def test_arrow_float64_table_is_borrowed_with_no_copy_at_all():
    pa = pytest.importorskip("pyarrow")
    table = pa.table({"a": pa.array(np.arange(ROWS, dtype=np.float64))})
    receipt = antecedent.data.ingestion_receipt(table)
    assert receipt.python_copy_count == 0
    assert receipt.ownership_bytes_copied == 0
    assert receipt.bytes_borrowed == COLUMN_BYTES


def test_arrow_integer_column_is_cast_once_then_borrowed():
    pa = pytest.importorskip("pyarrow")
    table = pa.table({"a": pa.array(np.arange(ROWS, dtype=np.int64))})
    receipt = antecedent.data.ingestion_receipt(table)
    assert _kinds(receipt) == ["arrow_cast"]
    assert receipt.python_bytes_copied == COLUMN_BYTES
    assert receipt.bytes_borrowed == COLUMN_BYTES
    assert receipt.ownership_bytes_copied == 0


def test_arrow_chunks_are_joined_once_by_the_native_loader_not_in_python():
    pa = pytest.importorskip("pyarrow")
    chunk = pa.array(np.arange(ROWS // 2, dtype=np.float64))
    chunked = pa.table({"a": pa.chunked_array([chunk, chunk])})
    receipt = antecedent.data.ingestion_receipt(chunked)
    # Python hands the chunks over as they are; the loader makes the table's single owned copy.
    assert receipt.python_copy_count == 0
    # The join writes the values once and the column's validity bitmap (one bit per row).
    assert receipt.ownership_bytes_copied == COLUMN_BYTES + (ROWS + 7) // 8
    assert receipt.bytes_borrowed == 0
    single = pa.table({"a": pa.chunked_array([pa.array(np.arange(ROWS, dtype=np.float64))])})
    one = antecedent.data.ingestion_receipt(single)
    assert one.python_copy_count == 0 and one.bytes_borrowed == COLUMN_BYTES


def test_columns_chunked_differently_are_joined_in_python_and_recorded():
    pa = pytest.importorskip("pyarrow")
    half = pa.array(np.arange(ROWS // 2, dtype=np.float64))
    whole = pa.array(np.arange(ROWS, dtype=np.float64))
    mixed = pa.table({"a": pa.chunked_array([half, half]), "b": pa.chunked_array([whole])})
    receipt = antecedent.data.ingestion_receipt(mixed)
    assert _kinds(receipt) == ["chunk_concatenation"]
    assert receipt.python_bytes_copied == COLUMN_BYTES
    # Agreeing layouts stay chunked: two columns of the same two chunks cost no Python copy.
    same = pa.table({"a": pa.chunked_array([half, half]), "b": pa.chunked_array([half, half])})
    agreeing = antecedent.data.ingestion_receipt(same)
    assert agreeing.python_copy_count == 0
    assert agreeing.ownership_bytes_copied == 2 * (COLUMN_BYTES + (ROWS + 7) // 8)


def test_arrow_nulls_are_filled_into_a_new_buffer_only_when_numpy_is_read():
    pa = pytest.importorskip("pyarrow")
    table = pa.table({"a": pa.array([1.0, None, 3.0] * 100)})
    ledger = antecedent._data._CopyLedger()
    _, columns = antecedent._data.as_columns(table, receipt=ledger)
    assert [copy.kind for copy in ledger.copies] == ["arrow_to_numpy"]
    assert np.isnan(columns[0][1])
    clean = pa.table({"a": pa.array(np.arange(ROWS, dtype=np.float64))})
    quiet = antecedent._data._CopyLedger()
    antecedent._data.as_columns(clean, receipt=quiet)
    assert quiet.copies == []


def test_a_multi_chunk_table_keeps_its_own_chunks_instead_of_reslicing():
    pa = pytest.importorskip("pyarrow")
    chunk = pa.array(np.arange(100, dtype=np.float64))
    # Three producer chunks, none near the engine's batch size: re-slicing and joining them
    # would add a copy, so the table's own chunks are used.
    table = pa.table({"a": pa.chunked_array([chunk, chunk, chunk])})
    ledger = antecedent._data._CopyLedger()
    names, columns = antecedent._data.try_as_arrow_c_columns(table, receipt=ledger)
    assert names == ["a"]
    assert ledger.copies == []
    assert len(columns[0].chunks) == 3


def test_polars_frame_is_taken_through_arrow_without_resliced_copies():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"a": np.arange(ROWS, dtype=np.float64)})
    receipt = antecedent.data.ingestion_receipt(frame)
    assert receipt.python_copy_count == 0
    assert receipt.ownership_bytes_copied + receipt.bytes_borrowed == COLUMN_BYTES


def test_panel_columns_are_copied_once_into_per_unit_windows():
    lengths = [300, 500, 200]
    columns = [np.arange(ROWS, dtype=np.float64) + k for k in range(3)]
    info = antecedent.data.load_float64_panel_columns(["a", "b", "c"], columns, lengths)
    # Every element is copied once, straight from the pooled column into its unit's window:
    # not once into a pooled batch and again into each window.
    assert info.bytes_copied == 3 * COLUMN_BYTES
    assert info.diagnostic_count == 3 * len(lengths)
    assert info.row_count == ROWS
    assert info.bytes_borrowed == 0


def test_a_strided_panel_column_is_gathered_per_unit_in_the_same_pass():
    wide = np.arange(2 * ROWS, dtype=np.float64)
    info = antecedent.data.load_float64_panel_columns(["a"], [wide[::2]], [400, 600])
    assert info.bytes_copied == COLUMN_BYTES
    assert info.diagnostic_count == 2


def test_panel_unit_lengths_must_cover_the_pooled_rows():
    columns = [np.arange(ROWS, dtype=np.float64)]
    for lengths in ([], [0, ROWS], [ROWS - 1], [ROWS, 1]):
        with pytest.raises(Exception, match="unit_lengths"):
            antecedent.data.load_float64_panel_columns(["a"], columns, lengths)
