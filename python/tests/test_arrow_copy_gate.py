"""gate: Python loads the shared Arrow/float fixture with measured copy.

Fixture: conformance/gates/arrow_copy_fixture.json (shared with Rust).
"""

from __future__ import annotations

import antecedent
import numpy as np

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
