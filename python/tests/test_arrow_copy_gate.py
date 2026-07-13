"""Phase 0 gate: Python loads the same Arrow/float fixture with measured copy.

Values match crates/causal-data/tests/arrow_copy_gate.rs.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("causal")
import causal


def test_arrow_load_reports_measured_copy():
    names = ["t", "y", "z"]
    columns = [
        np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64),
        np.array([1.0, 3.0, 1.5, 3.5, 2.0, 4.0, 2.5, 4.5], dtype=np.float64),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64),
    ]
    info = causal.load_float64_columns(names, columns)
    assert info.row_count == 8
    assert info.column_count == 3
    assert info.bytes_copied > 0
    assert info.diagnostic_count > 0
