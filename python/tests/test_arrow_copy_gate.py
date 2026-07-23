"""gate: Python loads the shared Arrow/float fixture with measured copy.

Fixture: conformance/gates/arrow_copy_fixture.json (shared with Rust).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent

FIXTURE = (
    Path(__file__).resolve().parents[2] / "conformance" / "gates" / "arrow_copy_fixture.json"
)


def test_arrow_load_reports_measured_copy():
    payload = json.loads(FIXTURE.read_text())
    names = list(payload["column_names"])
    columns = [np.asarray(payload["columns"][name], dtype=np.float64) for name in names]
    info = antecedent.load_float64_columns(names, columns)
    assert info.row_count == payload["row_count"]
    assert info.column_count == len(names)
    assert info.column_names == names
    assert info.bytes_copied > 0
    assert info.diagnostic_count > 0
