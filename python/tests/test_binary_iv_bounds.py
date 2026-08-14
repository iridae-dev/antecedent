"""Python surface for sharp binary-IV Balke–Pearl ATE bounds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from antecedent.identify import binary_iv_bounds


def test_binary_iv_bounds_matches_frozen_bpbounds_table1() -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "conformance"
            / "response"
            / "binary_iv_balke_pearl_table1"
            / "expected.json"
        ).read_text(encoding="utf-8")
    )
    cells = fixture["cells"]
    expected = fixture["expected"]
    atol = fixture["tolerance"]["atol"]

    bounds = binary_iv_bounds(cells)

    assert bounds.method == "identify.binary_iv_bounds"
    assert bounds.lower == pytest.approx(expected["lower"], abs=atol)
    assert bounds.upper == pytest.approx(expected["upper"], abs=atol)
