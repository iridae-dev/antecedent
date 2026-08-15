"""Python surface for sharp binary-IV Balke–Pearl ATE bounds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from antecedent.errors import CausalIdentifyError
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


def test_binary_iv_bounds_names_the_instrumental_inequality_on_violation() -> None:
    """A law that violates the Balke-Pearl instrumental inequality must raise a distinct,
    named error rather than a generic failure indistinguishable from malformed input.

    Z=0 forces every response type in the population to have (d0=0, y0=1); Z=1 forces every
    response type to have (d1=0, y0=0). Since exogeneity requires one population-wide
    response-type distribution to explain both arms, and y0 cannot be both 1 and 0, no
    distribution over the 16 latent types can reproduce this law (hand-verified violation,
    mirrored from the Rust unit test in antecedent-identify's bounds.rs).
    """
    cells = [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]

    with pytest.raises(CausalIdentifyError, match="instrumental inequality"):
        binary_iv_bounds(cells)
