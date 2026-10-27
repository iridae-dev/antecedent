"""Composition across horizons: coupled draws compose, independent marginals do not."""

from __future__ import annotations

import numpy as np
import pytest
from antecedent.horizon_composition import compose_horizons


def _coupled(n: int = 200):
    x = np.linspace(-8.0, 8.0, n)
    return [x, 2.0 * x + 1.0]


def test_a_contrast_of_coupled_horizons_has_the_spread_of_the_realizations():
    columns = _coupled()
    composed = compose_horizons(
        [1, 2], columns, evidence_id="series", alignment_id="replicates:200",
        operation="contrast", coefficients=[-1.0, 1.0],
    )  # fmt: skip
    # h2 - h1 = x + 1 on every row: the spread of x, not sqrt(var(x) + var(2x + 1)).
    assert composed.variance == pytest.approx(np.var(columns[0], ddof=1), rel=1e-9)
    assert composed.coordinate.endswith("|horizons=1,2")
    assert composed.label == "uncalibrated_spread"
    assert composed.route == "aligned_joint_law"


def test_a_nonlinear_map_composes_the_same_joint_draws():
    columns = _coupled()
    composed = compose_horizons(
        [1, 2], columns, evidence_id="series", alignment_id="replicates:200",
        operation="nonlinear_functional", map=lambda values: max(values),
    )  # fmt: skip
    expected = np.maximum(columns[0], columns[1])
    assert np.allclose(composed.draws, expected)


def test_a_malformed_declaration_is_refused():
    columns = _coupled()
    with pytest.raises(
        ValueError, match="give coefficients for a linear composition or map for a nonlinear one"
    ):
        compose_horizons(
            [1, 2], columns, evidence_id="e", alignment_id="a", operation="sum",
        )  # fmt: skip
    with pytest.raises(Exception, match="horizon"):
        compose_horizons(
            [1, 1], columns, evidence_id="e", alignment_id="a", operation="sum",
            coefficients=[1.0, 1.0],
        )  # fmt: skip


def test_a_wrong_coefficient_count_is_refused_and_no_composed_quantity_is_returned():
    columns = _coupled()
    with pytest.raises(Exception, match="give one coefficient per horizon"):
        compose_horizons(
            [1, 2], columns, evidence_id="series", alignment_id="replicates:200",
            operation="weighted_sum", coefficients=[1.0],
        )  # fmt: skip


def test_a_map_over_a_linear_operation_or_with_a_non_finite_value_is_refused():
    columns = _coupled()
    # A linear combination goes through coefficients, never an arbitrary map.
    with pytest.raises(Exception, match="compose a linear combination of horizons"):
        compose_horizons(
            [1, 2], columns, evidence_id="series", alignment_id="replicates:200",
            operation="sum", map=lambda values: values[0],
        )  # fmt: skip
    # A map that returns a non-finite value for a realization is refused, not averaged.
    with pytest.raises(Exception, match="finite map"):
        compose_horizons(
            [1, 2], columns, evidence_id="series", alignment_id="replicates:200",
            operation="nonlinear_functional", map=lambda values: float("nan"),
        )  # fmt: skip
    peak = compose_horizons(
        [1, 2], columns, evidence_id="series", alignment_id="replicates:200",
        operation="nonlinear_functional", map=lambda values: max(values),
    )  # fmt: skip
    assert peak.label == "uncalibrated_spread"
    assert peak.route == "aligned_joint_law"
    assert peak.coordinate.endswith("|horizons=1,2")
