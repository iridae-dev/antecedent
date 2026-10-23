"""Bias-aware dose-grid intervals and band through the Python facade.

The oracle is the quadratic response ``m(d) = 1 + 0.5 d + 0.25 d**2`` (so ``m''' = 0``); an
alternating deterministic perturbation gives non-degenerate standard errors. A declared bound on
``|m'''|`` widens each interval by the worst-case smoothing bias and, on a grid, adds a sup-t band
replayed from its seed by a fresh consumer. Coverage is deliberately NOT measured here.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent.dose_grid import (
    DoseBiasAware,
    DoseClaims,
    DoseGridRefusal,
    consume_dose_grid_artifact,
    dose_functional,
)

N = 201
GRID = [0.5, 1.0, 2.0, 3.5]


def truth_level(d):
    return 1.0 + 0.5 * d + 0.25 * d * d


def rows():
    dose = np.linspace(0.0, 4.0, N)
    sign = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    return dose, truth_level(dose) + 0.1 * sign


def run(functional="level", **kwargs):
    dose, outcome = rows()
    kwargs.setdefault("bandwidth", 0.4)
    kwargs.setdefault("bandwidth_range", (0.1, 1.0))
    if functional == "contrast":
        kwargs.setdefault("contrast", (1.0, 3.0))
    else:
        kwargs.setdefault("grid", GRID)
    return dose_functional(dose, outcome, functional=functional, **kwargs)


def test_without_a_bound_nothing_bias_aware_is_reported() -> None:
    result = run()
    assert result.bias_aware is None
    assert result.simultaneous_band == "closed"
    assert result.to_dict()["bias_aware"] is None


def test_a_declared_bound_widens_intervals_and_carries_a_replayable_band() -> None:
    claims = DoseClaims(pointwise_level=True, simultaneous_band=True)
    result = run(claims=claims, curvature_bound=0.2, band_replicates=500, band_seed=7)
    section = result.bias_aware
    assert isinstance(section, DoseBiasAware)
    assert result.simultaneous_band == "bias_aware"
    assert section.curvature_bound == pytest.approx(0.2)
    assert section.band_replicates == 500
    assert section.band_critical_value is not None and section.band_critical_value > 0.0
    assert section.band_lower is not None and section.band_upper is not None
    for j, point in enumerate(result.levels):
        assert section.smoothing_bias_bound[j] > 0.0
        assert section.lower[j] <= point.lower and point.upper <= section.upper[j]
        assert section.band_lower[j] <= section.lower[j]
        assert section.upper[j] <= section.band_upper[j]
        assert section.band_lower[j] <= truth_level(point.dose) <= section.band_upper[j]
    assert "calibration unmeasured" in result.explain()
    assert result.to_dict()["bias_aware"]["band_replicates"] == 500

    fresh = consume_dose_grid_artifact(result.export())
    assert fresh.bias_aware == section
    assert fresh.simultaneous_band == "bias_aware"


def test_a_contrast_under_a_bound_has_bias_aware_interval_and_no_band() -> None:
    result = run("contrast", curvature_bound=0.2)
    section = result.bias_aware
    assert section is not None and len(section.lower) == 1
    assert section.band_lower is None and section.band_critical_value is None
    assert result.simultaneous_band == "closed"
    assert section.lower[0] <= result.contrast.lower <= result.contrast.upper <= section.upper[0]


def test_a_band_claim_on_a_contrast_refuses_closed() -> None:
    claims = DoseClaims(pointwise_level=True, simultaneous_band=True)
    with pytest.raises(DoseGridRefusal) as info:
        run("contrast", claims=claims, curvature_bound=0.2)
    assert info.value.detail == "dose_grid.simultaneous_band_closed"


def test_the_quadratic_premise_and_a_bound_are_separate_rows() -> None:
    from antecedent.dose_grid import _dose_functional

    dose, outcome = rows()
    with pytest.raises(DoseGridRefusal):
        _dose_functional(
            dose,
            outcome,
            bandwidth=0.4,
            bandwidth_range=(0.1, 1.0),
            grid=GRID,
            quadratic_mean=True,
            curvature_bound=0.2,
        )


def test_band_arguments_are_type_checked() -> None:
    with pytest.raises(TypeError, match="band_replicates must be an int"):
        run(curvature_bound=0.2, band_replicates=True)
