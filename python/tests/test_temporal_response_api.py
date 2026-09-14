"""Python facade known-truth pin for temporal ResponseCurve / InterventionResponse."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from _repo_text import load_json

pytest.importorskip("antecedent")
import antecedent
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.estimation import PreparedAnalysis
from antecedent.intervention import Sequence, Set, Soft

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = load_json(_ROOT / "conformance" / "response" / "temporal_dose_horizon" / "expected.json")


def _fixture_data() -> dict[str, np.ndarray]:
    n = int(_FIXTURE["generation"]["n"])
    t = np.array([0.0 if i % 4 in (0, 2) else (1.0 if i % 4 == 1 else -1.0) for i in range(n)])
    y = np.zeros(n)
    for i in range(n):
        y[i] = 1.0 + 2.0 * (t[i - 1] if i >= 1 else 0.0) + 3.0 * (t[i - 2] if i >= 2 else 0.0)
    return {"t": t, "y": y}


_EDGES = [("t", 1, "y", 0), ("t", 2, "y", 0)]
_ATOL = float(_FIXTURE["tolerance"]["atol"])
_MEAN = np.asarray(_FIXTURE["contract"]["surface"]["mean"], dtype=float)
_SET_PATH = np.asarray(_FIXTURE["contract"]["intervention_paths"]["set_1"], dtype=float)
_SOFT_CONST = np.asarray(_FIXTURE["contract"]["intervention_paths"]["soft_constant_1"], dtype=float)


def _response_means(result: Any) -> np.ndarray:
    assert result.response is not None
    return np.asarray([row[0] for row in result.response.values], dtype=float)


def test_temporal_response_curve_matches_fixture_and_prepared_reuse():
    data = _fixture_data()
    query = antecedent.ResponseCurve(
        "t",
        "y",
        grid=[0.0, 1.0],
        horizons=[1, 2],
        policy="pulse",
        treatment_lag=1,
    )
    direct = antecedent.analyze(data, graph=_EDGES, query=query, refute=False, bootstrap=0, seed=21)
    np.testing.assert_allclose(_response_means(direct), _MEAN, atol=_ATOL)
    assert direct.support.status == "supported"
    assert list(direct.support.point_status) == ["supported"] * 4

    prepared = PreparedAnalysis.prepare(data, graph=_EDGES, query=query, refute=False, seed=21)
    click = prepared.estimate(data, seed=21)
    np.testing.assert_allclose(_response_means(click), _MEAN, atol=_ATOL)
    assert prepared.structure_source == "explicit"


def test_temporal_intervention_set_and_single_step_sequence_match_fixture():
    data = _fixture_data()
    set_query = antecedent.InterventionResponse(
        "y",
        intervention=Set("t", 1.0),
        horizons=[1, 2],
        policy="pulse",
        treatment_lag=1,
    )
    seq_query = antecedent.InterventionResponse(
        "y",
        intervention=Sequence([Set("t", 1.0)]),
        horizons=[1, 2],
        policy="pulse",
        treatment_lag=1,
    )
    soft_query = antecedent.InterventionResponse(
        "y",
        intervention=Soft("t", "constant", parameters=[1.0]),
        horizons=[1, 2],
        policy="pulse",
        treatment_lag=1,
    )
    set_result = antecedent.analyze(
        data, graph=_EDGES, query=set_query, refute=False, bootstrap=0, seed=22
    )
    seq_result = antecedent.analyze(
        data, graph=_EDGES, query=seq_query, refute=False, bootstrap=0, seed=22
    )
    soft_result = antecedent.analyze(
        data, graph=_EDGES, query=soft_query, refute=False, bootstrap=0, seed=22
    )
    np.testing.assert_allclose(_response_means(set_result), _SET_PATH, atol=_ATOL)
    np.testing.assert_allclose(_response_means(seq_result), _SET_PATH, atol=_ATOL)
    np.testing.assert_allclose(_response_means(soft_result), _SOFT_CONST, atol=_ATOL)

    prepared = PreparedAnalysis.prepare(data, graph=_EDGES, query=seq_query, refute=False, seed=22)
    click = prepared.estimate(data, seed=22)
    np.testing.assert_allclose(_response_means(click), _SET_PATH, atol=_ATOL)


def test_multi_step_sequence_matches_two_step_truth_and_does_not_collapse():
    data = _fixture_data()
    two_step = np.asarray(
        _FIXTURE["contract"]["intervention_paths"]["sequence_two_step_set_1"], dtype=float
    )
    last_step = _SET_PATH
    np.testing.assert_raises(AssertionError, np.testing.assert_allclose, two_step, last_step)
    query = antecedent.InterventionResponse(
        "y",
        intervention=Sequence([Set("t", 1.0), Set("t", 1.0)]),
        horizons=[1, 2],
        policy="pulse",
        treatment_lag=1,
    )
    result = antecedent.analyze(data, graph=_EDGES, query=query, refute=False, bootstrap=0, seed=22)
    np.testing.assert_allclose(_response_means(result), two_step, atol=_ATOL)
    np.testing.assert_raises(
        AssertionError, np.testing.assert_allclose, _response_means(result), last_step
    )
    prepared = PreparedAnalysis.prepare(data, graph=_EDGES, query=query, refute=False, seed=22)
    click = prepared.estimate(data, seed=22)
    np.testing.assert_allclose(_response_means(click), two_step, atol=_ATOL)


def test_nested_sequence_refuses():
    data = _fixture_data()
    query = antecedent.InterventionResponse(
        "y",
        intervention=Sequence([Sequence([Set("t", 1.0)])]),
        horizons=[1],
        policy="pulse",
        treatment_lag=1,
    )
    with pytest.raises(CausalUnsupportedError, match="nested Sequence"):
        antecedent.analyze(data, graph=_EDGES, query=query, refute=False, bootstrap=0)


def test_pulse_projection_matches_surface_contrast():
    data = _fixture_data()
    surface = antecedent.analyze(
        data,
        graph=_EDGES,
        query=antecedent.ResponseCurve(
            "t",
            "y",
            grid=[0.0, 1.0],
            horizons=[1, 2],
            policy="pulse",
            treatment_lag=1,
        ),
        refute=False,
        bootstrap=0,
        seed=23,
    )
    pulse = antecedent.analyze(
        data,
        graph=_EDGES,
        query=antecedent.PulseEffect("t", "y", treatment_lag=1, horizon_steps=1),
        refute=False,
        bootstrap=0,
        seed=23,
    )
    means = _response_means(surface)
    # dose-major: mean[dose=1,h=1] - mean[dose=0,h=1] == index 2 - index 0
    contrast = float(means[2] - means[0])
    expected = float(_FIXTURE["contract"]["pulse_effect_projection"]["contrast"])
    assert contrast == pytest.approx(expected, abs=_ATOL)
    assert pulse.ate == pytest.approx(expected, abs=_ATOL)


_SURFACE_QUERY = antecedent.ResponseCurve(
    "t",
    "y",
    grid=[0.0, 1.0],
    horizons=[1, 2],
    policy="pulse",
    treatment_lag=1,
)
_BAND_WITHHELD = "estimate.temporal_response.band_withheld"


def _column(rows: Any) -> np.ndarray:
    return np.asarray([row[0] for row in rows], dtype=float)


def _assert_band_withheld(result: Any) -> None:
    """Zero replicates: point surface only, with the withheld band diagnosed."""
    np.testing.assert_allclose(_response_means(result), _MEAN, atol=_ATOL)
    assert result.uncertainty.kind == "none"
    assert result.uncertainty.lower is None and result.uncertainty.upper is None
    assert result.simultaneous_band is None
    assert any("no pointwise or simultaneous band" in w for w in result.support.warnings)
    assert any(d.startswith(f"{_BAND_WITHHELD}: ") for d in result.diagnostics)


def _assert_block_bands(result: Any, replicates: int) -> None:
    """Replicates: positive-width pointwise band inside the simultaneous band."""
    means = _response_means(result)
    np.testing.assert_allclose(means, _MEAN, atol=_ATOL)
    assert result.uncertainty.kind == "pointwise"
    assert result.uncertainty.level == pytest.approx(0.95)
    assert result.uncertainty.lower is not None and result.uncertainty.upper is not None
    lower = _column(result.uncertainty.lower)
    upper = _column(result.uncertainty.upper)
    # Strictly positive at every cell, including dose zero at horizon 1 (the old
    # zero-width-at-dose-0 regression guard).
    assert np.all(upper - lower > 0.0)
    assert np.all(lower <= means) and np.all(means <= upper)
    band = result.simultaneous_band
    assert band is not None
    assert band.level == pytest.approx(0.95)
    assert band.replicates == replicates
    assert band.critical >= 1.959
    assert len(band) == len(means)
    sim_lower = _column(band.lower)
    sim_upper = _column(band.upper)
    assert np.all(sim_lower <= lower) and np.all(upper <= sim_upper)
    assert not any(d.startswith(f"{_BAND_WITHHELD}: ") for d in result.diagnostics)


def test_temporal_response_zero_bootstrap_withholds_band():
    """The pre-1.9 analytic band in the fixture treats lag-aligned rows as independent.

    ``surface.lower`` / ``surface.upper`` in the fixture record that retired band;
    with ``bootstrap=0`` the surface keeps its point values and publishes no band.
    """
    data = _fixture_data()
    result = antecedent.analyze(
        data, graph=_EDGES, query=_SURFACE_QUERY, refute=False, bootstrap=0, seed=21
    )
    _assert_band_withheld(result)
    assert result.identification.method == "temporal.backdoor.unfolded"
    assert "identify.temporal_backdoor" in result.provenance["operation_ids"]
    assert result.validation is not None
    assert any(check.id == "refute.temporal_response.skipped" for check in result.validation.checks)

    prepared = PreparedAnalysis.prepare(
        data, graph=_EDGES, query=_SURFACE_QUERY, refute=False, seed=21, bootstrap=0
    )
    _assert_band_withheld(prepared.estimate(data, seed=21))


@pytest.mark.parametrize("bootstrap", [None, 60])
def test_temporal_response_bootstrap_publishes_block_bands(bootstrap: int | None):
    data = _fixture_data()
    replicates = 199 if bootstrap is None else bootstrap
    kwargs: dict[str, Any] = {} if bootstrap is None else {"bootstrap": bootstrap}
    direct = antecedent.analyze(
        data, graph=_EDGES, query=_SURFACE_QUERY, refute=False, seed=21, **kwargs
    )
    _assert_block_bands(direct, replicates)
    again = antecedent.analyze(
        data, graph=_EDGES, query=_SURFACE_QUERY, refute=False, seed=21, **kwargs
    )
    assert again.uncertainty.lower == direct.uncertainty.lower
    assert again.uncertainty.upper == direct.uncertainty.upper

    if bootstrap is None:
        # Prepared temporal responses follow the latency tier like Pulse /
        # Sustained: the default interactive tier publishes no band, standard
        # runs 199 replicates, the same count as the Study default.
        interactive = PreparedAnalysis.prepare(
            data, graph=_EDGES, query=_SURFACE_QUERY, refute=False, seed=21
        )
        _assert_band_withheld(interactive.estimate(data, seed=21))
        kwargs = {"latency": "standard"}
    prepared = PreparedAnalysis.prepare(
        data, graph=_EDGES, query=_SURFACE_QUERY, refute=False, seed=21, **kwargs
    )
    click = prepared.estimate(data, seed=21)
    _assert_block_bands(click, replicates)


def test_temporal_intervention_path_bootstrap_publishes_block_bands():
    data = _fixture_data()
    query = antecedent.InterventionResponse(
        "y", intervention=Set("t", 1.0), horizons=[1, 2], policy="pulse", treatment_lag=1
    )
    withheld = antecedent.analyze(data, graph=_EDGES, query=query, refute=False, bootstrap=0)
    assert withheld.uncertainty.kind == "none"
    assert any(d.startswith(f"{_BAND_WITHHELD}: ") for d in withheld.diagnostics)
    banded = antecedent.analyze(data, graph=_EDGES, query=query, refute=False, bootstrap=60)
    np.testing.assert_allclose(_response_means(banded), _SET_PATH, atol=_ATOL)
    assert banded.uncertainty.kind == "pointwise"
    assert banded.simultaneous_band is not None
    assert banded.simultaneous_band.replicates == 60


def test_temporal_default_lag_matches_pulse():
    curve = antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1])
    path = antecedent.InterventionResponse("y", intervention=Set("t", 1.0), horizons=[1])
    assert curve.treatment_lag == 1
    assert path.treatment_lag == 1
    assert antecedent.PulseEffect("t", "y").treatment_lag == 1


def test_temporal_horizons_reject_bool_and_dynamic_policy():
    with pytest.raises(CausalValueError, match="positive integers"):
        antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[True])
    with pytest.raises(CausalValueError, match="pulse"):
        antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1], policy="dynamic")


def test_prepared_temporal_rejects_wrong_identifier_and_refute_click():
    data = _fixture_data()
    query = antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1], policy="pulse")
    with pytest.raises(CausalValueError, match="temporal.backdoor.unfolded"):
        PreparedAnalysis.prepare(data, graph=_EDGES, query=query, identifier="response.backdoor")
    prepared = PreparedAnalysis.prepare(data, graph=_EDGES, query=query, refute=False)
    with pytest.raises(CausalUnsupportedError, match="ResponseCurve cheap/full/placebo"):
        prepared.refute(data)
