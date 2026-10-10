"""2.3 B1 dose-grid functional row through the Python facade.

The oracle is the closed-form quadratic response ``m(d) = 1 + 0.5 d + 0.25 d**2``, so
``m'(d) = 0.5 + 0.5 d`` and ``m(3) - m(1) = 4.75 - 1.75 = 3``. A Gaussian-kernel local quadratic
reproduces a quadratic exactly on noise-free rows; an alternating deterministic perturbation of
the outcome gives non-degenerate standard errors. Calibration (coverage of any interval) is
deliberately NOT measured here, and every result says so.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent.dose_grid import (
    DoseClaims,
    DoseGridRefusal,
    consume_dose_grid_artifact,
    dose_functional,
    dose_support_table,
)
from antecedent.errors import CausalSerializationError, CausalUnsupportedError

from _refusal import assert_registered_refusal

N = 201
GRID = [0.5, 1.0, 2.0, 3.5]


def truth_level(d):
    return 1.0 + 0.5 * d + 0.25 * d * d


def truth_derivative(d):
    return 0.5 + 0.5 * d


def design(noise: float = 0.0):
    dose = np.linspace(0.0, 4.0, N)
    sign = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    return dose, truth_level(dose) + noise * sign


def run(functional="level", noise=0.0, **kwargs):
    dose, outcome = design(noise)
    kwargs.setdefault("bandwidth", 0.4)
    kwargs.setdefault("bandwidth_range", (0.1, 1.0))
    kwargs.setdefault("minimum_local_ess", 10.0)
    if functional == "contrast":
        kwargs.setdefault("contrast", (1.0, 3.0))
    else:
        kwargs.setdefault("grid", GRID)
    return dose_functional(dose, outcome, functional=functional, **kwargs)


def refused(call, detail: str, code: str) -> DoseGridRefusal:
    with pytest.raises(DoseGridRefusal) as info:
        call()
    error = info.value
    assert isinstance(error, CausalUnsupportedError)
    assert error.detail == detail
    assert error.reason_code == code
    assert_registered_refusal(error)
    return error


def test_b1_level_recovers_closed_form_with_unmeasured_calibration() -> None:
    result = run("level")
    expected = [1.3125, 1.75, 3.0, 5.8125]
    assert [p.dose for p in result.levels] == GRID
    for point, truth in zip(result.levels, expected, strict=True):
        assert point.value == pytest.approx(truth, abs=1e-6)
        assert truth == pytest.approx(truth_level(point.dose), abs=1e-12)
        assert point.support == "supported"
        assert point.calibration == "unmeasured"
        assert not point.smoothing_bias_included
    assert result.derivatives == () and result.contrast is None
    assert result.functional == "level"
    assert result.design == "randomized_dose"
    assert result.calibration == "unmeasured"
    assert result.inference_claim == "pointwise_calibration_unmeasured"
    assert result.claims == DoseClaims(pointwise_level=True)
    assert result.max_abs_influence_sum < 1e-9
    assert result.to_dict()["calibration"] == "unmeasured"


def test_b1_derivative_recovers_closed_form() -> None:
    result = run("derivative")
    expected = [0.75, 1.0, 1.5, 2.25]
    for point, truth in zip(result.derivatives, expected, strict=True):
        assert point.value == pytest.approx(truth, abs=1e-5)
        assert truth == pytest.approx(truth_derivative(point.dose), abs=1e-12)
    assert result.levels == () and result.contrast is None
    assert result.claims == DoseClaims(derivative=True)


def test_b1_noisy_rows_give_finite_pointwise_intervals_without_a_coverage_claim() -> None:
    result = run("level", noise=0.1)
    for point in result.levels:
        assert point.value == pytest.approx(truth_level(point.dose), abs=0.05)
        assert np.isfinite(point.standard_error) and point.standard_error > 0.0
        assert point.lower < point.value < point.upper
        assert point.nominal_level == 0.95
        assert point.calibration == "unmeasured"


def test_b1_contrast_between_named_doses_matches_closed_form() -> None:
    exact = run("contrast", contrast=(1.0, 3.0))
    assert exact.contrast is not None
    assert exact.contrast.estimate == pytest.approx(3.0, abs=1e-6)
    assert (exact.contrast.from_support, exact.contrast.to_support) == ("supported", "supported")
    noisy = run("contrast", noise=0.1)
    assert noisy.contrast is not None
    assert noisy.contrast.estimate == pytest.approx(3.0, abs=0.05)
    assert noisy.contrast.standard_error > 0.0
    assert noisy.contrast.lower < noisy.contrast.estimate < noisy.contrast.upper
    assert noisy.contrast.calibration == "unmeasured"
    assert noisy.levels == () and noisy.derivatives == ()
    assert [s.dose for s in noisy.support_table] == [1.0, 3.0]


def test_b1_claims_are_separate_and_the_simultaneous_band_refuses_closed() -> None:
    level = run("level", noise=0.1)
    assert level.simultaneous_band == "closed"
    assert level.derivatives == ()
    band = DoseClaims(pointwise_level=True, simultaneous_band=True)
    refused(
        lambda: run("level", noise=0.1, claims=band),
        "dose_grid.simultaneous_band_closed",
        "route_not_supported",
    )
    # A level claim does not license a derivative, nor a derivative claim a level.
    refused(
        lambda: run("derivative", noise=0.1, claims=DoseClaims(pointwise_level=True)),
        "dose_grid.derivative_without_claim",
        "option_not_applicable",
    )
    refused(
        lambda: run("level", noise=0.1, claims=DoseClaims(derivative=True)),
        "dose_grid.level_without_claim",
        "option_not_applicable",
    )


def test_b1_unsupported_doses_refuse_and_the_support_table_labels_them() -> None:
    dose, _ = design()
    table = dose_support_table(dose, [-1.0, 2.0, 10.0], bandwidth=0.4, minimum_local_ess=10.0)
    assert [s.label for s in table] == [
        "outside_empirical_support",
        "supported",
        "outside_empirical_support",
    ]
    weak = dose_support_table(dose, [2.0], bandwidth=0.4, minimum_local_ess=1000.0)
    assert weak[0].label == "weak_overlap"
    assert 10.0 < weak[0].local_ess < 1000.0
    refused(
        lambda: run("level", grid=[*GRID, 10.0]),
        "dose_grid.unsupported_dose",
        "cell_not_licensed",
    )
    refused(
        lambda: run("level", minimum_local_ess=1000.0),
        "dose_grid.insufficient_local_weight",
        "cell_not_licensed",
    )


def test_b1_observational_design_and_bad_inputs_refuse() -> None:
    refused(
        lambda: run("level", design="observational"),
        "dose_grid.graph_not_certified",
        "effect_not_identified",
    )
    refused(
        lambda: run("level", bandwidth=2.0),
        "dose_grid.bandwidth_outside_range",
        "invalid_argument",
    )
    refused(
        lambda: run("level", grid=[0.5, float("nan")]),
        "dose_grid.invalid_request",
        "invalid_argument",
    )
    dose, outcome = design()
    refused(
        lambda: dose_functional(
            dose[:2],
            outcome[:2],
            grid=[0.5],
            bandwidth=0.4,
            bandwidth_range=(0.1, 1.0),
        ),
        "dose_grid.invalid_request",
        "invalid_argument",
    )


def test_b1_export_consume_round_trip_recomputes_the_result() -> None:
    result = run("level", noise=0.1)
    artifact = result.export()
    assert isinstance(artifact, bytes)
    fresh = consume_dose_grid_artifact(artifact)
    assert fresh.levels == result.levels
    assert fresh.support == result.support
    assert fresh.premises_digest == result.premises_digest
    assert fresh.data_digest == result.data_digest
    assert fresh.export() == artifact
    assert fresh.calibration == "unmeasured" and fresh.simultaneous_band == "closed"
    contrast = consume_dose_grid_artifact(run("contrast", noise=0.1).export())
    assert contrast.contrast is not None and contrast.levels == ()


def test_b1_consumer_limits_and_corrupt_artifacts_refuse() -> None:
    artifact = run("level", noise=0.1).export()
    with pytest.raises(DoseGridRefusal) as info:
        consume_dose_grid_artifact(artifact, max_rows=10)
    assert info.value.detail == "dose_grid.consumer_limit_exceeded"
    assert_registered_refusal(info.value)
    with pytest.raises(CausalSerializationError):
        consume_dose_grid_artifact(artifact[:-5])
    with pytest.raises(CausalSerializationError):
        consume_dose_grid_artifact(b"not an artifact")
    corrupt = bytearray(artifact)
    corrupt[len(corrupt) // 2] ^= 0xFF
    with pytest.raises((CausalSerializationError, DoseGridRefusal)):
        consume_dose_grid_artifact(bytes(corrupt))
    with pytest.raises(TypeError, match="artifact must be bytes"):
        consume_dose_grid_artifact("text")  # type: ignore[arg-type]


@pytest.mark.parametrize("functional", ["level", "derivative", "contrast"])
@pytest.mark.parametrize("bandwidth", [0.2, 0.4, 0.8])
def test_b1_quadratic_scope_recovers_functional_across_fixed_bandwidths(
    functional: str, bandwidth: float
) -> None:
    from antecedent.dose_grid import quadratic_dose_functional

    dose, outcome = design()
    options = {"contrast": (1.0, 3.0)} if functional == "contrast" else {"grid": GRID}
    result = quadratic_dose_functional(
        dose,
        outcome,
        functional=functional,
        bandwidth=bandwidth,
        bandwidth_range=(0.1, 1.0),
        **options,
    )
    assert result.quadratic_mean and result.smoothing_bias_included
    assert result.calibration == "unmeasured" and result.simultaneous_band == "closed"
    for point in result.levels:
        assert point.value == pytest.approx(truth_level(point.dose), abs=1e-6)
        assert point.smoothing_bias_included
    for point in result.derivatives:
        assert point.value == pytest.approx(truth_derivative(point.dose), abs=1e-5)
        assert point.smoothing_bias_included
    if result.contrast is not None:
        assert result.contrast.estimate == pytest.approx(3.0, abs=1e-6)
        assert result.contrast.smoothing_bias_included
    fresh = consume_dose_grid_artifact(result.export())
    assert fresh == result
    assert fresh.quadratic_mean
    assert fresh.to_dict()["quadratic_mean"] is True


def test_b1_quadratic_scope_keeps_sampling_uncertainty_and_support_restrictions() -> None:
    from antecedent.dose_grid import quadratic_dose_functional

    dose, outcome = design(0.1)
    scoped = quadratic_dose_functional(
        dose, outcome, grid=GRID, bandwidth=0.4, bandwidth_range=(0.1, 1.0)
    )
    generic = run("level", noise=0.1)
    assert not generic.quadratic_mean and not generic.smoothing_bias_included
    assert scoped.premises_digest != generic.premises_digest
    assert scoped.data_digest == generic.data_digest
    assert [p.standard_error for p in scoped.levels] == [p.standard_error for p in generic.levels]
    assert all(p.standard_error > 0 for p in scoped.levels)
    refused(
        lambda: quadratic_dose_functional(
            dose, outcome, grid=[10.0], bandwidth=0.4, bandwidth_range=(0.1, 1.0)
        ),
        "dose_grid.unsupported_dose",
        "cell_not_licensed",
    )
    refused(
        lambda: quadratic_dose_functional(
            dose,
            outcome,
            grid=GRID,
            bandwidth=0.4,
            bandwidth_range=(0.1, 1.0),
            claims=DoseClaims(pointwise_level=True, simultaneous_band=True),
        ),
        "dose_grid.simultaneous_band_closed",
        "route_not_supported",
    )


def test_b1_quadratic_scope_fresh_process_replays_attested_artifact() -> None:
    import json
    import subprocess
    import sys

    from antecedent.dose_grid import quadratic_dose_functional

    dose, outcome = design()
    produced = quadratic_dose_functional(
        dose, outcome, grid=[1.0], bandwidth=0.4, bandwidth_range=(0.1, 1.0)
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys; "
                "from antecedent.dose_grid import consume_dose_grid_artifact; "
                "r=consume_dose_grid_artifact(sys.stdin.buffer.read()); "
                "print(json.dumps({'value':r.levels[0].value, "
                "'quadratic_mean':r.quadratic_mean, 'bias':r.smoothing_bias_included, "
                "'calibration':r.calibration, 'premises':r.premises_digest}))"
            ),
        ],
        input=produced.export(),
        capture_output=True,
        check=True,
    )
    replayed = json.loads(child.stdout)
    assert replayed["value"] == pytest.approx(1.75, abs=1e-6)
    assert replayed["quadratic_mean"] is True and replayed["bias"] is True
    assert replayed["calibration"] == "unmeasured"
    assert replayed["premises"] == produced.premises_digest
