"""Missing-not-at-random sensitivity: closed-form oracles and refusals.

Mirrors ``crates/antecedent-validate/tests/mnar_sensitivity.rs``.

Shift data: treated arm of 10 rows with outcomes 1..6 observed and 4 missing (mean 3.5,
missing fraction 0.4); control arm of 10 rows with outcomes 0..4 observed and 5 missing (mean 2,
missing fraction 0.5). The MAR effect is 1.5 and the shift model gives
``effect(delta) = 1.5 + delta * (l1 * 0.4 - l0 * 0.5)``.

Tilt data: treated arm of 10 rows, 6 observed (three ones, three zeros) and 4 missing; control
arm of 8 rows all observed with mean 0.25. With the tilt on the treated arm,
``effect(theta) = 0.3 + 0.4 * expit(theta) - 0.25``.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from antecedent import mnar_sensitivity as mnar
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError


def _shift_data() -> tuple[list[float], list[float | None]]:
    treatment = [1.0] * 10 + [0.0] * 10
    outcome: list[float | None] = [float(v) for v in range(1, 7)] + [None] * 4
    outcome += [float(v) for v in range(0, 5)] + [None] * 5
    return treatment, outcome


def _tilt_data() -> tuple[list[float], list[float | None]]:
    treatment = [1.0] * 10 + [0.0] * 8
    outcome: list[float | None] = [1.0] * 3 + [0.0] * 3 + [None] * 4 + [1.0] * 2 + [0.0] * 6
    return treatment, outcome


def _shift(**options):
    options.setdefault("model", "delta_pattern_mixture")
    options.setdefault("lower", -10.0)
    options.setdefault("upper", 10.0)
    return mnar.mnar_ate_sensitivity(*_shift_data(), **options)


def test_mnar_delta_zero_reproduces_the_mar_estimate() -> None:
    result = _shift(loading="common")
    assert result.mar.effect == pytest.approx(1.5, abs=1e-12)
    assert result.mar.effect == pytest.approx(result.observed_case_effect, abs=1e-12)
    assert result.mar.treated_mean == pytest.approx(3.5)
    assert result.mar.control_mean == pytest.approx(2.0)


def test_mnar_shift_changes_the_mean_by_delta_times_the_missing_fraction() -> None:
    result = _shift(loading="treated_only")
    for point in result.grid:
        assert point.treated_mean == pytest.approx(3.5 + 0.4 * point.parameter, abs=1e-12)
        assert point.control_mean == pytest.approx(2.0, abs=1e-12)
        assert point.effect == pytest.approx(1.5 + 0.4 * point.parameter, abs=1e-12)


def test_mnar_tipping_points_match_the_closed_form() -> None:
    result = _shift(loading="treated_only", decisions={"treat": 1.0})
    sign = result.tipping_point("sign")
    assert sign.at_mar == 1
    assert sign.negative is not None
    assert sign.negative.parameter == pytest.approx(-3.75, abs=1e-9)
    assert sign.negative.changes_to <= 0
    assert sign.positive is None
    treat = result.decisions[0]
    assert treat.acts_at_mar and treat.changes_within_range
    assert treat.tipping.negative is not None
    assert treat.tipping.negative.parameter == pytest.approx(-1.25, abs=1e-9)
    assert result.tipping_point("treat") == treat.tipping

    common = _shift(loading="common", lower=-20.0, upper=20.0)
    assert common.tipping_point("sign").positive.parameter == pytest.approx(15.0, abs=1e-9)


def test_mnar_sampling_interval_is_conditional_and_separate_from_the_range() -> None:
    result = _shift(loading="common")
    point = next(p for p in result.grid if p.parameter == 2.0)
    variance = 3.5 / 6 + 4 * 0.24 / 10 + 2.5 / 5 + 4 * 0.25 / 10
    assert point.sampling.standard_error == pytest.approx(math.sqrt(variance), abs=1e-12)
    assert point.sampling.upper - point.sampling.lower == pytest.approx(
        2 * 1.959963984540054 * math.sqrt(variance), abs=1e-9
    )
    envelope = result.assumption_range
    effects = [p.effect for p in result.grid]
    assert (envelope.effect_lower, envelope.effect_upper) == (min(effects), max(effects))
    assert min(p.sampling.lower for p in result.grid) < envelope.effect_lower


def test_mnar_tilt_selection_matches_the_closed_form() -> None:
    result = mnar.mnar_ate_sensitivity(
        *_tilt_data(),
        model="exponential_tilting_selection",
        loading="treated_only",
        lower=-5.0,
        upper=5.0,
        decisions={"treat": 0.3},
    )
    for point in result.grid:
        expit = 1.0 / (1.0 + math.exp(-point.parameter))
        assert point.effect == pytest.approx(0.05 + 0.4 * expit, abs=1e-12)
    treat = result.decisions[0]
    assert not treat.acts_at_mar
    assert treat.tipping.positive is not None
    assert treat.tipping.positive.parameter == pytest.approx(math.log(5 / 3), abs=1e-9)
    assert result.tipping_point("sign").negative is None
    assert result.tipping_point("sign").positive is None


def test_mnar_ranges_are_never_labelled_as_confidence_intervals() -> None:
    result = _shift(loading="treated_only")
    assert result.inference_claim == "assumption_range"
    assert result.empirical_status == "not_empirically_established"
    assert result.assumption_range.is_confidence_interval is False
    assert result.assumption_range.empirically_established is False
    assert "not a confidence interval" in result.interpretation
    departure = next(a for a in result.assumptions if a.id.startswith("outcome_missingness"))
    assert departure.status == "untestable"
    names = {
        f.name
        for cls in (mnar.MnarAssumptionRange, mnar.MnarResult)
        for f in dataclasses.fields(cls)
    }
    # The only field that names a confidence interval is the explicit negative flag.
    labelled = {n for n in names if "confidence" in n or n in {"ci", "ci_lower", "ci_upper"}}
    assert labelled == {"is_confidence_interval"}
    # The sampling interval is its own type, never the range's.
    assert mnar.MnarSamplingInterval is not mnar.MnarAssumptionRange


def test_mnar_refuses_undeclared_unbounded_and_unsupported() -> None:
    with pytest.raises(mnar.MnarSensitivityRefusal) as info:
        _shift(model=None)
    assert info.value.detail == "mnar_sensitivity.mechanism_not_declared"
    assert info.value.code == "required_option_missing"

    with pytest.raises(mnar.MnarSensitivityRefusal) as info:
        _shift(model="logit_selection_on_covariates")
    assert info.value.detail == "mnar_sensitivity.unsupported_mechanism"

    with pytest.raises(mnar.MnarSensitivityRefusal) as info:
        _shift(estimator="causal_forest")
    assert info.value.detail == "mnar_sensitivity.unsupported_estimator"
    assert isinstance(info.value, CausalUnsupportedError)

    for lower, upper in ((-math.inf, 1.0), (-1.0, math.inf), (-1.0, 1.0e13)):
        with pytest.raises(mnar.MnarSensitivityRefusal) as info:
            _shift(lower=lower, upper=upper)
        assert info.value.detail == "mnar_sensitivity.unbounded_range"

    with pytest.raises(mnar.MnarSensitivityRefusal) as info:
        _shift(lower=1.0, upper=2.0)
    assert info.value.detail == "mnar_sensitivity.range_excludes_mar"

    with pytest.raises(mnar.MnarSensitivityRefusal) as info:
        _shift(sampling_composition="add the interval to the range")
    assert info.value.detail == "mnar_sensitivity.composition_not_licensed"
    assert info.value.code == "cell_not_licensed"


def test_mnar_argument_errors_are_domain_errors() -> None:
    with pytest.raises(CausalTypeError):
        mnar.mnar_ate_sensitivity("abc", [1.0], model="delta_pattern_mixture", lower=-1, upper=1)
    with pytest.raises(CausalValueError):
        _shift(loading="sideways")
    with pytest.raises(mnar.MnarSensitivityRefusal) as info:
        _shift(level=1.0)
    assert info.value.detail == "mnar_sensitivity.invalid_level"
    assert isinstance(info.value, CausalValueError)
    with pytest.raises(CausalValueError):
        _shift(loading="treated_only").tipping_point("nonexistent")


def test_mnar_nan_outcomes_are_missing() -> None:
    treatment, outcome = _shift_data()
    with_nan = [math.nan if v is None else v for v in outcome]
    result = mnar.mnar_ate_sensitivity(
        treatment, with_nan, model="delta_pattern_mixture", lower=-1.0, upper=1.0
    )
    assert result.treated_missing == 4 and result.control_missing == 5


def _adjusted_data() -> tuple[list[float], list[float | None], list[float]]:
    """Stratified data whose adjusted effect is ``1.5 + 0.25 * delta`` (treated-only shift)."""
    treatment: list[float] = []
    outcome: list[float | None] = []
    covariate: list[float] = []
    cells = [
        (1.0, 0.0, (1.0, 2.0, 3.0), 3, 3),
        (1.0, 1.0, (4.0, 5.0, 6.0), 2, 2),
        (0.0, 0.0, (0.0, 1.0, 2.0), 2, 2),
        (0.0, 1.0, (2.0, 3.0, 4.0), 3, 3),
    ]
    for arm, cell, values, repeats, missing in cells:
        for _ in range(repeats):
            for value in values:
                treatment.append(arm)
                outcome.append(value)
                covariate.append(cell)
        for _ in range(missing):
            treatment.append(arm)
            outcome.append(None)
            covariate.append(cell)
    return treatment, outcome, covariate


@pytest.mark.parametrize("estimator", ["regression_adjusted", "ipw", "aipw"])
def test_mnar_adjusted_estimators_match_the_stratified_closed_form(estimator: str) -> None:
    treatment, outcome, covariate = _adjusted_data()
    result = mnar.mnar_ate_sensitivity(
        treatment,
        outcome,
        model="delta_pattern_mixture",
        loading="treated_only",
        lower=-10.0,
        upper=10.0,
        estimator=estimator,
        covariates=[covariate],
        bootstrap_replicates=40,
        seed=3,
        decisions={"treat": 1.0},
    )
    for point in result.grid:
        assert point.effect == pytest.approx(1.5 + 0.25 * point.parameter, abs=1e-9)
        assert point.sampling.standard_error > 0.0
    assert result.tipping_point("sign").negative.parameter == pytest.approx(-6.0, abs=1e-7)
    assert result.decisions[0].tipping.negative.parameter == pytest.approx(-2.0, abs=1e-7)


def test_mnar_adjusted_estimators_require_covariates() -> None:
    treatment, outcome, _ = _adjusted_data()
    with pytest.raises(mnar.MnarSensitivityRefusal) as info:
        mnar.mnar_ate_sensitivity(
            treatment,
            outcome,
            model="delta_pattern_mixture",
            lower=-1.0,
            upper=1.0,
            estimator="aipw",
        )
    assert info.value.detail == "mnar_sensitivity.covariates_required"


def test_mnar_result_replays_identically_with_its_assumption_records() -> None:
    treatment, outcome, covariate = _adjusted_data()
    options = {
        "model": "delta_pattern_mixture",
        "loading": "treated_only",
        "lower": -10.0,
        "upper": 10.0,
        "estimator": "aipw",
        "covariates": [covariate],
        "bootstrap_replicates": 30,
        "seed": 11,
    }
    first = mnar.mnar_ate_sensitivity(treatment, outcome, **options)
    second = mnar.mnar_ate_sensitivity(treatment, outcome, **options)
    assert first == second
    statuses = {a.id: a.status for a in first.assumptions}
    assert statuses["outcome_missingness_departure_from_mar"] == "untestable"
    assert statuses["treatment_independent_of_potential_outcomes"] == "declared"


def _analysis_data() -> dict[str, np.ndarray]:
    return {
        "a": np.array([0.0] * 8 + [1.0] * 8),
        "y": np.array(
            [
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                np.nan,
                np.nan,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                np.nan,
                1.0,
                0.0,
            ]
        ),
    }


def test_mnar_result_api_requests_the_sensitivity_of_the_analysed_effect() -> None:
    from antecedent import analyze
    from antecedent.outcomes import require_success
    from antecedent.query import AverageEffect

    data = _analysis_data()
    result = require_success(
        analyze(
            {"a": data["a"], "y": np.nan_to_num(data["y"], nan=0.0)},
            query=AverageEffect("a", "y"),
            graph=[("a", "y")],
        )
    )
    sensitivity = result.mnar_sensitivity(
        data=data, model="delta_pattern_mixture", lower=-1.0, upper=1.0, decisions={"act": 0.0}
    )
    assert sensitivity.estimator == "difference_in_means"
    assert sensitivity.mar.effect == pytest.approx(sensitivity.observed_case_effect)
    assert sensitivity.assumption_range.is_confidence_interval is False
    assert sensitivity.decisions[0].name == "act"
