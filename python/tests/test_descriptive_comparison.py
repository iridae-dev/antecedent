"""2.2 E7: descriptive raw-versus-adjusted comparison and reporting-scale transforms.

The oracles are hand arithmetic and a pure-Python central-difference Jacobian; nothing here
calls the Rust delta-method code to check itself.
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest
from antecedent.descriptive import (
    attribute_gap_to_columns,
    raw_reporting_transform,
    raw_vs_adjusted,
    replay_descriptive_comparison,
    transform_mean_pair,
)
from antecedent.errors import CausalSerializationError, CausalTypeError, CausalUnsupportedError

from _refusal import assert_registered_refusal

# Control [1, 2, 3] (mean 2, s^2 1), active [4, 6, 8] (mean 6, s^2 4).
OUTCOME = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
TREATMENT = [0, 0, 0, 1, 1, 1]
# Active [1, 1, 0, 1] (p 0.75), control [0, 0, 1, 0] (p 0.25); both s^2/n = 0.0625.
BINARY_OUTCOME = [1, 1, 0, 1, 0, 0, 1, 0]
BINARY_TREATMENT = [1, 1, 1, 1, 0, 0, 0, 0]


def refusal(call):
    with pytest.raises(CausalUnsupportedError) as caught:
        call()
    assert_registered_refusal(caught.value)
    return caught.value


def compare(adjusted=3.5, **options):
    return raw_vs_adjusted(
        outcome=OUTCOME, treatment=TREATMENT, adjusted_estimate=adjusted, **options
    )


def test_the_raw_contrast_and_the_gap_match_the_hand_computation():
    result = compare(adjusted_standard_error=0.4)
    assert (result.active.n, result.control.n) == (3, 3)
    assert math.isclose(result.active.mean, 6.0) and math.isclose(result.control.mean, 2.0)
    assert math.isclose(result.active.sum_squares, 8.0)
    assert math.isclose(result.control.sum_squares, 2.0)
    assert math.isclose(result.raw_difference, 4.0)
    assert math.isclose(result.raw_standard_error, math.sqrt(5.0 / 3.0), rel_tol=1e-12)
    assert math.isclose(result.gap, 0.5)
    assert result.adjusted_standard_error == 0.4
    assert result.claim == "point_only"
    assert result.interpretation == "descriptive_not_causal_decomposition"
    # The gap has no interval: the typed reason is carried, never a standard error.
    assert result.gap_interval_unavailable == (
        "cell_not_licensed",
        "descriptive_comparison.gap_interval_unavailable",
    )


def test_a_gap_is_zero_when_the_adjusted_estimate_equals_the_raw_contrast():
    assert compare(adjusted=4.0).gap == 0.0


def test_another_estimand_coding_or_population_is_refused_not_compared():
    for options, code, detail in (
        ({"active": 2.0}, "route_not_supported", "estimand_coding_mismatch"),
        ({"control": 1.0}, "route_not_supported", "estimand_coding_mismatch"),
        ({"adjusted_scale": "log_risk_ratio"}, "route_not_supported", "estimand_coding_mismatch"),
        ({"adjusted_scale": "hazard_ratio"}, "route_not_supported", "scale_not_supported"),
        ({"population": "treated"}, "population_not_estimable", "population_not_all_observed"),
        ({"adjusted_standard_error": -1.0}, "invalid_argument", "invalid_data"),
    ):
        error = refusal(lambda options=options: compare(**options))
        assert error.reason_code == code, options
        assert f"descriptive_comparison.{detail}" in str(error), options
    error = refusal(lambda: compare(adjusted=float("nan")))
    assert error.reason_code == "invalid_argument"


def test_malformed_rows_and_empty_arms_are_refused():
    for outcome, treatment in (
        ([], []),
        ([1.0, 2.0], [0]),
        ([1.0, float("nan")], [0, 1]),
        ([1.0, 2.0, 3.0], [0, 1, 2]),
    ):
        error = refusal(
            lambda o=outcome, t=treatment: raw_vs_adjusted(
                outcome=o, treatment=t, adjusted_estimate=1.0
            )
        )
        assert error.reason_code == "invalid_argument"
    error = refusal(
        lambda: raw_vs_adjusted(outcome=[1.0, 2.0], treatment=[1, 1], adjusted_estimate=1.0)
    )
    assert error.reason_code == "arm_not_populated"
    with pytest.raises(CausalTypeError):
        raw_vs_adjusted(outcome=["a", "b"], treatment=[0, 1], adjusted_estimate=1.0)


def test_a_one_row_arm_has_no_standard_error_and_no_covariance():
    result = raw_vs_adjusted(outcome=[1.0, 2.0, 5.0], treatment=[0, 0, 1], adjusted_estimate=3.0)
    assert math.isclose(result.raw_difference, 3.5)
    assert result.raw_standard_error is None
    transform = raw_reporting_transform(
        outcome=[0.2, 0.4, 0.6], treatment=[0, 0, 1], scales=["mean_difference"]
    )
    assert transform.covariance is None
    assert transform.covariance_unavailable == (
        "invalid_argument",
        "descriptive_comparison.arm_too_small",
    )


def test_the_gap_is_never_attributed_to_adjustment_columns():
    for call in (
        lambda: attribute_gap_to_columns(),
        lambda: attribute_gap_to_columns(columns=["age", "sex"], method="leave_one_out"),
    ):
        error = refusal(call)
        assert error.reason_code == "effect_not_identified"
        assert "descriptive_comparison.column_attribution_not_identified" in str(error)


def test_log_risk_ratio_matches_the_hand_computation():
    # Means (0.4, 0.2), Sigma [[0.01, 0.002], [0.002, 0.004]]: gradient (2.5, -5), variance
    # 6.25 * 0.01 + 2 * (2.5 * -5) * 0.002 + 25 * 0.004 = 0.1125.
    result = transform_mean_pair(
        0.4, 0.2, scales=["log_risk_ratio"], covariance=[[0.01, 0.002], [0.002, 0.004]]
    )
    assert result.claim == "point_only"
    assert math.isclose(result.values[0], math.log(2.0), rel_tol=1e-12)
    assert result.gradients[0] == (1.0 / 0.4, -1.0 / 0.2)
    assert math.isclose(result.covariance[0][0], 0.1125, rel_tol=1e-12)
    assert math.isclose(result.standard_error("log_risk_ratio"), math.sqrt(0.1125), rel_tol=1e-12)
    assert result.covariance_unavailable is None


def _fd_jacobian(f, a, c, h=1e-6):
    return (
        (f(a + h, c) - f(a - h, c)) / (2 * h),
        (f(a, c + h) - f(a, c - h)) / (2 * h),
    )


def test_the_family_covariance_matches_an_independent_finite_difference_jacobian():
    a, c = 0.4, 0.2
    sigma = [[0.01, 0.002], [0.002, 0.004]]
    functions = {
        "mean_difference": lambda x, y: x - y,
        "risk_difference": lambda x, y: x - y,
        "log_risk_ratio": lambda x, y: math.log(x / y),
        "log_odds_ratio": lambda x, y: math.log(x / (1 - x)) - math.log(y / (1 - y)),
    }
    scales = list(functions)
    result = transform_mean_pair(a, c, scales=scales, covariance=sigma)
    jac = [_fd_jacobian(functions[s], a, c) for s in scales]
    for i, scale in enumerate(scales):
        assert math.isclose(result.values[i], functions[scale](a, c), abs_tol=1e-12)
        for j in range(len(scales)):
            expected = sum(jac[i][p] * sigma[p][q] * jac[j][q] for p in (0, 1) for q in (0, 1))
            assert math.isclose(result.covariance[i][j], expected, rel_tol=1e-6, abs_tol=1e-9)
            assert result.covariance[i][j] == result.covariance[j][i]


def test_raw_arm_means_carry_the_independent_groups_covariance():
    result = raw_reporting_transform(
        outcome=BINARY_OUTCOME,
        treatment=BINARY_TREATMENT,
        scales=["log_risk_ratio", "log_odds_ratio"],
    )
    assert math.isclose(result.values[0], math.log(3.0), rel_tol=1e-12)
    assert math.isclose(result.values[1], 2.0 * math.log(3.0), rel_tol=1e-12)
    # Hand: log RR gradient (4/3, -4), variance 16/9 / 16 + 16 / 16 = 10/9; log OR gradient
    # (16/3, -16/3), variance 2 * (256/9) / 16 = 32/9.
    assert math.isclose(result.standard_error("log_risk_ratio"), math.sqrt(10 / 9), rel_tol=1e-12)
    assert math.isclose(result.standard_error("log_odds_ratio"), math.sqrt(32 / 9), rel_tol=1e-12)


def test_a_missing_joint_covariance_gives_points_and_a_typed_reason_never_a_zero():
    result = transform_mean_pair(0.4, 0.2, scales=["log_risk_ratio", "risk_difference"])
    assert math.isclose(result.values[0], math.log(2.0), rel_tol=1e-12)
    assert math.isclose(result.values[1], 0.2, rel_tol=1e-12)
    assert result.covariance is None
    assert result.standard_error("log_risk_ratio") is None
    assert result.covariance_unavailable == (
        "required_option_missing",
        "descriptive_comparison.joint_covariance_unavailable",
    )


def test_an_interval_on_a_transformed_scale_is_a_typed_refusal():
    error = refusal(lambda: transform_mean_pair(0.4, 0.2, level=0.95))
    assert error.reason_code == "cell_not_licensed"
    assert "descriptive_comparison.interval_withheld" in str(error)
    error = refusal(
        lambda: raw_reporting_transform(
            outcome=BINARY_OUTCOME, treatment=BINARY_TREATMENT, level=0.95
        )
    )
    assert error.reason_code == "cell_not_licensed"


def test_a_mean_pair_outside_a_scales_domain_or_a_bad_covariance_is_refused():
    for args, kwargs in (
        ((0.4, 0.0), {"scales": ["log_risk_ratio"]}),
        ((1.0, 0.2), {"scales": ["log_odds_ratio"]}),
        ((1.2, 0.2), {"scales": ["risk_difference"]}),
        ((0.4, 0.2), {"scales": []}),
        ((0.4, 0.2), {"scales": ["log_risk_ratio"], "covariance": [[0.01, 0.02], [0.02, 0.004]]}),
        ((0.4, 0.2), {"scales": ["log_risk_ratio"], "covariance": [[-0.01, 0], [0, 0.004]]}),
    ):
        error = refusal(lambda args=args, kwargs=kwargs: transform_mean_pair(*args, **kwargs))
        assert error.reason_code == "invalid_argument", (args, kwargs)
    error = refusal(lambda: transform_mean_pair(0.4, 0.2, scales=["hazard_ratio"]))
    assert error.reason_code == "route_not_supported"
    # A mean difference has no domain restriction.
    assert math.isclose(
        transform_mean_pair(-3.0, 12.5, scales=["mean_difference"]).values[0], -15.5
    )
    with pytest.raises(CausalTypeError):
        transform_mean_pair(0.4, 0.2, covariance=[[0.01, 0.002], [0.003, 0.004]])
    with pytest.raises(CausalTypeError):
        transform_mean_pair(0.4, 0.2, scales="log_risk_ratio")


def sealed(body):
    body = {k: v for k, v in body.items() if k != "digest"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return json.dumps({**body, "digest": hashlib.sha256(canonical.encode()).hexdigest()})


def test_an_exported_comparison_is_replayed_by_an_independent_consumer():
    result = compare(adjusted_standard_error=0.4)
    artifact = result.export()
    body = json.loads(artifact)
    assert body["format"] == "descriptive_comparison_v1"
    assert body["claim"] == "point_only"
    again = replay_descriptive_comparison(artifact)
    assert again == result


def test_a_tampered_artifact_is_refused_for_the_right_reason():
    body = json.loads(compare().export())
    for edit in (
        {"gap": body["gap"] + 1e-9},
        {"raw_difference": 4.5},
        {"active": [3, 6.0, 9.0]},
    ):
        with pytest.raises(CausalSerializationError, match="digest"):
            replay_descriptive_comparison(json.dumps({**body, **edit}))
        # Re-sealed edits keep a valid digest but no longer reproduce.
        with pytest.raises(CausalSerializationError, match="reproduce"):
            replay_descriptive_comparison(sealed({**body, **edit}))
    for forged in (
        {"format": "descriptive_comparison_v2"},
        {"claim": "calibrated"},
        {"interpretation": "causal_decomposition"},
        {"scale": "log_risk_ratio"},
        {"interval": [0.0, 1.0]},
        {"active": [0, 6.0, 8.0]},
        {"active": [3, 6.0, -1.0]},
        {"active": "nope"},
        {"adjusted_estimate": "x"},
        {"adjusted_standard_error": -0.1},
    ):
        with pytest.raises(CausalSerializationError):
            replay_descriptive_comparison(sealed({**body, **forged}))
    with pytest.raises(CausalSerializationError):
        replay_descriptive_comparison("not json")
