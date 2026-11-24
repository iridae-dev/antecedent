"""The descriptive comparison and the matched case-control odds ratio run through their
compiled programs.

``raw_vs_adjusted``, ``transform_mean_pair``, ``raw_reporting_transform`` and
``matched_case_control_odds_ratio`` reach the kernels through the compiled
``DescriptiveComparison`` and ``MatchedCaseControl`` program families, which call the same kernel
with the same inputs. Each call here emits no deprecation warning and still returns the
hand-arithmetic values the capability records cite, so moving the natives behind the compiler
changed no number. The oracles are hand arithmetic and closed forms; nothing here calls Rust to
check itself.
"""

from __future__ import annotations

import math
import warnings

from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.descriptive import raw_reporting_transform, raw_vs_adjusted, transform_mean_pair
from antecedent.matched import matched_case_control_odds_ratio

# Control [1, 2, 3] (mean 2, s^2 1), active [4, 6, 8] (mean 6, s^2 4).
OUTCOME = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
TREATMENT = [0, 0, 0, 1, 1, 1]
# Active [1, 1, 0, 1] (p 0.75), control [0, 0, 1, 0] (p 0.25); both s^2/n = 0.0625.
BINARY_OUTCOME = [1, 1, 0, 1, 0, 0, 1, 0]
BINARY_TREATMENT = [1, 1, 1, 1, 0, 0, 0, 0]


def no_deprecation(call):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntecedentDeprecationWarning)
        return call()


def test_raw_vs_adjusted_through_the_compiled_program_keeps_the_hand_values():
    result = no_deprecation(
        lambda: raw_vs_adjusted(
            outcome=OUTCOME,
            treatment=TREATMENT,
            adjusted_estimate=3.5,
            adjusted_standard_error=0.4,
        )
    )
    assert (result.active.n, result.control.n) == (3, 3)
    assert math.isclose(result.active.mean, 6.0) and math.isclose(result.control.mean, 2.0)
    assert math.isclose(result.raw_difference, 4.0)
    assert math.isclose(result.raw_standard_error, math.sqrt(5.0 / 3.0), rel_tol=1e-12)
    assert math.isclose(result.gap, 0.5)


def test_transform_mean_pair_through_the_compiled_program_keeps_the_hand_values():
    result = no_deprecation(
        lambda: transform_mean_pair(
            0.4,
            0.2,
            scales=["log_risk_ratio"],
            covariance=[[0.01, 0.002], [0.002, 0.004]],
            level=0.95,
        )
    )
    assert math.isclose(result.values[0], math.log(2.0), rel_tol=1e-12)
    assert result.gradients[0] == (1.0 / 0.4, -1.0 / 0.2)
    assert math.isclose(result.covariance[0][0], 0.1125, rel_tol=1e-12)
    half = 1.959963984540054 * math.sqrt(0.1125)
    lower, upper = result.intervals[0]
    assert math.isclose(lower, math.log(2.0) - half, rel_tol=1e-12)
    assert math.isclose(upper, math.log(2.0) + half, rel_tol=1e-12)


def test_raw_reporting_transform_through_the_compiled_program_keeps_the_hand_values():
    result = no_deprecation(
        lambda: raw_reporting_transform(
            outcome=BINARY_OUTCOME,
            treatment=BINARY_TREATMENT,
            scales=["log_risk_ratio", "log_odds_ratio"],
        )
    )
    assert math.isclose(result.values[0], math.log(3.0), rel_tol=1e-12)
    assert math.isclose(result.values[1], 2.0 * math.log(3.0), rel_tol=1e-12)
    assert math.isclose(result.standard_error("log_risk_ratio"), math.sqrt(10 / 9), rel_tol=1e-12)
    assert math.isclose(result.standard_error("log_odds_ratio"), math.sqrt(32 / 9), rel_tol=1e-12)


def test_matched_odds_ratio_through_the_compiled_program_keeps_the_closed_form():
    # 1:1 matching, 12 case-exposed and 4 control-exposed discordant pairs: theta = 3 and the
    # 95% log-scale interval is ln 3 -/+ 1.959963984540054 sqrt(1/12 + 1/4).
    sets = [[(1, 1), (0, 0)]] * 12 + [[(1, 0), (0, 1)]] * 4
    stratum, case, exposed = [], [], []
    for i, members in enumerate(sets):
        for y, x in members:
            stratum.append(f"set{i}")
            case.append(y)
            exposed.append(x)
    result = no_deprecation(
        lambda: matched_case_control_odds_ratio(
            stratum=stratum,
            case=case,
            exposed=exposed,
            sampling="matched_case_control",
            level=0.95,
        )
    )
    assert math.isclose(result.odds_ratio, 3.0, rel_tol=1e-9)
    half = 1.959963984540054 * math.sqrt(1 / 12 + 1 / 4)
    lower, upper = result.interval
    assert math.isclose(lower, 3.0 * math.exp(-half), rel_tol=1e-6)
    assert math.isclose(upper, 3.0 * math.exp(half), rel_tol=1e-6)
    assert result.interval_calibration["status"] == "not_measured"
