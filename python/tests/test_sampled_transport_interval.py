"""The delta-method Wald interval of a sampled counterfactual transport estimate.

Reference and tolerance. The estimate is a plug-in probability ``p`` from multinomial count
tables with a multinomial delta-method standard error ``se`` (the estimate already reports it,
labelled uncalibrated). The measured interval is the Wald interval of that same pair,
``p -/+ z * se`` with ``z = 1.959963984540054`` (the 0.975 normal quantile) clamped to ``[0, 1]``.
The identity is exact up to rounding, so ``1e-12`` absolute is the tolerance: the interval comes
from the replay of the sealed counts and must equal what the estimate's own point and standard
error imply, not a stored number. Calibration only annotates: the status is one of the four
documented labels and never changes the interval, and outside the measured construction (a
smoothed table) the interval is still reported with the rule named in its scope.
"""

from __future__ import annotations

import pytest
from antecedent.transport import advanced as transport

from test_transport_counterfactual import _emptied, _estimate, _joint_tables

Z_975 = 1.959963984540054
STATUSES = {"calibrated", "outside_measured_scope", "measured_below_nominal", "not_measured"}


def test_the_measured_interval_is_the_clamped_wald_interval_of_the_estimates_own_pair():
    derivation, tables = _joint_tables(seed=21, n=10_000)
    estimate = _estimate(derivation, tables)
    measured = estimate.measure_interval()

    assert measured.route == "sampled_counterfactual_transport"
    assert measured.calibration in STATUSES
    scalar = measured.scalar("target_probability")
    assert scalar.level == 0.95
    assert scalar.calibration in STATUSES
    assert scalar.point == pytest.approx(estimate.probability, abs=1e-15)

    se = estimate.standard_error["value"]
    lower, upper = scalar.interval
    assert lower == pytest.approx(max(0.0, estimate.probability - Z_975 * se), abs=1e-12)
    assert upper == pytest.approx(min(1.0, estimate.probability + Z_975 * se), abs=1e-12)
    assert lower < scalar.point < upper
    assert measured.validated_scope["method"] == "delta_wald_clamped"
    assert measured.validated_scope["unseen_rule"] == "refuse"

    # The original estimate still publishes no interval slot of its own.
    assert estimate.interval == {"available": True, "status": "reported"}
    assert estimate.standard_error["label"] == "uncalibrated_standard_error"

    # A fresh estimate of the same counts yields the identical interval (deterministic replay).
    again = _estimate(derivation, tables).measure_interval().scalar("target_probability")
    assert again.interval == scalar.interval


def test_a_smoothed_table_is_reported_with_its_rule_in_the_scope_never_refused_or_hidden():
    derivation, tables = _joint_tables(seed=21, n=10_000)
    observational = next(
        i for i, t in enumerate(tables) if t.population == "source" and not t.interventions
    )
    counts = tables[observational].counts
    cell = min((c for c in range(len(counts)) if counts[c] > 0), key=lambda c: counts[c])
    tables[observational] = _emptied(tables[observational], cell)
    smoothed = _estimate(
        derivation, tables, unseen_rule=transport.UnseenCellRule.additive_smoothing(0.5)
    )
    assert smoothed.point["smoothed"] is True
    measured = smoothed.measure_interval()
    # Calibration only annotates: the interval is still reported, with the rule that produced
    # it named in the scope so no reader mistakes it for the refused-unseen-cell construction.
    assert measured.validated_scope["unseen_rule"] == "additive_smoothing"
    scalar = measured.scalar("target_probability")
    assert scalar.calibration in STATUSES
    se = smoothed.standard_error["value"]
    assert scalar.interval[0] == pytest.approx(
        max(0.0, smoothed.probability - Z_975 * se), abs=1e-12
    )
    assert scalar.interval[1] == pytest.approx(
        min(1.0, smoothed.probability + Z_975 * se), abs=1e-12
    )
