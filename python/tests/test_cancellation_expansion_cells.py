"""Cancellation of the 2.2 E-cells that expose a ``cancel`` token to Python (cost and
cancellation checks): a cancelled token stops the cell with its typed cancellation refusal and
no partial result, and the same inputs then complete under a live token.

The finite-action inverse outcome has its own Python cancellation test in
``test_inverse_outcome.py``; the estimator-level cells (penalized AIPW, clustered DML, joint
cells) are covered by the Rust tests in ``crates/antecedent-estimate/tests``.
"""

from __future__ import annotations

import pytest
from antecedent.descriptive import raw_reporting_transform, raw_vs_adjusted
from antecedent.errors import CausalUnsupportedError
from antecedent.matched import matched_case_control_odds_ratio
from antecedent.state import CancellationToken

from _refusal import assert_registered_refusal

N = 10_000
OUTCOME = [float(i % 7) for i in range(N)]
TREATMENT = [1 if i % 3 == 0 else 0 for i in range(N)]
BINARY_OUTCOME = [float(i % 2) for i in range(N)]


def cancelled_token() -> CancellationToken:
    token = CancellationToken()
    token.cancel()
    return token


def refusal(call):
    with pytest.raises(CausalUnsupportedError) as caught:
        call()
    assert_registered_refusal(caught.value)
    return caught.value


def test_a_cancelled_descriptive_comparison_is_a_stop_never_a_contrast():
    error = refusal(
        lambda: raw_vs_adjusted(
            outcome=OUTCOME,
            treatment=TREATMENT,
            adjusted_estimate=0.1,
            adjusted_standard_error=0.02,
            cancel=cancelled_token(),
        )
    )
    assert error.reason_code == "cancelled_no_claim"
    assert "descriptive_comparison.cancelled" in str(error)
    live = raw_vs_adjusted(
        outcome=OUTCOME,
        treatment=TREATMENT,
        adjusted_estimate=0.1,
        adjusted_standard_error=0.02,
        cancel=CancellationToken(),
    )
    assert live.active.n + live.control.n == N


def test_a_cancelled_raw_reporting_transform_is_a_stop_never_a_transform():
    error = refusal(
        lambda: raw_reporting_transform(
            outcome=BINARY_OUTCOME,
            treatment=TREATMENT,
            scales=("risk_difference",),
            cancel=cancelled_token(),
        )
    )
    assert error.reason_code == "cancelled_no_claim"
    assert "descriptive_comparison.cancelled" in str(error)
    live = raw_reporting_transform(
        outcome=BINARY_OUTCOME,
        treatment=TREATMENT,
        scales=("risk_difference",),
        cancel=CancellationToken(),
    )
    assert live.scales == ("risk_difference",)


def test_a_cancelled_matched_solve_stops_before_the_tally_even_when_no_set_is_informative():
    # Every pair is exposure-concordant, so the solve is never reached: only the entry poll can
    # stop this call, and without it the refusal would be the data's own.
    stratum = [f"set{i // 2}" for i in range(20)]
    case = [float(i % 2) for i in range(20)]
    exposed = [1.0] * 20
    error = refusal(
        lambda: matched_case_control_odds_ratio(
            stratum=stratum,
            case=case,
            exposed=exposed,
            sampling="matched_case_control",
            cancel=cancelled_token(),
        )
    )
    assert error.reason_code == "transport_budget_cancel"
    assert "matched_case_control.budget" in str(error)
