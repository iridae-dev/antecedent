"""Evidence for the narrowly scoped right-censored survival utility."""

import antecedent
import numpy as np
import pandas as pd
import pytest
from antecedent.errors import CausalValueError


def _fixture() -> pd.DataFrame:
    # Control: KM steps to .75 at t=1 and .375 at t=3; RMST(4)=2.875.
    # Treated: KM steps to .75 at t=2 and .5 at t=4; RMST(4)=3.5.
    return pd.DataFrame(
        {
            "duration": [1, 2, 3, 4, 2, 4, 4, 6],
            "event": [1, 0, 1, 0, 1, 1, 0, 1],
            "treated": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


def test_randomized_survival_known_truth_with_right_censoring():
    query = antecedent.SurvivalOutcome(
        duration="duration", event_observed="event", treatment="treated", tau=4, randomized=True
    )
    result = antecedent.estimate_survival(_fixture(), query)

    assert result.rmst_control == pytest.approx(2.875)
    assert result.rmst_treated == pytest.approx(3.5)
    assert result.rmst_difference == pytest.approx(0.625)
    assert result.times == (0.0, 1.0, 2.0, 3.0, 4.0)
    assert result.control_survival == pytest.approx((1.0, 0.75, 0.75, 0.375, 0.375))
    assert result.treated_survival == pytest.approx((1.0, 1.0, 0.75, 0.75, 0.5))
    assert result.uncertainty == "point_only"
    assert result.support_status == "unlicensed_point_utility"
    assert "independent_right_censoring_within_arm" in result.assumptions


def test_survival_refuses_nonrandomized_and_invalid_inputs():
    data = _fixture()
    query = antecedent.SurvivalOutcome("duration", "event", "treated", tau=4)
    with pytest.raises(CausalValueError, match="require[s]? declared individual random assignment"):
        antecedent.estimate_survival(data, query)

    query = antecedent.SurvivalOutcome("duration", "event", "treated", tau=4, randomized=True)
    invalid_event = data.copy()
    invalid_event["event"] = np.full(len(data), 2)
    with pytest.raises(CausalValueError, match="event_observed values"):
        antecedent.estimate_survival(invalid_event, query)

    one_arm = data.assign(treated=0)
    with pytest.raises(CausalValueError, match="both treatment arms"):
        antecedent.estimate_survival(one_arm, query)


def test_survival_rejects_invalid_horizon_and_duration():
    with pytest.raises(CausalValueError, match="tau must be finite and positive"):
        antecedent.SurvivalOutcome("duration", "event", "treated", tau=float("inf"), randomized=True)

    data = _fixture()
    data.loc[0, "duration"] = -1
    query = antecedent.SurvivalOutcome("duration", "event", "treated", tau=4, randomized=True)
    with pytest.raises(CausalValueError, match="durations must be finite and non-negative"):
        antecedent.estimate_survival(data, query)

    data = _fixture()
    data.loc[data["treated"] == 0, "duration"] = 2
    query = antecedent.SurvivalOutcome("duration", "event", "treated", tau=4, randomized=True)
    with pytest.raises(CausalValueError, match="observed follow-up horizon"):
        antecedent.estimate_survival(data, query)


def _competing_fixture() -> pd.DataFrame:
    # Cause 1 is the target; cause 2 competes with it. The row censored at the
    # same time as the control target event stays in that event-time risk set.
    # The target CIF is .20 in control and .50 in treatment through tau=3.
    return pd.DataFrame(
        {
            "duration": [1, 2, 2, 3, 3, 1, 2, 3, 3],
            "cause": [2, 1, 0, 0, 0, 1, 1, 2, 0],
            "treated": [0, 0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


def test_cause_specific_cumulative_incidence_known_truth():
    query = antecedent.survival.CompetingRisksOutcome(
        duration="duration",
        event_cause="cause",
        treatment="treated",
        target_cause=1,
        tau=3,
        randomized=True,
    )
    result = antecedent.survival.estimate_cumulative_incidence(_competing_fixture(), query)

    assert result.target_cause == 1
    assert result.times == (0.0, 1.0, 2.0, 3.0)
    assert result.control_incidence == pytest.approx((0.0, 0.0, 0.2, 0.2))
    assert result.treated_incidence == pytest.approx((0.0, 0.25, 0.5, 0.5))
    assert result.incidence_difference == pytest.approx(0.3)
    assert result.uncertainty == "point_only"
    assert result.support_status == "unlicensed_point_utility"
    assert "all_event_causes_coded_distinctly" in result.assumptions


def test_cumulative_incidence_refuses_unrandomized_or_invalid_cause_codes():
    data = _competing_fixture()
    query = antecedent.survival.CompetingRisksOutcome(
        "duration", "cause", "treated", target_cause=1, tau=3
    )
    with pytest.raises(CausalValueError, match="require[s]? declared individual random assignment"):
        antecedent.survival.estimate_cumulative_incidence(data, query)

    query = antecedent.survival.CompetingRisksOutcome(
        "duration", "cause", "treated", target_cause=1, tau=3, randomized=True
    )
    invalid = data.copy()
    invalid.loc[0, "cause"] = -1
    with pytest.raises(CausalValueError, match="non-negative 64-bit integer codes"):
        antecedent.survival.estimate_cumulative_incidence(invalid, query)

    fractional = data.copy()
    fractional["cause"] = fractional["cause"].astype(float)
    fractional.loc[0, "cause"] = 1.5
    with pytest.raises(CausalValueError, match="non-negative integer codes"):
        antecedent.survival.estimate_cumulative_incidence(fractional, query)

    single_cause = data.assign(cause=1)
    with pytest.raises(CausalValueError, match="at least two observed event-cause codes"):
        antecedent.survival.estimate_cumulative_incidence(single_cause, query)

    with pytest.raises(CausalValueError, match="target_cause must occur"):
        absent_target = antecedent.survival.CompetingRisksOutcome(
            "duration", "cause", "treated", target_cause=3, tau=3, randomized=True
        )
        antecedent.survival.estimate_cumulative_incidence(data, absent_target)

    with pytest.raises(CausalValueError, match="positive integer event code"):
        antecedent.survival.CompetingRisksOutcome(
            "duration", "cause", "treated", target_cause=0, tau=3, randomized=True
        )
