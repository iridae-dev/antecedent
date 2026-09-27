"""Evidence for the narrowly scoped right-censored survival utility."""

import antecedent
import numpy as np
import pandas as pd
import pytest
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.observation import IndependentGiven


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
    query = antecedent.survival.SurvivalOutcome(
        duration="duration", event_observed="event", treatment="treated", tau=4,
        randomized=True, observation_assumption=IndependentGiven(()),
    )
    result = antecedent.analyze(_fixture(), query=query).survival

    assert result.rmst_control == pytest.approx(2.875)
    assert result.rmst_treated == pytest.approx(3.5)
    assert result.rmst_difference == pytest.approx(0.625)
    assert result.times == (0.0, 1.0, 2.0, 3.0, 4.0)
    assert result.control_survival == pytest.approx((1.0, 0.75, 0.75, 0.375, 0.375))
    assert result.treated_survival == pytest.approx((1.0, 1.0, 0.75, 0.75, 0.5))
    assert result.uncertainty == "point_only_no_interval"
    assert result.support_status == "unlicensed_point_utility"
    assert "independent_right_censoring_within_arm" in result.assumptions


def test_survival_refuses_nonrandomized_and_invalid_inputs():
    data = _fixture()
    query = antecedent.survival.SurvivalOutcome(
        "duration", "event", "treated", tau=4, observation_assumption=IndependentGiven(())
    )
    with pytest.raises(CausalUnsupportedError, match="randomized survival requires randomized=True"):
        antecedent.analyze(data, query=query)

    query = antecedent.survival.SurvivalOutcome(
        "duration", "event", "treated", tau=4, randomized=True,
        observation_assumption=IndependentGiven(()),
    )
    invalid_event = data.copy()
    invalid_event["event"] = np.full(len(data), 2)
    with pytest.raises(CausalUnsupportedError, match="survival events must be encoded"):
        antecedent.analyze(invalid_event, query=query)

    one_arm = data.assign(treated=0)
    with pytest.raises(CausalUnsupportedError, match="both randomized arms require observed units"):
        antecedent.analyze(one_arm, query=query)


def test_survival_rejects_invalid_horizon_and_duration():
    with pytest.raises(CausalValueError, match="tau must be finite and positive"):
        antecedent.survival.SurvivalOutcome("duration", "event", "treated", tau=float("inf"), randomized=True)

    data = _fixture()
    data.loc[0, "duration"] = -1
    query = antecedent.survival.SurvivalOutcome(
        "duration", "event", "treated", tau=4, randomized=True,
        observation_assumption=IndependentGiven(()),
    )
    with pytest.raises(CausalUnsupportedError, match="durations must be finite and nonnegative"):
        antecedent.analyze(data, query=query)

    data = _fixture()
    data.loc[data["treated"] == 0, "duration"] = 2
    query = antecedent.survival.SurvivalOutcome(
        "duration", "event", "treated", tau=4, randomized=True,
        observation_assumption=IndependentGiven(()),
    )
    with pytest.raises(CausalUnsupportedError, match="tau exceeds observed follow-up"):
        antecedent.analyze(data, query=query)


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
        observation_assumption=IndependentGiven(()),
    )
    result = antecedent.analyze(_competing_fixture(), query=query).survival

    assert result.target_cause == 1
    assert result.times == (0.0, 1.0, 2.0, 3.0)
    assert result.control_incidence == pytest.approx((0.0, 0.0, 0.2, 0.2))
    assert result.treated_incidence == pytest.approx((0.0, 0.25, 0.5, 0.5))
    assert result.incidence_difference == pytest.approx(0.3)
    assert result.uncertainty == "point_only_no_interval"
    assert result.support_status == "unlicensed_point_utility"
    assert "all_event_causes_coded_distinctly" in result.assumptions


def test_cumulative_incidence_refuses_unrandomized_or_invalid_cause_codes():
    data = _competing_fixture()
    query = antecedent.survival.CompetingRisksOutcome(
        "duration", "cause", "treated", target_cause=1, tau=3,
        observation_assumption=IndependentGiven(()),
    )
    with pytest.raises(CausalUnsupportedError, match="randomized survival requires randomized=True"):
        antecedent.analyze(data, query=query)

    query = antecedent.survival.CompetingRisksOutcome(
        "duration", "cause", "treated", target_cause=1, tau=3, randomized=True,
        observation_assumption=IndependentGiven(()),
    )
    invalid = data.copy()
    invalid.loc[0, "cause"] = -1
    with pytest.raises(CausalUnsupportedError, match="survival event codes must be finite nonnegative integers"):
        antecedent.analyze(invalid, query=query)

    fractional = data.copy()
    fractional["cause"] = fractional["cause"].astype(float)
    fractional.loc[0, "cause"] = 1.5
    with pytest.raises(CausalUnsupportedError, match="survival event codes must be finite nonnegative integers"):
        antecedent.analyze(fractional, query=query)

    single_cause = data.assign(cause=1)
    with pytest.raises(CausalUnsupportedError, match="two observed causes including the target"):
        antecedent.analyze(single_cause, query=query)

    with pytest.raises(CausalUnsupportedError, match="two observed causes including the target"):
        absent_target = antecedent.survival.CompetingRisksOutcome(
            "duration", "cause", "treated", target_cause=3, tau=3, randomized=True,
            observation_assumption=IndependentGiven(()),
        )
        antecedent.analyze(data, query=absent_target)

    with pytest.raises(CausalValueError, match="positive integer event code"):
        antecedent.survival.CompetingRisksOutcome(
            "duration", "cause", "treated", target_cause=0, tau=3, randomized=True
        )
