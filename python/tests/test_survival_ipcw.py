"""Evidence for caller-supplied censoring weighted survival summaries."""

import antecedent
import numpy as np
import pandas as pd
import pytest
from antecedent.errors import CausalValueError
from antecedent.observation import IndependentGiven


def _data():
    # At t=1 each arm has one event and one censor. Supplied G gives the
    # censored row weight 2 and the event row weight 1 (control), so weighted
    # KM survival is 2/3. Treated has two equal G=.5, giving survival 1/2.
    return pd.DataFrame({
        "duration": [1.0, 3.0, 1.0, 3.0],
        "event": [1, 0, 1, 0],
        "treated": [0, 0, 1, 1],
    })


def test_ipcw_survival_known_truth_and_explicit_point_semantics():
    query = antecedent.survival.SurvivalOutcome("duration", "event", "treated", 3, randomized=True)
    g = np.array([[1.0, 1.0, 1.0], [1.0, 0.5, 0.5],
                  [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]])
    result = antecedent.survival.estimate_survival_ipcw(
        _data(), query, times=[0.0, 1.0, 3.0], censoring_survival=g
    )
    assert result.control_survival == pytest.approx((1.0, 2 / 3, 2 / 3))
    assert result.treated_survival == pytest.approx((1.0, 0.5, 0.5))
    assert result.rmst_control == pytest.approx(7 / 3)
    assert result.rmst_treated == pytest.approx(2.0)
    assert result.rmst_difference == pytest.approx(-1 / 3)
    assert result.minimum_censoring_survival == pytest.approx(0.5)
    assert result.minimum_event_risk_set_control == 2
    assert result.uncertainty == "point_only"
    assert result.support_status == "unlicensed_point_utility"
    assert result.censoring_survival_provenance == "caller_supplied_not_fitted_or_verified"
    assert "correct_caller_supplied_conditional_censoring_survival" in result.assumptions


def test_ipcw_survival_refuses_unsupported_or_positivity_violating_inputs():
    query = antecedent.survival.SurvivalOutcome("duration", "event", "treated", 3, randomized=True)
    g = np.full((4, 3), 0.5)
    with pytest.raises(CausalValueError, match="include every observed duration"):
        antecedent.survival.estimate_survival_ipcw(
            _data(), query, times=[0.0, 2.0, 3.0], censoring_survival=g
        )
    g[0, 1] = 0.001
    with pytest.raises(CausalValueError, match="positivity floor"):
        antecedent.survival.estimate_survival_ipcw(
            _data(), query, times=[0.0, 1.0, 3.0], censoring_survival=g
        )
    delayed = antecedent.survival.SurvivalOutcome(
        "duration", "event", "treated", 3, randomized=True,
        delayed_entry="entry", observation_assumption=IndependentGiven(())
    )
    with pytest.raises(CausalValueError, match="refuses delayed-entry"):
        antecedent.survival.estimate_survival_ipcw(
            _data().assign(entry=0.0), delayed, times=[0.0, 1.0, 3.0],
            censoring_survival=np.full((4, 3), 0.5)
        )


def _competing_data():
    return pd.DataFrame({
        "duration": [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0],
        "cause": [1, 2, 0, 0, 1, 2, 0, 0],
        "treated": [0, 0, 0, 0, 1, 1, 1, 1],
    })


def test_ipcw_competing_risks_known_truth_and_diagnostics():
    query = antecedent.survival.CompetingRisksOutcome(
        "duration", "cause", "treated", target_cause=1, tau=2, randomized=True
    )
    # Control weighted risk at t=1 is 1 + 2 + 2 + 2 = 7; all-cause
    # weighted failure is 3 and target-cause failure is 1. Treated uses G=1.
    g = np.array([
        [1.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5],
        [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
    ])
    result = antecedent.survival.estimate_cumulative_incidence_ipcw(
        _competing_data(), query, times=[0.0, 1.0, 2.0], censoring_survival=g
    )
    assert result.control_incidence == pytest.approx((0.0, 1 / 7, 1 / 7))
    assert result.treated_incidence == pytest.approx((0.0, 1 / 4, 1 / 4))
    assert result.incidence_difference == pytest.approx(3 / 28)
    assert result.target_cause == 1
    assert result.minimum_censoring_survival == pytest.approx(0.5)
    assert result.minimum_event_risk_set_control == 4
    assert result.uncertainty == "point_only"
    assert result.support_status == "unlicensed_point_utility"
    assert result.censoring_survival_provenance == "caller_supplied_not_fitted_or_verified"
    assert "all_event_causes_coded_distinctly" in result.assumptions


def test_ipcw_competing_risks_refuses_invalid_codes_grid_and_positivity():
    query = antecedent.survival.CompetingRisksOutcome(
        "duration", "cause", "treated", target_cause=1, tau=2, randomized=True
    )
    data = _competing_data()
    with pytest.raises(CausalValueError, match="include every observed duration"):
        antecedent.survival.estimate_cumulative_incidence_ipcw(
            data, query, times=[0.0, 1.5, 2.0], censoring_survival=np.ones((8, 3))
        )
    bad_g = np.ones((8, 3))
    bad_g[0, 1] = 0.001
    with pytest.raises(CausalValueError, match="positivity floor"):
        antecedent.survival.estimate_cumulative_incidence_ipcw(
            data, query, times=[0.0, 1.0, 2.0], censoring_survival=bad_g
        )
    delayed = antecedent.survival.CompetingRisksOutcome(
        "duration", "cause", "treated", target_cause=1, tau=2, randomized=True,
        delayed_entry="entry", observation_assumption=IndependentGiven(()),
    )
    with pytest.raises(CausalValueError, match="refuses delayed-entry"):
        antecedent.survival.estimate_cumulative_incidence_ipcw(
            data.assign(entry=0.0), delayed, times=[0.0, 1.0, 2.0],
            censoring_survival=np.ones((8, 3)),
        )
