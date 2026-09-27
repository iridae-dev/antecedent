"""Known-truth and refusal evidence for left-truncated survival summaries."""

import antecedent
import pandas as pd
import pytest
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.observation import IndependentGiven


def _survival_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "duration": [2, 4, 3, 4, 4, 1, 4, 3, 4],
            # Entry at t=2 is excluded from the event-time risk set at t=2.
            "entry": [0, 0, 1, 1.5, 2, 0, 0, 2, 1],
            "event": [1, 0, 1, 0, 0, 1, 0, 1, 0],
            "treated": [0, 0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


def test_delayed_entry_km_and_rmst_known_truth():
    query = antecedent.survival.SurvivalOutcome(
        duration="duration",
        event_observed="event",
        treatment="treated",
        tau=4,
        randomized=True,
        delayed_entry="entry",
        observation_assumption=IndependentGiven(()),
    )
    result = antecedent.analyze(_survival_data(), query=query).survival

    # Counting-process risk sets are (entry, duration], so the entry=2
    # control censor is excluded at t=2 and included by t=3.
    assert result.rmst_control == pytest.approx(3.3125)
    assert result.rmst_treated == pytest.approx(7 / 3)
    assert result.rmst_difference == pytest.approx(7 / 3 - 3.3125)
    assert result.control_survival == pytest.approx((1.0, 1.0, 0.75, 0.5625, 0.5625))
    assert result.treated_survival == pytest.approx((1.0, 0.5, 0.5, 1 / 3, 1 / 3))
    assert "independent_delayed_entry_within_arm" in result.assumptions
    assert result.uncertainty == "point_only_no_interval"
    assert result.support_status == "unlicensed_point_utility"


def test_retained_delayed_entry_bootstrap_reports_pointwise_interval():
    rows = []
    for arm in (0, 1):
        for i in range(60):
            entry = 1.0 if i % 3 == 0 else 0.0
            duration = (
                1.0 if entry == 0.0 and i % 7 == 0 else 2.0 if i % (6 if arm else 5) == 0 else 3.0
            )
            rows.append((duration, float(duration < 3.0), float(arm), entry))
    data = pd.DataFrame(rows, columns=("duration", "event", "treated", "entry"))
    query = antecedent.survival.SurvivalOutcome(
        "duration",
        "event",
        "treated",
        3.0,
        randomized=True,
        delayed_entry="entry",
        observation_assumption=IndependentGiven(()),
    )
    prepared = antecedent.prepare(data, query=query, bootstrap=299, seed=279)
    result = prepared.estimate()
    section = result.survival
    assert section is not None
    assert section.uncertainty == "subject_stratified_percentile_bootstrap_pointwise_95"
    assert section.rmst_difference_interval is not None
    assert section.survival_at_tau_difference_interval is not None
    assert section.bootstrap_replicates_ok == 299
    assert section.support_status == "unlicensed_pointwise_interval"
    loaded = antecedent.load(prepared.export(artifact_id="left-truncated-survival-bootstrap"))
    assert loaded.answer.structured["rmst_difference_interval"] == list(
        section.rmst_difference_interval
    )


def _competing_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "duration": [2, 4, 3, 4, 4, 1, 4, 2, 4],
            "entry": [0, 0, 1, 1.5, 2, 0, 0, 1, 1.5],
            "cause": [1, 0, 2, 0, 0, 2, 0, 1, 0],
            "treated": [0, 0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


def test_delayed_entry_competing_risks_cif_known_truth():
    query = antecedent.survival.CompetingRisksOutcome(
        duration="duration",
        event_cause="cause",
        treatment="treated",
        target_cause=1,
        tau=3,
        randomized=True,
        delayed_entry="entry",
        observation_assumption=IndependentGiven(()),
    )
    result = antecedent.analyze(_competing_data(), query=query).survival

    assert result.control_incidence == pytest.approx((0.0, 0.0, 0.25, 0.25))
    assert result.treated_incidence == pytest.approx((0.0, 0.0, 1 / 6, 1 / 6))
    assert result.incidence_difference == pytest.approx(-1 / 12)
    assert "independent_delayed_entry_within_arm" in result.assumptions
    assert result.uncertainty == "point_only_no_interval"
    assert result.support_status == "unlicensed_point_utility"


@pytest.mark.parametrize("family", ["survival", "competing"])
def test_delayed_entry_requires_supported_observation_contract(family: str):
    if family == "survival":
        query_type = antecedent.survival.SurvivalOutcome
        args = ("duration", "event", "treated", 4)
        data = _survival_data()
    else:
        query_type = antecedent.survival.CompetingRisksOutcome
        args = ("duration", "cause", "treated", 1, 3)
        data = _competing_data()

    with pytest.raises(CausalValueError, match="explicit IndependentGiven"):
        no_assumption = query_type(*args, randomized=True, delayed_entry="entry")
        antecedent.analyze(data, query=no_assumption)

    with pytest.raises(CausalValueError, match="conditional delayed entry is unsupported"):
        conditional = query_type(
            *args,
            randomized=True,
            delayed_entry="entry",
            observation_assumption=IndependentGiven(("z",)),
        )
        antecedent.analyze(data, query=conditional)


def test_delayed_entry_refuses_invalid_intervals_and_unanchored_origin():
    query = antecedent.survival.SurvivalOutcome(
        "duration",
        "event",
        "treated",
        4,
        randomized=True,
        delayed_entry="entry",
        observation_assumption=IndependentGiven(()),
    )
    invalid = _survival_data()
    invalid.loc[0, "entry"] = invalid.loc[0, "duration"]
    with pytest.raises(CausalUnsupportedError, match="earlier than exit"):
        antecedent.analyze(invalid, query=query)

    negative_entry = _survival_data()
    negative_entry.loc[0, "entry"] = -0.5
    with pytest.raises(CausalUnsupportedError, match="entry times must be finite, nonnegative"):
        antecedent.analyze(negative_entry, query=query)

    unanchored = _survival_data()
    unanchored.loc[unanchored["treated"] == 0, "entry"] = 0.25
    with pytest.raises(CausalUnsupportedError, match="time-zero entrant in each arm"):
        antecedent.analyze(unanchored, query=query)
