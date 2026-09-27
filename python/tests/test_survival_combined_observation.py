"""Retained delayed entry with fixed known censoring survival."""

from __future__ import annotations

import antecedent as ant
import pandas as pd
import pytest
from antecedent.errors import CausalValueError
from antecedent.observation import IndependentGiven
from antecedent.survival import CompetingRisksOutcome, KnownCensoringSurvival, SurvivalOutcome


def _data() -> pd.DataFrame:
    rows = []
    for arm in (0, 1):
        for i in range(80):
            entry = 1.0 if i % 3 == 0 else 0.0
            if entry == 0.0 and i % 7 == 0:
                duration, cause = 1.0, 1
            elif i % 11 == 0:
                duration, cause = 1.5, 0
            elif i % 5 == 0:
                duration, cause = 2.0, 2
            elif i % (4 if arm else 3) == 0:
                duration, cause = 2.0, 1
            else:
                duration, cause = 3.0, 0
            rows.append((duration, cause, int(cause > 0), arm, entry,
                         1.0, 1.0, 1.0, 0.8, 0.8))
    return pd.DataFrame(rows, columns=(
        "duration", "cause", "event", "treated", "entry",
        "g0", "g1", "g15", "g2", "g3",
    ))


def _g() -> KnownCensoringSurvival:
    return KnownCensoringSurvival(
        (0.0, 1.0, 1.5, 2.0, 3.0), ("g0", "g1", "g15", "g2", "g3")
    )


@pytest.mark.parametrize("competing", [False, True])
def test_combined_entry_fixed_g_pointwise_interval_round_trip(competing: bool) -> None:
    data = _data()
    if competing:
        query = CompetingRisksOutcome(
            "duration", "cause", "treated", 1, 3.0, randomized=True,
            delayed_entry="entry", known_censoring=_g(),
            observation_assumption=IndependentGiven(()),
        )
    else:
        data.loc[data["cause"] == 2, "event"] = 1
        query = SurvivalOutcome(
            "duration", "event", "treated", 3.0, randomized=True,
            delayed_entry="entry", known_censoring=_g(),
            observation_assumption=IndependentGiven(()),
        )
    result = ant.analyze(data, query=query, bootstrap=299, seed=297)
    section = result.survival
    assert section is not None
    assert section.uncertainty == "subject_stratified_percentile_bootstrap_pointwise_95"
    assert section.censoring_survival_provenance == "caller_supplied_fixed_not_fitted_or_verified"
    assert section.difference_band is None
    assert "delayed entry" in section.band_unavailable_reason
    assert section.bootstrap_replicates_ok == 299
    if competing:
        assert section.incidence_difference_interval is not None
    else:
        assert section.rmst_difference_interval is not None
        assert section.survival_at_tau_difference_interval is not None
    body = ant.load(result.export()).artifact.payload
    assert body["query"]["survival"]["delayed_entry"] is not None
    assert len(body["query"]["survival"]["censoring_columns"]) == 5
    assert body["survival"]["difference_at_tau_interval"] is not None


def test_combined_entry_fixed_g_refuses_conditional_entry_and_bad_g() -> None:
    with pytest.raises(CausalValueError, match="conditional delayed entry"):
        SurvivalOutcome(
            "duration", "event", "treated", 3.0, randomized=True,
            delayed_entry="entry", known_censoring=_g(),
            observation_assumption=IndependentGiven(("baseline",)),
        )
    data = _data()
    data.loc[data["cause"] == 2, "event"] = 1
    data.loc[0, "g2"] = 0.0
    query = SurvivalOutcome(
        "duration", "event", "treated", 3.0, randomized=True,
        delayed_entry="entry", known_censoring=_g(),
        observation_assumption=IndependentGiven(()),
    )
    with pytest.raises(Exception, match="positivity|censoring survival"):
        ant.analyze(data, query=query, bootstrap=299, seed=297)
