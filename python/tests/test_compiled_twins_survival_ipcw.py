"""The direct IPCW survival utilities warn, and known-censoring ``analyze`` reproduces them."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.observation import IndependentGiven
from antecedent.outcomes import require_success
from antecedent.survival import (
    CompetingRisksOutcome,
    KnownCensoringSurvival,
    SurvivalOutcome,
    estimate_cumulative_incidence_ipcw,
    estimate_survival_ipcw,
)

N = 120
GRID = (0.0, 1.0, 2.0, 3.0)
TAU = 3.0


def _fixture():
    treatment, duration, cause, g = [], [], [], []
    for i in range(N):
        d = (1.0, 1.0, 2.0, 3.0, 3.0)[(i // 2) % 5]
        if d >= TAU or d == 2.0 and (i // 2) % 3 == 0:
            c = 0
        elif (i // 2) % 4 == 0:
            c = 2
        else:
            c = 1
        slope = 0.01 * (i % 5)
        g1 = 0.92 - slope
        g2 = g1 - 0.05 - slope / 2.0
        g3 = g2 - 0.04
        treatment.append(float(i % 2))
        duration.append(d)
        cause.append(float(c))
        g.append([1.0, g1, g2, g3])
    return treatment, duration, cause, np.array(g)


def _table(treatment, duration, cause, g):
    return {
        "duration": duration,
        "event": [float(c > 0) for c in cause],
        "cause": cause,
        "treatment": treatment,
        **{f"g{j}": g[:, j].tolist() for j in range(len(GRID))},
    }


def test_survival_direct_path_warns_and_known_censoring_analyze_matches_bit_for_bit():
    treatment, duration, cause, g = _fixture()
    table = _table(treatment, duration, cause, g)
    direct_query = SurvivalOutcome("duration", "event", "treatment", TAU, randomized=True)
    with pytest.warns(AntecedentDeprecationWarning, match="estimate_survival_ipcw"):
        direct = estimate_survival_ipcw(table, direct_query, times=GRID, censoring_survival=g)
    compiled_query = SurvivalOutcome(
        "duration",
        "event",
        "treatment",
        TAU,
        randomized=True,
        observation_assumption=IndependentGiven(()),
        known_censoring=KnownCensoringSurvival(GRID, tuple(f"g{j}" for j in range(len(GRID)))),
    )
    compiled = require_success(ant.analyze(table, query=compiled_query, refute="none")).survival
    assert compiled.times == direct.times
    assert compiled.control_survival == direct.control_survival
    assert compiled.treated_survival == direct.treated_survival
    assert compiled.rmst_control == direct.rmst_control
    assert compiled.rmst_treated == direct.rmst_treated
    assert compiled.rmst_difference == direct.rmst_difference
    assert compiled.uncertainty == "point_only_no_interval"


def test_cumulative_incidence_direct_path_warns_and_known_censoring_analyze_matches():
    treatment, duration, cause, g = _fixture()
    table = _table(treatment, duration, cause, g)
    direct_query = CompetingRisksOutcome(
        "duration", "cause", "treatment", target_cause=1, tau=TAU, randomized=True
    )
    with pytest.warns(AntecedentDeprecationWarning, match="estimate_cumulative_incidence_ipcw"):
        direct = estimate_cumulative_incidence_ipcw(
            table, direct_query, times=GRID, censoring_survival=g
        )
    compiled_query = CompetingRisksOutcome(
        "duration",
        "cause",
        "treatment",
        target_cause=1,
        tau=TAU,
        randomized=True,
        observation_assumption=IndependentGiven(()),
        known_censoring=KnownCensoringSurvival(GRID, tuple(f"g{j}" for j in range(len(GRID)))),
    )
    compiled = require_success(ant.analyze(table, query=compiled_query, refute="none")).survival
    assert compiled.times == direct.times
    assert compiled.control_incidence == direct.control_incidence
    assert compiled.treated_incidence == direct.treated_incidence
    assert compiled.incidence_difference == direct.incidence_difference
