"""The direct DiD utilities warn, and the compiled panel DiD queries reproduce them."""

from __future__ import annotations

import antecedent as ant
import pytest
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.outcomes import require_success
from antecedent.quasi import (
    DifferenceInDifferences,
    PanelDifferenceInDifferences,
    StaggeredAdoption,
    estimate_did,
    estimate_group_time_att,
)


def _staggered_panel():
    ids, periods, cohorts, outcome = [], [], [], []
    for cohort_index, cohort in enumerate((0, 3, 4)):
        for unit in range(8):
            for period in range(1, 6):
                wobble = (((unit * 7 + period * 13 + cohort_index * 5) % 11) - 5) * 0.173
                effect = 1.5 + 0.25 * (period - cohort) if cohort and period >= cohort else 0.0
                ids.append(f"u{cohort_index}-{unit}")
                periods.append(period)
                cohorts.append(cohort)
                outcome.append(2.0 + 0.4 * unit + 1.1 * period + wobble + effect)
    return {"y": outcome, "id": ids, "period": periods, "cohort": cohorts}


def test_group_time_att_direct_path_warns_and_the_event_study_reproduces_every_cell():
    data = _staggered_panel()
    query = StaggeredAdoption("y", "id", "period", "cohort")
    with pytest.warns(AntecedentDeprecationWarning, match="estimate_group_time_att"):
        direct = estimate_group_time_att(data, query)
    compiled = require_success(
        ant.analyze(data, query=StaggeredAdoption("y", "id", "period", "cohort", event_study=True))
    ).panel_did
    post = [effect for effect in compiled.effects if effect.event_time >= 0]
    assert [(e.cohort, e.period) for e in post] == [(e.cohort, e.period) for e in direct.effects]
    for got, want in zip(post, direct.effects, strict=True):
        assert got.estimate == want.estimate
        assert got.standard_error == want.standard_error
        assert got.treated_subjects == want.treated_subjects
        assert got.control_subjects == want.control_subjects
        assert got.clusters == want.clusters


def test_group_time_att_selected_cell_matches_the_direct_cell():
    data = _staggered_panel()
    with pytest.warns(AntecedentDeprecationWarning):
        direct = estimate_group_time_att(data, StaggeredAdoption("y", "id", "period", "cohort"))
    cell = next(e for e in direct.effects if (e.cohort, e.period) == (3, 4))
    selected = require_success(
        ant.analyze(
            data,
            query=StaggeredAdoption(
                "y", "id", "period", "cohort", target_cohort=3, target_period=4
            ),
        )
    ).panel_did
    assert selected.estimate == cell.estimate
    assert selected.standard_error == cell.standard_error


def test_did_direct_path_warns_and_the_repeated_cross_section_query_matches():
    n = 80
    treated = [bool((i // 2) % 2) for i in range(n)]
    post = [bool(i % 2) for i in range(n)]
    outcome = [
        10.0 + 3.0 * post[i] + 4.0 * (treated[i] and post[i]) + 0.1 * ((i * 7) % 13)
        for i in range(n)
    ]
    data = {
        "y": outcome,
        "unit": [f"u{i}" for i in range(n)],
        "group": treated,
        "after": post,
    }
    with pytest.warns(AntecedentDeprecationWarning, match="estimate_did"):
        direct = estimate_did(data, DifferenceInDifferences("y", "group", "after"))
    compiled = require_success(
        ant.analyze(
            data,
            query=PanelDifferenceInDifferences.repeated_cross_section(
                "y", "unit", "group", "after"
            ),
        )
    ).panel_did
    assert compiled.estimate == direct.estimate
