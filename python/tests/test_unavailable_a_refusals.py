"""Declarations the design routes cannot read are refused by name instead of ignored.

A treatment with a third level is refused by every route whose treatment enters as a boolean
assignment, and the static spellings of designs that need a time dimension (synthetic control,
staggered adoption, the augmented two-period panel) are refused on a table without one.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import interference
from antecedent.errors import CausalCompileError, CausalError, CausalValueError
from antecedent.outcomes import require_success
from antecedent.quasi import (
    AugmentedPanelDiD,
    DifferenceInDifferences,
    PanelDifferenceInDifferences,
    StaggeredAdoption,
    StaggeredRollout,
    SyntheticControl,
    estimate_did,
    estimate_group_time_att,
)

THREE_LEVELS = [0, 1, 2, 1]


@pytest.mark.parametrize("levels", [THREE_LEVELS, np.array(THREE_LEVELS)])
def test_two_arm_factorial_and_switchback_refuse_a_third_treatment_level(levels):
    with pytest.raises(CausalValueError, match="realized_assignment entries must be bool"):
        ant.experiment.ExperimentDesign(
            interference.BernoulliAssignment(0.5),
            levels,
            [f"a{i}" for i in range(4)],
            [f"r{i}" for i in range(4)],
        )
    with pytest.raises(CausalValueError, match="second_factor_assignment entries must be bool"):
        ant.experiment.FactorialRandomization(levels, (1, 1, 1, 1))
    with pytest.raises(CausalValueError, match="switchback realized_assignment entries must be bool"):
        ant.experiment.SwitchbackDesign(
            levels, ["s0", "s0", "s1", "s1"], ["p0", "p1", "p0", "p1"], [0.5] * 4
        )
    # The same designs run when the assignment is boolean.
    ant.experiment.SwitchbackDesign(
        [True, False, False, True], ["s0", "s0", "s1", "s1"], ["p0", "p1", "p0", "p1"], [0.5] * 4
    )


def test_every_difference_in_differences_form_refuses_a_third_treatment_level():
    with pytest.raises(CausalValueError, match="treated values must be bool or encoded as 0/1"):
        estimate_did(
            {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
             "group": [0, 0, 1, 1, 2, 2, 0, 1],
             "after": [0, 1, 0, 1, 0, 1, 1, 0]},
            DifferenceInDifferences("y", "group", "after"),
        )
    subjects = [f"s{i}" for i in range(8) for _ in range(2)]
    post = [p for _ in range(8) for p in (False, True)]
    group = [i % 3 for i in range(8) for _ in range(2)]
    with pytest.raises(CausalError, match="group values must be bool or encoded as 0/1"):
        require_success(
            ant.analyze(
                {"y": [float(i) for i in range(16)], "id": subjects, "group": group, "after": post},
                query=PanelDifferenceInDifferences("y", "id", "group", "after"),
            )
        )
    n = 60
    pre = np.linspace(0.0, 1.0, n)
    data = {
        "id": [f"s{i}" for i in range(n)],
        "pre": pre,
        "post": pre + 1.0,
        "treated": [i % 3 for i in range(n)],
        "p": np.full(n, 0.5),
        "m0": np.full(n, 1.0),
    }
    with pytest.raises(CausalError, match="treated values must be bool or encoded as 0/1"):
        require_success(
            ant.analyze(data, query=AugmentedPanelDiD("pre", "post", "id", "treated", "p", "m0", True))
        )
    data["treated"] = [i % 2 for i in range(n)]
    result = require_success(
        ant.analyze(data, query=AugmentedPanelDiD("pre", "post", "id", "treated", "p", "m0", True))
    )
    assert result.panel_did.estimate == pytest.approx(0.0, abs=1e-9)


def test_augmented_panel_refuses_a_repeated_cross_section_and_missing_period_outcomes():
    n = 60
    pre = np.linspace(0.0, 1.0, n)
    base = {
        "pre": pre,
        "post": pre + 1.0,
        "treated": [i % 2 for i in range(n)],
        "p": np.full(n, 0.5),
        "m0": np.full(n, 1.0),
    }
    query = AugmentedPanelDiD("pre", "post", "id", "treated", "p", "m0", True)
    # Subjects seen once per period (a repeated cross-section) share labels across rows.
    with pytest.raises(CausalValueError, match="one unique row per subject"):
        require_success(ant.analyze({**base, "id": [f"s{i // 2}" for i in range(n)]}, query=query))
    # A unit with no pre-period outcome has nothing to adjust its change by.
    missing = pre.copy()
    missing[::2] = np.nan
    with pytest.raises(CausalError, match="missing_data.policy_required"):
        require_success(
            ant.analyze(
                {**base, "pre": missing, "post": missing + 1.0, "id": [f"s{i}" for i in range(n)]},
                query=query,
            )
        )


def test_synthetic_control_and_staggered_adoption_are_refused_on_a_single_period_table():
    rows = {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "unit": ["t", "a", "b", "c", "d", "e"], "period": [1] * 6}
    for intervention in (1, 2):
        with pytest.raises(
            CausalCompileError,
            match="synthetic control requires at least two pre-periods and one post-period",
        ):
            require_success(
                ant.analyze(rows, query=SyntheticControl("y", "unit", "period", "t", intervention))
            )
    staggered = {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "id": list("abcdef"),
        "period": [1] * 6,
        "cohort": [0, 0, 0, 2, 2, 2],
    }
    with pytest.raises(CausalValueError, match="staggered group-time ATT requires at least two periods"):
        estimate_group_time_att(staggered, StaggeredAdoption("y", "id", "period", "cohort"))
    with pytest.raises(CausalError, match="a rollout needs at least two periods"):
        require_success(ant.analyze(staggered, query=StaggeredRollout("y", "id", "period", "cohort")))


def test_a_one_period_switchback_is_refused_because_it_never_switches_back():
    assignment = [True, False] * 4
    design = ant.experiment.SwitchbackDesign(
        assignment, [f"s{i}" for i in range(8)], ["p0"] * 8, [0.5] * 8
    )
    with pytest.raises(CausalError, match="never switches back") as caught:
        require_success(
            ant.analyze(
                {"outcome": np.arange(8.0)}, query=ant.experiment.SwitchbackEffect("outcome", design)
            )
        )
    assert caught.value.reason_code == "option_not_applicable"
    assert "Next attempt" in str(caught.value)
    # The same eight rows are a Bernoulli trial: the static randomized effect is the route for them.
    # With a second period for each sequence the design switches back and runs.
    switching = [True, False, False, True, True, False, False, True]
    design = ant.experiment.SwitchbackDesign(
        switching,
        ["s0", "s0", "s1", "s1", "s2", "s2", "s3", "s3"],
        ["p0", "p1"] * 4,
        [0.5] * 8,
    )
    # Outcome = row index plus a constant 2 for every treated row: with balanced arms (each
    # arm's index mean is 3.5) the difference of arm means is exactly 2.
    outcome = np.arange(8.0) + 2.0 * np.array(switching)
    fit = require_success(
        ant.analyze({"outcome": outcome}, query=ant.experiment.SwitchbackEffect("outcome", design))
    ).randomized_effect
    assert fit.assignment_design == "switchback"
    assert fit.effect == pytest.approx(2.0, abs=1e-12)
