"""Scalar and sequence boundaries for public Python analysis inputs."""

import antecedent as ant
import numpy as np
import pytest
from antecedent import data, experiment, interference, policy
from antecedent.derived import check_derived_treatment, factorized_joint_cells
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.outcomes import require_success
from antecedent.survival import CompetingRisksOutcome, SurvivalOutcome


def test_numpy_bool_and_integer_experiment_designs():
    primary = np.array([False, False, True, True, False, False, True, True])
    second = np.array([False] * 4 + [True] * 4)
    factorial = experiment.FactorialRandomization(second, tuple(np.int64(2) for _ in range(4)))
    design = experiment.ExperimentDesign(
        factorial, primary, [f"a{i}" for i in range(8)], [f"r{i}" for i in range(8)]
    )
    assert all(type(v) is bool for v in design.realized_assignment)
    assert all(type(v) is bool for v in factorial.second_factor_assignment)
    assert all(type(v) is int for v in factorial.cell_counts)
    result = require_success(ant.analyze(
        {"outcome": [0.0, 2.0, 2.0, 4.0, 1.0, 3.0, 5.0, 7.0]},
        query=experiment.RandomizedEffect("outcome", design), refute="none",
    ))
    assert result.answer.value == pytest.approx(3.0)
    assert experiment.StratifiedRandomization({"block": np.int64(2)}).treated_per_block == {"block": 2}
    with pytest.raises(CausalValueError):
        experiment.FactorialRandomization(second, (True, 2, 2, 2))
    with pytest.raises(CausalValueError):
        experiment.ExperimentDesign(factorial, [0] * 8, [str(i) for i in range(8)], [str(i) for i in range(8)])


def test_numpy_probability_reaches_retained_randomized_route():
    assignment = np.array([True, False] * 4)
    design = experiment.ExperimentDesign(
        interference.BernoulliAssignment(np.float32(0.5)), assignment,
        [f"a{i}" for i in range(8)], [f"r{i}" for i in range(8)],
    )
    result = require_success(ant.analyze(
        {"outcome": 2.0 * assignment.astype(float)},
        query=experiment.RandomizedEffect("outcome", design), refute="none",
    ))
    assert result.answer.value == pytest.approx(2.0)
    switchback = experiment.SwitchbackDesign(
        np.array([True, False, True, False]), ["a", "a", "b", "b"],
        ["0", "1", "0", "1"], [0.5] * 4,
    )
    assert all(type(v) is bool for v in switchback.realized_assignment)


def test_numpy_policy_scalar_propensity_and_costs_direct_and_retained():
    assignment = [False, True, False, True]
    selected = policy.BinaryPolicy(assignment, costs=np.float32(0.25))
    reference = policy.BinaryPolicy([False] * 4, costs=np.float32(0.5))
    values = {"y": [1.0, 3.0, 1.0, 3.0]}
    direct = policy.evaluate_policy(values, outcome="y", assignment=assignment,
                                    propensity=np.float32(0.5), policy=selected, reference=reference)
    query = policy.PolicyValue("y", assignment, np.float32(0.5), selected,
                               ["a", "b", "c", "d"], reference=reference)
    retained = require_success(ant.analyze(values, query=query, refute="none"))
    assert retained.policy_value.policy_value == pytest.approx(direct.policy_value)
    dr = policy.evaluate_policy_doubly_robust(
        values, outcome="y", assignment=assignment, propensity=np.float32(0.5),
        policy=selected, reference=reference, mu0=[1.0] * 4, mu1=[3.0] * 4,
        evaluation_subject_ids=["a", "b", "c", "d"], training_subject_ids=["train"],
    )
    assert np.isfinite(dr.policy_value)


@pytest.mark.parametrize("tau", [np.float32(2), np.int64(2)])
def test_numpy_survival_tau_is_native_float(tau):
    outcome = SurvivalOutcome("duration", "event", "treated", tau=tau)
    competing = CompetingRisksOutcome("duration", "cause", "treated", 1, tau=tau)
    assert type(outcome.tau) is float and outcome.tau == 2
    assert type(competing.tau) is float and competing.tau == 2


@pytest.mark.parametrize("key", [1.2, 1.7, "A", "1", True, np.bool_(True)])
def test_panel_rejects_noninteger_mapping_ids(key):
    with pytest.raises(CausalValueError, match="unit IDs must be integers"):
        data.panel({key: {"y": [1.0, 2.0]}})


def test_panel_preserves_numpy_integer_ids():
    frame = data.panel({np.int64(7): {"y": [1.0, 2.0]}})
    assert frame.unit_ids == (7,)
    assert type(frame.unit_ids[0]) is int


def test_derived_names_cannot_be_bare_strings():
    with pytest.raises(CausalTypeError, match="adjustment"):
        check_derived_treatment({}, None, outcome="y", adjustment="zw")
    for argument, value in (("adjustment", "zw"), ("ordering", "tw"), ("contrasts", "interaction")):
        kwargs = {"adjustment": [], argument: value}
        with pytest.raises(CausalTypeError, match=argument):
            factorized_joint_cells({}, None, outcome="y", **kwargs)
