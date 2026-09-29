from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalValueError


def test_top_k_policy_is_deterministic_and_respects_action_availability():
    policy = ant.policy.BinaryPolicy.top_k(
        [0.5, 1.0, 1.0, 3.0], 2, available=[True, True, False, True]
    )
    assert policy.actions == (False, True, False, True)


def test_binary_policy_evaluation_uses_heldout_randomized_rows_and_costs():
    assignment = [True, False, True, False]
    outcome = 5.0 + 2.0 * np.asarray(assignment, dtype=np.float64)
    policy = ant.policy.BinaryPolicy([True, True, False, False], costs=0.5)

    result = ant.policy.evaluate_policy(
        {"y": outcome},
        outcome="y",
        assignment=assignment,
        propensity=0.5,
        policy=policy,
    )

    assert result.policy_value == pytest.approx(5.75)
    assert result.reference_value == pytest.approx(5.0)
    assert result.incremental_value == pytest.approx(0.75)
    assert result.relative_value_gap == pytest.approx(-0.75)
    assert result.treatment_rate == pytest.approx(0.5)
    assert result.total_treatment_cost == pytest.approx(1.0)
    assert result.uncertainty == "point_only"


@pytest.mark.parametrize(
    ("policy", "kwargs", "message"),
    [
        (
            ant.policy.BinaryPolicy([True, True, False, False], max_treatment_rate=0.25),
            {},
            "capacity",
        ),
        (ant.policy.BinaryPolicy([True, True, False, False], costs=0.5, budget=0.5), {}, "budget"),
        (
            ant.policy.BinaryPolicy([True, True, False, False]),
            {"available": [False, True, True, True]},
            "unavailable",
        ),
        (ant.policy.BinaryPolicy([True, True, False, False]), {"propensity": 1.0}, "propensities"),
    ],
)
def test_binary_policy_constraints_and_support_fail_closed(policy, kwargs, message):
    with pytest.raises((CausalValueError, ValueError), match=message):
        ant.policy.evaluate_policy(
            {"y": [1.0, 2.0, 3.0, 4.0]},
            outcome="y",
            assignment=[True, False, True, False],
            propensity=kwargs.pop("propensity", 0.5),
            policy=policy,
            **kwargs,
        )


def test_binary_policy_rejects_non_binary_recommendations_and_row_mismatch():
    with pytest.raises(CausalValueError, match="bool values"):
        ant.policy.BinaryPolicy([0, 1])  # type: ignore[list-item]
    with pytest.raises(CausalValueError, match="match evaluation row count"):
        ant.policy.evaluate_policy(
            {"y": [1.0, 2.0]},
            outcome="y",
            assignment=[True, False],
            propensity=0.5,
            policy=ant.policy.BinaryPolicy([True]),
        )


def test_multi_action_policy_native_value_uses_action_propensities_and_reference():
    from antecedent.policy import MultiActionPolicy, evaluate_multi_action_policy

    labels = ("control", "A", "B")
    assigned = ["control", "A", "B"] * 3
    outcomes = [1.0, 2.0, 4.0] * 3
    policy = MultiActionPolicy(labels, ["control"] * 3 + ["A"] * 3 + ["B"] * 3)
    reference = MultiActionPolicy(labels, ["control"] * 9)
    result = evaluate_multi_action_policy(
        {"y": outcomes},
        outcome="y",
        assignment=assigned,
        propensities=np.full((9, 3), 1.0 / 3.0),
        policy=policy,
        reference=reference,
    )
    assert result.policy_value == pytest.approx(7.0 / 3.0)
    assert result.reference_value == pytest.approx(1.0)
    assert result.incremental_value == pytest.approx(4.0 / 3.0)
    assert result.treatment_rate == pytest.approx(2.0 / 3.0)
    assert result.uncertainty == "point_only"


def test_multi_action_top_score_policy_is_stable_and_respects_availability():
    from antecedent.policy import MultiActionPolicy

    policy = MultiActionPolicy.from_scores(
        [[0.4, 0.6, 0.6], [0.8, 0.7, 0.1]],
        ("control", "A", "B"),
        available=[[True, True, False], [True, True, True]],
    )
    assert policy.recommendations == ("A", "control")


def test_multi_action_policy_refuses_positivity_availability_and_capacity_failures():
    from antecedent.policy import MultiActionPolicy, evaluate_multi_action_policy

    policy = MultiActionPolicy(("control", "A", "B"), ["A", "A", "B"])
    args = dict(
        evaluation_data={"y": [1.0, 2.0, 3.0]},
        outcome="y",
        assignment=["A", "B", "control"],
        propensities=[[1 / 3] * 3] * 3,
        policy=policy,
    )
    with pytest.raises(ValueError, match="unavailable"):
        evaluate_multi_action_policy(**args, available=[[True, False, True]] * 3)
    with pytest.raises(ValueError, match="capacity"):
        evaluate_multi_action_policy(
            **{
                **args,
                "policy": MultiActionPolicy(
                    ("control", "A", "B"), ["A", "A", "B"], capacities=[3, 1, 3]
                ),
            }
        )
    with pytest.raises(ValueError, match="propensity"):
        evaluate_multi_action_policy(**{**args, "propensities": [[0.0, 0.5, 0.5]] * 3})


def test_uplift_view_recovers_effect_by_heldout_score_rank():
    assignment = [True, False] * 4
    scores = [9, 8, 7, 6, 3, 2, 1, 0]
    effects = np.array([5.0] * 4 + [1.0] * 4)
    outcome = effects * np.asarray(assignment, dtype=float)
    result = ant.policy.uplift_by_score(
        {"y": outcome},
        outcome="y",
        assignment=assignment,
        propensity=0.5,
        scores=scores,
        bins=2,
    )
    assert [item.effect for item in result] == pytest.approx([5.0, 1.0])
    assert [item.evaluation_rows for item in result] == [4, 4]
