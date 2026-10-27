"""Observational policy value against a closed-form truth.

One normal confounder ``z`` drives the treatment and the outcome: ``E[Y(0) | z] = 1 + z`` and
the effect is ``1.5 z``, so the policy "treat when z > 0" has value ``1 + 1.5 phi(0)`` with
``phi(0)`` the standard normal density at zero (``E[z; z > 0] = phi(0)``).
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import Execution, observational_policy
from antecedent.errors import CausalError

PHI_0 = 1.0 / np.sqrt(2.0 * np.pi)
TRUTH = 1.0 + 1.5 * PHI_0


def _draw(n: int = 6000, seed: int = 21) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(float)
    y = 1.0 + z + 1.5 * z * a + 0.5 * rng.normal(size=n)
    return {"y": y, "a": a, "z": z}


def test_a_prespecified_policy_recovers_its_value() -> None:
    data = _draw()
    result = observational_policy.policy_value(
        data["y"],
        data["a"],
        support=[0.0, 1.0],
        objective="maximize",
        evaluation="prespecified",
        policy_actions=(data["z"] > 0).astype(float),
        policy_origin="prespecified",
        reference=0.0,
        adjust={"z": data["z"]},
        execution=Execution(seed=0),
    )
    assert abs(result.value - TRUTH) < 4 * result.se
    assert result.reference is not None and result.reference.level == 0.0
    assert result.calibration == "unmeasured"
    assert result.evaluation == "prespecified"
    assert result.support == (0.0, 1.0)
    assert [a.level for a in result.action_values] == [0.0, 1.0]


def test_known_propensity_replaces_the_fitted_model() -> None:
    rng = np.random.default_rng(9)
    z = rng.normal(size=4000)
    a = (rng.uniform(size=4000) < 0.5).astype(float)
    y = 1.0 + z + 1.5 * z * a + 0.5 * rng.normal(size=4000)
    result = observational_policy.policy_value(
        y,
        a,
        support=[0.0, 1.0],
        objective="maximize",
        evaluation="prespecified",
        policy_actions=(z > 0).astype(float),
        policy_origin="prespecified",
        known_propensity=[0.5, 0.5],
        adjust={"z": z},
    )
    assert result.propensity_source == "known"
    assert abs(result.value - TRUTH) < 4 * result.se
    with pytest.raises(CausalError) as raised:
        observational_policy.policy_value(
            y,
            a,
            support=[0.0, 1.0],
            objective="maximize",
            evaluation="cross_fitted_learning",
            known_propensity=[0.5, 0.6],
            adjust={"z": z},
        )
    assert "policy_value.invalid_known_propensity" in str(raised.value)


def test_held_out_and_cross_fitted_learning_beat_never_treating() -> None:
    data = _draw(8000, seed=22)
    common = {
        "support": [0.0, 1.0],
        "objective": "maximize",
        "adjust": {"z": data["z"]},
        "execution": Execution(seed=3),
    }
    held_out = observational_policy.policy_value(
        data["y"], data["a"], evaluation="held_out", evaluation_fraction=0.4, **common
    )
    honest = observational_policy.policy_value(
        data["y"], data["a"], evaluation="cross_fitted_learning", **common
    )
    for result in (held_out, honest):
        assert result.value > 1.2
        # The optimum's value is 1 + 1.5 phi(0); the learned rule's regret is second order, so the
        # gap is sampling error: within 4 reported standard errors (plus 0.01 for the regret).
        assert abs(result.value - TRUTH) < 4 * result.se + 0.01
    assert held_out.evaluation_rows + held_out.learning_rows == 8000
    assert honest.evaluation_rows == 8000


def test_declarations_have_no_default_and_policy_reuse_is_refused() -> None:
    data = _draw(500, seed=5)
    kwargs = {"adjust": {"z": data["z"]}}
    for missing, detail in (
        ({"objective": "maximize", "evaluation": "cross_fitted_learning"}, "support_missing"),
        ({"support": [0.0, 1.0], "evaluation": "cross_fitted_learning"}, "objective_missing"),
        ({"support": [0.0, 1.0], "objective": "maximize"}, "evaluation_missing"),
    ):
        with pytest.raises(CausalError) as raised:
            observational_policy.policy_value(data["y"], data["a"], **missing, **kwargs)
        assert raised.value.reason_code == "required_option_missing"
        assert f"policy_value.{detail}" in str(raised.value)
    with pytest.raises(CausalError) as raised:
        observational_policy.policy_value(
            data["y"],
            data["a"],
            support=[0.0, 1.0],
            objective="maximize",
            evaluation="prespecified",
            policy_actions=(data["z"] > 0).astype(float),
            policy_origin="learned_on_evaluation_rows",
            **kwargs,
        )
    assert raised.value.reason_code == "policy_learned_on_evaluation_rows"
    with pytest.raises(CausalError) as raised:
        observational_policy.policy_value(
            data["y"],
            data["a"],
            support=[0.0, 1.0],
            objective="maximize",
            evaluation="prespecified",
            policy_actions=np.full(500, 3.0),
            policy_origin="prespecified",
            **kwargs,
        )
    assert raised.value.reason_code == "positivity_violated"
    assert "policy_value.action_outside_support" in str(raised.value)
