"""The same decision fixtures in Python as `apply_decision` in Rust.

The table is ``a in {0, 0, 1, 1}``, ``y in {0, 0, 2, 2}``, so the static average effect is 2.
The response is the line ``y = 0.5 + 1.5 t`` on doses 0.5 and 1.5. Refusals use the same
codes the Rust projection returns.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import AverageEffect, Frequentist, ResponseCurve, analyze, decision, prepare
from antecedent import _reason_details as reason_details
from antecedent.errors import CausalValueError
from antecedent.external import ExternalRefusal
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success
from antecedent.results._report import copy_model

UNITS = (
    "effect units must be declared (at most 256 bytes) and population must be "
    "target (the original observed sample)"
)


def _binary():
    return {
        "a": np.array([0.0, 0.0, 1.0, 1.0]),
        "y": np.array([0.0, 0.0, 2.0, 2.0]),
    }


def _curve():
    t = np.array([2.0 * i / 39.0 for i in range(40)])
    return {"t": t, "y": 0.5 + 1.5 * t}


def _effect(result: object) -> ScientificQuantity:
    return ScientificQuantity.from_effect(result, outcome_units="mmHg")


def _contrast(result, criterion):
    effect = _effect(result)
    return decision.Contract(
        (
            decision.Action("act", (effect,), decision.x(0)),
            decision.Action("wait", (effect,), decision.x(0) * 0.0),
        ),
        "mmHg",
        criterion,
        "target",
    )


def test_static_average_effect_is_the_observed_mean_contrast() -> None:
    result = require_success(
        analyze(
            _binary(),
            query=AverageEffect("a", "y"),
            graph=[("a", "y")],
            refute="none",
            inference=Frequentist(bootstrap=0),
        )
    )
    decided = require_success(
        _contrast(result, decision.Criterion.expected_utility()).evaluate(result)
    )
    assert result.estimate.ate == pytest.approx(2.0)
    assert decided.selected == ("act",)
    assert decided.actions[0].expected_utility == pytest.approx(2.0)
    assert decided.actions[1].expected_utility == pytest.approx(0.0)
    assert decided.evpi is None


def test_changed_effect_quantile_and_declaration_refuse() -> None:
    result = require_success(
        analyze(
            _binary(),
            query=AverageEffect("a", "y"),
            graph=[("a", "y")],
            refute="none",
            inference=Frequentist(bootstrap=0),
        )
    )
    changed = copy_model(result, estimate=copy_model(result.estimate, ate=99.0))
    with pytest.raises(ExternalRefusal) as mismatched:
        require_success(_contrast(changed, decision.Criterion.expected_utility()).evaluate(changed))
    assert mismatched.value.reason_code == "invalid_argument"
    assert mismatched.value.detail == reason_details.native_claims.PROJECTION_MISMATCH

    with pytest.raises(decision.DecisionRefusal) as quantile:
        require_success(
            _contrast(result, decision.Criterion.quantile(0.5)).evaluate(result)
        )
    assert quantile.value.reason_code == "decision_contract_unsatisfied"
    assert quantile.value.detail == reason_details.decision_evaluation.MEAN_SOURCE_INSUFFICIENT

    for units, population in (("", "target"), ("mmHg", "other")):
        with pytest.raises(CausalValueError) as declaration:
            result._raw.effect_source_json(units, population, result.as_point())
        assert str(declaration.value) == UNITS


def test_response_uses_the_native_mean_or_refuses_a_law() -> None:
    result = require_success(
        analyze(
            _curve(),
            query=ResponseCurve("t", "y", grid=[0.5, 1.5]),
            graph=[("t", "y")],
            refute="none",
            inference=Frequentist(bootstrap=0),
            outcome_units="mmHg",
        )
    )
    quantity = ScientificQuantity(
        variable_id="y",
        variable_name="y",
        role="outcome",
        units="mmHg",
        population_id="target",
        regime_id="do(t=0.5)",
        horizon=0,
        functional_id="mean",
    )
    high = ScientificQuantity(
        variable_id="y",
        variable_name="y",
        role="outcome",
        units="mmHg",
        population_id="target",
        regime_id="do(t=1.5)",
        horizon=0,
        functional_id="mean",
    )
    contract = decision.Contract(
        (
            decision.Action("low", (quantity,), decision.x(0)),
            decision.Action("high", (high,), decision.x(0)),
        ),
        "mmHg",
        decision.Criterion.expected_utility(),
        "target",
    )
    decided = require_success(contract.evaluate(result))
    assert decided.selected == ("high",)
    law = decision.Contract(
        contract.actions,
        "mmHg",
        decision.Criterion.quantile(0.5),
        "target",
    )
    with pytest.raises(ExternalRefusal) as refused:
        require_success(law.evaluate(result))
    assert refused.value.reason_code == "decision_contract_unsatisfied"
    assert refused.value.detail == reason_details.native_claims.SOURCE_NOT_SUPPLIED


def test_supplied_child_of_another_program_is_not_adopted() -> None:
    data = _curve()
    prepared = prepare(
        data,
        graph=[("t", "y")],
        query=ResponseCurve("t", "y", grid=[0.5, 1.5]),
        refute="none",
        inference=Frequentist(bootstrap=0),
    )
    quantity = ScientificQuantity(
        variable_id="y",
        variable_name="y",
        role="outcome",
        units="mmHg",
        population_id="target",
        regime_id="do(t=0.5)",
        horizon=0,
        functional_id="mean",
    )
    high = ScientificQuantity(
        variable_id="y",
        variable_name="y",
        role="outcome",
        units="mmHg",
        population_id="target",
        regime_id="do(t=1.5)",
        horizon=0,
        functional_id="mean",
    )
    contract = decision.Contract(
        (
            decision.Action("low", (quantity,), decision.x(0)),
            decision.Action("high", (high,), decision.x(0)),
        ),
        "mmHg",
        decision.Criterion.expected_utility(),
        "target",
    )
    compiled = require_success(contract.compile(prepared))
    foreign = require_success(
        analyze(
            data,
            query=ResponseCurve("t", "y", grid=[0.5, 1.5]),
            graph=[("t", "y")],
            refute="none",
            inference=Frequentist(bootstrap=0),
            outcome_units="mmHg",
        )
    )
    with pytest.raises(decision.DecisionRefusal) as refused:
        compiled.execute_supplied([foreign])
    assert refused.value.reason_code == "route_not_lowered"
    assert refused.value.detail == reason_details.preference_compile.EXECUTED_PROGRAM_MISMATCH
    with pytest.raises(decision.DecisionRefusal) as counted:
        compiled.execute_supplied([])
    assert counted.value.detail == reason_details.preference_compile.EXECUTED_PROGRAM_MISMATCH
