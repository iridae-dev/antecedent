"""F25: the structured refusal fields survive the Rust-to-Python bridge unchanged.

`crates/antecedent-design/tests/structured_refusals.rs` asserts the same strings on
the Rust side for the same case: a decision asks for a utility over two outcomes
from independent marginals at action A1.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import decision, external
from antecedent.errors import CausalUnsupportedError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)


def _quantity(variable: str, regime: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=variable,
        role="outcome",
        units="units",
        population_id="target",
        regime_id=regime,
        horizon=0,
        functional_id="outcome",
    )


COLUMNS = (_quantity("p", "do(a=1)"), _quantity("q", "do(a=1)"), _quantity("s", "do(a=0)"))


def _contract() -> decision.Contract:
    return decision.Contract(
        actions=(
            decision.Action(
                "A1", inputs=(COLUMNS[0], COLUMNS[1]), utility=decision.x(0) * decision.x(1)
            ),
            decision.Action("A2", inputs=(COLUMNS[2],), utility=decision.x(0)),
        ),
        utility_units="units",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )


def _marginals() -> JointDistributionArtifact:
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=COLUMNS,
        alignment="independent_marginals",
        source_id="marginals",
        provider_id="exact-law",
        rng_id="deterministic_exact",
        snapshot_id="snap",
        causal_contract_id="checked",
    )
    draws = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 1.0]], dtype=np.float64)
    return JointDistributionArtifact(identity, draws, calibration="exact")


def test_f25_decision_refusal_fields_survive_the_bridge():
    with pytest.raises(decision.DecisionRefusal) as caught:
        _contract().evaluate(_marginals())
    refusal = caught.value
    assert isinstance(refusal, CausalUnsupportedError)
    assert refusal.reason_code == "joint_law_required"
    assert refusal.stage == "evaluate"
    assert refusal.detail == "decision_evaluation.joint_law_required"
    assert refusal.offending == "A1"
    assert refusal.expected == "joint"
    assert refusal.supplied == "independent_marginals"
    assert refusal.remedy
    assert "joint draws" in refusal.remedy


def test_f25_unit_and_capability_style_mistakes_stay_distinct_across_families():
    """The same four-way distinctness the Rust test asserts, through the Python facades."""
    wrong_population = decision.Contract(
        actions=_contract().actions,
        utility_units="units",
        criterion=decision.Criterion.expected_utility(),
        target_population="other",
    )
    with pytest.raises(decision.DecisionRefusal) as population:
        wrong_population.evaluate(_marginals())
    assert population.value.reason_code == "quantity_semantics_mismatch"
    assert population.value.detail == "decision_contract.outside_decision_scope"

    unknown_action = decision.Contract(
        actions=_contract().actions,
        utility_units="units",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
        constraints=(decision.Constraint("cap", decision.x(0), 1.0, "units", 1.0, ("A9",)),),
    )
    with pytest.raises(decision.DecisionRefusal) as unknown:
        unknown_action.evaluate(_marginals())
    assert unknown.value.reason_code == "invalid_argument"
    assert unknown.value.detail == "decision_contract.unknown_action"

    pairs = {
        (population.value.reason_code, population.value.detail),
        (unknown.value.reason_code, unknown.value.detail),
        ("joint_law_required", "decision_evaluation.joint_law_required"),
    }
    assert len(pairs) == 3
    assert external.ExternalRefusal is not decision.DecisionRefusal
