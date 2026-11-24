"""Every decision result reads the same way: ``choice``, ``kind``, ``standing``, ``statement``.

The fixture is the two-action benefit/harm law of ``test_decision_value_contract``: treat leads
below a harm weight of 2 and wait above it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from antecedent import decision, decision_robust
from antecedent._decision_result import DecisionResult
from antecedent.decision_robust import Claim
from antecedent.errors import CausalValueError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.outcomes import require_success
from antecedent.results.sources import MeanSource


def _quantity(variable: str, units: str, regime: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=variable,
        role="outcome",
        units=units,
        population_id="target",
        regime_id=regime,
        horizon=0,
        functional_id="outcome",
    )


TREAT_BENEFIT = _quantity("benefit", "qaly", "do(a=1)")
TREAT_HARM = _quantity("harm", "events", "do(a=1)")
WAIT_BENEFIT = _quantity("benefit", "qaly", "do(a=0)")
WAIT_HARM = _quantity("harm", "events", "do(a=0)")
COLUMNS = (TREAT_BENEFIT, TREAT_HARM, WAIT_BENEFIT, WAIT_HARM)
ROWS = [[1.0, 0.0, 1.0, 0.0], [3.0, 1.0, 1.0, 0.0]]
BENEFIT = decision.attribute("benefit", units="qaly")
HARM = decision.attribute("harm", units="events")

# The standing of every native kind, by the family each kind belongs to.
STANDING_OF_KIND = {
    "uniquely_optimal": "best",
    "indistinguishable": "tied",
    "no_admissible_action": "no_admissible_action",
    "structurally_robust": "best",
    "support_robust": "best",
    "support_dependent": "depends",
    "graph_dependent_choice": "depends",
    "worst_case_choice": "policy_choice",
    "bayes_choice": "policy_choice",
    "unsupported_extrapolation": "insufficient_evidence",
    "insufficient_claims": "insufficient_evidence",
    "report_only": "report_only",
    "necessarily_best": "best",
    "never_beaten": "best",
    "tied": "tied",
    "preference_dependent": "depends",
    "structures_disagree": "depends",
    "per_sampled_member": "depends",
    "minimax_regret_choice": "policy_choice",
    "measure_choice": "policy_choice",
    "undetermined": "undetermined",
    "undetermined_on_sampled_measure": "undetermined",
    "insufficient_science": "insufficient_evidence",
    "best_on_sampled_measure": "sampled_estimate",
    "best_on_every_sampled_member": "sampled_estimate",
}


def _source(rows: list[list[float]] = ROWS) -> JointDistributionArtifact:
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=COLUMNS,
        alignment="joint",
        source_id="enumerated",
        provider_id="exact-law",
        rng_id="deterministic_exact",
        snapshot_id="enumeration-1",
        causal_contract_id="checked-contract",
    )
    return JointDistributionArtifact(
        identity, np.array(rows, dtype=np.float64), calibration="exact"
    )


def _actions() -> tuple[decision.Action, ...]:
    return (
        decision.Action("treat", inputs={"benefit": TREAT_BENEFIT, "harm": TREAT_HARM}),
        decision.Action("wait", inputs={"benefit": WAIT_BENEFIT, "harm": WAIT_HARM}),
    )


def _weight(lower: float, upper: float) -> decision.Tradeoff:
    return decision.tradeoff(
        "harm_weight", units="qaly/event", lower=lower, upper=upper, assumption="panel_judgment"
    )


def _contract(value: decision.Value | None = None, **kwargs: Any) -> decision.Contract:
    value = value or decision.Value(BENEFIT - 1.0 * HARM, units="qaly")
    return decision.Contract(actions=_actions(), value=value, target_population="target", **kwargs)


def _ranged(lower: float = 0.5, upper: float = 4.0, **kwargs: Any) -> decision.Contract:
    return _contract(decision.Value(BENEFIT - _weight(lower, upper) * HARM, units="qaly"), **kwargs)


def _claims() -> tuple[Claim, ...]:
    stronger = _source([[1.0, 0.0, 1.0, 0.0], [5.0, 1.0, 1.0, 0.0]])
    return (
        Claim.evaluated("direct", _source(), probability=0.6),
        Claim.evaluated("stronger", stronger, probability=0.4),
    )


def _results() -> dict[str, Any]:
    """One result of every class, from every engine."""
    equal = _source([[2.0, 0.0, 2.0, 0.0], [2.0, 0.0, 2.0, 0.0]])
    nothing = decision.Constraint("no_harm_at_all", HARM, bound=-1.0, units="events")
    views = decision.Measure.viewpoints(
        {"clinician": {"harm_weight": 1.0}, "patient": {"harm_weight": 3.0}},
        weights={"clinician": 3.0, "patient": 1.0},
        assumption="stakeholder_weights",
    )
    return {
        "ordinary_best": require_success(_contract().evaluate(_source())),
        "ordinary_tied": require_success(_contract().evaluate(equal)),
        "ordinary_none": require_success(_contract(constraints=(nothing,)).evaluate(_source())),
        "ranged_best": require_success(_ranged(0.5, 1.5).evaluate(_source())),
        "ranged_depends": require_success(_ranged().evaluate(_source())),
        "ranged_worst_case": require_success(
            _ranged(policy=decision.TradeoffPolicy.maximin()).evaluate(_source())
        ),
        "ranged_regret": require_success(
            _ranged(policy=decision.TradeoffPolicy.minimax_regret()).evaluate(_source())
        ),
        "ranged_measure": require_success(
            _ranged(policy=decision.TradeoffPolicy.weighted()).evaluate(_source(), measure=views)
        ),
        "ranged_report": require_success(
            _ranged(policy=decision.TradeoffPolicy.report_only()).evaluate(_source())
        ),
        "ranged_none": require_success(_ranged(constraints=(nothing,)).evaluate(_source())),
        "robust": require_success(
            _contract().evaluate(decision.Structures.weighted_atoms(_claims()))
        ),
    }


def test_every_result_has_choice_statement_kind_standing():
    results = _results()
    for name, result in results.items():
        assert isinstance(result, DecisionResult), name
        assert isinstance(result.kind, str), name
        assert isinstance(result.statement, str) and result.statement, name
        assert result.choice is None or isinstance(result.choice, str), name
        assert result.standing in STANDING_OF_KIND.values(), name
        assert result.contract_identity, name
        assert result.contract is not None, name
        assert result.explain(), name
    # One class per engine; the one base is what they share.
    assert type(results["ordinary_best"]) is decision.Decision
    assert type(results["ranged_best"]) is decision.TradeoffDecision
    assert isinstance(results["robust"], decision_robust.RobustDecision)
    assert isinstance(results["ordinary_best"], DecisionResult)
    assert isinstance(results["ranged_best"], DecisionResult)
    assert isinstance(results["robust"], DecisionResult)

    # The choice is the action a selecting result names, and nothing otherwise.
    assert results["ordinary_best"].choice == "treat"
    assert results["ordinary_best"].selected == ("treat",)
    assert results["ordinary_tied"].choice is None
    assert results["ranged_best"].choice == "treat"
    assert results["ranged_depends"].choice is None
    assert results["ranged_report"].choice is None
    assert results["ranged_measure"].choice == "treat"
    assert results["ranged_worst_case"].choice is not None
    assert results["ranged_regret"].choice is not None
    assert results["robust"].choice == results["robust"].selected
    # Ordinary results keep their own fields beside the shared ones.
    assert results["ordinary_best"].verdict.kind == results["ordinary_best"].kind
    assert results["ordinary_best"].evpi is not None
    assert results["ordinary_best"].effective_tie_tolerance == 0.0
    assert results["robust"].effective_tie_tolerance == 0.0
    # At least the declared 1e-9, widened by the numerical bound of the route.
    assert 1e-9 <= results["ranged_best"].effective_tie_tolerance < 1e-6


def test_standing_of_every_native_kind():
    results = _results()
    seen = set()
    for name, result in results.items():
        # A robust result's ``kind`` is the claim type it adapted; its verdict word is
        # ``verdict.kind``.
        native = (
            result.verdict.kind
            if isinstance(result, decision_robust.RobustDecision)
            else result.kind
        )
        assert native in STANDING_OF_KIND, (name, native)
        assert result.standing == STANDING_OF_KIND[native], (name, native)
        seen.add(native)
    # The fixture reaches every family of the table.
    assert {
        "uniquely_optimal",
        "indistinguishable",
        "no_admissible_action",
        "necessarily_best",
        "preference_dependent",
        "worst_case_choice",
        "minimax_regret_choice",
        "measure_choice",
        "report_only",
    } <= seen
    assert results["ordinary_none"].standing == "no_admissible_action"
    assert results["ordinary_none"].choice is None
    assert results["ranged_none"].standing == "no_admissible_action"
    assert results["ranged_none"].choice is None
    assert results["ranged_report"].standing == "report_only"
    assert results["ordinary_tied"].standing == "tied"


def test_refusal_hierarchy():
    assert issubclass(decision.PreferenceRefusal, decision.DecisionRefusal)

    # A ranged refusal is caught by the ordinary handler and by its own.
    contract = _ranged(policy=decision.TradeoffPolicy.weighted())
    for handler in (decision.DecisionRefusal, decision.PreferenceRefusal):
        with pytest.raises(handler) as refused:
            require_success(contract.evaluate(_source()))
        assert isinstance(refused.value, decision.PreferenceRefusal)
        assert refused.value.reason_code
        assert refused.value.detail

    # An ordinary refusal stays a decision refusal and is not a preference refusal: a mean
    # answers only an affine expectation.
    means = MeanSource(
        coordinates=tuple(
            ScientificQuantity(
                variable_id=q.variable_id,
                variable_name=q.variable_name,
                role="outcome",
                units=q.units,
                population_id="target",
                regime_id=q.regime_id,
                horizon=0,
                functional_id="mean",
            )
            for q in COLUMNS
        ),
        means=(2.0, 0.5, 1.0, 0.0),
        provider_id="grid",
        snapshot_id="grid-1",
        causal_contract_id="checked-contract",
    )
    quantile = _contract(criterion=decision.Criterion.quantile(0.5))
    with pytest.raises(decision.DecisionRefusal) as other:
        require_success(quantile.evaluate(means))
    assert not isinstance(other.value, decision.PreferenceRefusal)

    # A malformed argument is still a ValueError.
    with pytest.raises(CausalValueError):
        decision.tradeoff("w", units="qaly/event")


def test_receipt_only_on_compiled_results():
    # A decision evaluated on a source in hand was not compiled with an analysis.
    for name, result in _results().items():
        assert result.receipt is None, name
