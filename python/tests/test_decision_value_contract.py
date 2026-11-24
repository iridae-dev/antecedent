"""One contract spelling for fixed utilities and trade-off ranges.

Fixture: two actions over aligned joint draws of benefit (qaly) and harm (events).

    treat: benefit rows [1, 3] (mean 2), harm rows [0, 1] (mean 0.5)
    wait:  benefit rows [1, 1] (mean 1), harm rows [0, 0] (mean 0)

With value ``benefit - w * harm`` the margin of treat over wait is ``1 - 0.5 w``: treat leads
below ``w = 2`` and wait above it. A plain number for ``w`` is an ordinary contract; a range is
a trade-off decision.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap
from typing import Any

import numpy as np
import pytest
from antecedent import _reason_details as reason_details
from antecedent import decision, decision_robust
from antecedent.decision_robust import Claim
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.outcomes import require_success
from antecedent.results.sources import MeanSource


def _quantity(
    variable: str, units: str, regime: str, functional: str = "outcome"
) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=variable,
        role="outcome",
        units=units,
        population_id="target",
        regime_id=regime,
        horizon=0,
        functional_id=functional,
    )


TREAT_BENEFIT = _quantity("benefit", "qaly", "do(a=1)")
TREAT_HARM = _quantity("harm", "events", "do(a=1)")
WAIT_BENEFIT = _quantity("benefit", "qaly", "do(a=0)")
WAIT_HARM = _quantity("harm", "events", "do(a=0)")
COLUMNS = (TREAT_BENEFIT, TREAT_HARM, WAIT_BENEFIT, WAIT_HARM)
ROWS = [[1.0, 0.0, 1.0, 0.0], [3.0, 1.0, 1.0, 0.0]]

BENEFIT = decision.attribute("benefit", units="qaly")
HARM = decision.attribute("harm", units="events")


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


def _named_actions() -> tuple[decision.Action, ...]:
    return (
        decision.Action("treat", inputs={"benefit": TREAT_BENEFIT, "harm": TREAT_HARM}),
        decision.Action("wait", inputs={"benefit": WAIT_BENEFIT, "harm": WAIT_HARM}),
    )


def _positional_actions(utility: decision.Expr | None = None) -> tuple[decision.Action, ...]:
    utility = utility if utility is not None else decision.x(0) - 1.0 * decision.x(1)
    return (
        decision.Action("treat", inputs=(TREAT_BENEFIT, TREAT_HARM), utility=utility),
        decision.Action("wait", inputs=(WAIT_BENEFIT, WAIT_HARM), utility=utility),
    )


def _value_contract(
    value: Any = None, *, criterion: decision.Criterion | None = None, **kwargs: Any
) -> decision.Contract:
    """The contract over ``benefit - 1.0 * harm`` (a plain number) unless a value is given."""
    value = value if value is not None else decision.Value(BENEFIT - 1.0 * HARM, units="qaly")
    fields: dict[str, Any] = {
        "actions": _named_actions(),
        "value": value,
        "target_population": "target",
    }
    if criterion is not None:
        fields["criterion"] = criterion
    fields.update(kwargs)
    return decision.Contract(**fields)


def _hand_contract(criterion: decision.Criterion | None = None, **kwargs: Any) -> decision.Contract:
    fields: dict[str, Any] = {
        "actions": _positional_actions(),
        "utility_units": "qaly",
        "criterion": criterion or decision.Criterion.expected_utility(),
        "target_population": "target",
    }
    fields.update(kwargs)
    return decision.Contract(**fields)


def _range_value(lower: float = 0.5, upper: float = 4.0) -> decision.Value:
    weight = decision.tradeoff(
        "harm_weight", units="qaly/event", lower=lower, upper=upper, assumption="panel_judgment"
    )
    return decision.Value(BENEFIT - weight * HARM, units="qaly")


# --------------------------------------------------------------------------- plain numbers


@pytest.mark.parametrize(
    "criterion",
    [
        decision.Criterion.expected_utility(),
        decision.Criterion.expected_loss(),
        decision.Criterion.threshold_probability(1.5),
        decision.Criterion.quantile(0.5),
        decision.Criterion.regret(),
        decision.Criterion.expected_regret(),
    ],
)
def test_a_plain_number_trade_off_is_the_ordinary_contract(criterion):
    contract = _value_contract(criterion=criterion)
    hand = _hand_contract(criterion)
    assert contract.form == "ordinary"
    assert contract.route is None
    assert contract.identity == hand.identity

    result = require_success(contract.evaluate(_source()))
    expected = require_success(hand.evaluate(_source()))
    assert type(result) is decision.Decision
    assert result.verdict == expected.verdict
    assert result.actions == expected.actions
    assert result.evpi == expected.evpi
    assert result.to_dict() == expected.to_dict()
    assert result.contract_identity == hand.identity


@pytest.mark.parametrize("bound", [0.5, -0.5])
def test_attribute_constraints_match_positional_constraints(bound):
    by_name = decision.Constraint(
        "harm_cap", HARM, bound=bound, units="events", min_probability=1.0
    )
    by_position = decision.Constraint(
        "harm_cap", decision.x(1), bound=bound, units="events", min_probability=1.0
    )
    contract = _value_contract(constraints=(by_name,))
    hand = _hand_contract(constraints=(by_position,))
    assert contract.identity == hand.identity
    result = require_success(contract.evaluate(_source()))
    expected = require_success(hand.evaluate(_source()))
    assert result.verdict == expected.verdict
    assert result.actions == expected.actions

    with pytest.raises(CausalValueError, match="reads attributes by name"):
        _value_contract(
            constraints=(
                decision.Constraint("by_position", decision.x(1), bound=1.0, units="events"),
            )
        )
    with pytest.raises(CausalTypeError, match="Contract\\(value="):
        _hand_contract(constraints=(by_name,))
    with pytest.raises(CausalValueError, match="never a trade-off"):
        weight = decision.tradeoff("w", units="qaly/event", lower=0.0, upper=1.0, assumption="a")
        decision.Constraint("reads_a_weight", weight * HARM, bound=1.0, units="events")


def test_fixed_value_on_means_matches():
    mean_columns = tuple(_quantity(q.variable_id, q.units, q.regime_id, "mean") for q in COLUMNS)
    means = MeanSource(
        coordinates=mean_columns,
        means=(2.0, 0.5, 1.0, 0.0),
        provider_id="grid",
        snapshot_id="grid-1",
        causal_contract_id="checked-contract",
    )

    def on_means(contract: decision.Contract) -> decision.Decision:
        return require_success(contract.evaluate(means))

    actions = (
        decision.Action(
            "treat",
            inputs={"benefit": mean_columns[0], "harm": mean_columns[1]},
        ),
        decision.Action(
            "wait",
            inputs={"benefit": mean_columns[2], "harm": mean_columns[3]},
        ),
    )
    by_value = on_means(_value_contract(actions=actions))
    by_hand = on_means(
        _hand_contract(
            actions=(
                decision.Action(
                    "treat",
                    inputs=(mean_columns[0], mean_columns[1]),
                    utility=decision.x(0) - 1.0 * decision.x(1),
                ),
                decision.Action(
                    "wait",
                    inputs=(mean_columns[2], mean_columns[3]),
                    utility=decision.x(0) - 1.0 * decision.x(1),
                ),
            )
        )
    )
    assert type(by_value) is decision.Decision
    assert by_value.selected == by_hand.selected == ("treat",)
    assert by_value.to_dict() == by_hand.to_dict()


@pytest.mark.parametrize("kind", ["graph_dependent", "weighted_atoms", "finite_scenarios"])
@pytest.mark.parametrize(
    "structural_policy",
    ["report_only", "maximin", "bayes_over_structures", "require_invariant_best_action"],
)
def test_fixed_value_structures_match_decision_robust(kind, structural_policy):
    stronger = _source([[1.0, 0.0, 1.0, 0.0], [5.0, 1.0, 1.0, 0.0]])
    probabilities = (0.6, 0.4) if kind != "graph_dependent" else (None, None)
    claims = (
        Claim.evaluated("direct", _source(), probability=probabilities[0]),
        Claim.evaluated("stronger", stronger, probability=probabilities[1]),
    )
    contract = _value_contract(structural_policy=structural_policy)
    hand = _hand_contract(structural_policy=structural_policy)
    structures = getattr(decision.Structures, kind)(claims)

    result = contract.evaluate(structures)
    expected = decision_robust.robust(
        decision_robust.admissible_contract(hand), claims, kind=structures.kind
    )
    if structural_policy == "bayes_over_structures" and kind == "graph_dependent":
        # Weighing structures needs their probabilities; both spellings refuse alike.
        with pytest.raises(decision.DecisionRefusal) as refused:
            require_success(result)
        with pytest.raises(decision.DecisionRefusal) as hand_refused:
            require_success(expected)
        assert refused.value.detail == hand_refused.value.detail
        return
    result, expected = require_success(result), require_success(expected)
    assert type(result) is type(expected)
    assert result.kind == expected.kind
    assert result.selected == expected.selected
    assert result.choice == expected.selected
    # A robust result is identified by the contract with its admissibility rules.
    assert result.contract_identity == expected.contract_identity
    assert result.verdict == expected.verdict
    assert result.effective_tie_tolerance == 0.0


def test_fixed_value_ties_exactly():
    equal = [[2.0, 0.0, 2.0, 0.0], [2.0, 0.0, 2.0, 0.0]]
    tied = require_success(_value_contract().evaluate(_source(equal)))
    assert type(tied) is decision.Decision
    assert tied.verdict.kind == "indistinguishable"
    assert tied.kind == "indistinguishable"
    assert tied.choice is None
    assert tied.standing == "tied"
    assert tied.effective_tie_tolerance == 0.0

    close = [[2.0 + 1e-12, 0.0, 2.0, 0.0], [2.0 + 1e-12, 0.0, 2.0, 0.0]]
    apart = require_success(_value_contract().evaluate(_source(close)))
    assert apart.verdict.kind == "uniquely_optimal"
    assert apart.choice == "treat"
    assert apart.standing == "best"

    # A range reads the same 1e-12 gap as no difference worth a choice.
    # The harm is zero in every row, so the two actions are within the tie tolerance at every
    # admissible weight: a tie, not a choice.
    ranged = require_success(_value_contract(_range_value(0.0, 1.0)).evaluate(_source(close)))
    assert ranged.kind == "tied"
    assert ranged.choice is None
    assert ranged.standing == "tied"
    assert ranged.effective_tie_tolerance == pytest.approx(1e-9)


def test_fixed_value_export_bytes_equal_hand_written_and_replay_exactly():
    contract = _value_contract()
    hand = _hand_contract()
    assert bytes(contract.export()) == bytes(hand.export())
    loaded = decision.Contract.load(bytes(contract.export()), expected_identity=hand.identity)
    assert loaded.identity == hand.identity

    result = require_success(contract.evaluate(_source()))
    exported = result.export()
    replayed = require_success(decision.replay(exported, contract=contract, source=_source()))
    assert type(replayed) is decision.Decision

    def reading(decided: decision.Decision) -> tuple[Any, ...]:
        return (
            decided.verdict,
            decided.actions,
            decided.evpi,
            decided.contract_identity,
            decided.standing,
            decided.statement,
        )

    assert reading(replayed) == reading(result)
    assert bytes(replayed.export()) == bytes(exported)
    by_hand = require_success(decision.replay(exported, contract=hand, source=_source()))
    assert reading(by_hand) == reading(result)


def test_exp_of_an_attribute_without_trade_offs_uses_the_trade_off_engine():
    # The ordinary utility language has no exp or ln, so the value has no ordinary form.
    value = decision.Value(decision.exp(0.1 * BENEFIT) - 1.0 * HARM, units="qaly")
    contract = _value_contract(value)
    result = require_success(contract.evaluate(_source()))
    assert type(result) is decision.TradeoffDecision
    # exp of a quantity attribute has no term contract, so the certified route answers.
    assert result.route == "certified"
    # Expected values: treat (e^0.1 + e^0.3 - 1) / 2 = 0.73, wait e^0.1 = 1.11.
    assert result.choice == "wait"
    with pytest.raises(decision.DecisionRefusal) as refused:
        contract.export()
    assert refused.value.detail == reason_details.preference_contract.NOT_LOWERABLE


# --------------------------------------------------------------------------- ranges


def test_named_fixed_tradeoff_is_a_trade_off_decision():
    weight = decision.tradeoff("harm_weight", units="qaly/event", value=1.0)
    assert weight.is_fixed
    assert weight.assumption == "declared_value"
    contract = _value_contract(decision.Value(BENEFIT - weight * HARM, units="qaly"))
    assert contract.form == "tradeoff"
    result = require_success(contract.evaluate(_source()))
    assert type(result) is decision.TradeoffDecision
    assert result.kind == "necessarily_best"
    assert result.choice == "treat"
    assert "declared_value" in result.assumptions
    assert contract.identity != _hand_contract().identity

    with pytest.raises(CausalValueError):
        decision.tradeoff("w", units="qaly/event")
    with pytest.raises(CausalValueError):
        decision.tradeoff("w", units="qaly/event", lower=0.0, upper=1.0)
    with pytest.raises(CausalValueError):
        decision.tradeoff("w", units="qaly/event", lower=0.0, value=1.0, assumption="a")


@pytest.mark.parametrize(
    "policy",
    [
        None,
        decision.TradeoffPolicy.maximin(),
        decision.TradeoffPolicy.minimax_regret(),
        decision.TradeoffPolicy.report_only(),
    ],
)
def test_ranged_contract_is_decided_across_its_trade_offs(policy):
    kwargs = {} if policy is None else {"policy": policy}
    contract = _value_contract(_range_value(), **kwargs)
    assert contract.form == "tradeoff"
    assert contract.route == "exact"
    result = require_success(contract.evaluate(_source()))
    assert type(result) is decision.TradeoffDecision
    assert result.contract_identity == contract.identity
    # A named trade-off fixed at one value is still decided across trade-offs.
    fixed = decision.tradeoff("harm_weight", units="qaly/event", value=1.0)
    pinned = _value_contract(decision.Value(BENEFIT - fixed * HARM, units="qaly"), **kwargs)
    assert pinned.form == "tradeoff"
    assert type(require_success(pinned.evaluate(_source()))) is decision.TradeoffDecision


def test_ranged_contract_results():
    result = require_success(_value_contract(_range_value()).evaluate(_source()))
    assert result.kind == "preference_dependent"
    assert result.choice is None
    assert result.standing == "depends"
    (switch,) = result.switches
    assert switch.margin_range == pytest.approx((-1.0, 0.75))
    assert switch.threshold.value == pytest.approx(2.0)
    assert [t.id for t in _value_contract(_range_value()).trade_offs] == ["harm_weight"]

    narrow = require_success(_value_contract(_range_value(0.5, 1.5)).evaluate(_source()))
    assert narrow.kind == "necessarily_best"
    assert narrow.choice == "treat"
    assert narrow.standing == "best"

    worst = require_success(
        _value_contract(_range_value(), policy=decision.TradeoffPolicy.maximin()).evaluate(
            _source()
        )
    )
    assert worst.standing == "policy_choice"
    assert worst.choice is not None


def test_ranged_contract_with_a_measure_and_structures():
    value = _range_value(0.5, 1.5)
    measure = decision.Measure.viewpoints(
        {"clinician": {"harm_weight": 0.75}, "patient": {"harm_weight": 1.25}},
        assumption="panel_viewpoints",
    )
    claims = (
        Claim.evaluated("direct", _source(), probability=0.6),
        Claim.evaluated(
            "stronger",
            _source([[1.0, 0.0, 1.0, 0.0], [5.0, 1.0, 1.0, 0.0]]),
            probability=0.4,
        ),
    )
    unified = _value_contract(value)
    by_measure = require_success(unified.evaluate(_source(), measure=measure))
    assert by_measure.kind == "necessarily_best"
    by_structures = require_success(unified.evaluate(decision.Structures.weighted_atoms(claims)))
    assert {s.id for s in by_structures.structures} == {"direct", "stronger"}


# --------------------------------------------------------------------------- validation


def test_value_form_validation():
    value = decision.Value(BENEFIT - 1.0 * HARM, units="qaly")
    with pytest.raises(CausalTypeError, match="takes no utility="):
        decision.Action(
            "treat",
            inputs={"benefit": TREAT_BENEFIT, "harm": TREAT_HARM},
            utility=decision.x(0),
        )
    with pytest.raises(CausalTypeError, match="reads inputs by position"):
        decision.Action("treat", inputs=(TREAT_BENEFIT,), utility=BENEFIT)
    with pytest.raises(CausalTypeError, match="gives positional inputs"):
        decision.Contract(
            actions=(*_named_actions()[:1], _positional_actions()[1]),
            value=value,
            target_population="target",
        )
    with pytest.raises(CausalTypeError, match="Contract\\(value="):
        decision.Contract(
            actions=_named_actions(),
            utility_units="qaly",
            criterion=decision.Criterion.expected_utility(),
            target_population="target",
        )
    with pytest.raises(CausalTypeError, match="decision.Value"):
        _value_contract(BENEFIT - 1.0 * HARM)
    with pytest.raises(CausalValueError):
        _value_contract(utility_units="usd")
    with pytest.raises(CausalValueError) as ordinary_policy:
        _hand_contract(policy=decision.TradeoffPolicy.maximin())
    assert ordinary_policy.value.reason_code == "option_not_applicable"
    with pytest.raises(CausalValueError) as no_trade_offs:
        _value_contract(policy=decision.TradeoffPolicy.maximin())
    assert no_trade_offs.value.reason_code == "option_not_applicable"
    with pytest.raises(CausalTypeError):
        decision.Contract(actions=_positional_actions(), utility_units="qaly")
    # Each action names every attribute exactly once.
    missing = decision.Contract(
        actions=(
            decision.Action("treat", inputs={"benefit": TREAT_BENEFIT}),
            _named_actions()[1],
        ),
        value=value,
        target_population="target",
    )
    with pytest.raises(CausalValueError, match="exactly once"):
        missing.evaluate(_source())
    views = decision.Measure.viewpoints({"only": {"harm_weight": 1.0}}, assumption="a")
    with pytest.raises(CausalValueError) as measure_on_ordinary:
        _value_contract().evaluate(_source(), measure=views)
    assert measure_on_ordinary.value.reason_code == "option_not_applicable"
    with pytest.raises(CausalValueError) as admissibility_on_ranged:
        _value_contract(_range_value()).evaluate(
            _source(), admissibility=decision_robust.AdmissibilityRules()
        )
    assert admissibility_on_ranged.value.reason_code == "option_not_applicable"


def test_value_form_defaults_and_replace():
    contract = _value_contract()
    assert contract.horizon == 0
    assert contract.structural_policy == "report_only"
    # The value form leaves the criterion to its default: expected utility.
    explicit = _value_contract(criterion=decision.Criterion.expected_utility())
    assert contract.identity == explicit.identity

    replaced = dataclasses.replace(contract, structural_policy="maximin")
    assert replaced.structural_policy == "maximin"
    assert replaced.identity != contract.identity
    assert dataclasses.replace(contract, horizon=0).identity == contract.identity
    changed = dataclasses.replace(contract, criterion=decision.Criterion.expected_loss())
    assert changed.identity == _hand_contract(decision.Criterion.expected_loss()).identity

    ranged = _value_contract(_range_value())
    assert dataclasses.replace(ranged, target_population="target").identity == ranged.identity


def test_ranged_contract_has_no_ordinary_form():
    contract = _value_contract(_range_value())
    with pytest.raises(decision.DecisionRefusal) as exported:
        contract.export()
    assert exported.value.detail == reason_details.preference_contract.NO_ORDINARY_FORM
    with pytest.raises(decision.DecisionRefusal) as transported:
        require_success(contract.evaluate_transported(None, analysis=None))  # type: ignore[arg-type]
    assert transported.value.detail == reason_details.preference_contract.NO_ORDINARY_FORM


def test_a_value_action_takes_its_kind_third():
    inputs = {"benefit": WAIT_BENEFIT, "harm": 0.0}
    assert decision.Action("wait", inputs, "policy").kind == "policy"
    assert decision.Action("wait", inputs, "policy") == decision.Action(
        "wait", inputs, kind="policy"
    )
    with pytest.raises(CausalTypeError):
        decision.Action("wait", inputs, decision.x(0))


def test_a_constraint_bound_per_action_is_reported_under_its_declared_id():
    # treat knows its harm exactly (5 events), so the cap is bound per action; the exclusion is
    # reported under the id it was declared with, as the trade-off route reports it.
    actions = (
        decision.Action("treat", inputs={"benefit": TREAT_BENEFIT, "harm": 5.0}),
        decision.Action("wait", inputs={"benefit": WAIT_BENEFIT, "harm": WAIT_HARM}),
    )
    cap = decision.Constraint("cap", HARM, bound=1.0, units="events")
    contract = _value_contract(actions=actions, constraints=(cap,))
    decided = require_success(contract.evaluate(_source()))
    assert type(decided) is decision.Decision
    by_id = {a.id: a for a in decided.actions}
    assert not by_id["treat"].admissible
    assert [e.constraint_id for e in by_id["treat"].exclusions] == ["cap"]
    assert decided.choice == "wait"

    ranged = _value_contract(_range_value(), actions=actions, constraints=(cap,))
    across = require_success(ranged.evaluate(_source()))
    assert [e.constraint_id for e in across.actions[0].exclusions] == ["cap"]


def test_a_cancelled_execution_refuses_the_ordinary_engine():
    import antecedent as ant
    from antecedent.errors import CausalCancelled

    token = ant.CancellationToken()
    token.cancel()
    cancelled = ant.Execution(cancel=token)
    contract = _value_contract()
    with pytest.raises(CausalCancelled):
        require_success(contract.evaluate(_source(), execution=cancelled))
    exported = require_success(contract.evaluate(_source())).export()
    with pytest.raises(CausalCancelled):
        require_success(
            decision.replay(exported, contract=contract, source=_source(), execution=cancelled)
        )


# --------------------------------------------------------------------------- separation


def test_deciding_with_stated_trade_offs_loads_no_learning():
    probe = textwrap.dedent(
        """
        import sys
        import numpy as np
        import antecedent
        from antecedent import decision
        from antecedent.joint_distribution import (
            DistributionIdentity, JointDistributionArtifact, ScientificQuantity,
        )
        from antecedent.outcomes import require_success

        def q(variable, units, regime):
            return ScientificQuantity(
                variable_id=variable, variable_name=variable, role="outcome", units=units,
                population_id="target", regime_id=regime, horizon=0, functional_id="outcome",
            )

        cols = (q("benefit", "qaly", "do(a=1)"), q("harm", "events", "do(a=1)"),
                q("benefit", "qaly", "do(a=0)"), q("harm", "events", "do(a=0)"))
        identity = DistributionIdentity(
            semantic="interventional_predictive", quantities=cols, alignment="joint",
            source_id="enumerated", provider_id="exact-law", rng_id="deterministic_exact",
            snapshot_id="enumeration-1", causal_contract_id="checked-contract",
        )
        source = JointDistributionArtifact(
            identity, np.array([[1.0, 0.0, 1.0, 0.0], [3.0, 1.0, 1.0, 0.0]]),
            calibration="exact",
        )
        benefit = decision.attribute("benefit", units="qaly")
        harm = decision.attribute("harm", units="events")
        w = decision.tradeoff("w", units="qaly/event", lower=0.5, upper=4.0, assumption="a")
        contract = decision.Contract(
            value=decision.Value(benefit - w * harm, units="qaly"),
            actions=(
                decision.Action("treat", inputs={"benefit": cols[0], "harm": cols[1]}),
                decision.Action("wait", inputs={"benefit": cols[2], "harm": cols[3]}),
            ),
            target_population="target",
        )
        assert require_success(contract.evaluate(source)).kind == "preference_dependent"
        assert callable(contract.compile)
        loaded = sorted(m for m in sys.modules if "learning" in m and m.startswith("antecedent"))
        assert "antecedent.decision_learning" not in sys.modules, loaded
        """
    )
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize(
    "module",
    [
        "antecedent.decision",
        "antecedent._decision_tradeoff",
        "antecedent._decision_tradeoff_results",
        "antecedent._decision_compile",
        "antecedent._decision_sources",
    ],
)
def test_importing_a_deciding_module_loads_no_learning(module):
    probe = (
        f"import sys\nimport {module}\nassert 'antecedent.decision_learning' not in sys.modules\n"
    )
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
