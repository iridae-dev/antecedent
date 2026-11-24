"""Decisions across explicitly stated trade-off ranges (no recorded choices).

Fixture: two actions over aligned joint draws of benefit (qaly) and harm (events).

    treat: benefit rows [1, 3] (mean 2), harm rows [0, 1] (mean 0.5)
    wait:  benefit rows [1, 1] (mean 1), harm rows [0, 0] (mean 0)

With value ``benefit - w * harm`` the margin of treat over wait is ``1 - 0.5 w``: treat leads
below ``w = 2`` and wait above it, so every answer below is known in closed form.
"""

from __future__ import annotations

import json
from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent import _reason_details as reason_details
from antecedent import decision
from antecedent._decision_value import StructuralPolicy as StructuralPolicyClass
from antecedent.errors import CausalValueError
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


def _weight(lower: float = 0.5, upper: float = 4.0) -> decision.Tradeoff:
    return decision.tradeoff(
        "harm_weight", units="qaly/event", lower=lower, upper=upper, assumption="panel_judgment"
    )


def _actions() -> tuple[decision.Action, ...]:
    return (
        decision.Action("treat", inputs={"benefit": TREAT_BENEFIT, "harm": TREAT_HARM}),
        decision.Action("wait", inputs={"benefit": WAIT_BENEFIT, "harm": WAIT_HARM}),
    )


def _contract(model: decision.Value, **kwargs: object) -> decision.Contract:
    return decision.Contract(
        value=model,
        actions=_actions(),
        target_population="target",
        **kwargs,  # type: ignore[arg-type]
    )


def _decide(contract: decision.Contract, **kwargs: object) -> decision.TradeoffDecision:
    return require_success(contract.evaluate(_source(), **kwargs))


def test_fixed_trade_off_reproduces_the_ordinary_decision():
    w = decision.tradeoff("harm_weight", units="qaly/event", value=1.0)
    result = _decide(_contract(decision.Value(BENEFIT - w * HARM, units="qaly")))

    ordinary = require_success(
        decision.Contract(
            actions=(
                decision.Action(
                    "treat",
                    inputs=(TREAT_BENEFIT, TREAT_HARM),
                    utility=decision.x(0) - 1.0 * decision.x(1),
                ),
                decision.Action(
                    "wait",
                    inputs=(WAIT_BENEFIT, WAIT_HARM),
                    utility=decision.x(0) - 1.0 * decision.x(1),
                ),
            ),
            utility_units="qaly",
            criterion=decision.Criterion.expected_utility(),
            target_population="target",
        ).evaluate(_source())
    )
    assert ordinary.selected == ("treat",)
    assert result.route == "exact"
    assert result.kind == "necessarily_best"
    assert result.selected == "treat"
    by_id = {a.id: a for a in result.actions}
    for outcome in ordinary.actions:
        low, high = by_id[outcome.id].value_range
        assert low == pytest.approx(outcome.expected_utility)
        assert high == pytest.approx(outcome.expected_utility)
    assert by_id["treat"].value_range == pytest.approx((1.5, 1.5))
    assert result.switches == ()
    assert result.undecided_fraction is None
    assert "treat" in result.statement


_ORDINARY_KIND = {
    "uniquely_optimal": "necessarily_best",
    "no_admissible_action": "no_admissible_action",
}


@pytest.mark.parametrize(
    "criterion",
    [
        decision.Criterion.expected_utility(),
        decision.Criterion.expected_loss(),
        decision.Criterion.expected_regret(),
        decision.Criterion.threshold_probability(1.5),
        decision.Criterion.quantile(0.5),
        decision.Criterion.regret(),
    ],
)
@pytest.mark.parametrize(
    "constraints",
    [
        (),
        (("harm_cap", 1, 0.5, "events", 1.0, ()),),
        (
            ("benefit_floor", 0, -2.0, "qaly", 0.5, ("treat",)),
            ("harm_cap", 1, 0.5, "events", 1.0, ()),
        ),
        (("benefit_floor", 0, -2.0, "qaly", 1.0, ()), ("harm_cap", 1, 0.5, "events", 1.0, ())),
    ],
)
def test_a_fixed_trade_off_decides_as_the_hand_written_contract(criterion, constraints):
    # Constraints as (id, attribute, bound, units, min_probability, applies_to); the benefit
    # floor reads minus the benefit.
    def ordinary_constraint(spec):
        cid, k, bound, units, p, applies = spec
        expr = -decision.x(0) if k == 0 else decision.x(k)
        return decision.Constraint(
            cid, expr, bound=bound, units=units, min_probability=p, applies_to=applies
        )

    def preference_constraint(spec):
        cid, k, bound, units, p, applies = spec
        expr = -BENEFIT if k == 0 else HARM
        return decision.Constraint(
            cid, expr, bound=bound, units=units, min_probability=p, applies_to=applies
        )

    w = decision.tradeoff("harm_weight", units="qaly/event", value=1.0)
    result = _decide(
        _contract(
            decision.Value(BENEFIT - w * HARM, units="qaly"),
            criterion=criterion,
            constraints=tuple(preference_constraint(c) for c in constraints),
        )
    )
    utility = decision.x(0) - 1.0 * decision.x(1)
    ordinary = require_success(
        decision.Contract(
            actions=(
                decision.Action("treat", inputs=(TREAT_BENEFIT, TREAT_HARM), utility=utility),
                decision.Action("wait", inputs=(WAIT_BENEFIT, WAIT_HARM), utility=utility),
            ),
            utility_units="qaly",
            criterion=criterion,
            target_population="target",
            constraints=tuple(ordinary_constraint(c) for c in constraints),
        ).evaluate(_source())
    )
    tie = "tied" if result.route == "exact" else "undetermined"
    assert result.kind == _ORDINARY_KIND.get(ordinary.verdict.kind, tie)
    assert result.selected == (
        ordinary.selected[0] if ordinary.verdict.kind == "uniquely_optimal" else None
    )
    by_id = {a.id: a for a in result.actions}
    for outcome in ordinary.actions:
        ours = by_id[outcome.id]
        assert ours.admissible == outcome.admissible
        assert ours.exclusions == outcome.exclusions
        if not outcome.admissible or ours.value_range is None:
            continue
        low, high = ours.value_range
        if result.route == "exact":
            assert low == pytest.approx(outcome.value, rel=1e-12)
            assert high == pytest.approx(outcome.value, rel=1e-12)
        else:
            assert low - 1e-12 <= outcome.value <= high + 1e-12


def test_a_range_that_crosses_the_switch_reports_where_the_choice_changes():
    model = decision.Value(BENEFIT - _weight() * HARM, units="qaly")
    result = _decide(_contract(model))
    assert result.kind == "preference_dependent"
    assert result.selected is None
    (switch,) = result.switches
    assert (switch.first, switch.second) == ("treat", "wait")
    # margin 1 - 0.5 w over [0.5, 4]: from -1 to 0.75
    assert switch.margin_range == pytest.approx((-1.0, 0.75))
    assert switch.favours_first_at["harm_weight"] < 2.0 < switch.favours_second_at["harm_weight"]
    assert switch.threshold is not None
    assert switch.threshold.parameter == "harm_weight"
    assert switch.threshold.value == pytest.approx(2.0)
    assert switch.threshold.first_leads_above is False
    assert "panel_judgment" in result.assumptions

    # The same model over a range below the switch decides.
    narrow = _decide(_contract(decision.Value(BENEFIT - _weight(0.5, 1.5) * HARM, units="qaly")))
    assert narrow.kind == "necessarily_best"
    assert narrow.selected == "treat"
    treat = {a.id: a for a in narrow.actions}["treat"]
    assert treat.value_range == pytest.approx((1.25, 1.75))


def test_a_relation_between_trade_offs_cuts_the_set():
    cost = decision.attribute("cost", units="usd")
    w_harm = decision.tradeoff(
        "harm_weight", units="qaly/event", lower=0.0, upper=1.0, assumption="panel"
    )
    w_cost = decision.tradeoff(
        "cost_weight", units="qaly/usd", lower=0.0, upper=1.0, assumption="payer"
    )
    value = BENEFIT - w_harm * HARM - w_cost * cost
    actions = (
        decision.Action(
            "treat", inputs={"benefit": TREAT_BENEFIT, "harm": TREAT_HARM, "cost": 1.0}
        ),
        decision.Action("wait", inputs={"benefit": WAIT_BENEFIT, "harm": WAIT_HARM, "cost": 0.0}),
    )

    def decide(model: decision.Value) -> decision.TradeoffDecision:
        contract = decision.Contract(value=model, actions=actions, target_population="target")
        return require_success(contract.evaluate(_source()))

    # Without a relation the margin 1 - 0.5 w_harm - w_cost reaches -0.5.
    assert decide(decision.Value(value, units="qaly")).kind == "preference_dependent"

    rule = decision.relation("harm_outweighs_cost", 4 * w_cost <= w_harm)
    assert rule.id == "harm_outweighs_cost"
    with pytest.raises(TypeError, match="not a truth value"):
        bool(w_harm >= w_cost)
    result = decide(decision.Value(value, units="qaly", relations=(rule,)))
    # w_cost <= w_harm / 4 <= 0.25, so the margin stays at least 0.25.
    assert result.kind == "necessarily_best"
    assert result.selected == "treat"
    assert result.parameter_ranges["cost_weight"] == pytest.approx((0.0, 0.25))

    with pytest.raises(CausalValueError, match="not the attribute 'harm'"):
        decision.relation("reads_an_attribute", w_harm >= HARM)


def test_a_curve_with_shape_rules_finds_the_switch_on_its_slope():
    low = decision.tradeoff(
        "slope_low", units="qaly/qaly", lower=0.0, upper=2.0, assumption="shape"
    )
    high = decision.tradeoff(
        "slope_high", units="qaly/qaly", lower=0.0, upper=2.0, assumption="shape"
    )
    curve = decision.Curve(BENEFIT, breakpoints=(0.0, 1.0, 3.0), slopes=(low, high))
    rules = (*curve.increasing(), *curve.diminishing())
    assert [r.id for r in curve.diminishing()] == ["benefit.curve.diminishing.0"]
    harm_weight = decision.tradeoff("harm_weight", units="qaly/event", value=1.0)
    model = decision.Value(curve - harm_weight * HARM, units="qaly", relations=rules)
    assert model.shape_parameters == ()
    result = _decide(_contract(model))
    # treat: segments [1, 1] and [0, 2] -> means 1 and 1; wait: 1 and 0; harm 0.5 vs 0.
    # margin = slope_high - 0.5, which changes sign inside [0, 2].
    assert result.route == "exact"
    assert result.kind == "preference_dependent"
    (switch,) = result.switches
    assert switch.threshold is not None
    assert switch.threshold.parameter == "slope_high"
    assert switch.threshold.value == pytest.approx(0.5)


def test_maximin_and_minimax_regret_choose_across_the_range():
    model = decision.Value(BENEFIT - _weight() * HARM, units="qaly")
    # treat ranges over [0, 1.75], wait is 1: the worst case of treat is lower.
    maximin = _decide(_contract(model, policy=decision.TradeoffPolicy.maximin()))
    assert maximin.kind == "worst_case_choice"
    assert maximin.selected == "wait"
    # Worst regret: treat 1 (at w = 4), wait 0.75 (at w = 0.5).
    regret = _decide(_contract(model, policy=decision.TradeoffPolicy.minimax_regret()))
    assert regret.kind == "minimax_regret_choice"
    assert regret.selected == "wait"
    by_id = {a.id: a for a in regret.actions}
    assert by_id["treat"].worst_case_regret == pytest.approx(1.0)
    assert by_id["wait"].worst_case_regret == pytest.approx(0.75)


def test_a_measure_of_named_viewpoints_weighs_the_range():
    model = decision.Value(BENEFIT - _weight() * HARM, units="qaly")
    views = decision.Measure.viewpoints(
        {"clinician": {"harm_weight": 1.0}, "patient": {"harm_weight": 3.0}},
        weights={"clinician": 3.0, "patient": 1.0},
        assumption="stakeholder_weights",
    )
    contract = _contract(model, policy=decision.TradeoffPolicy.weighted())
    result = _decide(contract, measure=views)
    # treat is 1.5 at w = 1 and 0.5 at w = 3; wait is 1 at both.
    assert result.kind == "measure_choice"
    assert result.selected == "treat"
    treat = {a.id: a for a in result.actions}["treat"]
    assert treat.measure_share == pytest.approx(0.75)
    assert treat.measure_value == pytest.approx(1.25)
    assert result.measure is not None
    assert result.measure.labels == ("clinician", "patient")

    with pytest.raises(decision.PreferenceRefusal) as missing:
        _decide(contract)
    assert missing.value.detail == reason_details.preference_decision.MEASURE_REQUIRED


def test_a_shape_parameter_is_answered_by_certified_subdivision():
    w = decision.tradeoff(
        "harm_weight", units="qaly/event", lower=0.5, upper=1.0, assumption="panel"
    )
    model = decision.Value(BENEFIT - w * w * HARM, units="qaly")
    assert model.shape_parameters == ("harm_weight",)
    contract = _contract(model)
    assert contract.route == "certified"
    result = _decide(contract)
    # treat is 2 - 0.5 w^2 in [1.5, 1.875], always above wait's 1.
    assert result.route == "certified"
    assert result.kind == "necessarily_best"
    assert result.selected == "treat"
    assert result.undecided_fraction == pytest.approx(0.0)
    assert result.regions and all(region.best == "treat" for region in result.regions)
    assert result.switches == ()

    means = MeanSource(
        coordinates=COLUMNS,
        means=(2.0, 0.5, 1.0, 0.0),
        provider_id="grid",
        snapshot_id="grid-1",
        causal_contract_id="checked-contract",
    )
    with pytest.raises(decision.PreferenceRefusal) as needs_draws:
        require_success(contract.evaluate(means))
    assert needs_draws.value.detail == reason_details.preference_evaluation.JOINT_DRAWS_REQUIRED


def test_a_mean_source_answers_the_exact_route():
    # A mean source carries means, so its coordinates are the ``mean`` functional.
    mean_columns = tuple(_quantity(q.variable_id, q.units, q.regime_id, "mean") for q in COLUMNS)
    model = decision.Value(BENEFIT - _weight(0.5, 1.5) * HARM, units="qaly")
    contract = decision.Contract(
        value=model,
        actions=(
            decision.Action("treat", inputs={"benefit": mean_columns[0], "harm": mean_columns[1]}),
            decision.Action("wait", inputs={"benefit": mean_columns[2], "harm": mean_columns[3]}),
        ),
        target_population="target",
    )
    means = MeanSource(
        coordinates=mean_columns,
        means=(2.0, 0.5, 1.0, 0.0),
        provider_id="grid",
        snapshot_id="grid-1",
        causal_contract_id="checked-contract",
    )
    result = require_success(contract.evaluate(means))
    assert result.kind == "necessarily_best"
    assert result.selected == "treat"
    assert result.n_draws == 0
    assert result.draw_share_best_everywhere is None


def test_invalid_declarations_refuse_with_their_detail():
    reversed_range = decision.tradeoff(
        "harm_weight", units="qaly/event", lower=2.0, upper=1.0, assumption="x"
    )
    with pytest.raises(decision.PreferenceRefusal) as caught:
        decision.Value(BENEFIT - reversed_range * HARM, units="qaly").validate()
    assert caught.value.detail == reason_details.preference_model.PARAMETER_RANGE_EMPTY

    model = decision.Value(BENEFIT - _weight() * HARM, units="qaly")
    one_action = decision.Contract(value=model, actions=_actions()[:1], target_population="target")
    with pytest.raises(decision.PreferenceRefusal) as too_few:
        require_success(one_action.evaluate(_source()))
    assert too_few.value.detail == reason_details.preference_contract.TOO_FEW_ACTIONS

    with pytest.raises(CausalValueError, match="exactly once"):
        decision.Contract(
            value=model,
            actions=(decision.Action("treat", inputs={"benefit": TREAT_BENEFIT}), _actions()[1]),
            target_population="target",
        )._tradeoff_wire()


def test_export_and_replay_recompute_the_decision():
    model = decision.Value(BENEFIT - _weight() * HARM, units="qaly")
    contract = _contract(model)
    result = _decide(contract)
    exported = result.export()
    replayed = require_success(decision.replay(exported, contract=contract, source=_source()))
    assert replayed.kind == result.kind
    assert replayed.contract_identity == contract.identity
    assert replayed.switches[0].margin_range == pytest.approx(result.switches[0].margin_range)

    other = _source([[1.0, 0.0, 1.0, 0.0], [5.0, 1.0, 1.0, 0.0]])
    with pytest.raises(decision.PreferenceRefusal) as other_draws:
        require_success(decision.replay(exported, contract=contract, source=other))
    assert other_draws.value.reason_code == "decision_contract_unsatisfied"
    assert other_draws.value.detail == reason_details.preference_artifact.SOURCE_EXPECTED


def test_replay_keeps_the_stored_evaluation_options():
    # A shape parameter takes the certified route, which reads both options.
    w = decision.tradeoff(
        "harm_weight", units="qaly/event", lower=0.5, upper=1.0, assumption="panel"
    )
    contract = _contract(decision.Value(BENEFIT - w * w * HARM, units="qaly"))
    result = _decide(contract, tie_tolerance=1e-3, max_boxes=64)
    exported = result.export()
    replayed = require_success(decision.replay(exported, contract=contract, source=_source()))
    assert replayed.kind == result.kind
    assert replayed.undecided_fraction == result.undecided_fraction
    # Re-exporting the replayed decision recomputes it under the stored options.
    assert bytes(replayed.export()) == bytes(exported)
    assert bytes(_decide(contract).export()) != bytes(exported)


def test_a_record_written_with_every_option_re_exports_byte_identically():
    # Options Python's `evaluate` cannot set (a certified tie tolerance other than the exact
    # route's, non-default linear-program tolerances) are written as a Rust caller would write
    # them, through the native exporter; replay hands every stored option back, so the replayed
    # decision re-exports to the same bytes.
    from antecedent._native import export_preference_decision

    w = decision.tradeoff(
        "harm_weight", units="qaly/event", lower=0.5, upper=1.0, assumption="panel"
    )
    for value in (BENEFIT - w * w * HARM, BENEFIT - w * HARM):
        contract = _contract(decision.Value(value, units="qaly"))
        options = {
            "tie_tolerance": 1e-7,
            "certified_tie_tolerance": 1e-3,
            "max_boxes": 64,
            "lp_feasibility_tolerance": 1e-10,
            "lp_optimality_tolerance": 1e-11,
            "lp_max_pivots": 5_000,
        }
        data, refusal = export_preference_decision(
            json.dumps(contract._tradeoff_wire()),
            None,
            json.dumps(options),
            _source()._native,
            "preference-decision",
        )
        assert refusal is None and data is not None
        written = bytes(data)
        replayed = require_success(decision.replay(written, contract=contract, source=_source()))
        assert bytes(replayed.export()) == written
        effective = replayed.effective_tie_tolerance
        assert effective is not None
        if replayed.route == "certified":
            assert effective == 1e-3
        else:
            assert effective >= 1e-7


def test_the_effective_tie_tolerance_is_reported():
    w = decision.tradeoff(
        "harm_weight", units="qaly/event", lower=0.5, upper=1.0, assumption="panel"
    )
    exact = _decide(_contract(decision.Value(BENEFIT - w * HARM, units="qaly")))
    assert exact.route == "exact"
    # The declared 1e-9, or the margins' numerical bound when that is larger.
    assert exact.effective_tie_tolerance is not None and exact.effective_tie_tolerance >= 1e-9
    certified = _decide(
        _contract(decision.Value(BENEFIT - w * w * HARM, units="qaly")),
        tie_tolerance=1e-4,
    )
    assert certified.route == "certified"
    assert certified.effective_tie_tolerance == 1e-4


# --------------------------------------------------------------------------- several outcomes


def _paired_law(
    columns: tuple[ScientificQuantity, ...], rows: list[list[float]], snapshot: str = "draws-2"
) -> JointDistributionArtifact:
    """One law per estimated outcome sharing one draw identity, so row ``i`` of each pairs."""
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=columns,
        alignment="joint",
        source_id="shared-posterior",
        provider_id="exact-law",
        rng_id="deterministic_exact",
        snapshot_id=snapshot,
        causal_contract_id="checked-contract",
    )
    return JointDistributionArtifact(
        identity, np.array(rows, dtype=np.float64), calibration="exact"
    )


# benefit: treat rows [2, 4], wait [1, 1] -> estimated difference 2
# harm:    treat rows [1, 3], wait [0.5, 0.5] -> estimated difference 1.5
BENEFITS = _paired_law((TREAT_BENEFIT, WAIT_BENEFIT), [[2.0, 1.0], [4.0, 1.0]])
HARMS = _paired_law((TREAT_HARM, WAIT_HARM), [[1.0, 0.5], [3.0, 0.5]])
RATIO = 2.0 / 1.5


@pytest.mark.parametrize(
    ("lower", "upper", "kind", "selected"),
    [
        (0.5, 1.0, "necessarily_best", "treat"),  # left of the ratio
        (2.0, 3.0, "necessarily_best", "wait"),  # right of it
        (0.5, 3.0, "preference_dependent", None),  # around it
    ],
)
def test_separately_estimated_outcomes_switch_at_the_ratio_of_their_differences(
    lower: float, upper: float, kind: str, selected: str | None
):
    model = decision.Value(BENEFIT - _weight(lower, upper) * HARM, units="qaly")
    result = require_success(_contract(model).evaluate([BENEFITS, HARMS]))
    assert result.kind == kind
    assert result.selected == selected
    if kind == "preference_dependent":
        (switch,) = result.switches
        assert switch.threshold is not None
        assert switch.threshold.value == pytest.approx(RATIO)
        assert switch.threshold.first_leads_above is False

    # The joined law is an ordinary joint law: the decision exports and replays on it.
    joined = decision.joint_law(BENEFITS, HARMS)
    replayed = require_success(
        decision.replay(result.export(), contract=_contract(model), source=joined)
    )
    assert replayed.kind == kind


def test_separately_sampled_laws_do_not_pair():
    harms = _paired_law((TREAT_HARM, WAIT_HARM), [[1.0, 0.5], [3.0, 0.5]], snapshot="draws-3")
    model = decision.Value(BENEFIT - _weight() * HARM, units="qaly")
    with pytest.raises(decision.PreferenceRefusal) as unpaired:
        require_success(_contract(model).evaluate([BENEFITS, harms]))
    assert unpaired.value.reason_code == "joint_law_required"


def test_a_propagation_chain_joins_its_outputs_under_the_declared_dependence():
    # A predictive chain is the outcome law under each action, so it answers `outcome` inputs.
    T_BEN, T_HARM, W_BEN, W_HARM = COLUMNS
    from antecedent import propagation_chain as pc

    shared = [("disturbance", "trial-shock")]
    sources = [
        pc.draws(name, values, meaning="predictive", alignment="joint:trial", factors=shared)
        for name, values in (
            ("treat_benefit_draws", [2.0, 4.0]),
            ("treat_harm_draws", [1.0, 3.0]),
            ("wait_benefit_draws", [1.0, 1.0]),
            ("wait_harm_draws", [0.5, 0.5]),
        )
    ]
    stages = [
        pc.affine(name, "derived_quantity", [f"{name}_draws"])
        for name in ("treat_benefit", "treat_harm", "wait_benefit", "wait_harm")
    ]
    chain = require_success(
        pc.propagate(
            sources,
            stages,
            output="treat_benefit",
            reading="predictive",
            output_quantity=T_BEN,
            joint_outputs=[
                ("treat_benefit", T_BEN),
                ("treat_harm", T_HARM),
                ("wait_benefit", W_BEN),
                ("wait_harm", W_HARM),
            ],
        )
    )
    model = decision.Value(BENEFIT - _weight(0.5, 3.0) * HARM, units="qaly")
    actions = (
        decision.Action("treat", inputs={"benefit": T_BEN, "harm": T_HARM}),
        decision.Action("wait", inputs={"benefit": W_BEN, "harm": W_HARM}),
    )
    contract = decision.Contract(value=model, actions=actions, target_population="target")
    result = require_success(contract.evaluate(chain))
    assert result.kind == "preference_dependent"
    (switch,) = result.switches
    assert switch.threshold is not None
    assert switch.threshold.value == pytest.approx(RATIO)


# --------------------------------------------------------------------------- one experiment


def _experiment(clusters: int = 40, rows: int = 12):
    """A cluster-randomized experiment measuring benefit and harm in each arm."""
    import math

    from antecedent import decision_evidence as de

    assigned = [c % 2 == 0 for c in range(clusters)]
    totals: list[list[float]] = [[], []]
    for c, treated in enumerate(assigned):
        shock = math.sin((c + 1.0) * 12.9898) * 0.5
        benefit = harm = 0.0
        for r in range(rows):
            noise = math.sin((c * 31 + r * 17) * 0.7) * 0.2
            benefit += 2.0 + (1.0 if treated else 0.0) + 0.6 * shock + noise
            harm += 1.0 + (0.5 if treated else 0.0) + 0.3 * shock + 0.5 * noise
        totals[0].append(benefit)
        totals[1].append(harm)
    return de.JointEvidence.from_cluster_totals(
        evidence_id="exp-benefit-harm",
        population_id="target",
        treated_regime_id="do(a=1)",
        control_regime_id="do(a=0)",
        horizon=0,
        components={"benefit": ("outcome.benefit", "qaly"), "harm": ("outcome.harm", "events")},
        totals=totals,
        assigned=assigned,
        outcome_rows=clusters * rows,
        exposure="no_between_unit_interference",
        exposure_assumption_id="assumption.no_spillover",
        cluster_ids=[f"cluster-{i}" for i in range(clusters)],
        cluster_sizes=[rows] * clusters,
        assignment_law_id="complete-randomization",
        treated_clusters=sum(assigned),
        consistency_assumption_id="consistency",
    )


def _experiment_contract(experiment, model: decision.Value) -> decision.Contract:
    def arm(name: str) -> dict[str, ScientificQuantity]:
        return {o: decision.experiment_quantity(experiment, o, name) for o in ("benefit", "harm")}

    return decision.Contract(
        value=model,
        actions=(
            decision.Action("treat", inputs=arm("treated")),
            decision.Action("wait", inputs=arm("control")),
        ),
        target_population="target",
    )


def test_one_experiment_switches_at_the_ratio_of_its_estimated_differences():
    experiment = _experiment()
    treated, control = experiment.wire["treated_means"], experiment.wire["control_means"]
    ratio = (treated[0] - control[0]) / (treated[1] - control[1])
    assert ratio > 0.0

    quantity = decision.experiment_quantity(experiment, "harm", "control")
    assert quantity.functional_id == "mean"
    assert quantity.regime_id == "do(a=0)"

    for lower, upper, kind, selected in (
        (0.1 * ratio, 0.5 * ratio, "necessarily_best", "treat"),
        (1.5 * ratio, 3.0 * ratio, "necessarily_best", "wait"),
        (0.5 * ratio, 2.0 * ratio, "preference_dependent", None),
    ):
        model = decision.Value(BENEFIT - _weight(lower, upper) * HARM, units="qaly")
        result = require_success(_experiment_contract(experiment, model).evaluate(experiment))
        assert (result.kind, result.selected) == (kind, selected)
        if kind == "preference_dependent":
            (switch,) = result.switches
            assert switch.threshold is not None
            assert switch.threshold.value == pytest.approx(ratio)


def test_one_experiment_cannot_answer_a_kinked_term_from_its_means():
    experiment = _experiment()
    kinked = decision.Value(BENEFIT - _weight() * decision.excess(HARM, 1.0), units="qaly")
    with pytest.raises(decision.PreferenceRefusal) as refused:
        require_success(_experiment_contract(experiment, kinked).evaluate(experiment))
    # The arm means answer only an affine expectation: the ordinary evaluation of the kinked
    # part refuses the mean source.
    assert refused.value.reason_code == "decision_contract_unsatisfied"
    assert refused.value.detail == reason_details.decision_evaluation.MEAN_SOURCE_INSUFFICIENT


# --------------------------------------------------------------------------- one compiled design
#
# Fixture: a cluster-randomized design of 64 clusters of 10 rows, alternately assigned. The
# treatment raises per-row benefit by 1.0 and harm by 0.5; each cluster carries a bounded shock.
# Randomization alone does not identify the cross-arm dependence of the arm-mean estimators, so
# a kinked value needs the declared premise that the clusters were sampled independently from a
# larger population; then the clusters are resampled within each arm.


SAMPLING = "assumption.clusters_sampled_from_a_larger_population"


def _cluster_design(clusters: int = 64, rows: int = 10):
    import math

    from antecedent import joint_panel as jp

    assigned = [c % 2 == 0 for c in range(clusters)]
    shocks = [math.sin((c + 1.0) * 12.9898) for c in range(clusters)]
    benefit = [
        rows * (2.0 + (1.0 if a else 0.0) + 0.2 * s) for a, s in zip(assigned, shocks, strict=True)
    ]
    harm = [
        rows * (1.0 + (0.5 if a else 0.0) + 0.1 * s) for a, s in zip(assigned, shocks, strict=True)
    ]
    return jp.prepare_native_joint_design(
        {
            "kind": "decision_cluster",
            "recipe": {
                "population": "target",
                "treated_regime_id": "do(a=1)",
                "control_regime_id": "do(a=0)",
                "horizon": 0,
                "cluster_ids": [f"cluster-{c}" for c in range(clusters)],
                "cluster_sizes": [rows] * clusters,
                "assignment_law": "complete randomization of half the clusters",
                "treated_clusters": sum(assigned),
                "consistency": "consistency",
                "assignments": assigned,
                "outcomes": [
                    {
                        "component_name": "benefit",
                        "variable_id": "benefit",
                        "units": "qaly",
                        "values": benefit,
                    },
                    {
                        "component_name": "harm",
                        "variable_id": "harm",
                        "units": "events",
                        "values": harm,
                    },
                ],
                "adjustments": [],
                "exposure": {"kind": "no_interference", "assumption": "assumption.no_spillover"},
            },
        }
    )


def _design_contract(model: decision.Value) -> decision.Contract:
    def arm(regime: str) -> dict[str, ScientificQuantity]:
        return {
            "benefit": _quantity("benefit", "qaly", regime, "mean"),
            "harm": _quantity("harm", "events", regime, "mean"),
        }

    return decision.Contract(
        value=model,
        actions=(
            decision.Action("treat", inputs=arm("do(a=1)")),
            decision.Action("wait", inputs=arm("do(a=0)")),
        ),
        target_population="target",
    )


def test_a_cluster_design_reads_its_independent_unit_bootstrap_under_a_declared_sampling_premise():
    design = _cluster_design()
    affine = _design_contract(decision.Value(BENEFIT - _weight(0.1, 0.3) * HARM, units="qaly"))
    means = require_success(affine.compile(design))
    assert means.inspect()["demand"]["representation"] == "mean"
    assert means.plans[0]["evidence"]["decision"]["preference"]["design_law"] is None
    assert require_success(means.execute()).selected == "treat"

    kinked = _design_contract(
        decision.Value(BENEFIT - _weight(0.1, 0.3) * decision.excess(HARM, 1.2), units="qaly")
    )
    for declared in ({}, {"refits": 99, "seed": 3}):
        with pytest.raises(decision.PreferenceRefusal) as undeclared:
            require_success(kinked.compile(design, **declared))  # type: ignore[arg-type]
        assert undeclared.value.reason_code == "joint_law_required"
        assert (
            undeclared.value.detail == reason_details.preference_compile.DESIGN_SAMPLING_UNDECLARED
        )
    with pytest.raises(CausalValueError, match="refits="):
        require_success(kinked.compile(design, sampling=SAMPLING))

    compiled = require_success(kinked.compile(design, refits=99, seed=3, sampling=SAMPLING))
    assert compiled.inspect()["demand"]["representation"] == "joint_draws"
    assert means.design_law is None
    assert compiled.design_law == decision.DesignLawPlan(
        kind="independent_unit_bootstrap", unit="cluster", replicates=99, seed=3, sampling=SAMPLING
    )
    (plan,) = compiled.plans
    assert plan["evidence"]["decision"]["carrier"] == "decision_draws"
    assert plan["evidence"]["decision"]["preference"]["design_law"] == {
        "kind": "independent_unit_bootstrap",
        "unit": "cluster",
        "replicates": 99,
        "seed": 3,
        "sampling": SAMPLING,
    }
    reseeded = require_success(kinked.compile(design, refits=99, seed=4, sampling=SAMPLING))
    assert reseeded.plans[0]["identity"] != plan["identity"]

    decided = require_success(compiled.execute())
    assert isinstance(decided.source, JointDistributionArtifact)
    assert (
        decided.source.identity.rng_id == "independent_unit_bootstrap:cluster:replicates=99:seed=3"
    )
    assert decided.selected == "treat"
    assert SAMPLING in decided.assumptions

    # Ten clusters in each arm are below the bootstrap's floor of thirty.
    with pytest.raises(decision.PreferenceRefusal) as few:
        require_success(
            _design_contract(kinked.value).compile(
                _cluster_design(20), refits=99, sampling=SAMPLING
            )
        )
    assert few.value.reason_code == "too_few_clusters"
    assert few.value.detail == reason_details.preference_compile.DESIGN_UNITS_TOO_FEW


# --------------------------------------------------------------------------- the compiled route
#
# Fixture: dose tables ``y = intercept + slope * t + noise`` on ``t`` in [0, 2] with the graph
# ``t -> y``, and a mean response curve of ``y`` under ``do(t = 0.5)`` and ``do(t = 1.5)``.
# The actions ``low`` and ``high`` read one dose's mean benefit each and carry a known cost of
# 0.5 and 1.5, valued ``benefit - price * cost``, so ``high`` leads exactly when the price is
# below the gain ``m(1.5) - m(0.5)``.


DOSES = [0.5, 1.5]
PRICE_UNITS = "points/usd"


def _dose_table(name: str = "y", rows: int = 240, slope: float = 1.5, rate: float = 1.7):
    t = np.array([2.0 * i / (rows - 1) for i in range(rows)])
    noise = np.array([np.sin(i * rate) for i in range(rows)])
    return {"t": t, name: 0.5 + slope * t + 0.8 * noise}


def _prepared(data, outcome: str = "y", *, bayes: bool = False, discovery=None):
    inference = (
        ant.Bayesian(backend="conjugate", n_draws=64) if bayes else ant.Frequentist(bootstrap=0)
    )
    return ant.prepare(
        data,
        graph=None if discovery is not None else [("t", outcome)],
        discovery=discovery,
        query=ant.ResponseCurve("t", outcome, grid=DOSES),
        refute="none",
        inference=inference,
        execution=ant.Execution(seed=11),
    )


def _dose(outcome: str, dose: float, units: str = "points") -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=outcome,
        variable_name=outcome,
        role="outcome",
        units=units,
        population_id="target",
        regime_id=f"do(t={dose})",
        horizon=0,
        functional_id="mean",
    )


def _priced(lower: float, upper: float, *, kink: float | None = None) -> decision.Value:
    benefit = decision.attribute("benefit", units="points")
    cost = decision.attribute("cost", units="usd")
    price = decision.tradeoff(
        "price", units=PRICE_UNITS, lower=lower, upper=upper, assumption="price"
    )
    value = benefit - price * cost
    if kink is not None:
        weight = decision.tradeoff(
            "shortfall_weight", units="points/points", lower=0.0, upper=2.0, assumption="floor"
        )
        value = value - weight * decision.shortfall(benefit, kink)
    return decision.Value(value, units="points")


def _dose_contract(model: decision.Value, **kwargs: object) -> decision.Contract:
    return decision.Contract(
        value=model,
        actions=(
            decision.Action("low", inputs={"benefit": _dose("y", 0.5), "cost": 0.5}),
            decision.Action("high", inputs={"benefit": _dose("y", 1.5), "cost": 1.5}),
        ),
        target_population="target",
        **kwargs,  # type: ignore[arg-type]
    )


def test_compile_shows_the_plan_before_anything_is_fitted():
    prepared = _prepared(_dose_table())
    compiled = require_success(_dose_contract(_priced(0.2, 1.0)).compile(prepared))
    described = compiled.inspect()
    assert described["demand"]["representation"] == "mean"
    assert described["demand"]["carrier"] == "mean"
    assert [q["quantity"]["regime_id"] for q in described["demand"]["quantities"]] == [
        "do(t=0.5)",
        "do(t=1.5)",
    ]
    assert described["unidentified_mass"] == 0.0
    (plan,) = compiled.plans
    preference = plan["evidence"]["decision"]["preference"]
    assert preference["bound_contract_identity"] == compiled.bound_identity
    assert preference["representation"] == "mean"
    assert [q["regime"] for q in preference["quantities"]] == ["do(t=0.5)", "do(t=1.5)"]
    assert preference["parameters"] == [
        {
            "id": "price",
            "units": PRICE_UNITS,
            "lower": 0.2,
            "upper": 1.0,
            "assumption": "price",
        }
    ]

    # The same facts, typed.
    assert compiled.representation == "mean"
    assert compiled.carrier == "mean"
    assert compiled.route == "exact"
    assert compiled.contract_identity == compiled.contract.identity
    assert [q.quantity.regime_id for q in compiled.quantities] == ["do(t=0.5)", "do(t=1.5)"]
    assert [q.source for q in compiled.quantities] == [0, 0]
    assert [q.read_by for q in compiled.quantities] == [
        (("low", "benefit"),),
        (("high", "benefit"),),
    ]
    assert compiled.trade_offs == (decision.TradeOffRange("price", PRICE_UNITS, 0.2, 1.0, "price"),)
    assert compiled.parameter_ranges == {"price": (0.2, 1.0)}
    assert compiled.joint_law is None
    assert compiled.design_law is None

    # The gain is about 1.5, so a price range below it leaves `high` best everywhere.
    decided = require_success(compiled.execute())
    assert decided.kind == "necessarily_best"
    assert decided.selected == "high"


def _edge(names: list[str], source: str, target: str) -> int:
    """The adjacency bit of ``source -> target`` over ``names``."""
    n, i, j = len(names), names.index(source), names.index(target)
    return 1 << (i * (n - 1) + (j if j < i else j - 1))


NAMES = ["t", "y", "z"]
# A directed cycle t -> y -> z -> t: no acyclic mixed graph, so it identifies nothing.
CYCLE = _edge(NAMES, "t", "y") | _edge(NAMES, "y", "z") | _edge(NAMES, "z", "t")


def _coded_table(rows: int = 240):
    """Coded columns for the mixed-graph route's discrete plug-in: a binary ``z``, ``t`` at the
    two doses (more often high when ``z`` is one) and the noise on three levels."""
    z = np.array([float(i % 2) for i in range(rows)])
    t = np.array([1.5 if (i // 2) % 5 < 2 + 2 * z[i] else 0.5 for i in range(rows)])
    noise = np.array([0.3 * ((i // 10) % 3 - 1) for i in range(rows)])
    return {"t": t, "y": 0.5 + 1.5 * t + z + noise, "z": z}


def test_compile_refuses_a_quantity_no_structure_identifies():
    # The only structure is a directed cycle: no structure identifies the curve. A DAG
    # posterior holds only DAG atoms, so the cycle is a mixed-graph atom.
    posterior = ant.discovery.GraphPosterior.from_atoms(NAMES, [1.0], [CYCLE], atom_kind="Admg")
    table = _dose_table()
    table["z"] = np.array([np.sin(i * 0.91) for i in range(len(table["t"]))])
    prepared = _prepared(table, bayes=True, discovery=posterior)
    with pytest.raises(decision.PreferenceRefusal) as refused:
        require_success(_dose_contract(_priced(0.2, 1.0)).compile(prepared))
    assert refused.value.reason_code == "effect_not_identified"
    assert refused.value.detail == reason_details.native_claims.RESPONSE_NOT_POINT_IDENTIFIED


def test_the_compiled_decision_agrees_with_evaluate_on_the_same_draws():
    prepared = _prepared(_dose_table(), bayes=True)
    contract = _dose_contract(_priced(0.2, 1.0, kink=2.0))
    compiled = require_success(contract.compile(prepared))
    assert compiled.inspect()["demand"]["representation"] == "joint_draws"
    decided = require_success(compiled.execute())
    assert isinstance(decided.source, JointDistributionArtifact)
    again = require_success(contract.evaluate(decided.source))
    assert again.kind == decided.kind
    assert again.selected == decided.selected
    assert [a.value_range for a in again.actions] == [a.value_range for a in decided.actions]


def test_compile_over_a_graph_posterior_keeps_the_unidentified_mass():
    # Mixed-graph atoms: `t -> y`, compiled as an ADMG study on the coded table, and the cycle.
    direct = _edge(NAMES, "t", "y")
    posterior = ant.discovery.GraphPosterior.from_atoms(
        NAMES, [0.7, 0.3], [direct, CYCLE], atom_kind="Admg"
    )
    prepared = _prepared(_coded_table(), bayes=True, discovery=posterior)
    contract = _dose_contract(_priced(0.2, 1.0), structural_policy="require_invariant_best_action")
    compiled = require_success(contract.compile(prepared))
    described = compiled.inspect()
    assert len(described["structures"]) == 2
    assert described["unidentified_mass"] == pytest.approx(0.3)
    assert described["structures"][1]["identification"]["kind"] == "not_identified"
    decided = require_success(compiled.execute())
    assert decided.unidentified_mass == pytest.approx(0.3)
    assert decided.kind == "insufficient_science"


def test_compile_across_several_studies():
    benefit = _prepared(_dose_table("b", 220, 1.2, 1.7), "b", bayes=True)
    harm = _prepared(_dose_table("h", 260, 0.5, 2.9), "h", bayes=True)
    rate = decision.tradeoff(
        "harm_rate", units="points/events", lower=0.1, upper=0.3, assumption="h"
    )
    weight = decision.tradeoff(
        "excess", units="points/events", lower=0.0, upper=2.0, assumption="x"
    )
    b = decision.attribute("benefit", units="points")
    h = decision.attribute("harm", units="events")
    affine = decision.Value(b - rate * h, units="points")
    kinked = decision.Value(b - rate * h - weight * decision.excess(h, 1.2), units="points")

    def contract(model: decision.Value) -> decision.Contract:
        return decision.Contract(
            value=model,
            actions=(
                decision.Action(
                    "high",
                    inputs={"benefit": _dose("b", 1.5), "harm": _dose("h", 1.5, "events")},
                ),
                decision.Action(
                    "low", inputs={"benefit": _dose("b", 0.5), "harm": _dose("h", 0.5, "events")}
                ),
            ),
            target_population="target",
        )

    means = require_success(contract(affine).compile([benefit, harm]))
    assert means.inspect()["joint_law"] is None
    assert require_success(means.execute()).selected == "high"

    with pytest.raises(decision.PreferenceRefusal) as undeclared:
        require_success(contract(kinked).compile([benefit, harm]))
    assert undeclared.value.reason_code == "joint_law_required"

    joined = require_success(contract(kinked).compile([benefit, harm], draws=2000, seed=7))
    joint = joined.inspect()["joint_law"]
    assert joint["dependence"] == "distinct_evidence"
    assert len(joint["inputs"]) == 4
    law = joined.joint_law
    assert law is not None
    assert (law.dependence, law.draws, law.seed, law.cross_blocks) == (
        "distinct_evidence",
        2000,
        7,
        (),
    )
    assert sorted(column.study for column in law.inputs) == [0, 0, 1, 1]
    assert means.joint_law is None
    decided = require_success(joined.execute())
    assert isinstance(decided.source, JointDistributionArtifact)

    # Two outcomes of one table share every row: not distinct evidence.
    table = _dose_table("b", 200, 1.2, 1.7)
    table["h"] = _dose_table("h", 200, 0.5, 2.9)["h"]
    first = _prepared(table, "b", bayes=True)
    second = _prepared(table, "h", bayes=True)
    with pytest.raises(decision.PreferenceRefusal) as shared:
        require_success(contract(kinked).compile([first, second], draws=2000))
    assert shared.value.reason_code == "evidence_double_counted"
    assert shared.value.detail == reason_details.preference_compile.STUDIES_SHARE_EVIDENCE


# Studies of different tables observe the same patients only through a declared unit key.


def _keyed(outcome: str, rows: int, slope: float, rate: float, patients: list[str], **kwargs):
    table = _dose_table(outcome, rows, slope, rate)
    table["patient"] = np.array(patients, dtype=object)
    return ant.prepare(
        table,
        graph=[("t", outcome)],
        query=ant.ResponseCurve("t", outcome, grid=DOSES),
        refute="none",
        inference=ant.Bayesian(backend="conjugate", n_draws=64),
        execution=ant.Execution(seed=11),
        **kwargs,
    )


def _joined_contract() -> decision.Contract:
    rate = decision.tradeoff(
        "harm_rate", units="points/events", lower=0.1, upper=0.3, assumption="h"
    )
    weight = decision.tradeoff(
        "excess", units="points/events", lower=0.0, upper=2.0, assumption="x"
    )
    b = decision.attribute("benefit", units="points")
    h = decision.attribute("harm", units="events")
    return decision.Contract(
        value=decision.Value(b - rate * h - weight * decision.excess(h, 1.2), units="points"),
        actions=(
            decision.Action(
                "high", inputs={"benefit": _dose("b", 1.5), "harm": _dose("h", 1.5, "events")}
            ),
            decision.Action(
                "low", inputs={"benefit": _dose("b", 0.5), "harm": _dose("h", 0.5, "events")}
            ),
        ),
        target_population="target",
    )


def test_unit_keys_judge_studies_of_different_tables_by_the_patients_they_share():
    key = ant.UnitKey("patient", namespace="hospital_mrn")
    benefit = _keyed("b", 200, 1.2, 1.7, [f"p{i}" for i in range(200)], unit_key=key)
    # Patients 100-199 are in both tables.
    harm = _keyed("h", 200, 0.5, 2.9, [f"p{i}" for i in range(100, 300)], unit_key=key)
    with pytest.raises(decision.PreferenceRefusal) as shared:
        require_success(_joined_contract().compile([benefit, harm], draws=2000, seed=7))
    assert shared.value.reason_code == "evidence_double_counted"
    assert shared.value.detail == reason_details.preference_compile.STUDIES_SHARE_EVIDENCE

    # No patient in both tables: distinct evidence.
    elsewhere = _keyed("h", 200, 0.5, 2.9, [f"q{i}" for i in range(200)], unit_key=key)
    joined = require_success(_joined_contract().compile([benefit, elsewhere], draws=2000, seed=7))
    assert joined.inspect()["joint_law"]["dependence"] == "distinct_evidence"

    # Keys of another identifier space, or no key at all, leave the overlap unknown.
    insurer = ant.UnitKey("patient", namespace="insurer_id")
    for other in (
        _keyed("h", 200, 0.5, 2.9, [f"q{i}" for i in range(200)], unit_key=insurer),
        _keyed("h", 200, 0.5, 2.9, [f"q{i}" for i in range(200)]),
    ):
        with pytest.raises(decision.PreferenceRefusal) as unknown:
            require_success(_joined_contract().compile([benefit, other], draws=2000, seed=7))
        assert unknown.value.reason_code == "evidence_overlap_unknown"


def test_a_unit_key_is_checked_where_it_is_declared():
    from antecedent.errors import CausalUnsupportedError

    with pytest.raises(CausalValueError):
        ant.UnitKey("patient", namespace=" ")
    with pytest.raises(CausalValueError):
        _keyed("b", 200, 1.2, 1.7, [f"p{i}" for i in range(200)], unit_key=ant.UnitKey("mrn", "h"))
    missing = [f"p{i}" for i in range(199)] + [None]
    with pytest.raises(CausalValueError):
        _keyed("b", 200, 1.2, 1.7, missing, unit_key=ant.UnitKey("patient", "h"))
    table = _dose_table("y")
    table["patient"] = np.array([f"p{i}" for i in range(len(table["t"]))], dtype=object)
    with pytest.raises(CausalUnsupportedError) as refused:
        ant.prepare(
            table,
            graph=[("t", "y")],
            query=ant.AverageEffect("t", "y", control_level=0.5, active_level=1.5),
            unit_key=ant.UnitKey("patient", namespace="hospital_mrn"),
        )
    assert refused.value.reason_code == "option_not_applicable"


def test_projections_of_one_table_keep_their_source_rows():
    # Each study keeps only its graph columns of one table: the rows are still the table's.
    table = _dose_table("b", 200, 1.2, 1.7)
    table["h"] = _dose_table("h", 200, 0.5, 2.9)["h"]
    first = _prepared(table, "b", bayes=True)
    second = _prepared(table, "h", bayes=True)
    with pytest.raises(decision.PreferenceRefusal) as shared:
        require_success(_joined_contract().compile([first, second], draws=2000, seed=7))
    assert shared.value.reason_code == "evidence_double_counted"
    # A copy of the table with other values is another source.
    other = {name: np.array(values, copy=True) for name, values in table.items()}
    other["h"] = other["h"] + 1.0
    third = _prepared(other, "h", bayes=True)
    joined = require_success(_joined_contract().compile([first, third], draws=2000, seed=7))
    assert joined.inspect()["joint_law"]["dependence"] == "distinct_evidence"


def test_a_projection_and_a_study_of_the_whole_table_share_its_rows():
    # One study reads every column of the table; the other drops a column. Both name the
    # same source table, so they are judged as sharing its rows.
    table = _dose_table("b", 200, 1.2, 1.7)
    table["h"] = _dose_table("h", 200, 0.5, 2.9)["h"]
    whole = ant.prepare(
        table,
        graph=[("t", "b"), ("t", "h")],
        query=ant.ResponseCurve("t", "b", grid=DOSES),
        refute="none",
        inference=ant.Bayesian(backend="conjugate", n_draws=64),
        execution=ant.Execution(seed=11),
    )
    projected = _prepared(table, "h", bayes=True)
    with pytest.raises(decision.PreferenceRefusal) as shared:
        require_success(_joined_contract().compile([whole, projected], draws=2000, seed=7))
    assert shared.value.reason_code == "evidence_double_counted"
    # Either order names the one table.
    with pytest.raises(decision.PreferenceRefusal) as reversed_order:
        require_success(_joined_contract().compile([projected, whole], draws=2000, seed=7))
    assert reversed_order.value.reason_code == "evidence_double_counted"


# --------------------------------------------------------------------------- arguments and seeds


def test_compile_refuses_an_argument_its_source_does_not_read():
    study = _prepared(_dose_table(), bayes=True)
    contract = _dose_contract(_priced(0.2, 1.0, kink=2.0))
    block = {(0, 1): [[1e-5, 1e-5], [1e-5, 1e-5]]}
    for arguments in (
        {"draws": 2000},
        {"shared": block},
        {"refits": 99},
        {"sampling": SAMPLING},
        {"seed": 3},
    ):
        with pytest.raises(CausalValueError) as one:
            require_success(contract.compile(study, **arguments))  # type: ignore[arg-type]
        assert one.value.reason_code == "option_not_applicable"

    benefit = _prepared(_dose_table("b", 220, 1.2, 1.7), "b", bayes=True)
    harm = _prepared(_dose_table("h", 260, 0.5, 2.9), "h", bayes=True)
    joined = _joined_contract()
    for arguments in ({"refits": 99}, {"sampling": SAMPLING}, {"seed": 3}):
        with pytest.raises(CausalValueError) as several:
            require_success(joined.compile([benefit, harm], **arguments))  # type: ignore[arg-type]
        assert several.value.reason_code == "option_not_applicable"

    design = _cluster_design()
    kinked = _design_contract(
        decision.Value(BENEFIT - _weight(0.1, 0.3) * decision.excess(HARM, 1.2), units="qaly")
    )
    for arguments in ({"draws": 2000}, {"shared": block}, {"seed": 3}):
        with pytest.raises(CausalValueError) as on_design:
            require_success(kinked.compile(design, **arguments))  # type: ignore[arg-type]
        assert on_design.value.reason_code == "option_not_applicable"


def test_shared_evidence_reads_a_block_of_rows_by_columns():
    # Two outcomes of one table share every row; the declared block is rows of the first
    # study's two doses by columns of the second's.
    table = _dose_table("b", 200, 1.2, 1.7)
    table["h"] = _dose_table("h", 200, 0.5, 2.9)["h"]
    first = _prepared(table, "b", bayes=True)
    second = _prepared(table, "h", bayes=True)
    block = [[1e-5, 2e-5], [3e-5, 4e-5]]
    compiled = require_success(
        _joined_contract().compile([first, second], shared={(0, 1): block}, seed=7)
    )
    assert compiled.joint_law is not None
    assert compiled.joint_law.dependence == "shared_evidence"
    assert compiled.joint_law.cross_blocks == ((0, 1),)
    declared = compiled.inspect()["cross_study_covariance"]
    assert declared == [{"first": 0, "second": 1, "covariance": [1e-5, 2e-5, 3e-5, 4e-5]}]

    # A NumPy block keyed the other way round is read as its transpose.
    swapped = require_success(
        _joined_contract().compile([first, second], shared={(1, 0): np.array(block).T}, seed=7)
    )
    assert swapped.inspect()["cross_study_covariance"] == declared

    for wrong in ([[1e-5, 1e-5, 1e-5], [1e-5, 1e-5, 1e-5]], [1e-5, 2e-5, 3e-5, 4e-5]):
        with pytest.raises(CausalValueError, match="one row per coordinate of study 0"):
            require_success(_joined_contract().compile([first, second], shared={(0, 1): wrong}))
    with pytest.raises(CausalValueError, match="two different studies"):
        require_success(_joined_contract().compile([first, second], shared={(0, 2): block}))


def test_the_execution_seed_seeds_the_chain_draws_and_the_refits():
    benefit = _prepared(_dose_table("b", 220, 1.2, 1.7), "b", bayes=True)
    harm = _prepared(_dose_table("h", 260, 0.5, 2.9), "h", bayes=True)
    contract = _joined_contract()

    def chain_seed(**arguments: Any) -> int:
        compiled = require_success(contract.compile([benefit, harm], draws=2000, **arguments))
        assert compiled.joint_law is not None
        return compiled.joint_law.seed

    assert chain_seed(execution=ant.Execution(seed=5)) == 5
    assert chain_seed(execution=ant.Execution(seed=6)) == 6
    assert chain_seed() == ant.Execution().seed
    # An explicit seed= wins over the execution's.
    assert chain_seed(seed=8, execution=ant.Execution(seed=5)) == 8

    kinked = _design_contract(
        decision.Value(BENEFIT - _weight(0.1, 0.3) * decision.excess(HARM, 1.2), units="qaly")
    )
    refits = require_success(
        kinked.compile(
            _cluster_design(), refits=99, sampling=SAMPLING, execution=ant.Execution(seed=4)
        )
    )
    assert refits.design_law is not None
    assert refits.design_law.seed == 4


# --------------------------------------------------------------------------- refusals of the route


def test_a_cancelled_compile_or_execution_is_a_preference_refusal():
    token = ant.CancellationToken()
    token.cancel()
    cancelled = ant.Execution(cancel=token)
    prepared = _prepared(_dose_table())
    contract = _dose_contract(_priced(0.2, 1.0))

    # The causal compiler stops at the cancelled token: a compile refusal, not a bare error.
    with pytest.raises(decision.PreferenceRefusal) as at_compile:
        require_success(contract.compile(prepared, execution=cancelled))
    assert at_compile.value.reason_code == "cancelled_no_claim"
    assert at_compile.value.detail == reason_details.preference_compile.CANCELLED

    compiled = require_success(contract.compile(prepared))
    with pytest.raises(decision.PreferenceRefusal) as at_execute:
        require_success(compiled.execute(execution=cancelled))
    assert at_execute.value.reason_code == "cancelled_no_claim"
    assert at_execute.value.detail == reason_details.preference_compile.CANCELLED

    # A compiled design is stopped the same way.
    affine = _design_contract(decision.Value(BENEFIT - _weight(0.1, 0.3) * HARM, units="qaly"))
    with pytest.raises(decision.PreferenceRefusal) as on_design:
        require_success(affine.compile(_cluster_design(), execution=cancelled))
    assert on_design.value.detail == reason_details.preference_compile.CANCELLED


def test_policies_are_checked_where_they_are_declared():
    with pytest.raises(CausalValueError, match="Policy.invariant_best"):
        decision.TradeoffPolicy("minimax")  # type: ignore[arg-type]
    model = decision.Value(BENEFIT - _weight() * HARM, units="qaly")
    with pytest.raises(CausalValueError, match="structural_policy"):
        _contract(model, structural_policy="minimax")
    with pytest.raises(CausalValueError):
        StructuralPolicyClass("minimax")  # type: ignore[arg-type]
    # The typed structural policy and its kind as a string are the same contract.
    typed = _contract(model, structural_policy=StructuralPolicyClass.maximin())
    named = _contract(model, structural_policy="maximin")
    assert typed.structural_policy == named.structural_policy == "maximin"
    assert typed.identity == named.identity


# --------------------------------------------------------------------------- hard constraints


def test_a_hard_constraint_excludes_an_action_whatever_the_trade_off():
    model = decision.Value(BENEFIT - _weight() * HARM, units="qaly")
    # treat has harm 0 or 1, each with weight one half; wait has harm 0.
    cap = decision.Constraint("harm_cap", HARM, bound=0.5, units="events")
    result = _decide(_contract(model, constraints=(cap,)))
    by_id = {a.id: a for a in result.actions}
    assert not by_id["treat"].admissible
    (exclusion,) = by_id["treat"].exclusions
    assert exclusion.constraint_id == "harm_cap"
    assert exclusion.probability == pytest.approx(0.5)
    assert exclusion.required == pytest.approx(1.0)
    assert by_id["wait"].admissible
    # Without the cap the choice depends on the trade-off; with it no trade-off buys treat back.
    assert _decide(_contract(model)).kind == "preference_dependent"
    assert result.selected == "wait"

    nothing = decision.Constraint("no_harm_at_all", HARM, bound=-1.0, units="events")
    none_left = _decide(_contract(model, constraints=(nothing,)))
    assert none_left.kind == "no_admissible_action"
    assert none_left.selected is None

    with pytest.raises(CausalValueError, match="never a trade-off"):
        decision.Constraint("reads_a_weight", _weight() * HARM, bound=1.0, units="events")


def test_a_hard_constraint_demands_draws_at_compile():
    # An affine value read by means alone; a constraint probability needs a law of the outcome.
    benefit = decision.attribute("benefit", units="points")
    model = _priced(0.2, 1.0)
    affine = _dose_contract(model)
    capped = _dose_contract(
        model,
        constraints=(decision.Constraint("benefit_cap", benefit, bound=100.0, units="points"),),
    )
    prepared = _prepared(_dose_table(), bayes=True)
    assert require_success(affine.compile(prepared)).representation == "mean"
    compiled = require_success(capped.compile(prepared))
    assert compiled.representation == "joint_draws"
    decided = require_success(compiled.execute())
    assert all(a.admissible for a in decided.actions)
    assert decided.selected == "high"


# --------------------------------------------------------------------------- structures in hand


def test_structures_in_hand_are_decided_with_the_trade_offs():
    from antecedent.decision_robust import Claim

    model = decision.Value(BENEFIT - _weight(0.5, 1.5) * HARM, units="qaly")
    # Under both structures treat leads for every weight in [0.5, 1.5]: margins 1 - 0.5 w and
    # 2 - 0.5 w.
    stronger = _source([[1.0, 0.0, 1.0, 0.0], [5.0, 1.0, 1.0, 0.0]])
    claims = (
        Claim.evaluated("direct", _source(), probability=0.6),
        Claim.evaluated("stronger", stronger, probability=0.3),
        Claim.unidentified("cycle", probability=0.1),
    )
    reported = require_success(
        _contract(model).evaluate(decision.Structures.weighted_atoms(claims))
    )
    by_id = {s.id: s for s in reported.structures}
    assert (by_id["direct"].status, by_id["direct"].selected) == ("evaluated", "treat")
    assert (by_id["stronger"].status, by_id["stronger"].selected) == ("evaluated", "treat")
    assert by_id["cycle"].status == "unidentified"
    assert reported.unidentified_mass == pytest.approx(0.1)
    # The kind as a string is the same set of structures.
    named = require_success(
        _contract(model).evaluate(decision.Structures("weighted_graph_posterior", claims))
    )
    assert [(s.id, s.kind) for s in named.structures] == [
        (s.id, s.kind) for s in reported.structures
    ]

    # Every structure evaluated, the worst case over them still leaves treat best everywhere.
    maximin = _contract(model, structural_policy="maximin")
    combined = require_success(maximin.evaluate(decision.Structures.weighted_atoms(claims[:2])))
    assert combined.selected == "treat"


# --------------------------------------------------------------------------- compiled artifacts


def test_a_compiled_decision_on_draws_exports_and_replays():
    prepared = _prepared(_dose_table(), bayes=True)
    contract = _dose_contract(_priced(0.2, 1.0, kink=2.0))
    decided = require_success(require_success(contract.compile(prepared)).execute())
    assert isinstance(decided.source, JointDistributionArtifact)
    exported = decided.export()
    replayed = require_success(decision.replay(exported, contract=contract, source=decided.source))
    assert replayed.kind == decided.kind
    assert replayed.selected == decided.selected
    assert [a.value_range for a in replayed.actions] == [a.value_range for a in decided.actions]

    # A decision the compiled route answered from means has no draws to replay on.
    by_means = require_success(
        require_success(_dose_contract(_priced(0.2, 1.0)).compile(prepared)).execute()
    )
    with pytest.raises(decision.PreferenceRefusal) as refused:
        by_means.export()
    assert refused.value.reason_code == "route_not_supported"
    assert refused.value.detail == reason_details.decision_evaluation.MEAN_SOURCE_NOT_REPLAYABLE


# --------------------------------------------------------------------------- preparation


def test_a_table_the_engine_cannot_read_whole_is_refused_not_identified_otherwise():
    # A column of two values per row cannot be ingested as a table column. A static response
    # study's rows are identified by the engine's identity of the whole table they were taken
    # from, so such a table is refused at preparation rather than identified by another scheme.
    table = _dose_table("b", 200, 1.2, 1.7)
    table["h"] = _dose_table("h", 200, 0.5, 2.9)["h"]
    table["notes"] = np.zeros((200, 2))
    with pytest.raises(CausalValueError) as refused:
        _prepared(table, "b", bayes=True)
    assert refused.value.reason_code == "invalid_argument"
    # Without the column the engine cannot read, both studies name the one table.
    del table["notes"]
    first = _prepared(table, "b", bayes=True)
    second = _prepared(table, "h", bayes=True)
    with pytest.raises(decision.PreferenceRefusal) as shared:
        require_success(_joined_contract().compile([first, second], draws=2000, seed=7))
    assert shared.value.reason_code == "evidence_double_counted"
