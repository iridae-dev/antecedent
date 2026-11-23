"""Decisions across explicitly stated trade-off ranges (no recorded choices).

Fixture: two actions over aligned joint draws of benefit (qaly) and harm (events).

    treat: benefit rows [1, 3] (mean 2), harm rows [0, 1] (mean 0.5)
    wait:  benefit rows [1, 1] (mean 1), harm rows [0, 0] (mean 0)

With value ``benefit - w * harm`` the margin of treat over wait is ``1 - 0.5 w``: treat leads
below ``w = 2`` and wait above it, so every answer below is known in closed form.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import _reason_details as reason_details
from antecedent import decision
from antecedent import preferences as pf
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

BENEFIT = pf.attribute("benefit", units="qaly")
HARM = pf.attribute("harm", units="events")


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


def _weight(lower: float = 0.5, upper: float = 4.0) -> pf.Parameter:
    return pf.strength(
        "harm_weight", units="qaly/event", lower=lower, upper=upper, assumption="panel_judgment"
    )


def _actions() -> tuple[pf.Action, ...]:
    return (
        pf.Action("treat", inputs={"benefit": TREAT_BENEFIT, "harm": TREAT_HARM}),
        pf.Action("wait", inputs={"benefit": WAIT_BENEFIT, "harm": WAIT_HARM}),
    )


def _contract(model: pf.ValueModel, **kwargs: object) -> pf.Contract:
    return pf.Contract(
        model=model,
        actions=_actions(),
        target_population="target",
        **kwargs,  # type: ignore[arg-type]
    )


def _decide(contract: pf.Contract, **kwargs: object) -> pf.PreferenceDecision:
    return require_success(contract.evaluate(_source(), **kwargs))


def test_fixed_trade_off_reproduces_the_ordinary_decision():
    w = pf.fixed("harm_weight", units="qaly/event", value=1.0)
    result = _decide(_contract(pf.ValueModel(BENEFIT - w * HARM, utility_units="qaly")))

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


def test_a_range_that_crosses_the_switch_reports_where_the_choice_changes():
    model = pf.ValueModel(BENEFIT - _weight() * HARM, utility_units="qaly")
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
    narrow = _decide(
        _contract(pf.ValueModel(BENEFIT - _weight(0.5, 1.5) * HARM, utility_units="qaly"))
    )
    assert narrow.kind == "necessarily_best"
    assert narrow.selected == "treat"
    treat = {a.id: a for a in narrow.actions}["treat"]
    assert treat.value_range == pytest.approx((1.25, 1.75))


def test_a_relation_between_trade_offs_cuts_the_set():
    cost = pf.attribute("cost", units="usd")
    w_harm = pf.strength(
        "harm_weight", units="qaly/event", lower=0.0, upper=1.0, assumption="panel"
    )
    w_cost = pf.strength("cost_weight", units="qaly/usd", lower=0.0, upper=1.0, assumption="payer")
    value = BENEFIT - w_harm * HARM - w_cost * cost
    actions = (
        pf.Action("treat", inputs={"benefit": TREAT_BENEFIT, "harm": TREAT_HARM, "cost": 1.0}),
        pf.Action("wait", inputs={"benefit": WAIT_BENEFIT, "harm": WAIT_HARM, "cost": 0.0}),
    )

    def decide(model: pf.ValueModel) -> pf.PreferenceDecision:
        contract = pf.Contract(model=model, actions=actions, target_population="target")
        return require_success(contract.evaluate(_source()))

    # Without a relation the margin 1 - 0.5 w_harm - w_cost reaches -0.5.
    assert decide(pf.ValueModel(value, utility_units="qaly")).kind == "preference_dependent"

    rule = pf.relation("harm_outweighs_cost", 4 * w_cost <= w_harm)
    assert rule.id == "harm_outweighs_cost"
    with pytest.raises(TypeError, match="not a truth value"):
        bool(w_harm >= w_cost)
    result = decide(pf.ValueModel(value, utility_units="qaly", relations=(rule,)))
    # w_cost <= w_harm / 4 <= 0.25, so the margin stays at least 0.25.
    assert result.kind == "necessarily_best"
    assert result.selected == "treat"
    assert result.parameter_ranges["cost_weight"] == pytest.approx((0.0, 0.25))

    with pytest.raises(CausalValueError, match="not the attribute 'harm'"):
        pf.relation("reads_an_attribute", w_harm >= HARM)


def test_a_curve_with_shape_rules_finds_the_switch_on_its_slope():
    low = pf.strength("slope_low", units="qaly/qaly", lower=0.0, upper=2.0, assumption="shape")
    high = pf.strength("slope_high", units="qaly/qaly", lower=0.0, upper=2.0, assumption="shape")
    curve = pf.Curve(BENEFIT, breakpoints=(0.0, 1.0, 3.0), slopes=(low, high))
    rules = (*curve.increasing(), *curve.diminishing())
    assert [r.id for r in curve.diminishing()] == ["benefit.curve.diminishing.0"]
    harm_weight = pf.fixed("harm_weight", units="qaly/event", value=1.0)
    model = pf.ValueModel(curve - harm_weight * HARM, utility_units="qaly", relations=rules)
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
    model = pf.ValueModel(BENEFIT - _weight() * HARM, utility_units="qaly")
    # treat ranges over [0, 1.75], wait is 1: the worst case of treat is lower.
    maximin = _decide(_contract(model, policy=pf.Policy.maximin()))
    assert maximin.kind == "worst_case_choice"
    assert maximin.selected == "wait"
    # Worst regret: treat 1 (at w = 4), wait 0.75 (at w = 0.5).
    regret = _decide(_contract(model, policy=pf.Policy.minimax_regret()))
    assert regret.kind == "minimax_regret_choice"
    assert regret.selected == "wait"
    by_id = {a.id: a for a in regret.actions}
    assert by_id["treat"].worst_case_regret == pytest.approx(1.0)
    assert by_id["wait"].worst_case_regret == pytest.approx(0.75)


def test_a_measure_of_named_viewpoints_weighs_the_range():
    model = pf.ValueModel(BENEFIT - _weight() * HARM, utility_units="qaly")
    views = pf.Measure.viewpoints(
        {"clinician": {"harm_weight": 1.0}, "patient": {"harm_weight": 3.0}},
        weights={"clinician": 3.0, "patient": 1.0},
        assumption="stakeholder_weights",
    )
    contract = _contract(model, policy=pf.Policy.weighted())
    result = _decide(contract, measure=views)
    # treat is 1.5 at w = 1 and 0.5 at w = 3; wait is 1 at both.
    assert result.kind == "measure_choice"
    assert result.selected == "treat"
    treat = {a.id: a for a in result.actions}["treat"]
    assert treat.measure_share == pytest.approx(0.75)
    assert treat.measure_value == pytest.approx(1.25)
    assert result.measure is not None
    assert result.measure.labels == ("clinician", "patient")

    with pytest.raises(pf.PreferenceRefusal) as missing:
        _decide(contract)
    assert missing.value.detail == reason_details.preference_decision.MEASURE_REQUIRED


def test_a_shape_parameter_is_answered_by_certified_subdivision():
    w = pf.strength("harm_weight", units="qaly/event", lower=0.5, upper=1.0, assumption="panel")
    model = pf.ValueModel(BENEFIT - w * w * HARM, utility_units="qaly")
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
    with pytest.raises(pf.PreferenceRefusal) as needs_draws:
        require_success(contract.evaluate(means))
    assert needs_draws.value.detail == reason_details.preference_evaluation.JOINT_DRAWS_REQUIRED


def test_a_mean_source_answers_the_exact_route():
    # A mean source carries means, so its coordinates are the ``mean`` functional.
    mean_columns = tuple(_quantity(q.variable_id, q.units, q.regime_id, "mean") for q in COLUMNS)
    model = pf.ValueModel(BENEFIT - _weight(0.5, 1.5) * HARM, utility_units="qaly")
    contract = pf.Contract(
        model=model,
        actions=(
            pf.Action("treat", inputs={"benefit": mean_columns[0], "harm": mean_columns[1]}),
            pf.Action("wait", inputs={"benefit": mean_columns[2], "harm": mean_columns[3]}),
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
    reversed_range = pf.strength(
        "harm_weight", units="qaly/event", lower=2.0, upper=1.0, assumption="x"
    )
    with pytest.raises(pf.PreferenceRefusal) as caught:
        pf.ValueModel(BENEFIT - reversed_range * HARM, utility_units="qaly").validate()
    assert caught.value.detail == reason_details.preference_model.PARAMETER_RANGE_EMPTY

    model = pf.ValueModel(BENEFIT - _weight() * HARM, utility_units="qaly")
    one_action = pf.Contract(model=model, actions=_actions()[:1], target_population="target")
    with pytest.raises(pf.PreferenceRefusal) as too_few:
        require_success(one_action.evaluate(_source()))
    assert too_few.value.detail == reason_details.preference_contract.TOO_FEW_ACTIONS

    with pytest.raises(CausalValueError, match="exactly once"):
        pf.Contract(
            model=model,
            actions=(pf.Action("treat", inputs={"benefit": TREAT_BENEFIT}), _actions()[1]),
            target_population="target",
        )._wire()


def test_export_and_replay_recompute_the_decision():
    model = pf.ValueModel(BENEFIT - _weight() * HARM, utility_units="qaly")
    contract = _contract(model)
    result = _decide(contract)
    exported = result.export()
    replayed = require_success(pf.replay(exported, contract=contract, source=_source()))
    assert replayed.kind == result.kind
    assert replayed.contract_identity == contract.identity
    assert replayed.switches[0].margin_range == pytest.approx(result.switches[0].margin_range)

    other = _source([[1.0, 0.0, 1.0, 0.0], [5.0, 1.0, 1.0, 0.0]])
    with pytest.raises(pf.PreferenceRefusal):
        require_success(pf.replay(exported, contract=contract, source=other))


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
    model = pf.ValueModel(BENEFIT - _weight(lower, upper) * HARM, utility_units="qaly")
    result = require_success(_contract(model).evaluate([BENEFITS, HARMS]))
    assert result.kind == kind
    assert result.selected == selected
    if kind == "preference_dependent":
        (switch,) = result.switches
        assert switch.threshold is not None
        assert switch.threshold.value == pytest.approx(RATIO)
        assert switch.threshold.first_leads_above is False

    # The joined law is an ordinary joint law: the decision exports and replays on it.
    joined = pf.joint_law(BENEFITS, HARMS)
    replayed = require_success(pf.replay(result.export(), contract=_contract(model), source=joined))
    assert replayed.kind == kind


def test_separately_sampled_laws_do_not_pair():
    harms = _paired_law((TREAT_HARM, WAIT_HARM), [[1.0, 0.5], [3.0, 0.5]], snapshot="draws-3")
    model = pf.ValueModel(BENEFIT - _weight() * HARM, utility_units="qaly")
    with pytest.raises(pf.PreferenceRefusal) as unpaired:
        require_success(_contract(model).evaluate([BENEFITS, harms]))
    assert unpaired.value.reason_code == "joint_law_required"


def test_a_propagation_chain_joins_its_outputs_under_the_declared_dependence():
    # A predictive law answers a mean-like functional, not an interventional outcome threshold.
    T_BEN, T_HARM, W_BEN, W_HARM = (
        _quantity(q.variable_id, q.units, q.regime_id, "mean") for q in COLUMNS
    )
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
    model = pf.ValueModel(BENEFIT - _weight(0.5, 3.0) * HARM, utility_units="qaly")
    actions = (
        pf.Action("treat", inputs={"benefit": T_BEN, "harm": T_HARM}),
        pf.Action("wait", inputs={"benefit": W_BEN, "harm": W_HARM}),
    )
    contract = pf.Contract(model=model, actions=actions, target_population="target")
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


def _experiment_contract(experiment, model: pf.ValueModel) -> pf.Contract:
    def arm(name: str) -> dict[str, ScientificQuantity]:
        return {o: pf.experiment_quantity(experiment, o, name) for o in ("benefit", "harm")}

    return pf.Contract(
        model=model,
        actions=(
            pf.Action("treat", inputs=arm("treated")),
            pf.Action("wait", inputs=arm("control")),
        ),
        target_population="target",
    )


def test_one_experiment_switches_at_the_ratio_of_its_estimated_differences():
    experiment = _experiment()
    treated, control = experiment.wire["treated_means"], experiment.wire["control_means"]
    ratio = (treated[0] - control[0]) / (treated[1] - control[1])
    assert ratio > 0.0

    quantity = pf.experiment_quantity(experiment, "harm", "control")
    assert quantity.functional_id == "mean"
    assert quantity.regime_id == "do(a=0)"

    for lower, upper, kind, selected in (
        (0.1 * ratio, 0.5 * ratio, "necessarily_best", "treat"),
        (1.5 * ratio, 3.0 * ratio, "necessarily_best", "wait"),
        (0.5 * ratio, 2.0 * ratio, "preference_dependent", None),
    ):
        model = pf.ValueModel(BENEFIT - _weight(lower, upper) * HARM, utility_units="qaly")
        result = require_success(_experiment_contract(experiment, model).evaluate(experiment))
        assert (result.kind, result.selected) == (kind, selected)
        if kind == "preference_dependent":
            (switch,) = result.switches
            assert switch.threshold is not None
            assert switch.threshold.value == pytest.approx(ratio)


def test_one_experiment_cannot_answer_a_kinked_term_from_its_means():
    experiment = _experiment()
    kinked = pf.ValueModel(BENEFIT - _weight() * pf.excess(HARM, 1.0), utility_units="qaly")
    with pytest.raises(pf.PreferenceRefusal):
        require_success(_experiment_contract(experiment, kinked).evaluate(experiment))
