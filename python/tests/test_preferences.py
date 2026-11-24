"""Decisions across explicitly stated trade-off ranges (no recorded choices).

Fixture: two actions over aligned joint draws of benefit (qaly) and harm (events).

    treat: benefit rows [1, 3] (mean 2), harm rows [0, 1] (mean 0.5)
    wait:  benefit rows [1, 1] (mean 1), harm rows [0, 0] (mean 0)

With value ``benefit - w * harm`` the margin of treat over wait is ``1 - 0.5 w``: treat leads
below ``w = 2`` and wait above it, so every answer below is known in closed form.
"""

from __future__ import annotations

import antecedent as ant
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


def _priced(lower: float, upper: float, *, kink: float | None = None) -> pf.ValueModel:
    benefit = pf.attribute("benefit", units="points")
    cost = pf.attribute("cost", units="usd")
    price = pf.strength("price", units=PRICE_UNITS, lower=lower, upper=upper, assumption="price")
    value = benefit - price * cost
    if kink is not None:
        weight = pf.strength(
            "shortfall_weight", units="points/points", lower=0.0, upper=2.0, assumption="floor"
        )
        value = value - weight * pf.shortfall(benefit, kink)
    return pf.ValueModel(value, utility_units="points")


def _dose_contract(model: pf.ValueModel, **kwargs: object) -> pf.Contract:
    return pf.Contract(
        model=model,
        actions=(
            pf.Action("low", inputs={"benefit": _dose("y", 0.5), "cost": 0.5}),
            pf.Action("high", inputs={"benefit": _dose("y", 1.5), "cost": 1.5}),
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
    with pytest.raises(pf.PreferenceRefusal) as refused:
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
    rate = pf.strength("harm_rate", units="points/events", lower=0.1, upper=0.3, assumption="h")
    weight = pf.strength("excess", units="points/events", lower=0.0, upper=2.0, assumption="x")
    b = pf.attribute("benefit", units="points")
    h = pf.attribute("harm", units="events")
    affine = pf.ValueModel(b - rate * h, utility_units="points")
    kinked = pf.ValueModel(b - rate * h - weight * pf.excess(h, 1.2), utility_units="points")

    def contract(model: pf.ValueModel) -> pf.Contract:
        return pf.Contract(
            model=model,
            actions=(
                pf.Action(
                    "high",
                    inputs={"benefit": _dose("b", 1.5), "harm": _dose("h", 1.5, "events")},
                ),
                pf.Action(
                    "low", inputs={"benefit": _dose("b", 0.5), "harm": _dose("h", 0.5, "events")}
                ),
            ),
            target_population="target",
        )

    means = require_success(contract(affine).compile([benefit, harm]))
    assert means.inspect()["joint_law"] is None
    assert require_success(means.execute()).selected == "high"

    with pytest.raises(pf.PreferenceRefusal) as undeclared:
        require_success(contract(kinked).compile([benefit, harm]))
    assert undeclared.value.reason_code == "joint_law_required"

    joined = require_success(contract(kinked).compile([benefit, harm], draws=2000, seed=7))
    joint = joined.inspect()["joint_law"]
    assert joint["dependence"] == "distinct_evidence"
    assert len(joint["inputs"]) == 4
    decided = require_success(joined.execute())
    assert isinstance(decided.source, JointDistributionArtifact)

    # Two outcomes of one table share every row: not distinct evidence.
    table = _dose_table("b", 200, 1.2, 1.7)
    table["h"] = _dose_table("h", 200, 0.5, 2.9)["h"]
    first = _prepared(table, "b", bayes=True)
    second = _prepared(table, "h", bayes=True)
    with pytest.raises(pf.PreferenceRefusal) as shared:
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


def _joined_contract() -> pf.Contract:
    rate = pf.strength("harm_rate", units="points/events", lower=0.1, upper=0.3, assumption="h")
    weight = pf.strength("excess", units="points/events", lower=0.0, upper=2.0, assumption="x")
    b = pf.attribute("benefit", units="points")
    h = pf.attribute("harm", units="events")
    return pf.Contract(
        model=pf.ValueModel(b - rate * h - weight * pf.excess(h, 1.2), utility_units="points"),
        actions=(
            pf.Action(
                "high", inputs={"benefit": _dose("b", 1.5), "harm": _dose("h", 1.5, "events")}
            ),
            pf.Action(
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
    with pytest.raises(pf.PreferenceRefusal) as shared:
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
        with pytest.raises(pf.PreferenceRefusal) as unknown:
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
    with pytest.raises(pf.PreferenceRefusal) as shared:
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
    with pytest.raises(pf.PreferenceRefusal) as shared:
        require_success(_joined_contract().compile([whole, projected], draws=2000, seed=7))
    assert shared.value.reason_code == "evidence_double_counted"
    # Either order names the one table.
    with pytest.raises(pf.PreferenceRefusal) as reversed_order:
        require_success(_joined_contract().compile([projected, whole], draws=2000, seed=7))
    assert reversed_order.value.reason_code == "evidence_double_counted"
