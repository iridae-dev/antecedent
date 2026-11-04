"""Composable experimental evidence for decision-relevant outcomes (Python parity).

Mirrors ``crates/antecedent/tests/decision_evidence_workflow.rs``: a cluster-randomized
experiment with limited independent clusters, three dependent outcomes and a nonlinear margin.
"""

from __future__ import annotations

import math

import pytest
from antecedent import _reason_details as reason_details
from antecedent import decision_evidence as de
from antecedent.decision_evidence import (
    Assumed,
    DecisionEvidenceRefusal,
    DecisionFrame,
    DerivedQuantity,
    Expr,
    Measured,
    Missing,
)
from antecedent.errors import CausalValueError

from _refusal import assert_registered_refusal

ROWS = 12
COMPONENTS = {
    "frequency": ("outcome.frequency", "orders"),
    "value": ("outcome.value", "usd"),
    "cost": ("outcome.cost", "usd"),
}
NO_SPILL = "assumption.no_cross_customer_spillover"


def _shock(cluster: int, salt: float) -> float:
    return math.sin((cluster + 1.0) * 12.9898 + salt) * 0.5


def _evidence(clusters: int, **kwargs) -> de.JointEvidence:
    assigned = [c % 2 == 0 for c in range(clusters)]
    totals: list[list[float]] = [[], [], []]
    for c, treated in enumerate(assigned):
        u = _shock(c, 0.0)
        f = v = k = 0.0
        for r in range(ROWS):
            noise = math.sin((c * 31 + r * 17) * 0.7) * 0.2
            f += 2.0 + (0.3 if treated else 0.0) + 0.6 * u + noise
            v += 10.0 + (1.0 if treated else 0.0) + 2.0 * u + 0.5 * noise
            k += 6.0 + 0.1 * _shock(c, 3.0)
        for series, total in zip(totals, (f, v, k), strict=True):
            series.append(total)
    options = {
        "exposure": "no_between_unit_interference",
        "exposure_assumption_id": NO_SPILL,
        "cluster_ids": [f"cluster-{i}" for i in range(clusters)],
        "cluster_sizes": [ROWS] * clusters,
        "assignment_law_id": "complete-randomization",
        "treated_clusters": sum(assigned),
        "consistency_assumption_id": "consistency",
        "adjustment_preassignment_id": "preassignment",
        **kwargs,
    }
    return de.JointEvidence.from_cluster_totals(
        evidence_id="exp-shipping-2026q3",
        population_id="active_customers",
        treated_regime_id="free_shipping",
        control_regime_id="standard_shipping",
        horizon=4,
        components=COMPONENTS,
        totals=totals,
        assigned=assigned,
        outcome_rows=clusters * ROWS,
        **options,
    )


def _margin(cost) -> DerivedQuantity:
    freq, value, unit_cost = Expr.input(0), Expr.input(1), Expr.input(2)
    return DerivedQuantity(
        "margin_per_customer",
        "usd",
        freq * (value - unit_cost),
        [Measured("frequency"), Measured("value"), cost],
    )


FRAME = DecisionFrame("launch_free_shipping", "keep_standard", 0.0)
COST_RANGE = Assumed(4.0, 12.0, "assumption.unit_cost_range")


def test_cluster_evidence_refuses_unbound_numeric_edits():
    ev = _evidence(80)
    assert ev.wire["compiled_source_artifact"]
    edited = dict(ev.wire)
    edited["treated_means"] = list(edited["treated_means"])
    edited["treated_means"][0] += 1.0
    with pytest.raises(DecisionEvidenceRefusal):
        de.evaluate(de.JointEvidence(edited), _margin(Measured("cost")))


def test_nonlinear_margin_has_a_sampling_interval_and_inherits_no_support():
    result = de.evaluate(_evidence(80), _margin(Measured("cost")))
    d = result.derived
    assert d["contrast"] > 0 and d["sampling"]["standard_error"] > 0
    assert (d["support"], d["calibration"]) == ("none", "unmeasured")
    assert d["standing"] == "sampled"


def test_joint_dependence_is_retained_and_marginals_alone_cannot_answer():
    ev = _evidence(80)
    joint = de.evaluate(ev, _margin(Measured("cost"))).derived
    marginal = de.evaluate(ev.marginals_only(), _margin(Measured("cost"))).derived
    assert [r["code"] for r in marginal["unresolved"]] == ["joint_law_missing"]
    assert marginal["sampling"] is None and marginal["contrast"] == pytest.approx(joint["contrast"])


def test_decision_is_robust_assumption_dependent_or_unresolved():
    ev = _evidence(80)
    robust = de.evaluate(ev, _margin(Measured("cost")), frame=FRAME)
    assert robust.verdict == "robust"
    assert robust.verdict_detail["action"] == "launch_free_shipping"

    dependent = de.evaluate(
        ev, _margin(COST_RANGE), frame=DecisionFrame("launch_free_shipping", "keep_standard", 2.5)
    )
    assert dependent.verdict == "assumption_dependent"
    assert dependent.verdict_detail["assumption_ids"] == ["assumption.unit_cost_range"]
    assert dependent.derived["sampling"] is not None  # reported apart from the range
    assert dependent.derived["assumption_range"]["exact"] is True

    few = de.evaluate(_evidence(24), _margin(Measured("cost")), frame=FRAME)
    assert few.verdict == "unresolved"
    assert few.verdict_detail["reasons"][0]["code"] == "too_few_independent_units"

    missing = de.evaluate(ev, _margin(Missing("unit cost")), frame=FRAME)
    assert missing.verdict == "unresolved"


def test_out_of_scope_requests_are_typed_refusals():
    ev, q = _evidence(80), _margin(Measured("cost"))
    with pytest.raises(DecisionEvidenceRefusal) as caught:
        de.evaluate(ev, q, population_id="all_customers")
    assert_registered_refusal(caught.value)
    assert caught.value.detail == reason_details.decision_evidence.INCOMPATIBLE_POPULATION
    assert caught.value.remedy
    with pytest.raises(DecisionEvidenceRefusal) as caught:
        de.evaluate(ev, q, deployment=True)
    assert caught.value.detail == reason_details.decision_evidence.DEPLOYMENT_BRIDGE_MISSING
    assert de.evaluate(ev, q, deployment=True, bridge_assumption_id="assumption.rollout").derived[
        "sampling"
    ]
    with pytest.raises(DecisionEvidenceRefusal) as caught:
        de.evaluate(_evidence(80, exposure="unspecified"), q)
    assert caught.value.detail == reason_details.decision_evidence.INTERFERENCE_UNSPECIFIED


def test_compile_states_admissible_adjustments_before_any_fit():
    compiled = de.compile_experiment(
        outcomes=COMPONENTS,
        assignment_design="cluster",
        population_id="active_customers",
        treated_regime_id="free_shipping",
        control_regime_id="standard_shipping",
        horizon=4,
        exposure="no_between_unit_interference",
        exposure_assumption_id=NO_SPILL,
        adjustments=[
            {"component": "value", "kind": "fixed_cuped", "covariate": "pre", "coefficient": 0.5}
        ],
    )
    assert compiled["unit_of_analysis"] == "assignment cluster"
    verdicts = {v["strategy"]: v["admissible"] for v in compiled["adjustment_verdicts"]}
    assert verdicts == {"none": True, "fixed_cuped": True, "learned_regression": False}
    with pytest.raises(DecisionEvidenceRefusal) as caught:
        de.compile_experiment(
            outcomes=COMPONENTS,
            assignment_design="stepped_wedge",
            population_id="p",
            treated_regime_id="a",
            control_regime_id="b",
            horizon=1,
            exposure="unspecified",
        )
    assert caught.value.detail == reason_details.decision_evidence.UNSUPPORTED_ASSIGNMENT_DESIGN


def test_planning_reuse_and_withdrawal():
    ev = _evidence(80)
    bar = DecisionFrame("launch_free_shipping", "keep_standard", 2.5)
    first = de.evaluate(
        ev,
        _margin(COST_RANGE),
        frame=bar,
        candidates=[
            {
                "kind": "resolve_assumption",
                "id": "audit_unit_cost",
                "assumption_id": "assumption.unit_cost_range",
                "narrowed_lower": 5.5,
                "narrowed_upper": 6.5,
            },
            {
                "kind": "more_units",
                "id": "double_clusters",
                "additional_units_per_arm": 40,
                "cost": 0.01,
            },
        ],
    )
    assert first.plan["resolution_rows"][0]["value"]["resolves"] is False
    assert "refused" in first.plan["evsi_rows"][0]["value"]

    second = de.evaluate(
        ev, _margin(COST_RANGE), frame=DecisionFrame("launch_free_shipping", "keep_standard", 3.0)
    )
    receipt = de.reuse_receipt(first, second)
    assert receipt.reused("derived_quantity") and not receipt.reused("decision")

    ledger = de.EvidenceLedger()
    ledger.record(ev.evidence_id)
    assert not ledger.unsupported_parts(ev.evidence_id)
    ledger.withdraw(ev.evidence_id, "assignment log found corrupted")
    assert ledger.unsupported_parts(ev.evidence_id)


def test_arm_spillover_bounds_support_nonlinear_margin():
    ev = _evidence(
        80,
        exposure="bounded_arm_spillover",
        exposure_assumption_id="arm-bias",
        treated_spillover_bias_bounds=[0.1, 0.2, 0.0],
        control_spillover_bias_bounds=[0.1, 0.2, 0.0],
    )
    result = de.evaluate(ev, _margin(Measured("cost"))).derived
    assert result["assumption_range"]["assumption_ids"] == ["arm-bias"]
    assert (
        result["assumption_range"]["lower"]
        < result["contrast"]
        < result["assumption_range"]["upper"]
    )


def test_centered_fixed_cuped_supports_nonlinear_margin():
    n = 80
    ev = _evidence(
        n,
        adjustments=[
            {
                "kind": "centered_fixed_cuped",
                "component": "frequency",
                "covariate": "pre",
                "coefficient": 0.4,
            }
        ],
        cluster_sizes=[ROWS] * n,
        pretreatment_totals={"pre": [ROWS * (1 + 0.01 * c) for c in range(n)]},
    )
    assert de.evaluate(ev, _margin(Measured("cost"))).derived["sampling"] is not None


def test_matched_pair_design_keeps_joint_arms_and_scope():
    n = 40
    design = {
        "kind": "matched_pairs",
        "cluster_ids": [f"geo-{i}" for i in range(2 * n)],
        "pairs": [[2 * i, 2 * i + 1] for i in range(n)],
        "matching_variables": ["pre"],
        "treatment_probabilities": [0.5] * (2 * n),
        "cluster_sizes": [1] * (2 * n),
        "preassignment_assumption_id": "fixed-pairs",
        "independent_pairs_assumption_id": "independent-pairs",
        "consistency_assumption_id": "consistency",
    }
    prepared = de.compile_experimental_design(design)
    assert prepared["identity"] and prepared["obligations"]
    assigned = [i % 2 == 0 for i in range(2 * n)]
    values = [
        i // 2 + (1.0 + 0.01 * math.sin(i // 2) if assigned[i] else 0.0) for i in range(2 * n)
    ]
    ev = de.JointEvidence.from_experimental_design(
        design=design,
        outcomes=[values],
        assigned=assigned,
        evidence_id="e",
        population_id="pop",
        treated_regime_id="t",
        control_regime_id="c",
        horizon=1,
        components={"x": ("x", "u")},
        exposure="no_between_unit_interference",
        exposure_assumption_id="no-spill",
    )
    q = DerivedQuantity("effect", "u", Expr.input(0), [Measured("x")])
    expected = 1.0 + sum(0.01 * math.sin(i) for i in range(n)) / n
    assert de.evaluate(ev, q).derived["contrast"] == pytest.approx(expected)
    assert ev.wire["design_details"]["pairs"] == design["pairs"]
    with pytest.raises(CausalValueError):
        ev.marginals_only()


def test_proxy_bridge_rejects_predictive_association():
    ev = _evidence(80)
    bridge = {
        "proxy_component": "frequency",
        "target_variable_id": "profit",
        "target_units": "currency",
        "population_id": "active_customers",
        "treated_regime_id": "free_shipping",
        "control_regime_id": "standard_shipping",
        "horizon": 4,
        "mapping": {
            "kind": "causal_effect_bounds",
            "multiplier": [1.0, 1.5],
            "residual": [0.0, 0.0],
            "assumption_id": "causal-map",
            "mapping_evidence_id": "validation",
        },
    }
    result = de.decide_with_proxy(ev, bridge, FRAME)
    assert result["evidence_ids"] == ["exp-shipping-2026q3", "validation"]
    bridge["mapping"] = {"kind": "predictive_association"}
    with pytest.raises(DecisionEvidenceRefusal):
        de.decide_with_proxy(ev, bridge, FRAME)


def test_switchback_joint_evidence_uses_sequences_not_periods():
    sequences, periods = 40, 4
    n = sequences * periods
    design = {
        "kind": "switchback",
        "sequence_ids": [f"seq-{i // periods}" for i in range(n)],
        "period_ids": [f"p-{i % periods}" for i in range(n)],
        "treatment_probabilities": [0.5] * n,
        "no_carryover_assumption_id": "carryover-none",
        "washout_rule_id": "washout-zero",
        "washout_periods": 0,
        "independent_sequences_assumption_id": "independent-sequences",
        "consistency_assumption_id": "consistency",
    }
    de.compile_experimental_design(design)
    assigned = [(i + (i // periods) % 3) % 2 == 0 for i in range(n)]
    freq = [
        2.0 + (i // periods) * 0.02 + (0.3 + 0.02 * math.sin(i // periods) if assigned[i] else 0)
        for i in range(n)
    ]
    value = [4.0 + (i // periods) * 0.01 + (0.5 if assigned[i] else 0) for i in range(n)]
    ev = de.JointEvidence.from_experimental_design(
        design=design,
        outcomes=[freq, value],
        assigned=assigned,
        evidence_id="switchback",
        population_id="pop",
        treated_regime_id="t",
        control_regime_id="c",
        horizon=1,
        components={"frequency": ("f", "u"), "value": ("v", "u")},
        exposure="no_between_unit_interference",
        exposure_assumption_id="no-spill",
    )
    q = DerivedQuantity(
        "margin", "u", Expr.input(0) * Expr.input(1), [Measured("frequency"), Measured("value")]
    )
    assert de.evaluate(ev, q).derived["sampling"] is not None
    assert ev.wire["design"]["treated_units"] == sequences


def test_joint_excursion_preserves_lag_and_distinguishes_regime():
    units, periods = 40, 5
    n = units * periods
    assigned = [(i * 73 + (i // periods) * 19) % 11 < 5 for i in range(n)]
    design = {
        "kind": "temporal_excursion",
        "unit_ids": [f"geo-{i // periods}" for i in range(n)],
        "periods": [i % periods for i in range(n)],
        "treatment_probabilities": [0.5] * n,
        "lag": 1,
        "min_probability": 0.1,
        "assignment_law_id": "weekly-law",
        "independent_units_assumption_id": "geos-independent",
        "administrative_followup_assumption_id": "end-of-study",
        "consistency_assumption_id": "consistency",
    }
    assert de.compile_experimental_design(design)["obligations"]
    y = [
        2.0
        + 0.2 * math.sin((i // periods) * 0.17)
        + (1.0 if i % periods and assigned[i - 1] else 0.0)
        for i in range(n)
    ]
    z = [
        5.0
        + 0.4 * math.sin((i // periods) * 0.17)
        + (0.6 if i % periods and assigned[i - 1] else 0.0)
        for i in range(n)
    ]
    ev = de.JointEvidence.from_experimental_design(
        design=design,
        outcomes=[y, z],
        assigned=assigned,
        evidence_id="pulse",
        population_id="pop",
        treated_regime_id="on",
        control_regime_id="off",
        horizon=4,
        components={"frequency": ("frequency", "u"), "value": ("value", "u")},
        exposure="no_between_unit_interference",
        exposure_assumption_id="no-spill",
    )
    result = de.evaluate(ev, DerivedQuantity("effect", "u", Expr.input(0), [Measured("frequency")]))
    assert result.derived["sampling"] is not None
    assert ev.wire["design"]["treated_units"] == units
    assert ev.wire["design_details"]["paired_decisions"] == units * (periods - 1)
    planned = de.evaluate(
        ev,
        DerivedQuantity("effect", "u", Expr.input(0), [Measured("frequency")]),
        frame=DecisionFrame("on", "off", result.derived["sampling"]["estimate"]),
        candidates=[
            {
                "kind": "more_independent_units",
                "id": "more-geos",
                "additional_units": 40,
                "same_design_assumption_id": "same-schedule-and-variance",
                "cost": 0.0,
            }
        ],
    )
    row = planned.plan["evsi_rows"][0]
    assert row["planning_assumption_id"] == "same-schedule-and-variance"
    assert row["value"]["gross_evsi"] > 0
    assert ev.wire["compiled_source_artifact"]
    source_quantity = ev.excursion_quantity("frequency")
    assert source_quantity.functional_id == "sequential_randomized_excursion_mean_contrast"
    assert source_quantity.regime_id.startswith("excursion:")
    edited = dict(ev.wire)
    edited["treated_means"] = list(edited["treated_means"])
    edited["treated_means"][0] += 1.0
    with pytest.raises(DecisionEvidenceRefusal):
        de.evaluate(
            de.JointEvidence(edited),
            DerivedQuantity("effect", "u", Expr.input(0), [Measured("frequency")]),
        )
    with pytest.raises(DecisionEvidenceRefusal):
        de.evaluate(
            ev,
            DerivedQuantity("effect", "u", Expr.input(0), [Measured("frequency")]),
            deployment=True,
            bridge_assumption_id="bare-id",
        )
    bridge = {
        "source": {"kind": "excursion", "lag": 1, "assignment_law_id": "weekly-law"},
        "target": {
            "kind": "fixed_schedule",
            "active_schedule_id": "sustained",
            "reference_schedule_id": "status-quo",
            "horizon": 12,
        },
        "source_component": "frequency",
        "target_variable_id": "frequency",
        "target_units": "u",
        "mapping": {
            "kind": "causal_effect_bounds",
            "multiplier": [0.8, 1.2],
            "residual": [-0.1, 0.1],
            "assumption_id": "bridge",
            "mapping_evidence_id": "bridge-study",
        },
    }
    answer = de.decide_with_temporal_bridge(ev, bridge, DecisionFrame("on", "off", 0))
    assert answer["evidence_ids"] == ["pulse", "bridge-study"]
    bridge["mapping"] = {"kind": "predictive_association"}
    with pytest.raises(DecisionEvidenceRefusal):
        de.decide_with_temporal_bridge(ev, bridge, DecisionFrame("on", "off", 0))


def test_static_regime_values_remain_point_only_without_joint_uncertainty():
    units, periods = 120, 2
    n = units * periods
    assigned = [((i // 2) % 2 == 0 if i % 2 == 0 else (i // 4) % 2 == 0) for i in range(n)]
    conf = [math.sin((i // 2) * 0.13) + 0.1 * (i % 2) for i in range(n)]
    y = [
        1.0
        + 0.5 * conf[i]
        + float(assigned[(i // 2) * 2])
        + 2.0 * float(assigned[(i // 2) * 2 + 1])
        for i in range(n)
    ]
    design = {
        "kind": "temporal_regime",
        "unit_ids": [f"u-{i // 2}" for i in range(n)],
        "periods": [i % 2 for i in range(n)],
        "realized_assignment": assigned,
        "confounder_ids": ["history"],
        "active_path": [True, True],
        "reference_path": [False, False],
        "method": "g_formula",
        "min_probability": 0.05,
        "positivity_assumption_id": "positivity",
        "exchangeability_assumption_id": "exchangeability",
        "consistency_assumption_id": "consistency",
        "followup_assumption_id": "complete",
        "model_assumption_id": "specified-model",
    }
    assert de.compile_experimental_design(design)["obligations"]
    ev = de.JointEvidence.from_experimental_design(
        design=design,
        outcomes=[y],
        assigned=assigned,
        confounders=[conf],
        evidence_id="regime-study",
        population_id="pop",
        treated_regime_id="active",
        control_regime_id="reference",
        horizon=periods,
        components={"value": ("value", "u")},
        exposure="no_between_unit_interference",
        exposure_assumption_id="no-spill",
    )
    answer = de.evaluate(
        ev,
        DerivedQuantity("regime-value", "u", Expr.input(0), [Measured("value")]),
        frame=DecisionFrame("active", "reference", 0),
    )
    assert answer.derived["contrast"] == pytest.approx(3.0, abs=0.1)
    assert answer.derived["sampling"] is None
    assert answer.verdict == "unresolved"
    assert answer.derived["unresolved"][0]["code"] == "joint_law_missing"
    bridge = {
        "source": {
            "kind": "static_regime",
            "active_regime_id": "active",
            "reference_regime_id": "reference",
            "horizon": 2,
            "exchangeability_assumption_id": "exchangeability",
        },
        "target": {
            "kind": "model_response",
            "model_id": "model-v1",
            "active_schedule_id": "active-path",
            "reference_schedule_id": "reference-path",
            "horizon": 2,
        },
        "source_component": "value",
        "target_variable_id": "value",
        "target_units": "u",
        "mapping": {
            "kind": "causal_effect_bounds",
            "multiplier": [1.0, 1.0],
            "residual": [0.0, 0.0],
            "assumption_id": "model-bridge",
            "mapping_evidence_id": "mapping-evidence",
        },
    }
    assert (
        de.decide_with_temporal_bridge(ev, bridge, DecisionFrame("active", "reference", 0))[
            "verdict"
        ]["class"]
        == "unresolved"
    )


def test_static_regime_aligned_bootstrap_retains_joint_sampling_law():
    n = 120
    assigned = [i % 2 == 0 for i in range(n)]
    x = [math.sin(i * 0.29) for i in range(n)]
    y = [2.0 + x[i] + (0.7 if assigned[i] else 0.0) + 0.1 * ((i * 7) % 13) for i in range(n)]
    z = [2.0 * value for value in y]
    design = {
        "kind": "temporal_regime",
        "unit_ids": [f"u{i}" for i in range(n)],
        "periods": [0] * n,
        "realized_assignment": assigned,
        "confounder_ids": ["x"],
        "active_path": [True],
        "reference_path": [False],
        "method": "g_formula",
        "min_probability": 0.05,
        "positivity_assumption_id": "positivity",
        "exchangeability_assumption_id": "exchangeability",
        "consistency_assumption_id": "consistency",
        "followup_assumption_id": "followup",
        "model_assumption_id": "specified-model",
        "bootstrap_replicates": 99,
        "bootstrap_seed": 123,
    }
    ev = de.JointEvidence.from_experimental_design(
        design=design,
        outcomes=[y, z],
        confounders=[x],
        assigned=assigned,
        evidence_id="e",
        population_id="pop",
        treated_regime_id="on",
        control_regime_id="off",
        horizon=1,
        components={"y": ("y", "u"), "z": ("z", "u")},
        exposure="no_between_unit_interference",
        exposure_assumption_id="isolation",
    )
    assert ev.wire["dependence"]["kind"] == "joint_arms"
    result = de.evaluate(ev, DerivedQuantity("y", "u", Expr.input(0), [Measured("y")])).derived
    assert result["sampling"] is not None
    assert (result["support"], result["calibration"]) == ("none", "unmeasured")


def test_compiled_evidence_obeys_execution_memory_limit():
    from antecedent.execution import Execution, ResourceLimits

    evidence = _evidence(80)
    constrained = Execution(limits=ResourceLimits(memory_bytes=1))
    with pytest.raises(DecisionEvidenceRefusal):
        de.evaluate(evidence, _margin(Measured("cost")), execution=constrained)


def test_adaptive_policy_compiler_decision_and_replay():
    episodes = []
    for a in range(2):
        for state in range(2):
            for b in range(2):
                for repeat in range(6 if a == state else 2):
                    y = 2 * a + 3 * b + 4 * state + a * b
                    episodes.append(
                        {
                            "id": f"{a}/{state}/{b}/{repeat}",
                            "states": [0, state],
                            "actions": [a, b],
                            "outcomes": [y, 2 * y],
                        }
                    )

    def policy(name, first, adaptive):
        return {
            "id": name,
            "rules": [
                {"states": [0], "past_actions": [], "action": first},
                {"states": [0, 0], "past_actions": [first], "action": 0},
                {"states": [0, 1], "past_actions": [first], "action": int(adaptive)},
            ],
        }

    recipe = {
        "population": "pop",
        "state_variable_id": "state",
        "action_variable_id": "action",
        "outcomes": [["y", "y", "u"], ["z", "z", "u"]],
        "policies": [policy("adaptive", 1, True), policy("reference", 0, False)],
        "episodes": episodes,
        "policy_prespecification": "fixed before evaluation",
        "positivity": "positivity",
        "exchangeability": "exchangeability",
        "consistency": "consistency",
        "followup": "complete",
        "isolation": "independent",
    }
    with pytest.raises(DecisionEvidenceRefusal):
        de.JointEvidence.from_adaptive_regime(
            {**recipe, "policy_prespecification": ""}, evidence_id="adaptive-study"
        )
    ev = de.JointEvidence.from_adaptive_regime(recipe, evidence_id="adaptive-study")
    q = DerivedQuantity("effect", "u", Expr.input(0), [Measured("y")])
    result = de.evaluate(
        ev,
        q,
        frame=DecisionFrame("adaptive", "reference", 7.0),
        candidates=[
            {
                "kind": "more_independent_units",
                "id": "more-subjects",
                "additional_units": 32,
                "same_design_assumption_id": "same-distribution",
                "cost": 0.0,
            }
        ],
    )
    assert result.derived["contrast"] == pytest.approx(7.0)
    assert result.derived["sampling"] is not None
    assert result.plan["evsi_rows"][0]["planning_assumption_id"] == "same-distribution"
    assert ev.wire["design"]["treated_units"] == len(episodes)
    bridged = de.decide_with_temporal_bridge(
        ev,
        {
            "source": {
                "kind": "adaptive_regime",
                "active_regime_id": "adaptive",
                "reference_regime_id": "reference",
                "horizon": 2,
            },
            "target": {
                "kind": "fixed_schedule",
                "active_schedule_id": "schedule",
                "reference_schedule_id": "baseline",
                "horizon": 2,
            },
            "source_component": "y",
            "target_variable_id": "y",
            "target_units": "u",
            "mapping": {
                "kind": "causal_effect_bounds",
                "multiplier": [0.8, 1.2],
                "residual": [-0.1, 0.1],
                "assumption_id": "mapping",
                "mapping_evidence_id": "mapping-study",
            },
        },
        DecisionFrame("adaptive", "reference", 0),
    )
    assert bridged["evidence_ids"] == ["adaptive-study", "mapping-study"]
    edited = dict(ev.wire)
    edited["treated_means"] = [0, 0]
    with pytest.raises(DecisionEvidenceRefusal):
        de.evaluate(de.JointEvidence(edited), q)
