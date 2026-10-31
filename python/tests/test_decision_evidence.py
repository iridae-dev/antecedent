"""Composable experimental evidence for decision-relevant outcomes (Python parity).

Mirrors ``crates/antecedent/tests/decision_evidence_workflow.rs``: a cluster-randomized
experiment with limited independent clusters, three dependent outcomes and a nonlinear margin.
"""

from __future__ import annotations

import math

import pytest
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


def test_nonlinear_margin_has_a_sampling_interval_and_inherits_no_license():
    result = de.evaluate(_evidence(80), _margin(Measured("cost")))
    d = result.derived
    assert d["contrast"] > 0 and d["sampling"]["standard_error"] > 0
    assert (d["license"], d["calibration"]) == ("none", "unmeasured")
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
    assert caught.value.detail == "decision_evidence.incompatible_population"
    assert caught.value.remedy
    with pytest.raises(DecisionEvidenceRefusal) as caught:
        de.evaluate(ev, q, deployment=True)
    assert caught.value.detail == "decision_evidence.deployment_bridge_missing"
    assert de.evaluate(ev, q, deployment=True, bridge_assumption_id="assumption.rollout").derived[
        "sampling"
    ]
    with pytest.raises(DecisionEvidenceRefusal) as caught:
        de.evaluate(_evidence(80, exposure="unspecified"), q)
    assert caught.value.detail == "decision_evidence.interference_unspecified"


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
    assert caught.value.detail == "decision_evidence.unsupported_assignment_design"


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
            {"kind": "more_units", "id": "double_clusters", "additional_units_per_arm": 40, "cost": 0.01},
        ],
    )
    assert first.plan["resolution_rows"][0]["value"]["resolves"] is False
    assert first.plan["evsi_rows"][0]["value"]["gross_evsi"] > 0

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
    ev = _evidence(80, exposure="bounded_arm_spillover", exposure_assumption_id="arm-bias",
                   treated_spillover_bias_bounds=[0.1, 0.2, 0.0],
                   control_spillover_bias_bounds=[0.1, 0.2, 0.0])
    result = de.evaluate(ev, _margin(Measured("cost"))).derived
    assert result["assumption_range"]["assumption_ids"] == ["arm-bias"]
    assert result["assumption_range"]["lower"] < result["contrast"] < result["assumption_range"]["upper"]


def test_centered_fixed_cuped_supports_nonlinear_margin():
    n = 80
    ev = _evidence(n, adjustments=[{"kind": "centered_fixed_cuped", "component": "frequency", "covariate": "pre", "coefficient": 0.4}],
                   cluster_sizes=[ROWS] * n, pretreatment_totals={"pre": [ROWS * (1 + 0.01 * c) for c in range(n)]})
    assert de.evaluate(ev, _margin(Measured("cost"))).derived["sampling"] is not None
