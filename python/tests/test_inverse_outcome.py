"""Finite-action inverse-outcome query (2.2 E6) from Python.

Forward responses are either run through a real response route on a known
structural law, or built as synthetic views of an analytic law
``E[Y | do(a)] = 1 + 2a`` so every classification has a closed-form oracle that
shares no code with the classifier.
"""

from __future__ import annotations

import dataclasses
import math

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.inverse import (
    Action,
    ChanceConstraint,
    ObservationalScenarios,
    TargetMean,
    TargetQuantile,
    inverse_outcome,
)
from antecedent.query import AverageDerivative, ResponseCurve
from antecedent.results import IdentificationView
from antecedent.results.response import (
    SIMULTANEOUS_BAND_CRITICAL,
    SIMULTANEOUS_BAND_LOWER,
    SIMULTANEOUS_BAND_UPPER,
    CausalResponseView,
    ResponseUncertainty,
    ResponseView,
    SupportDiagnostic,
    SupportReport,
)

from _refusal import assert_registered_refusal

GRID = [0.0, 0.5, 1.0, 1.5, 2.0]


def _law(dose: float) -> float:
    return 1.0 + 2.0 * dose


def _forward(
    law=_law,
    grid=GRID,
    *,
    assumptions=("backdoor adjustment on z",),
    support="supported",
    point_status=None,
    half_width=None,
    interpretation="confidence",
    kind="pointwise",
    status="NonparametricallyIdentified",
    estimand=None,
) -> CausalResponseView:
    means = [law(a) for a in grid]
    if half_width is None:
        uncertainty = ResponseUncertainty("none")
    else:
        uncertainty = ResponseUncertainty(
            kind,
            lower=[[m - half_width] for m in means],
            upper=[[m + half_width] for m in means],
            level=0.9,
            interpretation=interpretation,
        )
    return CausalResponseView(
        estimand=estimand or ResponseCurve("t", "y", grid=sorted(grid)),
        response=ResponseView(["t"], ["y"], [[a] for a in grid], [[m] for m in means]),
        estimate=None,
        uncertainty=uncertainty,
        support=SupportReport(support, {"t": (min(grid), max(grid))}, point_status=point_status),
        identification=IdentificationView(
            status=status,
            method="response.backdoor",
            adjustment_set=["z"],
            assumption_count=len(assumptions),
            derivation_step_count=0,
        ),
        assumptions=list(assumptions),
    )


def _actions(grid=GRID) -> list[Action]:
    return [Action(f"dose_{a}", a, cost=abs(a)) for a in grid]


def _refused(code: str, detail: str, call) -> None:
    with pytest.raises(CausalUnsupportedError) as caught:
        call()
    assert_registered_refusal(caught.value)
    assert caught.value.reason_code == code
    assert detail in str(caught.value)


def test_forward_inverse_round_trip_on_a_known_structural_law_through_a_real_route():
    n = 240
    z = np.array([math.sin(i / 17.0) for i in range(n)])
    t = z + np.array([math.cos(i / 11.0) for i in range(n)])
    data = {"t": t, "y": 1.0 + 2.0 * t + 0.8 * z, "z": z}
    grid = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    forward = ant.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.ResponseCurve("t", "y", grid=grid),
        estimator="response.kennedy_dr",
        refute="none",
        bootstrap=0,
        seed=4,
    ).estimate(data, seed=4)
    # The structural truth, computed from the generating law and the data's own mean of z.
    intercept = 1.0 + 0.8 * float(np.mean(z))
    # E[Y | do(a)] = intercept + 2a >= intercept + 0.75 exactly when a >= 0.375.
    threshold = intercept + 0.75
    report = inverse_outcome(
        forward,
        query=TargetMean(threshold),
        actions=[Action(f"dose_{a}", a, cost=abs(a)) for a in grid],
    )
    truth = {f"dose_{a}": intercept + 2.0 * a >= threshold for a in grid}
    assert {r.label: r.status == "feasible" for r in report.actions} == truth
    assert report.feasible == ("dose_0.5", "dose_0.75", "dose_1.0")
    assert report.cheapest_feasible == ("dose_0.5",)
    for result in report.actions:
        # Finite-sample nuisance tolerance around the exact structural margin.
        assert result.margin == pytest.approx(
            intercept + 2.0 * result.point[0] - threshold, abs=0.1
        )
    assert report.inference_claim == "point_only"
    assert report.enumerated == "reachable"
    assert report.forward_claim_id == forward.claim_id
    assert report.forward_program_id == forward.program_id
    assert report.forward_data_snapshot_id == forward.data_snapshot_id
    assert report.verify()


def test_the_boundary_action_and_the_opposite_goal_follow_the_analytic_inverse():
    forward = _forward()
    # 1 + 2a >= 3 exactly when a >= 1; the boundary action meets it within tolerance.
    report = inverse_outcome(forward, query=TargetMean(3.0), actions=_actions())
    assert report.feasible == ("dose_1.0", "dose_1.5", "dose_2.0")
    boundary = report.action("dose_1.0")
    assert (boundary.reason, boundary.within_tolerance, boundary.margin) == (
        "meets_target",
        True,
        0.0,
    )
    below = inverse_outcome(forward, query=TargetMean(3.0, "at_most"), actions=_actions())
    assert below.feasible == ("dose_0.0", "dose_0.5", "dose_1.0")


def test_several_feasible_actions_keep_every_cost_tie():
    actions = _actions()
    actions[3] = Action("dose_1.5", 1.5, cost=1.0)  # same cost as dose_1.0
    report = inverse_outcome(_forward(), query=TargetMean(3.0), actions=actions)
    assert report.feasible == ("dose_1.0", "dose_1.5", "dose_2.0")
    assert report.cheapest_feasible == ("dose_1.0", "dose_1.5")
    assert report.infeasible == ("dose_0.0", "dose_0.5")


def test_an_unreachable_target_is_unreachable_only_within_the_enumerated_set():
    report = inverse_outcome(_forward(), query=TargetMean(100.0), actions=_actions())
    assert report.feasible == () and report.cheapest_feasible == ()
    assert report.enumerated == "unreachable_within_set"
    assert {r.reason for r in report.actions} == {"misses_target"}
    assert "enumerated action set" in report.scope_note


def test_unsupported_actions_are_not_classified_by_their_estimate():
    cells = [
        "supported",
        "supported",
        "extrapolative",
        "missing_evidence",
        "outside_empirical_support",
    ]
    forward = _forward(support="outside_empirical_support", point_status=cells)
    report = inverse_outcome(forward, query=TargetMean(3.0), actions=_actions())
    assert report.support_basis == "per_point"
    assert report.unsupported == ("dose_1.5", "dose_2.0")
    assert report.action("dose_1.5").reason == "missing_evidence"
    assert report.action("dose_2.0").reason == "outside_empirical_support"
    assert report.action("dose_2.0").estimate is not None  # shown, never decided on
    extrapolated = report.action("dose_1.0")
    assert extrapolated.status == "feasible" and extrapolated.support_status == "extrapolative"
    only_unsupported = inverse_outcome(
        forward, query=TargetMean(3.0), actions=[a for a in _actions() if a.point >= 1.5]
    )
    assert only_unsupported.enumerated == "undetermined"


def test_a_static_curve_support_is_the_worst_case_copied_to_every_point():
    forward = _forward(support="outside_empirical_support", point_status=None)
    report = inverse_outcome(forward, query=TargetMean(3.0), actions=_actions())
    assert report.support_basis == "surface_worst_case"
    assert report.unsupported == tuple(sorted(a.label for a in _actions()))
    assert report.enumerated == "undetermined"


def test_an_incomplete_grid_leaves_the_missing_actions_unevaluated_not_interpolated():
    forward = _forward(grid=[0.0, 1.0, 2.0])
    actions = [Action("a0", 0.0), Action("a1", 1.0), Action("a2", 2.0)]
    actions += [Action("between", 1.25), Action("beyond", 9.0)]
    report = inverse_outcome(forward, query=TargetMean(3.0), actions=actions)
    assert report.unevaluated == ("between", "beyond")
    assert report.feasible == ("a1", "a2")
    assert report.enumerated == "reachable"
    missed = report.action("between")
    assert (missed.reason, missed.estimate, missed.margin) == ("not_on_evaluated_grid", None, None)
    hopeless = inverse_outcome(forward, query=TargetMean(100.0), actions=actions)
    assert hopeless.enumerated == "undetermined"
    assert hopeless.infeasible == ("a0", "a1", "a2")


def test_constraints_and_budget_make_an_action_infeasible():
    actions = _actions()
    actions[4] = Action("dose_2.0", 2.0, cost=2.0, constraints={"cap": False, "stock": True})
    report = inverse_outcome(_forward(), query=TargetMean(3.0), actions=actions, budget=1.2)
    capped = report.action("dose_2.0")
    assert (capped.reason, capped.violated_constraints) == ("constraint_violated", ("cap",))
    assert report.action("dose_1.5").reason == "over_budget"
    assert report.feasible == ("dose_1.0",)


def test_a_changed_assumption_or_adjustment_changes_the_verdict_and_the_identity():
    base = inverse_outcome(_forward(), query=TargetMean(3.0), actions=_actions())
    # Same law, an extra declared assumption: same verdict, different identity.
    reassumed = inverse_outcome(
        _forward(assumptions=("backdoor adjustment on z", "no unmeasured confounding")),
        query=TargetMean(3.0),
        actions=_actions(),
    )
    assert reassumed.feasible == base.feasible
    assert reassumed.identity != base.identity
    assert reassumed.assumptions[-1] == "no unmeasured confounding"
    # A different adjustment gives a different forward law (0.5 + 2a): the boundary action drops out.
    shifted = inverse_outcome(
        _forward(law=lambda a: 0.5 + 2.0 * a), query=TargetMean(3.0), actions=_actions()
    )
    assert shifted.feasible == ("dose_1.5", "dose_2.0")
    assert shifted.identity != base.identity


def test_action_and_forward_row_order_and_cost_ties_never_change_a_result():
    actions = _actions()
    actions[3] = Action("dose_1.5", 1.5, cost=1.0)
    base = inverse_outcome(_forward(), query=TargetMean(3.0), actions=actions)
    permuted = inverse_outcome(
        _forward(grid=list(reversed(GRID))),
        query=TargetMean(3.0),
        actions=list(reversed(actions)),
    )
    assert permuted.identity == base.identity
    assert (permuted.feasible, permuted.infeasible, permuted.cheapest_feasible) == (
        base.feasible,
        base.infeasible,
        base.cheapest_feasible,
    )
    for result in base.actions:
        assert permuted.action(result.label) == result
    assert base.verify() and permuted.verify()


def test_a_probability_target_is_refused_because_a_mean_is_not_a_chance_constraint():
    _refused(
        "cell_not_licensed",
        "inverse.probability_target",
        lambda: inverse_outcome(_forward(), query=ChanceConstraint(3.0, 0.9), actions=_actions()),
    )


def test_a_quantile_target_is_refused_until_the_interventional_distribution_cell():
    _refused(
        "cell_not_licensed",
        "inverse.quantile_target",
        lambda: inverse_outcome(_forward(), query=TargetQuantile(0.9, 3.0), actions=_actions()),
    )


def test_observational_scenarios_are_refused_not_turned_into_an_action():
    _refused(
        "cell_not_licensed",
        "inverse.observational_scenarios",
        lambda: inverse_outcome(_forward(), query=ObservationalScenarios(), actions=_actions()),
    )


def test_a_forward_route_that_is_not_a_point_identified_mean_curve_is_refused():
    _refused(
        "route_not_supported",
        "inverse.forward_not_point_identified",
        lambda: inverse_outcome(
            _forward(status="PartiallyIdentified"), query=TargetMean(3.0), actions=_actions()
        ),
    )
    derivative = _forward(estimand=AverageDerivative("t", "y"))
    _refused(
        "route_not_supported",
        "inverse.forward_not_mean_response",
        lambda: inverse_outcome(derivative, query=TargetMean(3.0), actions=_actions()),
    )
    with pytest.raises(CausalTypeError):
        inverse_outcome({"not": "a response"}, query=TargetMean(3.0), actions=_actions())  # type: ignore[arg-type]


def test_malformed_actions_and_targets_are_invalid_arguments():
    forward = _forward()
    bad_actions = {
        "empty": [],
        "duplicate": [Action("a", 0.0), Action("a", 1.0)],
        "dimension": [Action("a", (0.0, 1.0))],
        "cost": [Action("a", 0.0, cost=float("nan"))],
        "negative cost": [Action("a", 0.0, cost=-1.0)],
    }
    for name, actions in bad_actions.items():
        with pytest.raises(CausalUnsupportedError, match=r"inverse\.invalid_action") as caught:
            inverse_outcome(forward, query=TargetMean(3.0), actions=actions)
        assert caught.value.reason_code == "invalid_argument", name
    with pytest.raises(CausalValueError):
        inverse_outcome(forward, query=TargetMean(float("nan")), actions=_actions())
    with pytest.raises(CausalUnsupportedError, match=r"inverse\.invalid_target"):
        inverse_outcome(forward, query=TargetMean(3.0), actions=_actions(), tolerance=-1.0)
    with pytest.raises(CausalTypeError):
        inverse_outcome(forward, query=TargetMean(3.0), actions=[(0.0, 1.0)])  # type: ignore[list-item]
    too_many = [Action(f"a{i}", 0.0) for i in range(4097)]
    _refused(
        "route_not_supported",
        "inverse.bounds_exceeded",
        lambda: inverse_outcome(forward, query=TargetMean(3.0), actions=too_many),
    )


def test_a_published_interval_adds_only_a_robustly_feasible_flag():
    point_only = inverse_outcome(_forward(), query=TargetMean(3.0), actions=_actions())
    banded = inverse_outcome(_forward(half_width=0.5), query=TargetMean(3.0), actions=_actions())
    assert banded.feasible == point_only.feasible
    # The lower endpoint 1 + 2a - 0.5 clears 3 only from a >= 1.25.
    assert banded.robustly_feasible == ("dose_1.5", "dose_2.0")
    boundary = banded.action("dose_1.0")
    assert boundary.robustly_feasible is False and boundary.interval == (2.5, 3.5)
    assert banded.action("dose_0.0").robustly_feasible is None
    assert banded.interval == {"scope": "pointwise", "interpretation": "confidence", "level": 0.9}
    assert point_only.interval is None
    assert {r.robustly_feasible for r in point_only.actions} == {None}
    credible = inverse_outcome(
        _forward(half_width=0.5, interpretation="credible", kind="simultaneous"),
        query=TargetMean(3.0),
        actions=_actions(),
    )
    assert credible.interval == {
        "scope": "simultaneous",
        "interpretation": "credible",
        "level": 0.9,
    }
    upper = inverse_outcome(
        _forward(half_width=0.5), query=TargetMean(3.0, "at_most"), actions=_actions()
    )
    assert upper.robustly_feasible == ("dose_0.0", "dose_0.5")


def test_temporal_style_simultaneous_band_is_preferred_to_the_pointwise_band():
    forward = _forward(half_width=0.1)
    means = [_law(a) for a in GRID]
    support = SupportReport(
        "supported",
        {"t": (0.0, 2.0)},
        diagnostics=[
            SupportDiagnostic(
                SIMULTANEOUS_BAND_LOWER, [mean - 1.5 for mean in means], "joint band"
            ),
            SupportDiagnostic(
                SIMULTANEOUS_BAND_UPPER, [mean + 1.5 for mean in means], "joint band"
            ),
            SupportDiagnostic(SIMULTANEOUS_BAND_CRITICAL, [0.9, 2.5, 100.0], "joint band"),
        ],
    )
    forward = forward.model_copy(update={"support": support})
    report = inverse_outcome(forward, query=TargetMean(3.0), actions=_actions())
    assert report.interval == {
        "scope": "simultaneous",
        "interpretation": "confidence",
        "level": 0.9,
    }
    assert report.robustly_feasible == ("dose_2.0",)
    assert report.action("dose_1.5").interval == (2.5, 5.5)


def test_forward_claim_and_snapshot_identity_are_bound_and_unlicensed_input_refuses():
    forward = _forward().model_copy(
        update={
            "claim_id": "claim-a",
            "program_id": "program-a",
            "data_snapshot_id": "snapshot-a",
        }
    )
    report = inverse_outcome(forward, query=TargetMean(3.0), actions=_actions())
    assert (
        report.forward_claim_id,
        report.forward_program_id,
        report.forward_data_snapshot_id,
    ) == (
        "claim-a",
        "program-a",
        "snapshot-a",
    )
    assert report.verify()
    changed = forward.model_copy(update={"data_snapshot_id": "snapshot-b"})
    assert (
        inverse_outcome(changed, query=TargetMean(3.0), actions=_actions()).identity
        != report.identity
    )
    _refused(
        "cell_not_licensed",
        "inverse.forward_not_licensed",
        lambda: inverse_outcome(
            forward.model_copy(update={"evidence_status": "allowed_unlicensed"}),
            query=TargetMean(3.0),
            actions=_actions(),
        ),
    )


def test_finite_forward_values_cannot_publish_an_infinite_target_margin():
    forward = _forward(law=lambda _: float.fromhex("0x1.fffffffffffffp+1023"), grid=[0.0, 1.0])
    report = inverse_outcome(
        forward,
        query=TargetMean(-float.fromhex("0x1.fffffffffffffp+1023")),
        actions=[Action("extreme", 0.0)],
    )
    action = report.action("extreme")
    assert (action.status, action.reason, action.margin) == (
        "unevaluated",
        "numerical_margin_overflow",
        None,
    )
    assert report.enumerated == "undetermined"


def test_action_constraints_are_booleans_not_truthy_strings():
    with pytest.raises(CausalTypeError, match="boolean values"):
        inverse_outcome(
            _forward(),
            query=TargetMean(3.0),
            actions=[Action("unsafe", 1.0, constraints={"in_stock": "false"})],  # type: ignore[dict-item]
        )
    with pytest.raises(CausalTypeError, match="not booleans"):
        inverse_outcome(
            _forward(),
            query=TargetMean(3.0),
            actions=[Action("boolean_dose", [True])],  # type: ignore[list-item]
        )


def test_temporal_reverse_forecast_uses_one_declared_horizon():
    forward = _forward().model_copy(
        update={
            "estimand": ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1, 2]),
            "response": ResponseView(
                ["t"],
                ["y"],
                [[0.0, 1.0], [1.0, 1.0], [0.0, 2.0], [1.0, 2.0]],
                [[1.0], [3.0], [2.0], [4.0]],
            ),
            "support": SupportReport(
                "supported", {"t": (0.0, 1.0)}, point_status=["supported"] * 4
            ),
        }
    )
    report = inverse_outcome(
        forward,
        query=TargetMean(3.0),
        actions=[Action("low", (0.0, 1.0)), Action("high", (1.0, 1.0))],
    )
    assert report.horizons == (1,)
    assert report.feasible == ("high",)
    assert report.verify()
    assert not dataclasses.replace(report, horizons=(2,)).verify()
    assert not dataclasses.replace(report, horizons=None).verify()
    _refused(
        "invalid_argument",
        "one fixed outcome horizon",
        lambda: inverse_outcome(
            forward,
            query=TargetMean(3.0),
            actions=[Action("early", (0.0, 1.0)), Action("late", (1.0, 2.0))],
        ),
    )
    with pytest.raises(CausalValueError, match="one horizon declared"):
        inverse_outcome(forward, query=TargetMean(3.0), actions=[Action("unknown", (0.0, 3.0))])


def test_study_options_and_sensitivity_are_separate_views_outside_the_classification():
    plan = {
        "outcome": "sufficient",
        "minimal": True,
        "proposals": [
            {
                "rank": 0,
                "candidates": ["trial_z"],
                "cost_units": 5,
                "sample_budget": 100,
                "deliver": [{"regime": "r1"}],
                "repairs": [{"candidate": "trial_z"}],
                "stage": "ignored",
            }
        ],
    }
    sensitivity = {
        "estimand": "target active-minus-control mean outcome response",
        "baseline": 2.0,
        "assumption_range": {"minimum": 1.0, "maximum": 3.0},
        "interval_interpretation": "assumption range; not a sampling interval",
        "decision_threshold": 1.5,
        "tipping_fraction": 0.2,
    }
    plain = inverse_outcome(_forward(), query=TargetMean(3.0), actions=_actions())
    full = inverse_outcome(
        _forward(),
        query=TargetMean(3.0),
        actions=_actions(),
        study_options=plan,
        sensitivity=sensitivity,
    )
    assert (full.identity, full.feasible, full.actions) == (
        plain.identity,
        plain.feasible,
        plain.actions,
    )
    options = full.missing_evidence
    assert options is not None and plain.missing_evidence is None
    assert options.evidence_status == "hypothetical_catalog_delta"
    assert options.proposals[0] == {
        "rank": 0,
        "candidates": ["trial_z"],
        "cost_units": 5,
        "sample_budget": 100,
        "deliver": [{"regime": "r1"}],
        "repairs": [{"candidate": "trial_z"}],
    }
    assert "Not observed evidence" in options.note
    view = full.sensitivity
    assert view is not None and view.is_confidence_interval is False
    assert view.assumption_range == {"minimum": 1.0, "maximum": 3.0}
    assert view.fields["tipping_fraction"] == 0.2 and "not a confidence interval" in view.note
    # A stage object exposing plan() is projected the same way.
    stage = type("Stage", (), {"plan": staticmethod(lambda: plan)})()
    staged = inverse_outcome(
        _forward(), query=TargetMean(3.0), actions=_actions(), study_options=stage
    )
    assert staged.missing_evidence == full.missing_evidence
    with pytest.raises(CausalValueError):
        inverse_outcome(_forward(), query=TargetMean(3.0), actions=_actions(), study_options={})
    with pytest.raises(CausalValueError):
        inverse_outcome(
            _forward(), query=TargetMean(3.0), actions=_actions(), sensitivity={"baseline": 1.0}
        )


def test_verify_recomputes_the_report_and_detects_an_edited_one():
    report = inverse_outcome(_forward(), query=TargetMean(3.0), actions=_actions())
    assert report.verify()
    assert len(report.identity) == 64
    edited_call = {**report._call, "threshold": 2.0}
    edited = dataclasses.replace(report, _call=edited_call)
    assert not edited.verify()
    assert not dataclasses.replace(report, feasible=("fabricated",)).verify()
    changed_action = dataclasses.replace(report.actions[0], status="feasible")
    assert not dataclasses.replace(report, actions=(changed_action, *report.actions[1:])).verify()
    cancel = ant.state.CancellationToken()
    cancel.cancel()
    _refused(
        "transport_budget_cancel",
        "inverse.cancelled",
        lambda: inverse_outcome(
            _forward(), query=TargetMean(3.0), actions=_actions(), cancel=cancel
        ),
    )
