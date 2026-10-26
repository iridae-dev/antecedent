"""Native target refits preserve modalities, producer state and scientific declarations."""

from dataclasses import FrozenInstanceError, replace

import antecedent as ant
import numpy as np
import pytest
from antecedent import validation as v
from antecedent.errors import (
    CausalCancelledError,
    CausalResourceError,
    CausalTypeError,
    CausalValueError,
)
from antecedent.outcomes import Outcome, require_success


def _static():
    rng = np.random.default_rng(61)
    z = rng.normal(size=500)
    x = 0.5 * z + rng.normal(size=500)
    y = 1 + 2 * x + z + rng.normal(scale=0.3, size=500)
    return ant.prepare(
        {"x": x, "z": z, "y": y},
        graph=[("z", "x"), ("z", "y"), ("x", "y")],
        query=ant.PointDerivative("x", "y", at=0),
        # A declared local window for the linear response on the roughly unit-scale
        # treatment. Point derivatives must not inherit a level-optimal default.
        estimator_config={"bandwidth": 0.6},
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=61, threads=1),
    )


def _series(seed, n=400):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.zeros(n)
    y[1:] = 1 + 2 * x[:-1] + rng.normal(scale=0.25, size=n - 1)
    return {"x": x, "y": y}


def _reports(result):
    return {check.refuter: check for check in result.reports}


def test_prepared_and_completed_target_refits_preserve_exact_producer_and_original_execution_count():
    study = _static()
    checks = v.TargetRefutationSpec(suite="cheap", replicates=12, derivative_step=0.2)
    plan = study.plan_view()
    fresh = require_success(v.refute_target(study, checks=checks))
    assert fresh.original_executions == 1
    assert not fresh.reused_original_result
    original = ant.estimate(study)
    original_value = require_success(original)
    identities = dict(original.identities)
    retained = require_success(v.refute_target(original, checks=checks))
    assert retained.original_executions == 0
    assert retained.reused_original_result
    assert retained.target.source == retained.program == identities["lowered_program_id"]
    assert retained.target.identity == fresh.target.identity
    assert retained.certificate == fresh.certificate
    assert retained.reports == fresh.reports
    assert original.value is original_value
    assert dict(original.identities) == identities
    assert study.plan_view().identity == plan.identity
    assert retained.target.family == "response_derivative"
    placebo = _reports(retained)["target.placebo_treatment"]
    assert placebo.original[0] == pytest.approx(2, abs=0.1)
    assert abs(placebo.refuted[0]) < 0.5
    assert _reports(retained)["target.finite_difference_consistency"].passed
    assert retained.calibration == "unmeasured"
    assert retained.inspect().uncertainty.reason == "diagnostic_unmeasured"
    assert len(repr(retained)) < 350
    assert original.inspect().program_id == identities["program_id"]


@pytest.mark.parametrize("modality", ["series", "panel", "multi_env", "events"])
def test_temporal_refits_use_retained_native_modality_and_explicit_dependence_controls(modality):
    frames = [_series(41 + i) for i in range(3)]
    if modality == "panel":
        data = ant.data.panel({10 + i: frame for i, frame in enumerate(frames)})
    elif modality == "multi_env":
        data = ant.data.multi_env(frames)
    elif modality == "events":
        data = ant.data.event(frames[0], np.arange(400, dtype=np.int64), align_interval_ns=1)
    else:
        data = ant.data.table(frames[0])
    study = ant.prepare(
        data,
        graph=[("x", 1, "y", 0)],
        query=ant.PulseEffect("x", "y", treatment_lag=1, horizon_steps=1),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=38, threads=1),
    )
    original = ant.estimate(study)
    require_success(original)
    checks = v.TargetRefutationSpec(
        replicates=12,
        pre_trend_lead=2,
        gathered_resamples=True,
        common_cause_strength=0.2,
    )
    result = v.refute_target(original, checks=checks)
    report = require_success(result)
    assert report.reused_original_result and report.original_executions == 0
    assert "temporal" in report.target.quadrant
    assert report.target.source == report.program == original.identities["lowered_program_id"]
    assert original.inspect().program_id == original.identities["program_id"]
    reports = _reports(report)
    assert reports["target.placebo_treatment"].passed
    assert reports["target.bootstrap_stability"].passed
    assert reports["target.bootstrap_stability"].original[0] == pytest.approx(2, abs=0.15)
    assert all(check.target == report.target.identity for check in report.reports)
    assert all(refusal.reason and refusal.next_attempt for refusal in report.refused)
    # No graph-declared candidate exposure exists. This preserves a real partial suite.
    assert any(r.refuter == "target.negative_control_exposure" for r in report.refused)
    assert report.declaration == checks
    assert result.answer.kind == "partial"
    assert result.inspect().next_actions


def test_posterior_and_derivative_controls_refit_the_original_inference_declaration():
    study = ant.reconfigure(
        _static(), inference=ant.Bayesian(n_draws=40, prior_scale=10, backend="conjugate")
    )
    checks = v.TargetRefutationSpec(
        replicates=12,
        prior_scale_factors=(0.5, 2.0),
        prior_shift_tolerance=0.75,
        derivative_step=0.2,
        derivative_tolerance=0.15,
        linearity_tolerance=0.3,
    )
    report = require_success(v.refute_target(study, checks=checks))
    assert "bayesian" in report.target.quadrant
    reports = _reports(report)
    assert reports["target.prior_sensitivity"].original[0] == pytest.approx(2, abs=0.2)
    assert "target.posterior_predictive" in reports
    assert reports["target.finite_difference_consistency"].passed
    assert report.calibration == "unmeasured"


def test_inspection_does_not_refit_or_serialize_payload(monkeypatch):
    report = require_success(
        v.refute_target(_static(), checks=v.TargetRefutationSpec(suite="cheap", replicates=8))
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("inspection must not serialize or execute")

    monkeypatch.setattr(v.TargetRefutation, "to_dict", forbidden)
    monkeypatch.setattr(v.TargetRefutation, "_from_json", forbidden)
    assert report.inspect().target_id == report.target.identity
    assert report.answer.kind in ("partial", "structured")
    assert len(repr(report)) < 350


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"replicates": True}, CausalTypeError),
        ({"replicates": 0}, CausalValueError),
        ({"alpha": 1}, CausalValueError),
        ({"subset_fraction": float("nan")}, CausalValueError),
        ({"pre_trend_lead": 0}, CausalValueError),
        ({"prior_scale_factors": (-1,)}, CausalValueError),
        ({"derivative_step": 0}, CausalValueError),
        ({"gathered_resamples": 1}, CausalTypeError),
        ({"within_class": "implicit"}, CausalValueError),
    ],
)
def test_invalid_scientific_declarations_are_typed_arguments(changes, error):
    with pytest.raises(error):
        v.TargetRefutationSpec(**changes)


def test_declaration_is_immutable_and_no_live_producer_is_a_scientific_refusal():
    spec = v.TargetRefutationSpec(prior_scale_factors=[0.25, 4])
    assert spec.prior_scale_factors == (0.25, 4)
    with pytest.raises(FrozenInstanceError):
        spec.replicates = 8
    report = v.refute_target(Outcome(1.0), checks=spec)
    assert report.refusal.code == "route_not_supported"
    assert report.refusal.detail == "target_refutation.live_result_required"


def test_shared_execution_cancellation_and_admission_leave_source_usable():
    original = ant.estimate(_static())
    require_success(original)
    checks = v.TargetRefutationSpec(suite="cheap", replicates=8)
    cancelled = ant.CancellationToken()
    cancelled.cancel()
    with pytest.raises(CausalCancelledError):
        v.refute_target(original, checks=checks, execution=ant.Execution(cancel=cancelled))
    limited = ant.Execution(limits=ant.ResourceLimits(memory_bytes=1))
    with pytest.raises(CausalResourceError):
        v.refute_target(original, checks=checks, execution=limited)
    report = require_success(v.refute_target(original, checks=checks))
    assert report.original_executions == 0


def test_failed_suite_cannot_claim_pass_by_vacuous_truth():
    report = v.TargetRefutation(
        v.RefutationTarget("scalar_average_effect", "static.frequentist", ("ate",), "id", "source"),
        "cheap",
        (),
        (),
        (),
        "certificate",
        "unmeasured",
    )
    assert report.passed is None
    assert replace(report, reports=()).passed is None


def test_insufficient_vector_replicates_are_a_scientific_refusal_with_exact_next_actions():
    study = ant.reconfigure(_static(), query=ant.ResponseCurve("x", "y", grid=[-2, -1, 0, 1, 2]))
    result = v.refute_target(study, checks=v.TargetRefutationSpec(suite="cheap", replicates=3))
    assert result.refusal is not None
    assert result.refusal.code == "route_not_supported"
    assert result.refusal.detail == "target_refutation.no_applicable_refuters"
    assert result.refusal.causes
    assert all("5-coordinate" in cause["message"] for cause in result.refusal.causes)
    assert all("7" in action["action"] for action in result.refusal.next_actions)


def test_independently_loaded_outcome_cannot_reconstruct_a_live_refit():
    original = ant.estimate(_static())
    require_success(original)
    loaded = ant.load(original.export())
    require_success(loaded)
    assert loaded.study is None
    assert loaded.identities.get("lowered_program_id") is None
    result = v.refute_target(loaded, checks=v.TargetRefutationSpec(suite="cheap", replicates=8))
    assert result.refusal.detail == "target_refutation.live_result_required"


@pytest.mark.parametrize("scientific_exception", [False, True])
def test_target_refit_stage_observer_preserves_original_exception_and_source(scientific_exception):
    from antecedent.errors import StructuredRefusal

    rng = np.random.default_rng(94)
    z = rng.normal(size=240)
    x = (z + rng.normal(size=240) > 0).astype(float)
    y = 1 + 2 * x + z + rng.normal(size=240)
    study = ant.prepare(
        {"z": z, "x": x, "y": y},
        graph=[("z", "x"), ("z", "y"), ("x", "y")],
        query=ant.AverageEffect("x", "y"),
        inference=ant.Frequentist(bootstrap=8),
        refute="none",
        execution=ant.Execution(seed=94, threads=2, on_stage=lambda *_: None),
    )
    original = ant.estimate(study)
    require_success(original)
    identities = dict(original.identities)
    error = (
        StructuredRefusal(
            {
                "code": "route_not_supported",
                "detail": "test.observer_failure",
                "message": "observer failed",
            }
        )
        if scientific_exception
        else RuntimeError("observer failed")
    )
    called = []

    def fail(*_):
        called.append(True)
        raise error

    with pytest.raises(type(error)) as caught:
        v.refute_target(
            original,
            checks=v.TargetRefutationSpec(suite="cheap", replicates=8),
            execution=ant.Execution(seed=94, threads=2, on_stage=fail),
        )
    assert caught.value is error
    assert called
    assert dict(original.identities) == identities
    assert (
        require_success(
            v.refute_target(
                original,
                checks=v.TargetRefutationSpec(suite="cheap", replicates=8),
                execution=ant.Execution(seed=94, threads=2),
            )
        ).original_executions
        == 0
    )


def test_explicit_negative_controls_keep_their_scientific_roles():
    rng = np.random.default_rng(1)
    n = 1200
    z = rng.normal(size=n)
    x = (z + rng.normal(size=n) > 0).astype(float)
    negative_exposure = (z + rng.normal(size=n) > 0).astype(float)
    negative_outcome = z + rng.normal(size=n)
    y = 1 + 2 * x + z + rng.normal(size=n)
    study = ant.prepare(
        {"z": z, "x": x, "y": y, "nce": negative_exposure, "nco": negative_outcome},
        graph=[("z", "x"), ("z", "y"), ("x", "y"), ("z", "nce"), ("z", "nco")],
        query=ant.AverageEffect("x", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=1, threads=1),
    )
    checks = v.TargetRefutationSpec(
        replicates=12,
        negative_control_exposure="nce",
        negative_control_outcome="nco",
        common_cause_strength=0.5,
        alpha=0.05,
        subset_fraction=0.75,
    )
    report = require_success(v.refute_target(study, checks=checks))
    assert report.declaration == checks
    reports = _reports(report)
    assert reports["target.negative_control_exposure"].passed
    assert reports["target.negative_control_outcome"].passed
    assert reports["target.random_common_cause"].passed
    assert reports["target.random_common_cause"].original[0] == pytest.approx(2, abs=0.2)


def test_mediation_controls_assess_the_declared_joint_decomposition():
    rng = np.random.default_rng(5)
    n = 500
    x = rng.integers(0, 2, size=n).astype(float)
    m = 0.7 * x + rng.normal(size=n)
    y = 0.2 * x + 0.9 * m + rng.normal(size=n)
    study = ant.prepare(
        {"x": x, "m": m, "y": y},
        graph=[("x", "m"), ("x", "y"), ("m", "y")],
        query=ant.MediationEffect("x", "y", mediators=["m"]),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=5, threads=1),
    )
    checks = v.TargetRefutationSpec(
        suite="cheap", replicates=8, mediator="m", mediation_rho_bar=0.1
    )
    report = require_success(v.refute_target(study, checks=checks))
    assert report.target.family == "mediation"
    assert report.target.labels == ("total", "direct", "indirect")
    diagnostic = _reports(report)["target.sequential_ignorability"]
    assert diagnostic.original[0] == pytest.approx(
        diagnostic.original[1] + diagnostic.original[2], abs=1e-12
    )
    assert diagnostic.original[2] == pytest.approx(0.63, abs=0.2)
    assert report.declaration.mediator == "m"
    assert report.declaration.mediation_rho_bar == 0.1


def test_class_posterior_common_cause_requires_explicit_mass_allocation():
    from antecedent.discovery import GraphPosterior

    rng = np.random.default_rng(6)
    n = 900
    z, w = rng.normal(size=(2, n))
    x = w + rng.normal(size=n)
    t = (z + w + 0.5 * rng.normal(size=n) > 0).astype(float)
    y = 1 + 2 * t + rng.normal(size=n)
    names = ["t", "y", "z", "w", "x"]
    edges = [(2, 0), (3, 0), (0, 1), (3, 4), (4, 3)]
    # The packed directed adjacency omits diagonal slots.
    mask = sum(1 << (a * 4 + (b if b < a else b - 1)) for a, b in edges)
    posterior = GraphPosterior.from_atoms(names, [1.0], [mask], atom_kind="Cpdag")
    study = ant.prepare(
        {"t": t, "y": y, "z": z, "w": w, "x": x},
        discovery=posterior,
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=66, threads=1),
    )
    original = ant.estimate(study)
    require_success(original)
    undeclared = v.TargetRefutationSpec(replicates=8, common_cause_strength=0.5)
    partial = require_success(v.refute_target(original, checks=undeclared))
    refusal = next(r for r in partial.refused if r.refuter == "target.random_common_cause")
    assert "WithinClassDistribution" in refusal.reason
    assert refusal.next_attempt
    declared = replace(undeclared, within_class="uniform_over_members")
    report = require_success(v.refute_target(original, checks=declared))
    cause = _reports(report)["target.random_common_cause"]
    assert cause.informative
    assert cause.original[0] == pytest.approx(2, abs=0.2)
    assert report.target == partial.target
    assert report.inspect().assumptions.payload["within_class"] == "uniform_over_members"
    assert report.original_executions == 0


@pytest.mark.parametrize("change", ["data", "query"])
def test_mismatched_real_outcome_and_study_refuse_before_any_execution(change, monkeypatch):
    study = _static()
    original = ant.estimate(study)
    require_success(original)
    if change == "data":
        data = _series(17, n=500)
        data["z"] = np.linspace(-1, 1, 500)
        other_study = ant.rebind(study, data=data)
    else:
        other_study = ant.reconfigure(study, query=ant.PointDerivative("x", "y", at=0.5))
    other = ant.estimate(other_study)
    require_success(other)
    original_ids, other_ids = dict(original.identities), dict(other.identities)
    mismatched = replace(other, _study=original.study)

    def no_execution(*args, **kwargs):
        raise AssertionError("producer mismatch must refuse without any native execution")

    monkeypatch.setattr(ant.Execution, "_call", no_execution)
    result = v.refute_target(mismatched, checks=v.TargetRefutationSpec(suite="cheap", replicates=8))
    assert result.refusal.code == "route_not_supported"
    assert result.refusal.detail == "target_refutation.producer_mismatch"
    assert result.refusal.next_actions
    assert dict(original.identities) == original_ids
    assert dict(other.identities) == other_ids


def test_partial_reports_preserve_failed_checks_and_never_claim_an_unrun_suite_passed():
    target = v.RefutationTarget(
        "scalar_average_effect", "static.frequentist", ("ate",), "id", "source"
    )
    check = v.TargetCheck(
        "target.placebo_treatment",
        "id",
        (2.0,),
        (1.0,),
        "chi_square",
        4.0,
        1,
        0.04,
        False,
        True,
        8,
        "failed",
    )
    refusal = v.RefuterRefusal(
        "target.subset_stability", "rank deficient", "use a full-rank subset"
    )
    report = v.TargetRefutation(
        target,
        "cheap",
        (check.refuter, refusal.refuter),
        (check,),
        (refusal,),
        "certificate",
        "unmeasured",
    )
    assert report.passed is False
    assert report.answer.kind == "partial"
    assert report.inspect().next_actions == (
        {"refuter": refusal.refuter, "action": refusal.next_attempt},
    )
    # A passed runnable check does not erase the partial answer or its obligations.
    passed = replace(report, reports=(replace(check, passed=True, failure=None),))
    assert passed.passed is True  # explicitly the standing of informative reports
    assert passed.answer.kind == "partial"
    assert passed.refused == (refusal,)
    assert replace(report, reports=()).passed is None
