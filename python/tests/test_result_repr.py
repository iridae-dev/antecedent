"""``__repr__`` coverage for every view in :mod:`antecedent.results`.

Views are constructed directly (no native calls needed) so these tests
exercise the Python-only formatting layer independently of the compiled
extension.
"""

from __future__ import annotations

import math

from antecedent.results import (
    AnalysisResult,
    ConflictSummaryView,
    EffectEnvelope,
    EstimateView,
    IdentificationView,
    MediationView,
    PerformanceView,
    PhysicalPlanView,
    PlanView,
    PosteriorView,
    PredictiveCheckReport,
    PriorSensitivityReport,
    RefutationReport,
    ValidationView,
)


def _identification(**overrides: object) -> IdentificationView:
    base = dict(
        status="NonparametricallyIdentified",
        method="backdoor.adjustment",
        adjustment_set=["market_demand_index"],
        assumption_count=1,
        derivation_step_count=2,
    )
    base.update(overrides)
    return IdentificationView(**base)  # type: ignore[arg-type]


def _estimate(**overrides: object) -> EstimateView:
    base = dict(
        ate=0.412,
        se_analytic=0.031,
        se_bootstrap=None,
        estimator_id="ols",
        method="backdoor.adjustment",
    )
    base.update(overrides)
    return EstimateView(**base)  # type: ignore[arg-type]


def _refutation(passed: bool = True, refuter: str = "placebo_treatment") -> RefutationReport:
    return RefutationReport(
        refuter=refuter,
        original_ate=0.412,
        refuted_ate=0.001 if passed else 0.39,
        comparison=0.02,
        informative=True,
        passed=passed,
        failure_condition=None if passed else "|refuted| too close to original",
        replicates=100,
    )


def _validation(
    *, ran: bool = True, reports: list[RefutationReport] | None = None
) -> ValidationView:
    reports = reports if reports is not None else [_refutation(True), _refutation(True)]
    return ValidationView(
        passed=all(r.passed for r in reports) if reports else False,
        ran=ran,
        count=len(reports),
        reports=reports,
    )


def _performance(**overrides: object) -> PerformanceView:
    base: dict[str, object] = {}
    base.update(overrides)
    return PerformanceView(**base)  # type: ignore[arg-type]


def _result(
    *,
    identified: bool = True,
    posterior: PosteriorView | None = None,
    validation: ValidationView | None = None,
) -> AnalysisResult:
    ident = _identification() if identified else _identification(status="NotIdentified")
    return AnalysisResult(
        identification=ident,
        estimate=_estimate(),
        posterior=posterior,
        validation=validation if validation is not None else _validation(),
        performance=_performance(),
        diagnostics=[],
        provenance={"node_count": 3},
    )


# --- IdentificationView -----------------------------------------------------


def test_identification_view_repr_identified():
    view = _identification()
    text = repr(view)
    assert "identified" in text
    assert "not identified" not in text
    assert "backdoor.adjustment" in text


def test_identification_view_repr_not_identified():
    view = _identification(status="NotIdentified", adjustment_set=[])
    assert "not identified" in repr(view)


def test_identification_view_bool_known_identified_statuses():
    assert bool(_identification(status="NonparametricallyIdentified"))
    assert bool(_identification(status="gcm.parametric"))


def test_identification_view_bool_negated_statuses_are_false():
    for status in ("NotIdentified", "Unidentified", "PartiallyIdentified", "GraphDependent"):
        assert not bool(_identification(status=status)), status


def test_identification_view_bool_unknown_status_defaults_false():
    assert not bool(_identification(status="SomeFutureStatus"))


# --- MediationView -----------------------------------------------------


def test_mediation_view_repr_handles_none():
    view = MediationView(total=None, direct=None, mediated=None)
    text = repr(view)
    assert "None" in text
    assert "nan" not in text


def test_mediation_view_repr_formats_floats():
    view = MediationView(total=1.5, direct=0.9, mediated=0.6)
    text = repr(view)
    assert "1.500" in text


# --- EstimateView -----------------------------------------------------


def test_estimate_view_repr_shows_analytic_se_by_default():
    view = _estimate()
    text = repr(view)
    assert "analytic" in text
    assert "ols" in text
    assert "0.412" in text


def test_estimate_view_repr_prefers_bootstrap_se_when_present():
    view = _estimate(se_bootstrap=0.05)
    text = repr(view)
    assert "bootstrap" in text
    assert "0.050" in text


# --- ConflictSummaryView -----------------------------------------------------


def test_conflict_summary_view_repr():
    view = ConflictSummaryView(
        source_ids=["s1", "s2"],
        alphas_requested=[0.5, 0.5],
        alphas_applied=[0.3, 0.4],
    )
    text = repr(view)
    assert "s1" in text and "s2" in text
    assert "0.300" in text and "0.400" in text


# --- PosteriorView -----------------------------------------------------


def test_posterior_view_repr_empty():
    view = PosteriorView(
        effect_mean=None,
        effect_sd=None,
        q025=None,
        q975=None,
        n_draws=None,
        p_below_zero=None,
        backend=None,
    )
    assert repr(view) == "<PosteriorView empty>"


def test_posterior_view_repr_hides_zero_unidentified_mass():
    view = PosteriorView(
        effect_mean=2.29,
        effect_sd=0.03,
        q025=2.2,
        q975=2.4,
        n_draws=200,
        p_below_zero=0.0,
        backend="conjugate",
        unidentified_mass=0.0,
    )
    assert "unidentified_mass" not in repr(view)


def test_posterior_view_repr_shows_positive_unidentified_mass():
    view = PosteriorView(
        effect_mean=2.29,
        effect_sd=0.03,
        q025=2.2,
        q975=2.4,
        n_draws=200,
        p_below_zero=0.0,
        backend="conjugate",
        unidentified_mass=0.5,
    )
    text = repr(view)
    assert "unidentified_mass=50.0%" in text


# --- EffectEnvelope -----------------------------------------------------


def test_effect_envelope_repr_always_shows_mass():
    view = EffectEnvelope(
        effect_mean=2.29,
        effect_sd=0.03,
        q025=2.2,
        q975=2.4,
        unidentified_mass=0.5,
        n_draws=200,
    )
    assert "unidentified_mass=50.0%" in repr(view)


# --- PredictiveCheckReport -----------------------------------------------------


def test_predictive_check_report_repr():
    report = PredictiveCheckReport(
        kind="prior_predictive",
        observed=10.0,
        predictive_mean=9.5,
        predictive_sd=1.2,
        p_value=0.42,
        n_sims=500,
    )
    text = repr(report)
    assert "prior_predictive" in text
    assert "500" in text


# --- PriorSensitivityReport -----------------------------------------------------


def test_prior_sensitivity_report_repr_scales_mode():
    report = PriorSensitivityReport(
        scales=[0.5, 1.0, 2.0], effect_means=[1, 2, 3], effect_sds=[0.1] * 3
    )
    text = repr(report)
    assert "scales" in text
    assert "n=3" in text


def test_prior_sensitivity_report_repr_alphas_mode():
    report = PriorSensitivityReport(
        scales=[], effect_means=[1, 2], effect_sds=[0.1, 0.1], alphas=[0.5, 0.6]
    )
    assert "alphas" in repr(report)


# --- RefutationReport -----------------------------------------------------


def test_refutation_report_repr_pass_and_fail():
    passed = repr(_refutation(True))
    failed = repr(_refutation(False))
    assert "pass" in passed
    assert "fail" in failed
    assert "placebo_treatment" in passed


# --- ValidationView -----------------------------------------------------


def test_validation_view_repr_not_run():
    view = ValidationView(passed=False, ran=False, count=0)
    assert repr(view) == "<ValidationView not run>"


def test_validation_view_repr_pass_fail_counts():
    view = _validation(reports=[_refutation(True), _refutation(False)])
    text = repr(view)
    assert "2 refuters" in text
    assert "1 failed" in text


# --- PerformanceView -----------------------------------------------------


def test_performance_view_repr_no_data():
    assert repr(_performance()) == "<PerformanceView no timing data>"


def test_performance_view_repr_with_timing():
    view = _performance(wall_time_ns=1_500_000, peak_memory_bytes=2_000_000)
    text = repr(view)
    assert "wall=1.5ms" in text
    assert "peak_mem=2.0MB" in text


def test_performance_view_repr_flags():
    view = _performance(cancelled=True, early_stopped=True)
    text = repr(view)
    assert "cancelled" in text
    assert "early_stopped" in text


# --- PlanView / PhysicalPlanView -----------------------------------------------------


def test_plan_view_repr():
    view = PlanView(plan_id="p1", identifier="backdoor", estimator="ols")
    text = repr(view)
    assert "p1" in text
    assert "backdoor" in text


def test_physical_plan_view_repr():
    view = PhysicalPlanView(plan_id="p1", worker_threads=4, expected_python_crossings=2)
    text = repr(view)
    assert "worker_threads=4" in text
    assert "expected_python_crossings=2" in text


# --- AnalysisResult -----------------------------------------------------


def test_analysis_result_repr_identified_with_refutations():
    result = _result()
    text = repr(result)
    assert text.startswith("<AnalysisResult identified")
    assert "effect=0.412" in text
    assert "±0.031" in text
    assert "refute=2/2 pass" in text


def test_analysis_result_repr_not_identified():
    result = _result(identified=False)
    assert "<AnalysisResult not identified" in repr(result)


def test_analysis_result_repr_shows_unidentified_mass_when_positive():
    posterior = PosteriorView(
        effect_mean=2.29,
        effect_sd=0.03,
        q025=2.2,
        q975=2.4,
        n_draws=200,
        p_below_zero=0.0,
        backend="conjugate",
        unidentified_mass=0.5,
    )
    result = _result(posterior=posterior)
    assert "unidentified_mass=50.0%" in repr(result)


def test_analysis_result_repr_hides_unidentified_mass_when_absent_or_zero():
    result_no_posterior = _result(posterior=None)
    assert "unidentified_mass" not in repr(result_no_posterior)

    posterior_zero = PosteriorView(
        effect_mean=2.29,
        effect_sd=0.03,
        q025=2.2,
        q975=2.4,
        n_draws=200,
        p_below_zero=0.0,
        backend="conjugate",
        unidentified_mass=0.0,
    )
    result_zero = _result(posterior=posterior_zero)
    assert "unidentified_mass" not in repr(result_zero)


def test_analysis_result_repr_no_refutations_ran():
    result = _result(validation=ValidationView(passed=False, ran=False, count=0))
    assert "refute=" not in repr(result)


def test_fmt_float_handles_nan_and_none():
    from antecedent.results._format import fmt_float, fmt_pct

    assert fmt_float(None) == "None"
    assert fmt_float(float("nan")) == "nan"
    assert fmt_float(float("inf")) == "inf"
    assert fmt_float(float("-inf")) == "-inf"
    assert fmt_float(1.23456, ndigits=2) == "1.23"
    assert fmt_pct(None) == "None"
    assert fmt_pct(math.nan) == "nan"
    assert fmt_pct(0.5) == "50.0%"
