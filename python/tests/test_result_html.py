"""Golden tests for ``_repr_html_`` notebook rendering.

These construct views directly (no native calls) and assert on the
structural pieces the task calls out: verdict banner, effect row,
adjustment-set chips, refutation table, and the amber unidentified-mass
callout — plus that every interpolated value is escaped and that
rendering degrades to a ``repr()`` fallback rather than raising.
"""

from __future__ import annotations

from antecedent.results import (
    AnalysisResult,
    EstimateView,
    IdentificationView,
    PerformanceView,
    PosteriorView,
    RefutationReport,
    ValidationView,
)
from antecedent.results._html import _analysis_result_repr_html


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


def _refutation(passed: bool, refuter: str = "placebo_treatment") -> RefutationReport:
    return RefutationReport(
        refuter=refuter,
        original_ate=0.412,
        refuted_ate=0.001 if passed else 0.39,
        comparison=0.02,
        informative=True,
        passed=passed,
        failure_condition=None if passed else "too close to original",
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


def _result(
    *,
    identification: IdentificationView | None = None,
    posterior: PosteriorView | None = None,
    validation: ValidationView | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        identification=identification if identification is not None else _identification(),
        estimate=_estimate(),
        posterior=posterior,
        validation=validation if validation is not None else _validation(),
        performance=PerformanceView(),
        diagnostics=[],
        provenance={"node_count": 3},
    )


# --- attachment -----------------------------------------------------


def test_repr_html_is_attached_to_analysis_result():
    result = _result()
    assert hasattr(result, "_repr_html_")
    rendered = result._repr_html_()
    assert isinstance(rendered, str)
    assert "<div" in rendered


# --- verdict banner -----------------------------------------------------


def test_repr_html_identified_verdict():
    html = _result()._repr_html_()
    assert "Identified" in html
    assert "backdoor.adjustment" in html


def test_repr_html_not_identified_verdict():
    html = _result(identification=_identification(status="NotIdentified"))._repr_html_()
    assert "Not identified" in html


# --- effect row -----------------------------------------------------


def test_repr_html_effect_row_shows_point_estimate_and_estimator():
    html = _result()._repr_html_()
    assert "0.412" in html
    assert "0.031" in html
    assert "ols" in html
    assert "analytic" in html


# --- adjustment-set chips + escaping -----------------------------------------------------


def test_repr_html_adjustment_chips_render():
    html = _result()._repr_html_()
    assert "market_demand_index" in html


def test_repr_html_empty_adjustment_set_renders_placeholder():
    html = _result(identification=_identification(adjustment_set=[]))._repr_html_()
    assert "(none)" in html


def test_repr_html_escapes_script_injection_in_adjustment_set():
    malicious = "<script>alert(1)</script>"
    html = _result(identification=_identification(adjustment_set=[malicious]))._repr_html_()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_repr_html_escapes_estimator_id_and_method():
    html = _result(
        identification=_identification(method='backdoor" onmouseover="alert(1)'),
    )._repr_html_()
    assert 'onmouseover="alert(1)' not in html
    assert "&quot;" in html or "&#34;" in html


# --- refutation table -----------------------------------------------------


def test_repr_html_refutation_table_has_one_row_per_report():
    html = _result(
        validation=_validation(
            reports=[_refutation(True, "placebo"), _refutation(False, "bootstrap")]
        )
    )._repr_html_()
    assert "placebo" in html
    assert "bootstrap" in html
    assert "pass" in html
    assert "fail" in html


def test_repr_html_no_refutations_shows_muted_message():
    html = _result(validation=ValidationView(passed=False, ran=False, count=0))._repr_html_()
    assert "No refutations ran." in html


# --- amber unidentified-mass callout -----------------------------------------------------


def test_repr_html_amber_callout_present_when_mass_positive():
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
    html = _result(posterior=posterior)._repr_html_()
    assert '<div class="antecedent-ar-callout">' in html
    assert "gives no identified estimand" in html
    assert "50.0%" in html
    assert "no identified estimand" in html


def test_repr_html_amber_callout_absent_when_mass_zero():
    posterior = PosteriorView(
        effect_mean=2.29,
        effect_sd=0.03,
        q025=2.2,
        q975=2.4,
        n_draws=200,
        p_below_zero=0.0,
        backend="conjugate",
        unidentified_mass=0.0,
    )
    html = _result(posterior=posterior)._repr_html_()
    assert '<div class="antecedent-ar-callout">' not in html
    assert "gives no identified estimand" not in html


def test_repr_html_amber_callout_absent_when_posterior_none():
    html = _result(posterior=None)._repr_html_()
    assert '<div class="antecedent-ar-callout">' not in html
    assert "gives no identified estimand" not in html


# --- graceful degradation -----------------------------------------------------


def test_repr_html_frequentist_result_without_posterior_still_renders():
    html = _result(
        posterior=None, validation=ValidationView(passed=False, ran=False, count=0)
    )._repr_html_()
    assert "Identified" in html
    assert "No refutations ran." in html
    assert '<div class="antecedent-ar-callout">' not in html
    assert "gives no identified estimand" not in html


def test_repr_html_never_raises_on_malformed_result_and_falls_back_to_pre():
    class Broken:
        def __repr__(self) -> str:
            return "<Broken totally-unrenderable>"

        @property
        def identification(self) -> object:
            raise RuntimeError("boom")

    rendered = _analysis_result_repr_html(Broken())  # type: ignore[arg-type]
    assert rendered.startswith("<pre>")
    assert "&lt;Broken totally-unrenderable&gt;" in rendered
    assert "<Broken totally-unrenderable>" not in rendered


# --- self-contained markup -----------------------------------------------------


def test_repr_html_is_self_contained_no_external_assets():
    html = _result()._repr_html_()
    assert "http://" not in html
    assert "https://" not in html
    assert "<script" not in html.lower()
    assert "@import" not in html


# --- nested-view standalone rendering -----------------------------------------------------


def test_validation_view_repr_html_standalone():
    view = _validation(reports=[_refutation(True, "placebo")])
    html = view._repr_html_()
    assert "placebo" in html
    assert "pass" in html


def test_posterior_view_repr_html_standalone_empty():
    view = PosteriorView(
        effect_mean=None,
        effect_sd=None,
        q025=None,
        q975=None,
        n_draws=None,
        p_below_zero=None,
        backend=None,
    )
    html = view._repr_html_()
    assert "No posterior computed." in html


def test_posterior_view_repr_html_standalone_populated():
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
    html = view._repr_html_()
    assert "2.290" in html
    assert '<div class="antecedent-ar-callout">' in html
    assert "gives no identified estimand" in html
