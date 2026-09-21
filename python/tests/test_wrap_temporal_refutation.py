"""Regression test: a failing temporal refuter must report validation.passed is False.

`estimation._wrap_ate` (which the temporal DTO also wraps) once set `passed=ran`, True
whenever *any* refuter ran. The aggregate is owned by the native `ValidationSection`;
the wrapper must publish it unchanged. These tests exercise the wrapper directly
against a minimal stand-in for the native DTO's nested sections.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("antecedent")
from antecedent.estimation import _wrap_ate as _wrap_temporal


def _raw_temporal_result(*, refutations, passed, ran):
    """Minimal stand-in carrying the nested sections the native DTO exposes.

    ``validation.passed`` / ``ran`` are the native aggregate: the wrapper must
    carry them through and never recompute them from the per-refuter list.
    """
    return SimpleNamespace(
        identification=SimpleNamespace(
            status="NonparametricallyIdentified",
            method="temporal.linear.adjustment",
            adjustment_set=[],
            assumption_count=0,
            derivation_step_count=0,
        ),
        estimate=SimpleNamespace(
            ate=1.0,
            se_analytic=0.1,
            se_bootstrap=None,
            estimator_id="temporal.linear.adjustment",
            method="temporal.linear.adjustment",
            overlap_ess=None,
            overlap_propensity_min=None,
        ),
        posterior=SimpleNamespace(n_draws=None),
        validation=SimpleNamespace(
            passed=passed,
            ran=ran,
            count=len(refutations),
            reports=refutations,
        ),
        performance=SimpleNamespace(
            plan_id="temporal.plan",
            modality="temporal",
            peak_memory_bytes=None,
            latency_mode=None,
            wall_time_ns=None,
            bootstrap_replicates_requested=None,
            bootstrap_replicates_ok=None,
            n_draws=None,
            cancelled=False,
            early_stopped=False,
            stage_timings=None,
        ),
        mediation_total=None,
        mediation_mediated=None,
        mediation_direct=None,
        refutations=refutations,
        diagnostics=[],
        provenance_node_count=3,
        plan_id="temporal.plan",
        modality="temporal",
        discovery_algorithm=None,
        graph_review_required=False,
        plan_identifier=None,
        plan_estimator=None,
        validation_suite=None,
        peak_memory_bytes=None,
        worker_threads=1,
        expected_python_crossings=0,
    )


def _report(*, passed):
    return SimpleNamespace(
        refuter="placebo",
        original_ate=1.0,
        refuted_ate=1.0 if passed else 5.0,
        comparison=0.0 if passed else 4.0,
        informative=True,
        passed=passed,
        failure_condition=None if passed else "placebo effect not near zero",
        replicates=1,
    )


def test_failing_temporal_refuter_reports_validation_failed():
    raw = _raw_temporal_result(refutations=[_report(passed=False)], passed=False, ran=True)
    result = _wrap_temporal(raw)
    assert result.validation.ran is True
    assert result.validation.count == 1
    assert result.validation.passed is False, (
        "a failing temporal refuter must not report validation.passed=True"
    )


def test_passing_temporal_refuter_reports_validation_passed():
    raw = _raw_temporal_result(refutations=[_report(passed=True)], passed=True, ran=True)
    result = _wrap_temporal(raw)
    assert result.validation.ran is True
    assert result.validation.passed is True


def test_mixed_refuters_report_validation_failed():
    """Any refuter failing must fail the whole validation, not just the last one checked."""
    raw = _raw_temporal_result(
        refutations=[_report(passed=True), _report(passed=False)], passed=False, ran=True
    )
    result = _wrap_temporal(raw)
    assert result.validation.ran is True
    assert result.validation.count == 2
    assert result.validation.passed is False


def test_no_refuters_ran_reports_validation_not_passed():
    raw = _raw_temporal_result(refutations=[], passed=False, ran=False)
    result = _wrap_temporal(raw)
    assert result.validation.ran is False
    assert result.validation.passed is False
