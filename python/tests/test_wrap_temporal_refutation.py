"""Regression test: a failing temporal refuter must report validation.passed is False.

`estimation._wrap_temporal` used to set `passed=ran` — True whenever *any*
refuter ran, regardless of whether it actually passed — instead of reading
each refuter's real outcome. Unlike the static `AteAnalysisResult` DTO, the
temporal `AnalysisResult` DTO has no scalar `refutation_passed` field; the
only source of truth is the per-refuter `refutations` list (each entry has
its own `passed: bool`, confirmed against `python/antecedent/_native.pyi`).
This test exercises `_wrap_temporal` directly against a minimal stand-in for
that native DTO, rather than depending on a real refuter actually failing
end-to-end (which would be slower and less precisely targeted at the bug).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("antecedent")
from antecedent.estimation import _wrap_temporal


def _raw_temporal_result(*, refutations):
    """Minimal stand-in covering every attribute `_wrap_temporal` reads directly."""
    return SimpleNamespace(
        ate=1.0,
        se_analytic=0.1,
        se_bootstrap=None,
        identification_status="NonparametricallyIdentified",
        method="temporal.linear.adjustment",
        estimator_id="temporal.linear.adjustment",
        adjustment_set=[],
        assumption_count=0,
        derivation_step_count=0,
        posterior_n_draws=None,
        mediation_total=None,
        mediation_mediated=None,
        mediation_direct=None,
        refutation_count=len(refutations),
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
    raw = _raw_temporal_result(refutations=[_report(passed=False)])
    result = _wrap_temporal(raw)
    assert result.validation.ran is True
    assert result.validation.count == 1
    assert result.validation.passed is False, (
        "a failing temporal refuter must not report validation.passed=True"
    )


def test_passing_temporal_refuter_reports_validation_passed():
    raw = _raw_temporal_result(refutations=[_report(passed=True)])
    result = _wrap_temporal(raw)
    assert result.validation.ran is True
    assert result.validation.passed is True


def test_mixed_refuters_report_validation_failed():
    """Any refuter failing must fail the whole validation, not just the last one checked."""
    raw = _raw_temporal_result(refutations=[_report(passed=True), _report(passed=False)])
    result = _wrap_temporal(raw)
    assert result.validation.ran is True
    assert result.validation.count == 2
    assert result.validation.passed is False


def test_no_refuters_ran_reports_validation_not_passed():
    raw = _raw_temporal_result(refutations=[])
    result = _wrap_temporal(raw)
    assert result.validation.ran is False
    assert result.validation.passed is False
