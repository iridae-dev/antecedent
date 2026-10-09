"""Actual original native work on success, failure, cancellation and compatible reuse."""

import traceback
from dataclasses import replace

import pytest
from antecedent._native import CancellationToken
from antecedent.errors import CausalValueError
from antecedent.execution_attempt import observe_native_attempts
from antecedent.recalc import RecalcReceipt
from antecedent.recalc_adjusted import AdjustedSession, GlmModel
from antecedent.recalc_bayesian import BayesianSession
from antecedent.recalc_design import DesignSession
from antecedent.recalc_dr import DrSession
from antecedent.recalc_static import StaticResponseSession
from antecedent.recalc_temporal import TemporalSession

from test_recalc_adjusted import linear_request, logistic_request
from test_recalc_bayesian import request as bayesian_request
from test_recalc_composite import fixture as composite_fixture
from test_recalc_design import request as design_request
from test_recalc_dr import request as dr_request
from test_recalc_static import response_request
from test_recalc_temporal import request as temporal_request


@pytest.mark.parametrize(
    "session_factory,request_factory",
    [
        (AdjustedSession, linear_request),
        (DrSession, dr_request),
        (StaticResponseSession, response_request),
        (BayesianSession, bayesian_request),
        (TemporalSession, temporal_request),
        (DesignSession, design_request),
    ],
    ids=["adjusted", "doubly_robust", "static", "bayesian", "temporal", "design"],
)
def test_six_original_native_families_actual_first_work_and_complete_reuse_zero(
    session_factory, request_factory
):
    session, request = session_factory(), request_factory()
    first = observe_native_attempts(lambda: session.execute(request, seed=3))
    original = first.unwrap()
    assert first.error is None and first.value is original
    # Independently consume the original scientific receipt; the observation creates none.
    historical = RecalcReceipt.consume(original.receipt.export())
    assert historical.totals == original.receipt.totals
    assert not first.report.is_empty and first.report.is_complete
    assert len(first.report.operations) == 18
    assert all(row.failed == row.unfinished == 0 for row in first.report.operations.values())
    assert any(row.completed > 0 for row in first.report.operations.values())
    if session_factory in (AdjustedSession, DrSession):
        assert first.report.operations["law_summary"].completed == 1
    same = observe_native_attempts(lambda: session.execute(request, seed=3))
    assert same.unwrap().receipt.totals.total == 0
    assert same.report.is_empty and same.report.is_complete
    assert all(row.attempted == 0 for row in same.report.operations.values())
    # These are observations, not fit multipliers or new resumable scientific authority.
    assert same.report.scope == "synchronous_original_native_components"


def test_actual_glm_nonconvergence_reports_unsuccessful_work_and_preserves_original_state():
    session, request = AdjustedSession(), logistic_request()
    original = session.execute(request, seed=3)
    before = session.identities
    attempt = observe_native_attempts(
        lambda: session.execute(replace(request, model=GlmModel(max_iter=1)), seed=3)
    )
    assert attempt.value is None and attempt.error is not None
    assert attempt.error.reason_code == "route_not_supported"
    assert attempt.error.detail == "recalc.adjusted_glm_not_converged"
    assert "execute" in {frame.name for frame in traceback.extract_tb(attempt.error.__traceback__)}
    assert not attempt.report.is_empty and attempt.report.is_complete
    assert any(row.failed > 0 for row in attempt.report.operations.values())
    assert attempt.report.operations["adjusted_fit"].attempted == 1
    with pytest.raises(type(attempt.error)) as repeated:
        attempt.unwrap()
    assert repeated.value is attempt.error
    assert session.identities == before and session.is_live
    retry = observe_native_attempts(lambda: session.execute(request, seed=3))
    assert retry.report.is_empty and retry.unwrap().law == original.law


def test_preflight_cancellation_has_zero_component_work_and_preserves_combined_state():
    _, session, request, providers, calls = composite_fixture(selected_providers=True)
    original = session.execute(request, providers=providers)
    token = CancellationToken()
    token.cancel()
    stop = observe_native_attempts(lambda: session.execute(request, providers={}, cancel=token))
    assert stop.error is not None and stop.value is None
    assert stop.error.reason_code == "cancelled_no_claim"
    assert stop.report.is_empty and stop.report.is_complete
    assert [len(x) for x in calls] == [1, 1]
    same = observe_native_attempts(lambda: session.execute(request, providers={}))
    assert same.report.is_empty and same.unwrap().decision == original.decision


def test_nested_native_observers_each_see_original_work_once_and_python_errors_are_unchanged():
    session, request = AdjustedSession(), linear_request()
    outer = observe_native_attempts(
        lambda: observe_native_attempts(lambda: session.execute(request, seed=3))
    )
    inner = outer.unwrap()
    assert inner.unwrap().receipt.totals.model_fits == 1
    assert outer.report.operations == inner.report.operations
    error = ValueError("original Python exception, outside instrumented native work")
    calls = []

    def fail_once():
        calls.append(1)
        raise error

    attempt = observe_native_attempts(fail_once)
    assert calls == [1] and attempt.error is error
    assert "fail_once" in {frame.name for frame in traceback.extract_tb(error.__traceback__)}
    assert attempt.report.is_empty
    with pytest.raises(ValueError, match="original Python exception") as raised:
        attempt.unwrap()
    assert raised.value is error and calls == [1]

    with pytest.raises(CausalValueError, match="native_attempt.invalid_callable") as invalid:
        observe_native_attempts(5)
    assert invalid.value.reason_code == "invalid_argument"
