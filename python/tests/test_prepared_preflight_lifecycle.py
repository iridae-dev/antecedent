"""Prepared diagnostics use retained scientific designs and explicit execution boundaries."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import preflight as pf
from antecedent.errors import (
    CausalCancelledError,
    CausalResourceError,
    CausalTypeError,
    CausalUnsupportedError,
)
from antecedent.execution import Execution, ResourceLimits
from antecedent.execution_attempt import observe_native_attempts
from antecedent.outcomes import Outcome, capture_outcome, require_success


def _inputs(*, dependent=False):
    rng = np.random.default_rng(83)
    a, b = rng.normal(size=(2, 360))
    t = (rng.random(len(a)) < 1 / (1 + np.exp(-0.5 * a + 0.2 * b))).astype(float)
    y = 1.5 * t + a + 0.5 * b + rng.normal(scale=0.4, size=len(a))
    data = {"t": t, "y": y, "y2": 0.5 * y - a, "a": a, "b": b}
    if dependent:
        data.update(c=2 * a - 3 * b + 1, a_copy=a.copy())
    confounders = [name for name in data if name not in {"t", "y", "y2"}]
    graph = [(name, target) for name in confounders for target in ("t", "y", "y2")]
    graph.extend([("t", "y"), ("t", "y2")])
    return data, graph


def _study(*, dependent=False, execution=None):
    data, graph = _inputs(dependent=dependent)
    return ant.prepare(
        data,
        graph=graph,
        query=ant.AverageEffect("t", "y"),
        estimator="aipw",
        inference=ant.Frequentist(bootstrap=7),
        refute=False,
        execution=Execution(seed=19, threads=1) if execution is None else execution,
    )


def _assert_no_fits(attempt):
    assert attempt.report.is_complete
    for operation in (
        "glm_fit",
        "adjusted_fit",
        "learner_fit",
        "posterior_fit",
        "posterior_draw",
        "score_construction",
        "factor_construction",
        "integration",
        "decision",
    ):
        assert attempt.report.operations[operation].attempted == 0, operation


def test_prepared_counts_rank_drop_and_cost_read_the_certified_design_without_fitting():
    preparation = observe_native_attempts(lambda: _study(dependent=True))
    study = preparation.unwrap()
    _assert_no_fits(preparation)
    before = study.inspect().contract
    inspected = observe_native_attempts(lambda: pf.diagnose(study))
    report = inspected.unwrap()
    _assert_no_fits(inspected)
    assert report.treatment == ("t",) and report.outcome == "y"
    assert set(report.adjustment_set) == {"a", "b", "c", "a_copy"}
    assert report.rows_complete == report.rows_total == 360
    assert sum(arm.rows for arm in report.arms) == 360
    assert report.rank is not None
    data, _ = _inputs(dependent=True)
    independent_rank = np.linalg.matrix_rank(
        np.column_stack([np.ones(360), *(data[name] for name in report.adjustment_set)])
    )
    assert report.rank.numerical_rank == independent_rank == 3
    assert report.rank.design_columns == 5
    assert report.is_blocked
    assert any(set(group.columns) == {"a", "a_copy"} for group in report.duplicates)
    planned = observe_native_attempts(
        lambda: pf.plan_rank_drop(study, priority=["a", "b", "c", "a_copy"])
    )
    plan = planned.unwrap()
    _assert_no_fits(planned)
    assert plan.kept_adjustment == ("a", "b")
    dropped = {column.column: column for column in plan.dropped}
    assert set(dropped) == {"c", "a_copy"}
    relation = {term.column: term.coefficient for term in dropped["c"].explained_by}
    assert relation == pytest.approx({"(intercept)": 1, "a": 2, "b": -3}, abs=1e-8)
    counted = observe_native_attempts(lambda: pf.estimate_cost(study))
    cost = counted.unwrap()
    _assert_no_fits(counted)
    assert cost.planning_hint and cost.estimator == "aipw"
    assert cost.crossfit_folds == 5 and cost.nuisance_fits_per_pass == 15
    assert cost.inference.bootstrap_replicates == 7
    assert cost.passes_upper_bound == 8 and cost.nuisance_fits_upper_bound == 120
    assert study.inspect().contract == before
    assert study.inspect().stage == "prepared"


def test_diagnostic_fitting_returns_one_outcome_and_does_not_execute_the_effect():
    events = []
    policy = Execution(
        seed=23, threads=1, on_progress=lambda value, stage: events.append((value, stage))
    )
    study = _study(execution=policy)
    events.clear()
    before = study.inspect().contract
    attempted = observe_native_attempts(lambda: pf.diagnose_fit(study))
    outcome = attempted.unwrap()
    assert isinstance(outcome, Outcome)
    diagnostic = require_success(outcome)
    assert not isinstance(diagnostic, Outcome)
    assert attempted.report.operations["glm_fit"].completed == 1
    assert attempted.report.operations["score_construction"].attempted == 0
    assert diagnostic.status == "fitted" and diagnostic.fit is not None
    assert 0 < diagnostic.fit.min <= diagnostic.fit.max < 1
    assert sum(arm.rows for arm in diagnostic.fit.arm_ess) == 360
    assert all(0 < arm.ess <= arm.rows for arm in diagnostic.fit.arm_ess)
    assert events == [(0.0, "diagnose_fit"), (1.0, "diagnose_fit")]
    assert study.execution is policy
    assert study.inspect().contract == before and study.inspect().stage == "prepared"


def test_prepared_batch_diagnostics_keep_query_order_and_declared_budgets():
    from antecedent.estimation import prepare_many

    data, graph = _inputs()
    batch = prepare_many(
        data,
        graph=graph,
        queries=[ant.AverageEffect("t", "y"), ant.AverageEffect("t", "y2")],
        estimator="aipw",
        inference=ant.Frequentist(bootstrap=7),
        refute=False,
        execution=Execution(seed=29, threads=1),
    )
    inspected = observe_native_attempts(lambda: pf.diagnose(batch))
    report = inspected.unwrap()
    _assert_no_fits(inspected)
    assert isinstance(report, pf.BatchPreflightReport)
    assert [member.outcome for member in report.reports] == ["y", "y2"]
    assert report.blocked_plans == () and report.shares_covariates
    assert all(member.rows_complete == 360 for member in report.reports)
    planned = observe_native_attempts(lambda: pf.plan_rank_drop(batch))
    plans = planned.unwrap()
    _assert_no_fits(planned)
    assert len(plans) == 2 and all(plan.dropped == () for plan in plans)
    cost = pf.estimate_cost(batch)
    assert isinstance(cost, pf.BatchCostEstimate)
    assert cost.claims == 2 and cost.bootstrap_replicates_total == 14
    assert cost.nuisance_fits_upper_bound == 240
    fitted = observe_native_attempts(lambda: pf.diagnose_fit(batch))
    diagnostics = require_success(fitted.unwrap())
    assert len(diagnostics) == 2
    assert all(report.status == "fitted" for report in diagnostics)
    assert fitted.report.operations["glm_fit"].completed == 2
    assert batch._native.materialized_plan_count() == 0
    assert batch._native.retained_score_table_count() == 0


def test_diagnostic_execution_honors_resource_cancellation_and_original_observer_errors():
    study = _study()
    tiny = Execution(limits=ResourceLimits(memory_bytes=0))
    with pytest.raises(CausalResourceError):
        pf.diagnose_fit(study, execution=tiny)
    cancel = ant.CancellationToken()
    cancel.cancel()
    with pytest.raises(CausalCancelledError):
        pf.diagnose_fit(study, execution=Execution(cancel=cancel))
    original = CausalUnsupportedError("user progress failed", reason_code="effect_not_identified")

    def failed_progress(_fraction, _stage):
        raise original

    with pytest.raises(CausalUnsupportedError) as raised:
        capture_outcome(pf.diagnose_fit, study, execution=Execution(on_progress=failed_progress))
    assert raised.value is original
    assert any(frame.name == "failed_progress" for frame in raised.traceback)
    with pytest.raises(CausalUnsupportedError) as unavailable:
        pf.diagnose_fit(study, execution=Execution(on_stage=lambda *_args: None))
    assert unavailable.value.reason_code == "stage_stream_unavailable"
    assert require_success(pf.diagnose_fit(study)).status == "fitted"
    assert study.inspect().stage == "prepared"


def test_prepared_rank_plans_reject_role_reconstruction_and_loose_execution_keywords():
    study = _study()
    with pytest.raises(CausalTypeError):
        pf.plan_rank_drop(study, treatment="t", outcome="y", adjustment=["a", "b"])
    with pytest.raises(TypeError, match="unexpected keyword"):
        pf.plan_rank_drop(study, seed=1)
    with pytest.raises(CausalTypeError):
        pf.diagnose(study, execution={"seed": 1})
    with pytest.raises(CausalTypeError):
        pf.diagnose_fit({"t": [0, 1], "y": [1, 2]})
    with pytest.raises(CausalUnsupportedError) as refused:
        pf.plan_rank_drop(study, priority=["a"])
    assert refused.value.reason_code == "rank_drop_not_licensed"
    assert refused.value.refusal_fields["implicated_columns"] == ["b"]


def test_absent_diagnostic_fit_keeps_missing_scores_explicit():
    study = _study(dependent=True)
    attempted = observe_native_attempts(lambda: pf.diagnose_fit(study))
    diagnostic = require_success(attempted.unwrap())
    assert diagnostic.status == "absent" and diagnostic.fit is None
    assert diagnostic.reason is not None
    assert diagnostic.numerical_rank == 3 and diagnostic.design_columns == 5
    assert attempted.report.operations["glm_fit"].attempted == 1
    assert attempted.report.operations["score_construction"].attempted == 0
    assert study.inspect().stage == "prepared"


def test_unsupported_diagnostic_family_returns_a_stable_scientific_refusal():
    data, graph = _inputs()
    study = ant.prepare(
        data,
        graph=graph,
        query=ant.ResponseCurve("t", "y", grid=[0.0, 1.0]),
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
    )
    attempted = observe_native_attempts(lambda: pf.diagnose_fit(study))
    result = attempted.unwrap()
    assert result.value is None and result.refusal is not None
    assert result.refusal.code == "route_not_supported"
    _assert_no_fits(attempted)
