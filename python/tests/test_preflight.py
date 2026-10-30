"""Fit-free preflight, fitted diagnostics, rank-drop plans and cost counts.

Mirrors ``crates/antecedent/tests/preflight_reproducers.rs`` through the Python surface.
Numeric expectations are hand-derived in the test (exact column relations) or recomputed
with NumPy, never read back from the code under test.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent import preflight as pf
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.outcomes import require_success

from _refusal import assert_registered_refusal


def _normals(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(n) for _ in range(k)]


def _frame(t, y, **columns) -> dict[str, np.ndarray]:
    return {"t": np.asarray(t, dtype=np.float64), "y": np.asarray(y, dtype=np.float64)} | {
        name: np.asarray(values, dtype=np.float64) for name, values in columns.items()
    }


def test_rank_174_of_175_is_named_by_the_fit_free_report_and_agrees_with_numpy() -> None:
    n = 400
    covariates = _normals(n, 173, seed=1)
    covariates.append(covariates[5].copy())
    rng = np.random.default_rng(2)
    t = (rng.random(n) < 0.5).astype(np.float64)
    y = t + covariates[0] + rng.standard_normal(n)
    data = _frame(t, y, **{f"x{i}": c for i, c in enumerate(covariates)})
    adjustment = [f"x{i}" for i in range(174)]

    report = pf.preflight(data, treatment="t", outcome="y", adjustment=adjustment)
    assert report.adjustment_set == tuple(adjustment)
    assert report.rank is not None
    assert (report.rank.design_columns, report.rank.numerical_rank) == (175, 174)
    # Independent oracle: NumPy's SVD rank of the same design.
    design = np.column_stack([np.ones(n), *covariates])
    assert np.linalg.matrix_rank(design) == report.rank.numerical_rank
    (dependent,) = report.rank.dependent
    assert dependent.column == "x173"
    (term,) = dependent.explained_by
    assert term.column == "x5"
    assert term.coefficient == pytest.approx(1.0, abs=1e-8)
    assert report.duplicates[0].columns == ("x5", "x173")
    assert report.is_blocked
    assert [f.code for f in report.blocking] == ["design_rank_deficient"]

    # The fit that would follow fails before any score exists: absent, never reconstructed.
    fitted = pf.fit_diagnostics(data, treatment="t", outcome="y", adjustment=adjustment)
    assert fitted.status == "absent"
    assert fitted.fit is None
    assert (fitted.numerical_rank, fitted.design_columns) == (174, 175)


def test_exact_collinearity_reports_the_hand_computed_relation() -> None:
    n = 200
    a, b = _normals(n, 2, seed=3)
    combo = 3.0 * a - 2.0 * b + 5.0
    rng = np.random.default_rng(4)
    t = (rng.random(n) < 0.5).astype(np.float64)
    data = _frame(t, t + a, a=a, b=b, combo=combo, constant=np.full(n, 7.0))
    report = pf.preflight(
        data, treatment="t", outcome="y", adjustment=["a", "b", "combo", "constant"]
    )
    assert report.rank is not None
    assert report.rank.numerical_rank == 3
    by_name = {
        d.column: {w.column: w.coefficient for w in d.explained_by} for d in report.rank.dependent
    }
    assert set(by_name) == {"combo", "constant"}
    assert by_name["combo"]["(intercept)"] == pytest.approx(5.0, abs=1e-8)
    assert by_name["combo"]["a"] == pytest.approx(3.0, abs=1e-8)
    assert by_name["combo"]["b"] == pytest.approx(-2.0, abs=1e-8)
    assert by_name["constant"]["(intercept)"] == pytest.approx(7.0, abs=1e-8)


def test_near_duplicates_and_leakage_like_columns_only_flag_for_review() -> None:
    n = 300
    z0, noise, z3 = _normals(n, 3, seed=5)
    rng = np.random.default_rng(6)
    t = (rng.random(n) < 0.5).astype(np.float64)
    near = z0 + 1e-4 * noise
    y = 3.0 * z3 + 1e-3 * rng.standard_normal(n)
    data = _frame(t, y, z0=z0, near=near, z3=z3, treated_copy=t.copy())
    adjustment = ["z0", "near", "z3", "treated_copy"]
    report = pf.preflight(data, treatment="t", outcome="y", adjustment=adjustment)
    assert not report.is_blocked
    codes = {f.code for f in report.review_flags}
    assert {
        "near_collinear_column",
        "adjustment_column_tracks_outcome",
        "adjustment_column_tracks_treatment",
        "adjustment_duplicates_query_variable",
    } <= codes
    near_flag = next(f for f in report.findings if f.code == "near_collinear_column")
    assert near_flag.columns == ("near",)
    assert near_flag.measure is not None and 1e-6 < near_flag.measure < 1e-3
    # Review flags never change the adjustment set.
    assert report.adjustment_set == tuple(adjustment)


def test_separation_and_empty_arms_are_flagged_or_refused() -> None:
    n = 200
    z, w = _normals(n, 2, seed=7)
    t = (z > 0).astype(np.float64)
    report = pf.preflight(
        _frame(t, t + w, z=z, w=w), treatment="t", outcome="y", adjustment=["z", "w"]
    )
    flag = next(f for f in report.findings if f.code == "column_separates_arms")
    assert flag.columns == ("z",)
    assert flag.severity == "review"
    assert not report.is_blocked
    fitted = pf.fit_diagnostics(
        _frame(t, t + w, z=z, w=w), treatment="t", outcome="y", adjustment=["z", "w"]
    )
    # Treatment is the sign of z, so the logistic MLE does not exist: the fit is reported
    # as separated, saturated at the boundary and not converged, never as a clean fit.
    assert fitted.status == "fitted" and fitted.fit is not None
    assert fitted.fit.separated and fitted.fit.boundary_saturated
    assert not fitted.fit.converged
    assert fitted.fit.min < 1e-6 and fitted.fit.max > 1.0 - 1e-6

    empty = pf.preflight(_frame(np.zeros(n), w, z=z), treatment="t", outcome="y", adjustment=["z"])
    assert [a.rows for a in empty.arms] == [n, 0]
    assert [f.code for f in empty.blocking] == ["arm_not_populated"]
    absent = pf.fit_diagnostics(
        _frame(np.zeros(n), w, z=z), treatment="t", outcome="y", adjustment=["z"]
    )
    assert absent.status == "absent"
    assert absent.reason is not None and "no complete-case rows" in absent.reason


def test_rank_drop_follows_the_declared_priority_and_refuses_query_columns() -> None:
    n = 150
    a, c = _normals(n, 2, seed=8)
    b = 2.0 * a
    rng = np.random.default_rng(9)
    t = (rng.random(n) < 0.5).astype(np.float64)
    data = _frame(t, t + a + c, a=a, b=b, c=c)
    plan = pf.plan_rank_drop(data, treatment="t", outcome="y", adjustment=["a", "b", "c"])
    assert [d.column for d in plan.dropped] == ["b"]
    assert plan.dropped[0].explained_by[0].column == "a"
    assert plan.dropped[0].explained_by[0].coefficient == pytest.approx(2.0, abs=1e-8)
    assert plan.kept_adjustment == ("a", "c")
    assert plan.design_identity == "adjustment=[a,c];dropped=[b];priority=[a,b,c]"

    flipped = pf.plan_rank_drop(
        data, treatment="t", outcome="y", adjustment=["a", "b", "c"], priority=["b", "c", "a"]
    )
    assert [d.column for d in flipped.dropped] == ["a"]
    assert flipped.dropped[0].explained_by[0].coefficient == pytest.approx(0.5, abs=1e-8)

    with pytest.raises(CausalUnsupportedError) as raised:
        pf.plan_rank_drop(
            data, treatment="t", outcome="y", adjustment=["a", "b", "c"], priority=["a"]
        )
    assert raised.value.reason_code == "rank_drop_not_licensed"
    assert_registered_refusal(raised.value)
    assert raised.value.refusal_fields["stage"] == "rank_drop"
    assert raised.value.refusal_fields["implicated_columns"] == ["b", "c"]

    aliased = _frame(t, t + a, a=a, alias=t.copy())
    with pytest.raises(CausalUnsupportedError) as raised:
        pf.plan_rank_drop(aliased, treatment="t", outcome="y", adjustment=["a", "alias"])
    assert raised.value.reason_code == "rank_drop_not_licensed"
    assert "alias" in raised.value.refusal_fields["implicated_columns"]


def _confounded(n: int = 500, seed: int = 12):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    t = (rng.random(n) < 1.0 / (1.0 + np.exp(-0.9 * z))).astype(np.float64)
    t2 = (rng.random(n) < 1.0 / (1.0 + np.exp(0.5 * z))).astype(np.float64)
    y = t + 0.5 * t2 + z + 0.4 * rng.standard_normal(n)
    return {"t": t, "t2": t2, "y": y, "z": z}


def test_a_prepared_plan_diagnoses_and_counts_its_cost() -> None:
    data = _confounded()
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    prepared = antecedent.prepare(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        estimator="aipw",
        refute=False,
        inference=antecedent.Frequentist(bootstrap=0),
    )
    report = prepared.diagnose()
    assert "z" in report.adjustment_set
    assert report.rows_complete == 500
    assert sum(a.rows for a in report.arms) == 500
    assert not report.is_blocked

    fitted = require_success(prepared.diagnose_fit())
    assert fitted.status == "fitted" and fitted.fit is not None
    t = data["t"]
    # Independent oracle: unpenalised logistic MLE of treatment on z by Newton steps, and the
    # Kish effective sample size of the inverse-probability weights within each arm.
    design = np.column_stack([np.ones(len(t)), data["z"]])
    beta = np.zeros(2)
    for _ in range(60):
        q = 1.0 / (1.0 + np.exp(-design @ beta))
        beta += np.linalg.solve(design.T @ (design * (q * (1.0 - q))[:, None]), design.T @ (t - q))
    q = 1.0 / (1.0 + np.exp(-design @ beta))
    assert fitted.fit.min == pytest.approx(q.min(), abs=1e-7)
    assert fitted.fit.max == pytest.approx(q.max(), abs=1e-7)
    expected_ess = {}
    for arm_value, weight in ((0.0, 1.0 / (1.0 - q)), (1.0, 1.0 / q)):
        w_arm = weight[t == arm_value]
        expected_ess[int((t == arm_value).sum())] = w_arm.sum() ** 2 / (w_arm**2).sum()
    for arm in fitted.fit.arm_ess:
        assert arm.ess == pytest.approx(expected_ess[arm.rows], rel=1e-6)
    assert sum(arm.rows for arm in fitted.fit.arm_ess) == len(t)

    plan = prepared.plan_rank_drop()
    assert plan.dropped == ()

    cost = prepared.estimate_cost()
    assert cost.planning_hint is True
    assert cost.estimator == "aipw"
    assert cost.fit_route == "aipw" and cost.cluster_labels is None
    assert cost.crossfit_folds == 5
    assert cost.nuisance_fits_per_pass == 15
    assert cost.nuisance_fits_upper_bound == 15
    # Seconds appear only with a named benchmark file (parity/cost_model.toml), and say which.
    named = cost.seconds_basis.startswith("planning hint from named local benchmark")
    assert (cost.seconds is not None) == named and cost.seconds_basis
    assert cost.inference.mode == "frequentist"
    assert cost.inference.refit_warning is None


def test_a_prepared_batch_diagnoses_and_costs_every_claim() -> None:
    data = _confounded(seed=13)
    graph = antecedent.Dag.from_edges(
        ["t", "t2", "y", "z"],
        [("z", "t"), ("z", "t2"), ("z", "y"), ("t", "y"), ("t2", "y")],
    )
    queries = [
        antecedent.AverageEffect(treatment="t", outcome="y"),
        antecedent.AverageEffect(treatment="t2", outcome="y"),
    ]
    batch = antecedent.estimation.prepare_many(
        data,
        graph=graph,
        queries=queries,
        estimator="aipw",
        refute="none",
        inference=antecedent.Frequentist(bootstrap=10),
    )
    assert batch._native.materialized_plan_count() == 0
    report = batch.diagnose()
    assert len(report.reports) == 2
    assert report.blocked_plans == ()
    assert report.reports[0].treatment == ("t",)
    assert report.reports[1].treatment == ("t2",)
    cost = batch.estimate_cost()
    assert cost.claims == 2
    # 5 folds x 3 fits x (1 + 10 replicates) per claim, an upper bound.
    assert cost.nuisance_fits_upper_bound == 330
    assert cost.bootstrap_replicates_total == 20
    assert cost.plans[0].inference.refit_warning is not None
    assert batch._native.materialized_plan_count() == 0
    assert len(require_success(batch.diagnose_fit())) == 2
    assert len(batch.plan_rank_drop()) == 2
    assert batch._native.materialized_plan_count() == 0


def _collinear(n: int = 600, seed: int = 31):
    """``c = 2a - 3b + 1`` exactly; treatment driven by ``a``; true effect 1.5."""
    rng = np.random.default_rng(seed)
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    c = 2.0 * a - 3.0 * b + 1.0
    t = (rng.random(n) < 1.0 / (1.0 + np.exp(-0.8 * a))).astype(np.float64)
    y = 1.5 * t + a + 0.5 * b + 0.3 * rng.standard_normal(n)
    return t, y, a, b, c


def _ols_treatment_coefficient(t, y, *covariates) -> float:
    design = np.column_stack([np.ones_like(t), t, *covariates])
    return float(np.linalg.lstsq(design, y, rcond=None)[0][1])


def test_estimate_with_rank_drop_matches_numpy_and_records_the_drop() -> None:
    t, y, a, b, c = _collinear()
    data = _frame(t, y, a=a, b=b, c=c)
    result = pf.estimate_with_rank_drop(
        data, treatment="t", outcome="y", adjustment=["a", "b", "c"]
    )
    assert [d.column for d in result.plan.dropped] == ["c"]
    assert result.plan.original_adjustment == ("a", "b", "c")
    assert result.plan.kept_adjustment == ("a", "b")
    assert result.plan.design_identity == "adjustment=[a,b];dropped=[c];priority=[a,b,c]"
    assert result.span_check.original_rank == result.span_check.retained_rank == 3
    assert result.projection_invariance == "exact"
    # Independent oracle: NumPy least squares on the hand-built reduced design.
    assert result.ate == pytest.approx(_ols_treatment_coefficient(t, y, a, b), abs=1e-8)

    flipped = pf.estimate_with_rank_drop(
        data,
        treatment="t",
        outcome="y",
        adjustment=["a", "b", "c"],
        priority=["c", "a", "b"],
    )
    assert [d.column for d in flipped.plan.dropped] == ["b"]
    assert flipped.plan.design_identity != result.plan.design_identity
    assert flipped.ate == pytest.approx(result.ate, abs=1e-9)

    aipw = pf.estimate_with_rank_drop(
        data, treatment="t", outcome="y", adjustment=["a", "b", "c"], estimator="aipw"
    )
    assert aipw.projection_invariance == "unpenalized_logistic_no_separation"
    assert aipw.ate == pytest.approx(1.5, abs=0.1)


def test_estimate_with_rank_drop_refuses_aliases_and_unlicensed_estimators() -> None:
    t, y, a, b, c = _collinear(seed=32)
    aliased = _frame(t, y, a=a, alias=t.copy())
    with pytest.raises(CausalUnsupportedError) as raised:
        pf.estimate_with_rank_drop(aliased, treatment="t", outcome="y", adjustment=["a", "alias"])
    assert raised.value.reason_code == "rank_drop_not_licensed"
    assert_registered_refusal(raised.value)

    data = _frame(t, y, a=a, b=b, c=c)
    with pytest.raises(CausalUnsupportedError) as raised:
        pf.estimate_with_rank_drop(
            data,
            treatment="t",
            outcome="y",
            adjustment=["a", "b", "c"],
            estimator="propensity.weighting",
        )
    assert raised.value.reason_code == "route_not_supported"

    with pytest.raises(CausalUnsupportedError) as raised:
        pf.estimate_with_rank_drop(
            data, treatment="t", outcome="y", adjustment=["a", "b", "c"], priority=["a"]
        )
    assert raised.value.reason_code == "rank_drop_not_licensed"


def test_fit_diagnostics_of_a_confounded_design_match_an_independent_logistic_fit() -> None:
    n = 500
    z, w = _normals(n, 2, seed=21)
    rng = np.random.default_rng(22)
    t = (rng.random(n) < 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * z - 0.5 * w)))).astype(np.float64)
    data = _frame(t, t + z + rng.standard_normal(n), z=z, w=w)
    fitted = pf.fit_diagnostics(data, treatment="t", outcome="y", adjustment=["z", "w"])
    assert fitted.status == "fitted" and fitted.fit is not None
    assert fitted.reason is None and fitted.numerical_rank is None
    # Independent oracle: unpenalised logistic maximum likelihood by Newton steps in NumPy.
    design = np.column_stack([np.ones(n), z, w])
    beta = np.zeros(3)
    for _ in range(60):
        p = 1.0 / (1.0 + np.exp(-design @ beta))
        beta += np.linalg.solve(design.T @ (design * (p * (1.0 - p))[:, None]), design.T @ (t - p))
    p = 1.0 / (1.0 + np.exp(-design @ beta))
    assert fitted.fit.converged and not fitted.fit.separated
    assert fitted.fit.min == pytest.approx(p.min(), abs=1e-5)
    assert fitted.fit.max == pytest.approx(p.max(), abs=1e-5)
    assert sorted(arm.rows for arm in fitted.fit.arm_ess) == sorted(
        [int((t == 0).sum()), int((t == 1).sum())]
    )
    assert all(0 < arm.ess <= arm.rows for arm in fitted.fit.arm_ess)


_COST_TOML = """
benchmark = "unit-bench"

[machine]
platform = "test-box"
cpu_count = 8

[coefficients."aipw"]
a = 0.5
b = 0.000001
"""


def _aipw_plan(**options):
    data = _confounded()
    return antecedent.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        estimator="aipw",
        refute=False,
        inference=antecedent.Frequentist(bootstrap=0),
        **options,
    )


def test_a_declared_cost_model_backs_the_seconds_of_a_plan() -> None:
    prepared = _aipw_plan()
    model = pf.CostModel.from_toml(_COST_TOML)
    assert (model.benchmark, model.platform, model.cpu_count) == ("unit-bench", "test-box", 8)
    assert dict(model.coefficients) == {"aipw": (0.5, 1e-6)}
    assert pf.CostModel("unit-bench", "test-box", 8, {"aipw": (0.5, 1e-6)}) == model

    cost = pf.estimate_cost(prepared, cost_model=model)
    # seconds per counted fit = a + b * rows * columns^2, times the 15 counted nuisance fits.
    assert cost.nuisance_fits_upper_bound == 15
    assert cost.rows == 500 and cost.design_columns is not None
    assert cost.seconds == pytest.approx(15 * (0.5 + 1e-6 * 500 * cost.design_columns**2))
    assert cost.seconds_basis == (
        "planning hint from named local benchmark unit-bench, machine test-box (8 cpus)"
    )
    # The model is a value: the same declaration gives the same seconds and counts are unchanged.
    again = pf.estimate_cost(prepared, cost_model=model)
    assert again.seconds == cost.seconds
    assert pf.estimate_cost(prepared).nuisance_fits_upper_bound == cost.nuisance_fits_upper_bound

    other = pf.CostModel("unit-bench", "test-box", 8, {"linear.adjustment.ate": (0.1, 1e-7)})
    uncovered = pf.estimate_cost(prepared, cost_model=other)
    assert uncovered.seconds is None
    assert "no coefficients for estimator aipw" in uncovered.seconds_basis


def test_a_declared_cost_model_prices_every_plan_of_a_batch() -> None:
    data = _confounded(seed=13)
    graph = antecedent.Dag.from_edges(
        ["t", "t2", "y", "z"],
        [("z", "t"), ("z", "t2"), ("z", "y"), ("t", "y"), ("t2", "y")],
    )
    batch = antecedent.estimation.prepare_many(
        data,
        graph=graph,
        queries=[
            antecedent.AverageEffect(treatment="t", outcome="y"),
            antecedent.AverageEffect(treatment="t2", outcome="y"),
        ],
        estimator="aipw",
        refute="none",
        inference=antecedent.Frequentist(bootstrap=0),
    )
    model = pf.CostModel.from_toml(_COST_TOML)
    cost = pf.estimate_cost(batch, cost_model=model)
    assert cost.claims == 2
    for plan in cost.plans:
        assert plan.rows is not None and plan.design_columns is not None
        assert plan.nuisance_fits_upper_bound == 15
        assert plan.seconds == pytest.approx(15 * (0.5 + 1e-6 * plan.rows * plan.design_columns**2))


def test_a_malformed_or_non_monotone_cost_model_is_refused() -> None:
    with pytest.raises(CausalValueError):
        pf.CostModel("unit-bench", "test-box", 8, {"aipw": (-0.1, 1e-6)})
    with pytest.raises(CausalValueError):
        pf.CostModel("unit-bench", "test-box", 8, {"aipw": (0.1, float("nan"))})
    with pytest.raises(CausalValueError):
        pf.CostModel("unit-bench", "test-box", 8, {})
    with pytest.raises(CausalValueError):
        pf.CostModel("", "test-box", 8, {"aipw": (0.1, 1e-6)})
    with pytest.raises(CausalValueError):
        pf.CostModel("unit-bench", "test-box", 0, {"aipw": (0.1, 1e-6)})
    with pytest.raises(CausalValueError, match="machine"):
        pf.CostModel.from_toml('benchmark = "unit-bench"\n')
    with pytest.raises(CausalValueError):
        pf.CostModel.from_toml(
            'benchmark = "b"\n[machine]\nplatform = "p"\ncpu_count = 2\n'
            '[coefficients."aipw"]\na = 0.1\n'
        )
    with pytest.raises(CausalTypeError):
        pf.estimate_cost(_aipw_plan(), cost_model="not a model")  # type: ignore[arg-type]


def _confounded_table(n: int = 300, seed: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    t = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(np.float64)
    y = 2.0 * t + z + 0.3 * rng.standard_normal(n)
    return _frame(t, y, z=z, dup=z.copy())


def test_the_table_path_takes_execution_and_no_loose_seed_or_threads() -> None:
    from antecedent.execution import Execution

    data = _confounded_table()
    run = Execution(seed=5, threads=1)
    first = pf.fit_diagnostics(data, treatment="t", outcome="y", adjustment=["z"], execution=run)
    again = pf.fit_diagnostics(
        data, treatment="t", outcome="y", adjustment=["z"], execution=Execution(seed=5, threads=1)
    )
    assert first == again and first.status == "fitted"
    report = pf.preflight(data, treatment="t", outcome="y", adjustment=["z"], execution=run)
    assert report.adjustment_set == ("z",)
    # The rank-drop table path accepts an execution too; it names the exact duplicate.
    plan = pf.plan_rank_drop(
        data, treatment="t", outcome="y", adjustment=["z", "dup"], execution=run
    )
    assert [d.column for d in plan.dropped] == ["dup"]
    estimate = pf.estimate_with_rank_drop(
        data, treatment="t", outcome="y", adjustment=["z", "dup"], execution=run
    )
    assert [d.column for d in estimate.plan.dropped] == ["dup"]
    # Controls live only on Execution: a loose keyword is a plain TypeError.
    for call in (pf.preflight, pf.fit_diagnostics, pf.plan_rank_drop, pf.estimate_with_rank_drop):
        for loose in ({"seed": 5}, {"threads": 1}):
            with pytest.raises(TypeError, match="unexpected keyword"):
                call(data, treatment="t", outcome="y", adjustment=["z"], **loose)
    with pytest.raises(CausalTypeError):
        pf.preflight(data, treatment="t", outcome="y", adjustment=["z"], execution={"seed": 1})


def test_arm_weight_ess_availability_marks_unavailable_numeric_sentinel():
    from antecedent.preflight import _fit_diagnostics

    raw = {"subject": "test", "propensity": {"status": "fitted", "converged": True,
           "separated": True, "boundary_saturated": True, "iterations": 1,
           "min": 0.0, "max": 1.0, "quantiles": [],
           "arm_ess": [{"label": "treated", "rows": 3, "ess": None},
                       {"label": "control", "rows": 3, "ess": 2.5}]}}
    arms = _fit_diagnostics(raw).fit.arm_ess
    assert arms[0].ess is None and arms[0].ess_status == "unavailable"
    assert arms[1].ess == 2.5 and arms[1].ess_status == "available"


def test_overflowed_relation_coefficient_is_explicitly_unavailable():
    data = _frame([0, 1, 0, 1], [0, 1, 2, 3],
                  small=[-5e-156, 5e-156, -5e-156, 5e-156],
                  large=[-5e153, 5e153, -5e153, 5e153])
    report = pf.preflight(data, treatment="t", outcome="y", adjustment=["small", "large"])
    assert report.rank.numerical_rank == 2
    large = next(d for d in report.rank.dependent if d.column == "large")
    term = next(w for w in large.explained_by if w.column == "small")
    assert term.coefficient is None
    assert term.coefficient_status == "unavailable"

    path = f"/rank/dependent/{report.rank.dependent.index(large)}/explained_by/{large.explained_by.index(term)}/coefficient"
    assert path in report.nonfinite_numeric_fields
