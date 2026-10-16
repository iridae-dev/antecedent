"""Declared missing-data handling: refusal by default, known-truth execution, row accounting."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import interference
from antecedent.errors import CausalError, CausalValueError
from antecedent.missing import MissingPolicy, coerce_missing
from antecedent.quasi import PanelDifferenceInDifferences

TRUE_EFFECT = 0.3


def _trial(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    arm = np.arange(n) % 2 == 1
    baseline = np.clip(rng.normal(size=n), -2.0, 2.0)
    outcome = 1.0 + TRUE_EFFECT * arm + 2.0 * baseline + rng.normal(size=n)
    # Complete-case bias is about 2 * (-0.406 + 0.233) = -0.35 (probit approximation).
    eta = 0.8 - (0.8 + 0.8 * arm) * baseline
    responded = rng.random(n) < 1.0 / (1.0 + np.exp(-eta))
    observed = np.where(responded, outcome, np.nan)
    return arm, baseline, outcome, responded, observed


def _query(arm: np.ndarray) -> ant.experiment.RandomizedEffect:
    n = len(arm)
    design = ant.experiment.ExperimentDesign(
        interference.BernoulliAssignment(0.5),
        [bool(a) for a in arm],
        [f"participant-{i}" for i in range(n)],
        [f"row-{i}" for i in range(n)],
    )
    return ant.experiment.RandomizedEffect("outcome", design)


def test_policy_declaration_is_validated():
    assert coerce_missing(None) is None
    assert coerce_missing("complete_case") == MissingPolicy.complete_case()
    assert MissingPolicy.inverse_probability(["a", "b"]).covariates == ("a", "b")
    with pytest.raises(CausalValueError):
        MissingPolicy("listwise")  # type: ignore[arg-type]
    with pytest.raises(CausalValueError, match="observation model"):
        MissingPolicy("complete_case", ("a",))
    with pytest.raises(CausalValueError):
        MissingPolicy.inverse_probability(["a", "a"])


def test_randomized_trial_refuses_missing_outcomes_without_a_policy():
    arm, baseline, _, responded, observed = _trial(400)
    first = int(np.flatnonzero(~responded)[0])
    data = {"outcome": observed, "baseline": baseline}
    with pytest.raises(CausalError) as raised:
        ant.analyze(data, query=_query(arm), refute="none")
    assert getattr(raised.value, "reason_code", None) == "missing_policy_required"
    assert f"column `outcome` at row {first}" in str(raised.value)
    assert "complete_case, inverse_probability" in str(raised.value)


def test_inverse_probability_recovers_the_effect_and_reports_accounting():
    arm, baseline, _, responded, observed = _trial(30_000, seed=3)
    data = {"outcome": observed, "baseline": baseline}
    weighted = ant.analyze(
        data,
        query=_query(arm),
        missing=MissingPolicy.inverse_probability(["baseline"]),
        refute="none",
    )
    fit = weighted.randomized_effect
    se = fit.variance_upper_bound**0.5
    assert abs(fit.effect - TRUE_EFFECT) < 4 * se
    assert fit.interval_95 is None
    report = weighted.missing_data
    assert report is not None and report.declared
    assert report.policy == "inverse_probability"
    assert report.covariates == ("baseline",)
    assert "missing at random given arm" in report.assumption
    accounting = report.accounting
    assert accounting.rows_total == 30_000
    assert accounting.rows_in_denominator == 30_000
    assert accounting.rows_kept == int(responded.sum())
    assert accounting.dropped_by_variable["outcome"] == int((~responded).sum())
    assert accounting.dropped_rows == tuple(int(i) for i in np.flatnonzero(~responded))
    assert sum(group.total for group in accounting.groups) == 30_000
    assert fit.control_units + fit.treatment_units == 30_000

    naive = ant.analyze(data, query=_query(arm), missing="complete_case", refute="none")
    assert abs(naive.randomized_effect.effect - TRUE_EFFECT) > 0.2
    assert naive.missing_data.accounting.rows_in_denominator == int(responded.sum())


def test_dropped_rows_are_named_by_participant_label_with_a_digest():
    arm, baseline, _, responded, observed = _trial(600, seed=5)
    labels = [f"P{i:04d}" for i in range(600)]
    data = {"outcome": observed, "baseline": baseline, "pid": labels}
    result = ant.analyze(
        data, query=_query(arm), missing="complete_case", participant="pid", refute="none"
    )
    accounting = result.missing_data.accounting
    expected = tuple(labels[i] for i in np.flatnonzero(~responded))
    assert accounting.dropped_labels == expected
    assert accounting.dropped_labels_digest and len(accounting.dropped_labels_digest) == 64
    assert len(accounting.dropped_digest) == 64


def test_foreign_policy_is_refused_when_the_study_is_built():
    arm, baseline, _, _, observed = _trial(200)
    with pytest.raises(CausalError) as raised:
        ant.analyze(
            {"outcome": observed, "baseline": baseline},
            query=_query(arm),
            missing="balanced_panel",
            refute="none",
        )
    assert getattr(raised.value, "reason_code", None) == "missing_policy_not_applicable"


def _panel(missing: list[tuple[int, bool]]):
    units = 60
    subjects, treated, post, outcome = [], [], [], []
    for unit in range(units):
        for after in (False, True):
            level = 0.1 * (unit % 7) + 0.3 * after
            value = level + (0.5 if unit % 2 == 0 and after else 0.0)
            outcome.append(np.nan if (unit, after) in missing else value)
            subjects.append(f"u{unit}")
            treated.append(unit % 2 == 0)
            post.append(after)
    return {"y": outcome, "id": subjects, "group": treated, "after": post}


def test_panel_did_balances_by_whole_units_and_never_by_single_cells():
    query = PanelDifferenceInDifferences("y", "id", "group", "after")
    data = _panel([(5, True), (8, False), (31, True)])
    with pytest.raises(CausalError) as raised:
        ant.analyze(data, query=query, refute="none")
    assert getattr(raised.value, "reason_code", None) == "missing_policy_required"
    with pytest.raises(CausalError) as foreign:
        ant.analyze(data, query=query, missing="complete_case", refute="none")
    assert getattr(foreign.value, "reason_code", None) == "missing_policy_not_applicable"

    result = ant.analyze(data, query=query, missing="balanced_panel", refute="none")
    assert result.panel_did.estimate == pytest.approx(0.5)
    accounting = result.missing_data.accounting
    assert (accounting.units_total, accounting.units_dropped) == (60, 3)
    assert accounting.rows_total == 120 and accounting.rows_kept == 114
    assert result.panel_did.interval_95 is None


def test_notears_refuses_then_runs_on_declared_complete_cases():
    rng = np.random.default_rng(1)
    n = 600
    x = rng.normal(size=n)
    y = 0.8 * x + rng.normal(size=n)
    z = 0.8 * y + rng.normal(size=n)
    y_missing = y.copy()
    holes = np.arange(4, n, 9)
    y_missing[holes] = np.nan
    data = {"x": x, "y": y_missing, "z": z}
    with pytest.raises(CausalError) as raised:
        ant.discovery.NOTEARS().run(data)
    assert getattr(raised.value, "reason_code", None) == "missing_policy_required"
    assert f"column `y` at row {holes[0]}" in str(raised.value)

    declared = ant.discovery.NOTEARS(missing="complete_case").run(data)
    keep = np.setdiff1d(np.arange(n), holes)
    reference = ant.discovery.NOTEARS().run({"x": x[keep], "y": y[keep], "z": z[keep]})
    assert [(e.source, e.target) for e in declared.links] == [
        (e.source, e.target) for e in reference.links
    ]
    accounting = declared.row_accounting
    assert (accounting["rows_total"], accounting["rows_kept"]) == (n, len(keep))
    assert accounting["dropped_rows"] == [int(i) for i in holes]
    assert reference.row_accounting["rows_dropped"] == 0
    with pytest.raises(CausalValueError):
        ant.discovery.NOTEARS(missing="pairwise")  # type: ignore[arg-type]


def test_temporal_gaps_refuse_then_drop_only_the_windows_that_read_them():
    n = 300
    t = np.arange(n)
    x = np.sin(0.37 * t) + 0.3 * np.cos(1.9 * t)
    y = np.zeros(n)
    y[1:] = 2.0 * x[:-1] + 0.3 * np.sin(7.0 * t[1:])
    gaps = [100, 101, 202]
    x_obs = x.copy()
    x_obs[gaps] = np.nan
    data = {"x": x_obs, "y": y}
    kwargs = dict(
        graph=[("x", 1, "y", 0)], query=ant.PulseEffect("x", "y", treatment_lag=1, horizon_steps=1)
    )
    with pytest.raises(CausalError, match="IncompleteSeries|incomplete series"):
        ant.analyze(data, bootstrap=0, refute="none", **kwargs)
    with pytest.raises(CausalError) as raised:
        ant.analyze(data, missing="error", bootstrap=0, refute="none", **kwargs)
    assert getattr(raised.value, "reason_code", None) == "missing_policy_required"
    assert "column `x` at row 100" in str(raised.value)
    with pytest.raises(CausalError) as refuted:
        ant.analyze(data, missing="complete_case", bootstrap=0, refute="cheap", **kwargs)
    assert getattr(refuted.value, "reason_code", None) == "missing_policy_not_applicable"
    assert "refuters" in str(refuted.value)
    boot = ant.analyze(data, missing="complete_case", bootstrap=150, refute="none", **kwargs)
    assert boot.estimate.se_bootstrap and boot.estimate.se_bootstrap > 0

    result = ant.analyze(data, missing="complete_case", bootstrap=0, refute="none", **kwargs)
    kept = [i for i in range(1, n) if (i - 1) not in gaps]
    xs = x[[i - 1 for i in kept]]
    ys = y[kept]
    slope = np.cov(xs, ys, bias=True)[0, 1] / np.var(xs)
    assert result.estimate.ate == pytest.approx(slope, abs=1e-9)
    accounting = result.missing_data.accounting
    assert accounting.rows_total == n - 1
    assert accounting.dropped_rows == (101, 102, 203)


def test_average_effect_reports_dropped_rows_and_error_is_a_declared_refusal():
    rng = np.random.default_rng(7)
    n = 800
    z = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    y = t + 0.7 * z + rng.normal(size=n)
    z_obs = z.copy()
    holes = np.arange(3, n, 10)
    z_obs[holes] = np.nan
    data = {"t": t, "y": y, "z": z_obs}
    graph = [("z", "t"), ("z", "y"), ("t", "y")]
    query = ant.AverageEffect("t", "y")
    default = ant.analyze(data, query=query, graph=graph, refute="none", bootstrap=0)
    explicit = ant.analyze(
        data, query=query, graph=graph, refute="none", bootstrap=0, missing="complete_case"
    )
    assert default.estimate.ate == explicit.estimate.ate
    assert default.missing_data.declared is False and explicit.missing_data.declared is True
    accounting = default.missing_data.accounting
    assert accounting.rows_total == n and accounting.rows_dropped == len(holes)
    assert accounting.dropped_by_variable["z"] == len(holes)
    assert accounting.dropped_by_reason == {"missing_value": len(holes)}
    with pytest.raises(CausalError) as raised:
        ant.analyze(data, query=query, graph=graph, refute="none", bootstrap=0, missing="error")
    assert getattr(raised.value, "reason_code", None) == "missing_policy_required"
