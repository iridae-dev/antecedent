"""Retained summary refresh, retained bootstrap replicates, retention policies and spill.

A prepared study's frozen estimator decides which summary its result rests on. The oracle is an
independent weighted least-squares fit (``numpy``) over the rows the refresh should hold after
each change, so an in-place append, a recomputed window/evict/weight/mask/replace and a reload
from disk are each compared with a fresh solve, not with the library's own reconstruction.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalResourceError, CausalStateError, CausalValueError
from antecedent.execution import Execution, ResourceLimits
from antecedent.prepared import prepare
from antecedent.state import (
    RefreshMutation,
    RefreshObservation,
    RetainedBootstrap,
    RetainedRefresh,
    RetentionPolicy,
    SpillRegistry,
)


def _study(estimator: str | None, inference=None, execution: Execution | None = None):
    t = np.tile([0.0, 1.0], 32)
    z = np.repeat(np.linspace(-1.0, 1.0, 32), 2)
    graph = ant.Dag.from_edges(["t", "z", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    return prepare(
        {"t": t, "z": z, "y": 2.0 * t + z},
        graph=graph,
        query=ant.AverageEffect("t", "y"),
        estimator=estimator,
        inference=inference,
        execution=execution,
        refute=False,
    )


def _row(i: int, weight: float = 1.0, masked: bool = False) -> RefreshObservation:
    t = float(i % 7) / 3.0
    y = 1.0 + 2.0 * t + 0.1 * np.sin(1.7 * i)
    return RefreshObservation(i, (1.0, t), float(y), weight, masked)


def _fit(rows: list[RefreshObservation]) -> tuple[np.ndarray, float]:
    """Independent weighted least squares over the live rows: (beta, total weight)."""
    live = [r for r in rows if not r.masked and r.weight > 0]
    design = np.array([r.x for r in live])
    y = np.array([r.y for r in live])
    w = np.array([r.weight for r in live])
    root = np.sqrt(w)
    beta = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
    return beta, float(w.sum())


def _assert_matches(refresh: RetainedRefresh, rows: list[RefreshObservation], source: str) -> None:
    beta, weight = _fit(rows)
    result = refresh.result()
    assert result["source"] == source
    assert result["beta"] == pytest.approx(beta, rel=1e-9, abs=1e-9)
    assert result["n"] == round(weight)
    assert result["live_rows"] == len([r for r in rows if not r.masked and r.weight > 0])
    assert result["rows"] == len(rows)
    refresh.verify()
    assert refresh.fresh_result()["beta"] == pytest.approx(result["beta"], rel=1e-9, abs=1e-9)


def test_least_squares_refresh_follows_every_change_like_a_fresh_fit():
    study = _study("linear.adjustment.ate")
    assert RetainedRefresh.family_of(study) == "ols"
    rows = [_row(i) for i in range(20)]
    refresh = RetainedRefresh.start(study, source="s0", width=2, rows=rows)
    assert refresh.family == "ols"
    _assert_matches(refresh, rows, "s0")

    appended = [_row(i) for i in range(20, 30)]
    assert refresh.apply(RefreshMutation.append("s0", appended)) == "updated"
    rows = rows + appended
    _assert_matches(refresh, rows, "s0")

    assert refresh.apply(RefreshMutation.window(15)) == "recomputed"
    rows = rows[-15:]
    _assert_matches(refresh, rows, "s0")

    assert refresh.apply(RefreshMutation.evict(20)) == "recomputed"
    rows = [r for r in rows if r.id != 20]
    _assert_matches(refresh, rows, "s0")

    assert refresh.apply(RefreshMutation.set_mask(21, True)) == "recomputed"
    rows = [_row(r.id, r.weight, True) if r.id == 21 else r for r in rows]
    _assert_matches(refresh, rows, "s0")

    assert refresh.apply(RefreshMutation.set_weight(22, 3)) == "recomputed"
    rows = [_row(r.id, 3.0, r.masked) if r.id == 22 else r for r in rows]
    _assert_matches(refresh, rows, "s0")

    replacement = [_row(100 + i) for i in range(12)]
    assert refresh.apply(RefreshMutation.replace("s1", replacement)) == "recomputed"
    _assert_matches(refresh, replacement, "s1")


def test_a_refused_change_leaves_the_summary_exactly_as_it_was():
    study = _study("linear.adjustment.ate")
    rows = [_row(i) for i in range(12)]
    refresh = RetainedRefresh.start(study, source="s0", width=2, rows=rows)
    before = dict(refresh.result().values)
    refusals = [
        RefreshMutation.append("other-source", [_row(50)]),
        RefreshMutation.append("s0", [_row(3)]),
        RefreshMutation.evict(999),
        RefreshMutation.set_weight(2, 1.5),
        RefreshMutation.append("s0", [RefreshObservation(70, (1.0, 2.0, 3.0), 1.0)]),
    ]
    for mutation in refusals:
        with pytest.raises(CausalError):
            refresh.apply(mutation)
        assert dict(refresh.result().values) == before
    with pytest.raises(CausalValueError):
        refresh.apply(RefreshMutation._make("rotate"))  # type: ignore[attr-defined]
    with pytest.raises(CausalError):
        RetainedRefresh.start(
            study, source="s0", width=2, rows=[_row(1), RefreshObservation(1, (1.0, 0.0), 0.0)]
        )


def test_weighted_least_squares_refresh_carries_real_weights():
    study = _study("aipw", inference=ant.Frequentist(bootstrap=0))
    assert RetainedRefresh.family_of(study) == "weighted_ols"
    rows = [_row(i, weight=0.5 + (i % 4) * 0.25) for i in range(24)]
    refresh = RetainedRefresh.start(study, source="w", width=2, rows=rows)
    beta, total = _fit(rows)
    result = refresh.result()
    assert result["family"] == "weighted_ols"
    assert result["beta"] == pytest.approx(beta, rel=1e-9, abs=1e-9)
    assert result["total_weight"] == pytest.approx(total)
    weights = np.array([r.weight for r in rows])
    assert result["effective_sample_size"] == pytest.approx(weights.sum() ** 2 / (weights**2).sum())
    assert refresh.apply(RefreshMutation.set_weight(3, 2.75)) == "recomputed"
    rows = [_row(r.id, 2.75) if r.id == 3 else r for r in rows]
    assert refresh.result()["beta"] == pytest.approx(_fit(rows)[0], rel=1e-9, abs=1e-9)
    refresh.verify()


def test_conjugate_refresh_updates_the_normal_inverse_gamma_posterior_block():
    study = _study("bayesian.gcomp", inference=ant.Bayesian(n_draws=64, backend="conjugate"))
    assert RetainedRefresh.family_of(study) == "conjugate_blocks"
    m0 = np.array([0.0, 0.0])
    lambda0 = np.eye(2)
    prior = {
        "mean": m0.tolist(),
        "precision": lambda0.ravel().tolist(),
        "shape": 2.0,
        "rate": 1.5,
        "label": "outcome",
    }
    rows = [_row(i) for i in range(16)]
    refresh = RetainedRefresh.start(study, source="s0", width=2, rows=rows, prior=prior)

    def posterior(held: list[RefreshObservation]) -> tuple[np.ndarray, float, float]:
        x = np.array([r.x for r in held])
        y = np.array([r.y for r in held])
        precision = lambda0 + x.T @ x
        eta = lambda0 @ m0 + x.T @ y
        mean = np.linalg.solve(precision, eta)
        shape = 2.0 + len(held) / 2.0
        rate = 1.5 + 0.5 * (y @ y + m0 @ lambda0 @ m0 - eta @ mean)
        return mean, shape, rate

    def check(held: list[RefreshObservation]) -> None:
        mean, shape, rate = posterior(held)
        result = refresh.result()
        assert result["mean"] == pytest.approx(mean, rel=1e-9, abs=1e-9)
        assert result["shape"] == pytest.approx(shape)
        assert result["rate"] == pytest.approx(rate, rel=1e-9)
        refresh.verify()

    check(rows)
    more = [_row(i) for i in range(16, 24)]
    mutation = RefreshMutation.in_block("outcome", RefreshMutation.append("s0", more))
    # rows that join an existing block rebuild that block from the retained rows; only a new block
    # extends the summary in place
    assert refresh.apply(mutation) == "recomputed"
    check(rows + more)
    with pytest.raises(CausalError):
        refresh.apply(RefreshMutation.in_block("missing-block", RefreshMutation.evict(0)))
    check(rows + more)


def test_retention_policies_name_what_they_no_longer_serve():
    study = _study("linear.adjustment.ate")
    rows = [_row(i) for i in range(20)]
    refresh = RetainedRefresh.start(study, source="s0", width=2, rows=rows)
    raw = RetentionPolicy.raw_history()
    assert (raw.kind, raw.max_retained_rows, raw.result_mode) == (
        "raw_history",
        None,
        "full reusable state",
    )
    assert raw.no_longer_serves == ()
    assert refresh.with_retention(raw).result().values == refresh.result().values

    window = RetentionPolicy.bounded_window(8)
    assert window.max_retained_rows == 8
    assert window.result_mode == "full reusable state"
    windowed = refresh.with_retention(window)
    assert windowed.result()["beta"] == pytest.approx(_fit(rows[-8:])[0], rel=1e-9, abs=1e-9)
    assert windowed.result()["rows"] == 8

    for policy in (RetentionPolicy.sufficient_statistics(), RetentionPolicy.bounded_window(0)):
        assert policy.result_mode == "summary-only"
        assert "refit from retained rows" in policy.no_longer_serves
        with pytest.raises(CausalStateError, match="summary-only cannot serve refit"):
            policy.require("refit_from_retained_rows")
        policy.require("reuse_summary")
        kept = refresh.with_retention(policy)
        assert kept.result()["beta"] == pytest.approx(refresh.result()["beta"])
        with pytest.raises(CausalStateError, match="summary-only"):
            kept.apply(RefreshMutation.evict(3))
        with pytest.raises(CausalStateError, match="summary-only"):
            kept.verify()
        with pytest.raises(CausalStateError, match="summary-only"):
            kept.fresh_result()
        assert kept.result().values == refresh.summary_only().result().values
    with pytest.raises(CausalValueError):
        RetentionPolicy.bounded_window(-1)
    with pytest.raises(CausalValueError):
        refresh.require("teleport")


def test_a_refresh_spills_under_the_admission_budget_and_reloads_identically(tmp_path):
    study = _study("linear.adjustment.ate")
    rows = [_row(i) for i in range(30)]
    refresh = RetainedRefresh.start(study, source="s0", width=2, rows=rows)
    frame = refresh.frame_bytes
    assert frame > 0
    registry = SpillRegistry(tmp_path / "spill", 1 << 20)
    spilled = refresh.spill(registry, "study")
    assert spilled.mode == "disk-backed output"
    assert spilled.on_disk and spilled.evicted == ()
    assert registry.contains("study")
    assert registry.stats["entries"] == 1 and registry.stats["spills"] == 1
    # the frame size is what admission is asked for: an upper bound of the stored frame
    assert 0 < registry.stats["used_bytes"] <= frame
    spilled.require("read_disk_artifact")
    with pytest.raises(CausalStateError, match="disk-backed"):
        spilled.require("refit_from_retained_rows")
    back = spilled.reload()
    assert back.result().values == refresh.result().values
    assert back.family == "ols"
    assert registry.stats["reloads"] == 1
    # The reloaded summary keeps its rows: a change applies and verifies against a fresh fit.
    held = dict(refresh.result().values)
    assert back.apply(RefreshMutation.evict(0)) == "recomputed"
    assert back.result()["beta"] == pytest.approx(_fit(rows[1:])[0], rel=1e-9, abs=1e-9)
    # The reload is a branch: the in-memory summary and the spilled artifact are unchanged.
    assert dict(refresh.result().values) == held
    assert spilled.reload().result().values == held

    # A registry that holds one frame evicts the older result; its reload is refused.
    tight = SpillRegistry(tmp_path / "tight", frame + frame // 2)
    first = refresh.spill(tight, "a")
    second = refresh.spill(tight, "b")
    assert second.evicted == ("a",)
    assert not first.on_disk and second.on_disk
    assert tight.keys_by_recency() == ("b",)
    with pytest.raises(CausalError):
        first.reload()
    assert second.reload().result().values == refresh.result().values
    # A frame larger than the whole registry budget is refused without evicting anything.
    with pytest.raises(CausalResourceError):
        refresh.spill(SpillRegistry(tmp_path / "small", frame // 2), "x")
    # A summary without its rows cannot be written as the rows' spill.
    with pytest.raises(CausalStateError, match="summary-only"):
        refresh.summary_only().spill(registry, "no-rows")


def test_the_execution_memory_limit_bounds_the_registry_and_the_spill(tmp_path):
    study = _study("linear.adjustment.ate")
    refresh = RetainedRefresh.start(study, source="s0", width=2, rows=[_row(i) for i in range(30)])
    frame = refresh.frame_bytes
    roomy = Execution(limits=ResourceLimits(memory_bytes=4 * frame))
    registry = SpillRegistry(tmp_path / "roomy", 1 << 40, execution=roomy)
    assert registry.stats["budget_bytes"] == 4 * frame
    spilled = refresh.spill(registry, "k", execution=roomy)
    assert spilled.reload(execution=roomy).result().values == refresh.result().values
    # No byte of admission budget: the registry itself is refused.
    with pytest.raises(CausalResourceError, match="admission budget"):
        SpillRegistry(tmp_path / "zero", 0)
    # A pool smaller than the frame refuses the write with the raise-the-limit next attempt.
    starved = Execution(limits=ResourceLimits(memory_bytes=frame // 2))
    tiny = SpillRegistry(tmp_path / "starved", 1 << 40, execution=starved)
    with pytest.raises(CausalResourceError, match="admission budget"):
        refresh.spill(tiny, "k", execution=starved)
    assert not tiny.contains("k")
    # The study's own execution bounds the spill when none is given.
    # (a 1 MiB pool still prepares the study, whose checked quote is under it, and is short of the
    # frame of a longer retained history)
    short = Execution(limits=ResourceLimits(memory_bytes=1 << 20))
    bound = _study("linear.adjustment.ate", execution=short)
    long_rows = [_row(i) for i in range(20_000)]
    held = RetainedRefresh.start(bound, source="s0", width=2, rows=long_rows)
    assert held.frame_bytes > 1 << 20
    with pytest.raises(CausalResourceError, match="admission budget"):
        held.spill(registry, "k2")


def _replicates(n: int = 200) -> np.ndarray:
    return 1.0 + np.sin(np.arange(n) * 0.37)


def test_bootstrap_retention_keeps_the_summary_and_drops_replicates_on_request():
    draws = _replicates()
    boot = RetainedBootstrap(draws.tolist(), failed=2)
    assert boot.retention == "full reusable state"
    assert boot.mean == pytest.approx(draws.mean())
    assert boot.accounting["se"] == pytest.approx(draws.std(ddof=1))
    assert (boot.accounting["replicates_ok"], boot.accounting["replicates_failed"]) == (200, 2)
    low, high = boot.percentile_interval(0.95)
    assert (low, high) == pytest.approx(tuple(np.quantile(draws, [0.025, 0.975])))
    boot.verify()

    more = _replicates(250)[200:]
    boot.append(more.tolist(), failed=1)
    everything = np.concatenate([draws, more])
    assert boot.mean == pytest.approx(everything.mean())
    assert boot.accounting["se"] == pytest.approx(everything.std(ddof=1))
    assert boot.accounting["replicates_failed"] == 3
    assert len(boot.estimates) == 250
    boot.verify()

    summary = boot.summary_only()
    assert summary.retention == "summary-only"
    assert summary.estimates == ()
    assert summary.mean == pytest.approx(boot.mean)
    assert summary.accounting["se"] == pytest.approx(boot.accounting["se"])
    summary.append(more.tolist(), failed=0)
    assert summary.accounting["replicates_ok"] == 300
    with pytest.raises(CausalError, match="summary-only"):
        summary.percentile_interval(0.95)
    with pytest.raises(CausalError, match="summary-only"):
        summary.verify()
    with pytest.raises(CausalError):
        boot.percentile_interval(1.5)
    with pytest.raises(CausalError):
        RetainedBootstrap([1.0, float("nan")])
    with pytest.raises(CausalError):
        boot.append([float("inf")])
    assert len(boot.estimates) == 250


def test_a_bootstrap_spills_to_disk_and_reloads_its_replicates(tmp_path):
    draws = _replicates()
    boot = RetainedBootstrap(draws.tolist(), failed=1)
    registry = SpillRegistry(tmp_path / "boot", 1 << 20)
    spilled = boot.spill(registry, "boot")
    assert spilled.mode == "disk-backed output" and spilled.on_disk
    spilled.require("read_disk_artifact")
    with pytest.raises(CausalStateError, match="disk-backed"):
        spilled.require("row_reconstruction")
    back = spilled.reload()
    assert back.estimates == pytest.approx(tuple(draws))
    assert back.accounting["replicates_failed"] == 1
    assert back.accounting["se"] == pytest.approx(boot.accounting["se"])
    with pytest.raises(CausalError):
        boot.summary_only().spill(registry, "no-rows")
    stored = registry.stats["used_bytes"]
    assert stored > 8 * len(draws)
    tight = SpillRegistry(tmp_path / "tight", stored + 8)
    older = boot.spill(tight, "a")
    newer = boot.spill(tight, "b")
    assert newer.evicted == ("a",) and not older.on_disk
    with pytest.raises(CausalError):
        older.reload()
    with pytest.raises(CausalResourceError):
        boot.spill(SpillRegistry(tmp_path / "small", 16), "x")


def test_scatter_refresh_keeps_the_covariance_of_the_instrumented_observations():
    i = np.arange(80, dtype=float)
    z = (np.arange(80) % 2).astype(float)
    t = (0.2 + 0.6 * z + 0.15 * np.sin(i * 1.3) > 0.5).astype(float)
    study = prepare(
        {"z": z, "t": t, "y": 1.5 * t + 0.2 * np.cos(i * 0.9)},
        graph=[("z", "t"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        identifier="iv",
        estimator="iv.wald",
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
    )
    assert RetainedRefresh.family_of(study) == "covariance"

    def observation(k: int, masked: bool = False) -> RefreshObservation:
        return RefreshObservation(
            k, (float(k % 2), float((k + k // 3) % 2), 1.5 * (k % 5) / 4.0), masked=masked
        )

    rows = [observation(k) for k in range(30)]
    refresh = RetainedRefresh.start(study, source="iv", width=3, rows=rows)

    def check(held: list[RefreshObservation]) -> None:
        live = np.array([r.x for r in held if not r.masked])
        result = refresh.result()
        assert result["n"] == len(live)
        assert result["mean"] == pytest.approx(live.mean(axis=0))
        covariance = np.array(result["sample_covariance"]).reshape(3, 3)
        assert covariance == pytest.approx(np.cov(live.T), rel=1e-9, abs=1e-12)
        refresh.verify()

    check(rows)
    more = [observation(k) for k in range(30, 40)]
    assert refresh.apply(RefreshMutation.append("iv", more)) == "updated"
    rows += more
    check(rows)
    assert refresh.apply(RefreshMutation.evict(4)) == "recomputed"
    rows = [r for r in rows if r.id != 4]
    check(rows)
    assert refresh.apply(RefreshMutation.set_mask(7, True)) == "recomputed"
    rows = [observation(r.id, True) if r.id == 7 else r for r in rows]
    check(rows)
    before = dict(refresh.result().values)
    with pytest.raises(CausalError):
        refresh.apply(RefreshMutation.set_weight(8, 2))
    assert dict(refresh.result().values) == before
