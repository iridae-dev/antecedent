"""One omitted-default table, owned by the Rust study builder.

`analyze` / `prepare` / `PreparedAnalysis.prepare` never fill an omitted
`refute` / `bootstrap` / `latency`: the omission reaches the builder, which
applies its own table, the latency-tier mapping and the refute downgrade. These
tests read the executed plan back, so a change to the builder's table changes
what they observe — they do not compare a Python literal with itself.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent._defaults import OMITTED, is_temporal_query, resolve_threads
from antecedent._native import default_user_threads, omitted_defaults
from antecedent.estimation import PreparedAnalysis
from antecedent.inference import Bayesian

GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]


def _data(n: int = 80, seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 0.5).astype(float)
    return {"t": t, "y": t + z, "z": z}


def _analyze(**kwargs):
    kwargs.setdefault("graph", GRAPH)
    kwargs.setdefault("query", ant.AverageEffect("t", "y"))
    return ant.analyze(_data(), **kwargs)


def test_omitted_threads_use_the_machine():
    assert default_user_threads() >= 1
    assert resolve_threads(None) == default_user_threads()
    assert resolve_threads(1) == 1


def test_table_is_the_builders_own():
    """The module constant is whatever the extension reports, with no second copy."""
    assert omitted_defaults() == OMITTED
    assert set(OMITTED) == {
        "bootstrap",
        "refute",
        "latency",
        "n_draws",
        "n_draws_hmc",
        "discovery_alpha",
        "discovery_max_cond_size",
        "prior_scale",
        "overlap_clip",
        "overlap_trim",
        "transport_bootstrap",
        "transport_coverage_level",
    }
    assert OMITTED["latency"] is None


def test_omitted_budget_is_the_table_value():
    result = _analyze(refute="none")
    assert result.performance.bootstrap_requested == OMITTED["bootstrap"]


def test_omitted_refute_runs_the_table_suite():
    result = _analyze(bootstrap=0)
    refuters = {report.refuter for report in result.validation.reports}
    assert refuters, "the omitted refute suite ran no refuters"
    assert any("placebo" in name for name in refuters) == (OMITTED["refute"] == "placebo")


def test_a_tier_maps_the_omitted_budget_but_never_an_explicit_one():
    interactive = _analyze(latency="interactive")
    report = _analyze(latency="report")
    explicit = _analyze(latency="interactive", bootstrap=7, refute="none")
    assert interactive.performance.bootstrap_replicates_requested == 0
    assert report.performance.bootstrap_replicates_requested == 200
    assert explicit.performance.bootstrap_replicates_requested == 7


def test_static_responses_do_not_report_a_replicate_budget():
    """A static response has analytic uncertainty, so the omitted budget is zero."""
    rng = np.random.default_rng(4)
    z = rng.normal(size=200)
    t = z + rng.normal(scale=0.4, size=200)
    result = ant.analyze(
        {"t": t, "y": t + z + rng.normal(scale=0.2, size=200), "z": z},
        graph=GRAPH,
        query=ant.ResponseCurve("t", "y", grid=[-0.5, 0.0, 0.5]),
    )
    assert result.uncertainty.replicates in (None, 0)


def test_temporal_kind_helper():
    assert is_temporal_query(kind="pulse")
    assert is_temporal_query(kind="sustained")
    assert is_temporal_query(kind="temporal_mediation")
    assert not is_temporal_query(kind="average")
    assert is_temporal_query(ant.PulseEffect("t", "y"))


def test_bayesian_omitted_n_draws_is_not_explicit():
    assert Bayesian().n_draws is None
    assert Bayesian().n_draws_explicit is False
    assert Bayesian(n_draws=32).n_draws_explicit is True


def test_entry_points_agree():
    data = _data()
    query = ant.AverageEffect("t", "y")
    analyzed = ant.analyze(data, graph=GRAPH, query=query, refute="none")
    prepared = PreparedAnalysis.prepare(data, graph=GRAPH, query=query, refute="none").estimate()
    workflow = ant.prepare(data, graph=GRAPH, query=query, refute="none").estimate()
    assert analyzed.performance.bootstrap_requested == prepared.performance.bootstrap_requested
    assert prepared.performance.bootstrap_requested == workflow.performance.bootstrap_requested
    assert analyzed.performance.latency_mode == prepared.performance.latency_mode
    assert analyzed.claim_id == prepared.claim_id == workflow.claim_id


def test_explicit_budget_reaches_plan():
    result = _analyze(
        inference=ant.Bayesian(n_draws=1000), latency="interactive", refute="none", bootstrap=0
    )
    assert result.performance.n_draws == 1000
    joined = " ".join(str(d) for d in result.diagnostics)
    assert "latency.explicit_budget_kept" in joined


def test_omitted_bayesian_draws_follow_the_backend_default():
    result = _analyze(inference=ant.Bayesian(), refute="none", bootstrap=0)
    assert result.performance.n_draws == OMITTED["n_draws"]


@pytest.mark.parametrize("latency", ["interactive", "standard", "report"])
def test_a_tier_cuts_the_omitted_draw_budget(latency):
    result = _analyze(inference=ant.Bayesian(), latency=latency, refute="none", bootstrap=0)
    assert result.performance.n_draws is not None
    if latency == "interactive":
        assert result.performance.n_draws < OMITTED["n_draws"]


def _native_defaults(parameter: str):
    """`(function, default)` for every native routine that exposes `parameter`."""
    import inspect

    from antecedent import _native

    for name in sorted(dir(_native)):
        obj = getattr(_native, name)
        if not inspect.isroutine(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        if parameter in sig.parameters:
            yield name, sig.parameters[parameter].default


def test_native_signature_literals_equal_the_rust_constants():
    """The `#[pyo3(signature)]` literals repeat facade constants; they must not drift."""
    stale: list[str] = []
    for name, default in _native_defaults("bootstrap"):
        # 0 (no resampling) and None (route decides) are deliberate; a budget is the table's.
        if isinstance(default, int) and default not in (0,) and default != OMITTED["bootstrap"]:
            stale.append(f"{name}: bootstrap={default!r} != {OMITTED['bootstrap']}")
    for name, default in _native_defaults("prior_scale"):
        if default != OMITTED["prior_scale"]:
            stale.append(f"{name}: prior_scale={default!r} != {OMITTED['prior_scale']}")
    for name, default in _native_defaults("alpha"):
        if name.startswith("discover_") and default != OMITTED["discovery_alpha"]:
            stale.append(f"{name}: alpha={default!r} != {OMITTED['discovery_alpha']}")
    for name, default in _native_defaults("max_cond_size"):
        # LiNGAM / NOTEARS prune with their own conditioning bound.
        if name.startswith("discover_") and default not in (
            OMITTED["discovery_max_cond_size"],
            8,
        ):
            stale.append(f"{name}: max_cond_size={default!r}")
    assert not stale, "\n".join(stale)
