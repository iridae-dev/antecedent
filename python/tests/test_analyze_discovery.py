"""discovery= on temporal analyze() and enriched result fields."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _two_regime_lag1_series(n: int = 120, seed: int = 3):
    """Genuinely two-regime series: the lag-1 coefficient flips sign at the midpoint.

    A single-regime series with an artificial half-split no longer exercises the
    refuse-to-collapse property below: alternating refinement now fits each regime's
    equation on its own rows, correctly detects that both halves share one model, and
    merges them — leaving a single graph and nothing to refuse.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = rng.normal()
    mid = n // 2
    for t in range(1, n):
        coef = 0.8 if t < mid else -0.8
        y[t] = coef * x[t - 1] + 0.05 * rng.normal()
    return {"x": x, "y": y}


def _lag1_series(n: int = 120, seed: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = 0.6 * x[t - 1] + 0.2 * rng.normal()
    return {"x": x, "y": y}


def test_analyze_discovery_pcmci_smoke():
    data = _lag1_series()
    result = antecedent.analyze(
        data,
        discovery=antecedent.discovery.PCMCI(max_lag=1, alpha=0.2, fdr=False),
        query=antecedent.PulseEffect(
            treatment="x",
            outcome="y",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        bootstrap=0,
        seed=1,
        refute=False,
    )
    assert isinstance(result.ate, float)
    assert result.performance.plan_id
    assert isinstance(result.diagnostics, list)
    assert "node_count" in result.provenance


def test_analyze_discovery_jpcmci_plus_two_env():
    n = 80
    rng = np.random.default_rng(4)
    envs = []
    for _ in range(2):
        x = rng.normal(size=n)
        y = np.empty(n)
        y[0] = rng.normal()
        for t in range(1, n):
            y[t] = 0.55 * x[t - 1] + 0.2 * rng.normal()
        envs.append({"x": x, "y": y})
    # J-PCMCI+ discovery is CPDAG-shaped (`accept_temporal_cpdag_review` in
    # `python/src/lib.rs`): accept_discovered=True (the default here) only
    # auto-accepts already-directed pending edges, so a leftover undirected/circle
    # mark on this 2-env system can still block with ReviewRequired; a downstream
    # CausalIdentifyError is the other legitimate fail-closed outcome. Only these
    # two exceptions are acceptable — anything else is a real wiring break.
    try:
        result = antecedent.analyze(
            envs,
            discovery=antecedent.discovery.JPCMCIPlus(max_lag=1, alpha=0.2, fdr=False),
            query=antecedent.PulseEffect(
                treatment="x",
                outcome="y",
                treatment_lag=1,
                horizon_steps=1,
                active_level=1.0,
            ),
            bootstrap=0,
            seed=1,
            refute=False,
        )
        assert np.isfinite(result.ate)
    except (
        antecedent.errors.CausalReviewError,
        antecedent.errors.CausalIdentifyError,
        antecedent.errors.CausalCompileError,
    ) as exc:
        assert str(exc)
        if isinstance(exc, antecedent.errors.CausalReviewError):
            assert exc.kind
            assert exc.hint


def test_analyze_discovery_rpcmci_regimes():
    """Two explicit regime labels must fail closed, never silently produce a number.

    Two regimes yield two per-regime CPDAGs, and `accept_rpcmci_review`
    (`python/src/lib.rs`) refuses to collapse them into a single accepted graph.
    Whether the refusal surfaces as `ReviewRequired` or as `CausalIdentifyError`
    depends on which gate is reached first — temporal unfolding currently hits its
    history cap before the review gate — so both are accepted here. What is pinned
    is that one of them fires, with its documented reason.
    """
    data = _two_regime_lag1_series(n=200, seed=5)
    n = len(data["x"])
    regimes = [0] * (n // 2) + [1] * (n - n // 2)
    with pytest.raises(
        (antecedent.errors.ReviewRequired, antecedent.errors.CausalIdentifyError),
        match="a single accepted graph requires exactly one|not certified",
    ):
        antecedent.analyze(
            data,
            discovery=antecedent.discovery.RPCMCI(max_lag=1, alpha=0.2, fdr=False),
            regimes=regimes,
            query=antecedent.PulseEffect(
                treatment="x",
                outcome="y",
                treatment_lag=1,
                horizon_steps=1,
                active_level=1.0,
            ),
            bootstrap=0,
            seed=1,
            refute=False,
        )


def test_analyze_discovery_pc_smoke():
    n = 250
    rng = np.random.default_rng(7)
    z = rng.normal(size=n)
    t = z + rng.normal(size=n) * 0.3
    y = 1.5 * t + z + rng.normal(size=n) * 0.3
    # z, t, y form a fully connected triangle (z->t, z->y, t->y): every pair stays
    # dependent under every conditioning set available here, so PC's skeleton phase
    # can remove no edge, and with no unshielded triple to seed a v-structure it can
    # orient none of the three either. The CPDAG-shaped review
    # (`accept_cpdag_review` in `python/src/lib.rs`) only auto-accepts already
    # directed pending edges, so a fully undirected triangle still blocks with
    # ReviewRequired even at accept_discovered=True (the default). A clean
    # Ready-estimate is the other legitimate outcome if discovery manages to orient
    # the triangle after all.
    try:
        result = antecedent.analyze(
            {"t": t, "y": y, "z": z},
            discovery=antecedent.discovery.PC(alpha=0.2, fdr=False, max_cond_size=2),
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            refute=False,
            bootstrap=0,
            seed=1,
        )
        assert np.isfinite(result.ate)
        assert result.performance.plan_id
    except antecedent.errors.CausalReviewError as exc:
        assert exc.kind
        assert exc.hint
        assert exc.pending_edge_count > 0


def test_analyze_ate_enriched_fields():
    n = 200
    rng = np.random.default_rng(1)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + rng.normal(size=n) * 0.3
    result = antecedent.analyze(
        {"t": t, "y": y, "z": z},
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert result.performance.modality
    assert result.performance.plan_id
    assert isinstance(result.diagnostics, list)
    assert result.provenance["node_count"] >= 0
