"""discovery= on temporal analyze() and enriched result fields."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.outcomes import require_success


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
    result = require_success(
        antecedent.analyze(
            data,
            graph=antecedent.discovery.discover(
                data,
                method=antecedent.discovery.PCMCI(max_lag=1, alpha=0.2, fdr=False),
                execution=antecedent.Execution(seed=1),
                accept_discovered=True,
            ),
            query=antecedent.PulseEffect(
                treatment="x",
                outcome="y",
                treatment_lag=1,
                horizon_steps=1,
                active_level=1.0,
            ),
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
            refute=False,
        )
    )
    # y_t = 0.6 x_{t-1} + 0.2 noise, so a unit pulse of x at lag 1 moves y one step later
    # by 0.6. The OLS SE at n = 120 is about 0.02, so 0.1 is five SEs: the discovered
    # lag-1 graph must give the structural effect, not merely a float.
    assert abs(result.ate - 0.6) < 0.1
    assert result.performance.plan_id
    assert isinstance(result.diagnostics, tuple)
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
    envs = antecedent.data.multi_env(envs)
    # J-PCMCI+ discovery on this 2-env system returns a temporal equivalence class,
    # and the prepared study licenses multi-environment temporal effects only on a
    # TemporalDag, so the support matrix refuses the combination (fail closed with
    # a registered reason code, never a silent number).
    with pytest.raises(
        antecedent.errors.CausalUnsupportedError,
        match="multi-environment TemporalEffect on a TemporalDag",
    ) as caught:
        require_success(
            antecedent.analyze(
                envs,
                graph=antecedent.discovery.discover(
                    envs,
                    method=antecedent.discovery.JPCMCIPlus(max_lag=1, alpha=0.2, fdr=False),
                    execution=antecedent.Execution(seed=1),
                    accept_discovered=True,
                ),
                query=antecedent.PulseEffect(
                    treatment="x",
                    outcome="y",
                    treatment_lag=1,
                    horizon_steps=1,
                    active_level=1.0,
                ),
                inference=antecedent.Frequentist(bootstrap=0),
                execution=antecedent.Execution(seed=1),
                refute=False,
            )
        )
    assert caught.value.reason_code == "data_modality_not_licensed"


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
        antecedent.errors.ReviewRequired,
        match="a single accepted graph requires exactly one|not certified",
    ):
        require_success(
            antecedent.analyze(
                data,
                graph=antecedent.discovery.discover(
                    data,
                    method=antecedent.discovery.RPCMCI(max_lag=1, alpha=0.2, fdr=False),
                    execution=antecedent.Execution(seed=1),
                    regimes=regimes,
                    accept_discovered=True,
                ),
                query=antecedent.PulseEffect(
                    treatment="x",
                    outcome="y",
                    treatment_lag=1,
                    horizon_steps=1,
                    active_level=1.0,
                ),
                inference=antecedent.Frequentist(bootstrap=0),
                execution=antecedent.Execution(seed=1),
                refute=False,
            )
        )


def _v_structure_data(n: int = 1000, seed: int = 7):
    """``t -> y <- w`` with independent causes: PC must orient both edges into ``y``.

    ``t`` and ``w`` are marginally independent and become dependent given their
    collider ``y``, so the skeleton drops ``t - w`` and the unshielded triple is a
    v-structure; nothing is left undirected, and the ATE of ``t`` is its coefficient.
    """
    rng = np.random.default_rng(seed)
    t = rng.normal(size=n)
    w = rng.normal(size=n)
    y = 1.5 * t + 1.0 * w + rng.normal(size=n) * 0.3
    return {"t": t, "y": y, "w": w}


def test_analyze_discovery_pc_orients_v_structure_and_recovers_effect():
    data = _v_structure_data()
    result = require_success(
        antecedent.analyze(
            data,
            graph=antecedent.discovery.discover(
                data,
                method=antecedent.discovery.PC(alpha=0.001, fdr=False, max_cond_size=2),
                execution=antecedent.Execution(seed=1),
                accept_discovered=True,
            ),
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    # `w` is a second cause of `y`, not a confounder, so the adjustment set is empty and
    # the OLS SE of the t coefficient is sqrt(1.0**2 + 0.3**2) / sqrt(n) = 0.033. The
    # estimate must sit within four SEs of the structural coefficient 1.5, so a
    # discovered graph that adjusts wrongly or an estimate that is garbage cannot pass.
    se = result.estimate.se_analytic
    assert 0.02 < se < 0.05
    assert abs(result.ate - 1.5) < 4.0 * se
    assert result.performance.plan_id


def test_analyze_discovery_pc_refuses_unorientable_triangle():
    n = 250
    rng = np.random.default_rng(7)
    z = rng.normal(size=n)
    t = z + rng.normal(size=n) * 0.3
    y = 1.5 * t + z + rng.normal(size=n) * 0.3
    # z, t, y form a fully connected triangle (z->t, z->y, t->y): every pair stays
    # dependent under every conditioning set available here, so PC's skeleton phase
    # can remove no edge, and with no unshielded triple to seed a v-structure it can
    # orient none of the three either. `accept_cpdag_review` (`python/src/lib.rs`)
    # only ever gates on *directed* pending edges — there are none here — so
    # `CpdagReview::into_accepted` (`crates/antecedent/src/accepted.rs`) accepts the
    # fully undirected CPDAG outright at accept_discovered=True (the default):
    # undirected marks are the MEC, not incompleteness, and generalized adjustment
    # (`identify.cpdag.envelope`) is exactly the machinery built to consume them.
    # So this must not refuse: it must return a graph-dependent envelope that
    # mixes the identified and unidentified completions of the triangle's MEC,
    # never a single confident number.
    result = require_success(
        antecedent.analyze(
            {"t": t, "y": y, "z": z},
            graph=antecedent.discovery.discover(
                {"t": t, "y": y, "z": z},
                method=antecedent.discovery.PC(alpha=0.2, fdr=False, max_cond_size=2),
                execution=antecedent.Execution(seed=1),
                accept_discovered=True,
            ),
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.answer.kind == "bounds"
    assert result.answer.value is None
    assert result.answer.bounds is not None
    assert result.structural_identified_mass == pytest.approx(0.5)
    assert result.structural_unidentified_mass == pytest.approx(0.5)
    assert result.identification.status == "GraphDependent"


def test_analyze_ate_enriched_fields():
    n = 200
    rng = np.random.default_rng(1)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + rng.normal(size=n) * 0.3
    result = require_success(
        antecedent.analyze(
            {"t": t, "y": y, "z": z},
            graph=[("z", "t"), ("z", "y"), ("t", "y")],
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.performance.modality
    assert result.performance.plan_id
    assert isinstance(result.diagnostics, tuple)
    assert result.provenance["node_count"] >= 0


def test_discover_requires_explicit_native_review_acceptance():
    data = _v_structure_data()
    method = antecedent.discovery.PC(alpha=0.001, fdr=False, max_cond_size=2)
    with pytest.raises(antecedent.errors.ReviewRequired):
        antecedent.discovery.discover(data, method=method)
    accepted = antecedent.discovery.discover(data, method=method, accept_discovered=True)
    assert accepted.algorithm_id == "pc"
    assert set(accepted.graph.nodes()) == set(data)


def test_discover_rejects_untyped_execution_and_inapplicable_regimes():
    method = antecedent.discovery.PC()
    with pytest.raises(antecedent.errors.CausalTypeError, match="Execution"):
        antecedent.discovery.discover(_v_structure_data(), method=method, execution={"seed": 1})
    with pytest.raises(antecedent.errors.CausalUnsupportedError) as caught:
        antecedent.discovery.discover(_v_structure_data(), method=method, regimes=[0])
    assert caught.value.reason_code == "option_not_applicable"


def test_discover_honors_execution_cancellation_and_observer_exceptions():
    method = antecedent.discovery.PC()
    token = antecedent.CancellationToken()
    token.cancel()
    with pytest.raises(antecedent.errors.CausalCancelledError):
        antecedent.discovery.discover(
            _v_structure_data(), method=method, execution=antecedent.Execution(cancel=token)
        )
    failure = RuntimeError("discovery observer failure")

    def observe(*_):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        antecedent.discovery.discover(
            _v_structure_data(), method=method, execution=antecedent.Execution(on_progress=observe)
        )
    assert caught.value is failure


def test_preparation_requires_explicit_discovery_execution():
    with pytest.raises(antecedent.errors.CausalTypeError, match="run discovery.discover first"):
        antecedent.prepare(
            _v_structure_data(),
            query=antecedent.AverageEffect("t", "y"),
            discovery=antecedent.discovery.PC(),
        )
