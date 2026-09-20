"""Latency tiers + cancel mid-bootstrap (Python dual of Rust backlog A)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _confounded_scm(n: int = 600, seed: int = 7):
    rng = random.Random(seed)
    z = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-(-0.4 + 0.9 * zi)))
        ti = 1.0 if rng.random() < p else 0.0
        yi = 2.0 * ti + zi + rng.gauss(0.0, 0.4)
        z[i] = zi
        t[i] = ti
        y[i] = yi
    return {"t": t, "y": y, "z": z}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_interactive_vs_standard_effort():
    data, edges = _confounded_scm()
    interactive = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        latency="interactive",
        seed=1,
    )
    assert math.isfinite(interactive.ate)
    assert abs(interactive.ate - 2.0) < 0.5
    assert interactive.performance.latency_mode == "interactive"
    assert interactive.performance.bootstrap_replicates_requested == 0
    assert interactive.identification.status

    standard = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        latency="standard",
        refute=False,
        seed=1,
    )
    assert math.isfinite(standard.ate)
    assert standard.performance.latency_mode == "standard"
    assert standard.performance.bootstrap_replicates_requested == 199
    assert standard.performance.bootstrap_replicates_ok == 199
    assert not standard.performance.early_stopped
    assert standard.estimate.se_bootstrap is not None


def test_cancel_mid_bootstrap_partial():
    data, edges = _confounded_scm(n=400, seed=11)
    requested = 80
    full = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        bootstrap=requested,
        refute=False,
        seed=3,
    )
    full_ok = full.performance.bootstrap_replicates_ok or 0
    # Production contexts evaluate the full request; there is no early-stop budget.
    assert full_ok == requested
    assert not full.performance.early_stopped

    token = antecedent.state.CancellationToken()
    seen = {"bootstrap": False}

    def on_progress(fraction: float, stage: str) -> None:
        if stage == "bootstrap" and not seen["bootstrap"]:
            seen["bootstrap"] = True
            token.cancel()

    partial = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        bootstrap=requested,
        refute=False,
        seed=3,
        threads=1,
        cancel=token,
        on_progress=on_progress,
    )
    assert math.isfinite(partial.ate)
    assert partial.performance.cancelled
    ok = partial.performance.bootstrap_replicates_ok or 0
    assert ok < requested
    assert ok != full_ok


def test_progressive_stages_callback_order():
    data, edges = _confounded_scm(400, 11)
    stages: list[str] = []
    point_ate: list[float] = []

    def on_stage(stage: str, payload: dict) -> None:
        stages.append(stage)
        if stage == "estimate_point":
            assert payload.get("se_bootstrap") is None
            point_ate.append(float(payload["ate"]))
        if stage == "uncertainty":
            assert payload.get("se_bootstrap") is not None

    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        bootstrap=40,
        refute=False,
        seed=11,
        on_stage=on_stage,
    )
    assert stages == ["identify", "estimate_point", "uncertainty", "validate"]
    assert point_ate and math.isfinite(point_ate[0])
    assert abs(point_ate[0] - result.ate) < 1e-12
    assert result.estimate.se_bootstrap is not None


def _star_cpdag_study(n_leaves: int = 17, n: int = 200):
    """Undirected star centred on ``t``: one completion per root, more than the
    Interactive tier's 16-atom envelope budget."""
    rows = np.arange(n, dtype=np.float64)
    leaves = {f"l{k}": np.sin(rows / (3.0 + 1.7 * k)) for k in range(n_leaves)}
    noise = ((np.arange(n) * 7919) % 101) / 101.0 - 0.5
    t = 0.3 * leaves["l0"] + 0.2 * leaves["l1"] + 0.5 * noise
    y = 1.0 + 2.0 * t + 0.3 * np.roll(noise, 5)
    data = {"t": t, "y": y, **leaves}
    names = list(data)
    graph = antecedent.Cpdag.from_directed_undirected(
        names, [], [("t", name) for name in names if name != "t"]
    )
    return data, graph


def test_interactive_class_envelope_reports_subsampled_out_mass_separately():
    data, graph = _star_cpdag_study()

    def run(latency: str):
        return antecedent.analyze(
            data,
            graph=graph,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=64, backend="conjugate"),
            latency=latency,
            refute=False,
            bootstrap=0,
            seed=5,
        )

    standard = run("standard")
    assert standard.posterior is not None
    unidentified = standard.posterior.unidentified_mass
    assert unidentified is not None
    assert standard.posterior.subsampled_out_mass == 0.0

    interactive = run("interactive")
    posterior = interactive.posterior
    assert posterior is not None
    # 19 equally weighted completions; the 16 mixed ones plus the skipped and
    # the unidentified ones account for all of the class.
    cases = len(data)
    assert posterior.unidentified_mass == pytest.approx(unidentified, abs=1e-12)
    assert posterior.subsampled_out_mass > 0.0
    mixed = 16 / cases
    assert mixed + posterior.unidentified_mass + posterior.subsampled_out_mass == pytest.approx(
        1.0, abs=1e-12
    )
    assert posterior.envelope is not None
    assert posterior.envelope.subsampled_out_mass == posterior.subsampled_out_mass
    assert "subsampled_out_mass=" in repr(posterior)
    assert interactive.identification.status == "GraphDependent"
    assert any(
        "reported as subsampled_out_mass" in diagnostic for diagnostic in interactive.diagnostics
    )
