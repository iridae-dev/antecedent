"""Intrinsic causal influence on a linear-Gaussian chain with closed-form contributions.

``x -> y -> z`` with ``y = 2x + e_y`` (unit noise) and ``z = 1.5y + e_z`` (noise scale 0.5): the
total coefficients of the noise terms on ``z`` are 3, 1.5 and 1, so the contributions are
``9``, ``2.25`` and ``0.25`` and sum to ``Var(z)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import Execution
from antecedent.gcm import IntrinsicInfluence, intrinsic_influence

EDGES = [("x", "y"), ("y", "z")]
TRUTH = {"x": 9.0, "y": 2.25, "z": 0.25}


def _data(rows: int = 2000, seed: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=rows)
    y = 2.0 * x + rng.normal(size=rows)
    z = 1.5 * y + 0.5 * rng.normal(size=rows)
    return {"x": x, "y": y, "z": z}


def _by_source(result: IntrinsicInfluence):
    return {s.source: s for s in result.sources}


def test_contributions_match_the_closed_form_within_the_reported_uncertainty():
    result = intrinsic_influence(
        _data(),
        EDGES,
        "z",
        independent_noise=True,
        outer_draws=400,
        inner_draws=20,
        fit_bootstrap=20,
        execution=Execution(seed=3),
    )
    assert isinstance(result, IntrinsicInfluence)
    assert result.method["shapley"] == {"kind": "exact"}
    for name, source in _by_source(result).items():
        parts = {u.component: u for u in source.uncertainty}
        assert set(parts) == {"permutation_sampling", "simulation", "mechanism_fit"}
        assert parts["permutation_sampling"].status == "not_applicable"
        assert parts["simulation"].status == "quantified"
        assert parts["mechanism_fit"].status == "quantified"
        # Components add in quadrature here only because all that arise are quantified.
        stderr = np.hypot(parts["simulation"].stderr, parts["mechanism_fit"].stderr)
        assert abs(source.contribution - TRUTH[name]) <= 6 * max(stderr, 0.05 * TRUTH[name])
    assert result.budget["refits"] == 20


def test_efficiency_gap_is_reported_and_small():
    result = intrinsic_influence(
        _data(), EDGES, "z", independent_noise=True, execution=Execution(seed=1)
    )
    total = sum(s.contribution for s in result.sources)
    assert result.allocation_residual == pytest.approx(result.total_variance - total)
    assert abs(result.allocation_residual) <= 6 * max(result.allocation_residual_stderr, 0.3)
    assert result.total_variance == pytest.approx(sum(TRUTH.values()), rel=0.1)


def test_unmeasured_mechanism_fit_is_marked_unquantified_not_zero():
    result = intrinsic_influence(_data(), EDGES, "z", independent_noise=True)
    for source in result.sources:
        fit = source.component("mechanism_fit")
        assert fit.status == "unquantified" and fit.stderr is None
        assert fit.reason == "bootstrap_not_requested"
    sampled = intrinsic_influence(
        _data(),
        EDGES,
        "z",
        independent_noise=True,
        allocation="permutation",
        n_permutations=30,
        fit_bootstrap=10,
        outer_draws=100,
        inner_draws=10,
    )
    for source in sampled.sources:
        assert source.component("permutation_sampling").status == "quantified"
        assert source.component("mechanism_fit").status == "quantified"
        whole = source.whole_procedure
        assert whole is not None and whole.status == "quantified"
        # The whole procedure keeps the noise the mechanism-fit component removes.
        assert whole.stderr >= source.component("mechanism_fit").stderr
        assert whole.ci_low <= source.contribution <= whole.ci_high
    for source in result.sources:
        assert source.whole_procedure is not None
        assert source.whole_procedure.status == "unquantified"


def test_the_posterior_over_the_mechanisms_covers_the_closed_form_shares():
    result = intrinsic_influence(
        _data(),
        EDGES,
        "z",
        independent_noise=True,
        outer_draws=200,
        inner_draws=20,
        posterior_draws=20,
        execution=Execution(seed=3),
    )
    posterior = result.posterior
    assert posterior is not None and posterior.draws == 20
    assert result.method["fit_resampling"] == "dirichlet_row_weights"
    assert result.budget["refits"] == 20
    for name, truth in TRUTH.items():
        share = posterior.source(name)
        assert share.sd > 0.0
        assert share.ci_low <= share.mean <= share.ci_high
        assert abs(share.mean - truth) <= 6 * max(share.sd, 0.05 * truth)


def test_a_frequentist_run_carries_no_posterior():
    result = intrinsic_influence(_data(), EDGES, "z", independent_noise=True)
    assert result.posterior is None
    assert result.method["fit_resampling"] == "row_bootstrap"


def test_same_seed_reproduces_and_a_different_seed_does_not():
    kwargs = {"independent_noise": True, "outer_draws": 100, "inner_draws": 10}
    a = intrinsic_influence(_data(), EDGES, "z", execution=Execution(seed=2), **kwargs)
    b = intrinsic_influence(_data(), EDGES, "z", execution=Execution(seed=2), **kwargs)
    c = intrinsic_influence(_data(), EDGES, "z", execution=Execution(seed=3), **kwargs)
    assert [s.contribution for s in a.sources] == [s.contribution for s in b.sources]
    assert [s.contribution for s in a.sources] != [s.contribution for s in c.sources]


def test_undeclared_and_latent_confounded_noise_is_refused_with_its_reason():
    with pytest.raises(Exception, match="intrinsic_influence.noise_independence_undeclared") as e:
        intrinsic_influence(_data(), EDGES, "z")
    assert getattr(e.value, "reason_code", None) == "route_not_supported"
    with pytest.raises(Exception, match="intrinsic_influence.correlated_noise") as e:
        intrinsic_influence(_data(), EDGES, "z", bidirected_edges=[("x", "z")])
    assert getattr(e.value, "reason_code", None) == "route_not_supported"


def test_exact_enumeration_reports_exactly_zero_permutation_error_and_keeps_simulation_error():
    # a -> b -> c -> y over 100000 rows: b = 2a + e_b, c = 1.5b + e_c, y = 0.5c + e_y with
    # noise scales (1, 0.5, 1, 0.8). The total coefficients of the noise terms on y are
    # (1.5, 0.75, 0.5, 1), so the exact contributions are c_i^2 sigma_i^2.
    rng = np.random.default_rng(3192000)
    rows = 100_000
    a = rng.normal(size=rows)
    b = 2.0 * a + 0.5 * rng.normal(size=rows)
    c = 1.5 * b + rng.normal(size=rows)
    y = 0.5 * c + 0.8 * rng.normal(size=rows)
    truth = {"a": 2.25, "b": 0.140625, "c": 0.25, "y": 0.64}
    result = intrinsic_influence(
        {"a": a, "b": b, "c": c, "y": y},
        [("a", "b"), ("b", "c"), ("c", "y")],
        "y",
        independent_noise=True,
        allocation="exact",
        outer_draws=2500,
        inner_draws=10,
        execution=Execution(seed=2),
    )
    assert result.method["shapley"] == {"kind": "exact"}
    assert result.budget["permutations"] == 0 and result.budget["exact_coalitions"] == 16
    for name, source in _by_source(result).items():
        permutation = source.component("permutation_sampling")
        assert permutation.status == "not_applicable" and permutation.stderr == 0.0
        simulation = source.component("simulation")
        assert simulation.status == "quantified"
        assert simulation.stderr is not None and simulation.stderr > 0.0
        assert abs(source.contribution - truth[name]) <= 6 * max(
            simulation.stderr, 0.02 * truth[name]
        )
