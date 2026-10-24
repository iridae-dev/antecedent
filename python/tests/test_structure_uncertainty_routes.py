"""Uncertainty over a discovered structure runs through the compiled discovery route.

A class posterior collapses a DAG posterior into a posterior over Markov equivalence
classes, and a stability search re-runs a frequentist search on resamples drawn under the
data's sampling design and tabulates what it returned. Both keep the compiled route they ran
on: its contract, its claim and its inspection record.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import _native
from antecedent.discovery import (
    GES,
    NOTEARS,
    PC,
    PCMCI,
    BootstrapStability,
    EquivalenceClassPosterior,
    ExactDagPosterior,
    LiNGAM,
    StructureMcmc,
)
from antecedent.errors import CausalUnsupportedError, CausalValueError


def _static_rows(seed: int = 4, n: int = 300) -> tuple[list[str], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=n)
    t = 0.8 * z + rng.uniform(-1.0, 1.0, size=n)
    # No direct z -> y edge: with one, the partial correlation of z and t given y nearly
    # cancels (about 0.06) and PC drops the z - t edge in most resamples.
    y = 1.5 * t + rng.uniform(-1.0, 1.0, size=n)
    return ["z", "t", "y"], [z, t, y]


def _series_rows(seed: int = 7, n: int = 240) -> tuple[list[str], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        x[i] = rng.normal()
        y[i] = 0.8 * x[i - 1] + 0.2 * rng.normal()
    return ["x", "y"], [x, y]


def _edges(mask: int, n: int) -> set[tuple[int, int]]:
    """Directed edges of a packed atom: bit ``i * (n - 1) + (j if j < i else j - 1)`` is ``i -> j``."""
    return {
        (i, j)
        for i in range(n)
        for j in range(n)
        if i != j and (mask >> (i * (n - 1) + (j if j < i else j - 1))) & 1
    }


def _assert_contract(result, inference: str) -> None:
    contract = result.contract
    assert contract["query_kind"] == "Discover"
    assert contract["inference"] == inference
    assert contract["lowered_program"] == result.route.program_identity
    assert len(result.claim["claim_id"]) == 64


def test_a_class_posterior_collapses_the_exact_posterior():
    names, cols = _static_rows(n=200)
    dags = _native.discover_exact_dag_posterior(names, cols, seed=1)
    classes = _native.discover_exact_dag_posterior(names, cols, seed=1, equivalence_classes=True)
    assert dags.atom_kind == "Dag"
    assert classes.atom_kind == "Cpdag"
    _assert_contract(classes, "bayesian")
    assert classes.n_graphs <= 11 < dags.n_graphs
    assert sum(classes.weights) == pytest.approx(1.0)
    assert classes.algorithm == "class_posterior(exact_dag_posterior)"
    # The class program is not the DAG program it collapses.
    assert classes.claim["claim_id"] != dags.claim["claim_id"]
    again = _native.discover_exact_dag_posterior(names, cols, seed=1, equivalence_classes=True)
    assert classes.claim["claim_id"] == again.claim["claim_id"]
    assert classes.weights == again.weights


def test_a_class_posterior_pools_the_mass_of_markov_equivalent_dags():
    # A chain with no collider has three DAGs in one class; summing them is exact.
    rng = np.random.default_rng(3)
    a = rng.normal(size=400)
    b = 0.9 * a + 0.4 * rng.normal(size=400)
    c = 0.9 * b + 0.4 * rng.normal(size=400)
    names, cols = ["a", "b", "c"], [a, b, c]
    dags = _native.discover_exact_dag_posterior(names, cols, seed=1)
    classes = _native.discover_exact_dag_posterior(names, cols, seed=1, equivalence_classes=True)
    # The DAGs of the chain class: skeleton a - b - c and no collider at b.
    chain_mass = 0.0
    for mask, weight in zip(dags.adjacency, dags.weights, strict=True):
        edges = _edges(mask, 3)
        skeleton = {frozenset(edge) for edge in edges}
        collider = (0, 1) in edges and (2, 1) in edges
        if skeleton == {frozenset((0, 1)), frozenset((1, 2))} and not collider:
            chain_mass += weight
    assert chain_mass > max(dags.weights)
    # Pooling is exact: the best class carries precisely the mass of its DAGs.
    assert max(classes.weights) == pytest.approx(chain_mass, abs=1e-9)


def test_the_python_wrapper_runs_the_class_posterior_over_each_base():
    names, cols = _static_rows(n=200)
    data = dict(zip(names, cols, strict=True))
    exact = EquivalenceClassPosterior(ExactDagPosterior()).run(data, seed=1)
    assert exact.atom_kind == "Cpdag"
    sampled = EquivalenceClassPosterior(StructureMcmc(n_chains=2, n_warmup=100, n_draws=300)).run(
        data, seed=1
    )
    assert sampled.atom_kind == "Cpdag"
    assert sum(sampled.weights) == pytest.approx(1.0)
    with pytest.raises(CausalUnsupportedError):
        EquivalenceClassPosterior(PC())  # a frequentist search lists no graphs to collapse


@pytest.mark.parametrize(
    "base",
    ["pc", "ges", "direct_lingam", "notears"],
)
def test_a_static_stability_search_tabulates_its_resamples(base):
    names, cols = _static_rows()
    result = _native.discover_bootstrap_stability(names, cols, base=base, replicates=20, seed=1)
    _assert_contract(result, "frequentist")
    assert result.algorithm_id == "bootstrap_stability"
    assert result.base_algorithm == base
    assert result.replicates_requested == 20
    assert result.replicates_succeeded + result.replicates_failed == 20
    assert result.replicates_succeeded >= 10
    assert result.reference_edges
    assert sum(s.frequency for s in result.structures) == pytest.approx(1.0)
    assert result.adjacency_frequency("z", "t") >= 0.9
    assert result.adjacency_frequency("t", "y") >= 0.9
    for edge in result.edges:
        assert 0.0 <= edge.frequency <= 1.0
        assert sum(m.frequency for m in edge.marks) == pytest.approx(edge.frequency)
    # Resamples come from keyed streams, so a seed fixes the table.
    again = _native.discover_bootstrap_stability(names, cols, base=base, replicates=20, seed=1)
    assert result.claim["claim_id"] == again.claim["claim_id"]
    assert [(e.source, e.target, e.frequency) for e in result.edges] == [
        (e.source, e.target, e.frequency) for e in again.edges
    ]


def test_a_stability_search_matches_the_search_it_wraps_on_the_full_sample():
    names, cols = _static_rows()
    direct = _native.discover_pc(names, cols, alpha=0.05, fdr=True, max_cond_size=2, seed=1)
    result = _native.discover_bootstrap_stability(names, cols, base="pc", replicates=10, seed=1)
    reference = {(e.source, e.target, e.at_source, e.at_target) for e in result.reference_edges}
    full = {(e.source, e.target, e.at_source, e.at_target) for e in direct.graph_edges}
    unordered = lambda edges: {frozenset((a, b)) for a, b, *_ in edges}  # noqa: E731
    assert unordered(reference) == unordered(full)


def test_a_series_stability_search_resamples_circular_blocks():
    names, cols = _series_rows()
    result = _native.discover_bootstrap_stability(
        names, cols, base="pcmci", replicates=12, max_lag=1, seed=1
    )
    _assert_contract(result, "frequentist")
    assert result.structure_class == "temporal_dag"
    assert result.replicates_requested == 12
    lagged = [e for e in result.edges if e.in_reference and e.frequency >= 0.8]
    assert lagged


def test_a_panel_stability_search_resamples_whole_units():
    names, cols = _series_rows(n=120)
    names2, cols2 = _series_rows(seed=9, n=120)
    pooled = [np.concatenate([a, b]) for a, b in zip(cols, cols2, strict=True)]
    result = _native.discover_bootstrap_stability(
        names, pooled, base="pcmci", replicates=10, max_lag=1, seed=1, unit_lengths=[120, 120]
    )
    assert result.replicates_requested == 10
    assert result.replicates_succeeded >= 5


def test_the_python_wrapper_validates_and_runs_the_stability_search():
    names, cols = _static_rows()
    data = dict(zip(names, cols, strict=True))
    for base in (PC(), GES(), LiNGAM(), NOTEARS()):
        result = BootstrapStability(base, replicates=10).run(data, seed=2)
        assert result.replicates_requested == 10
    series = dict(zip(*_series_rows(), strict=True))
    lagged = BootstrapStability(PCMCI(max_lag=1), replicates=8).run(series, seed=2)
    assert lagged.structure_class == "temporal_dag"
    with pytest.raises(CausalValueError):
        BootstrapStability(PC(), replicates=1)
    with pytest.raises(CausalUnsupportedError):
        BootstrapStability(ExactDagPosterior())  # a posterior has its own sampling error
    with pytest.raises(CausalUnsupportedError):
        BootstrapStability(PC(ci=lambda *args: [])).run(data)  # a callable test is not re-entered


def test_unsupported_bases_are_refused_with_a_next_attempt():
    names, cols = _static_rows()
    with pytest.raises(ValueError, match="unknown base"):
        _native.discover_bootstrap_stability(names, cols, base="rpcmci")
    with pytest.raises(ValueError, match="unit_lengths"):
        _native.discover_bootstrap_stability(names, cols, base="pc", unit_lengths=[300])
    with_missing = [c.copy() for c in cols]
    with_missing[0][3] = np.nan
    with pytest.raises(Exception):  # noqa: B017, PT011 - any typed refusal: rows are incomplete
        _native.discover_bootstrap_stability(names, with_missing, base="pc", replicates=5)


def test_a_template_posterior_collapses_onto_temporal_classes():
    names, cols = _series_rows(n=200)
    data = dict(zip(names, cols, strict=True))
    from antecedent.discovery import DbnPosterior

    classes = EquivalenceClassPosterior(
        DbnPosterior(max_lag=1, n_chains=2, n_warmup=50, n_draws=100)
    ).run(data, seed=1)
    assert classes.atom_kind == "Cpdag"
    assert sum(classes.weights) == pytest.approx(1.0)


def test_a_latent_variable_search_is_resampled():
    from antecedent.discovery import LPCMCI

    names, cols = _series_rows(n=200)
    result = _native.discover_bootstrap_stability(
        names, cols, base="lpcmci", replicates=8, max_lag=1, seed=1, fdr=False
    )
    assert result.structure_class == "temporal_pag"
    data = dict(zip(names, cols, strict=True))
    assert (
        BootstrapStability(LPCMCI(max_lag=1), replicates=8).run(data, seed=1).replicates_requested
        == 8
    )


def test_laplace_noise_orients_an_edge_the_gaussian_posterior_cannot():
    rng = np.random.default_rng(5)
    x = rng.laplace(size=1500)
    y = 1.5 * x + rng.laplace(size=1500)
    laplace = _native.discover_order_mcmc(
        ["x", "y"],
        [x, y],
        n_warmup=200,
        n_draws=400,
        require_diagnostics_gate=False,
        noise="laplace",
        seed=1,
    )
    forward, backward = laplace.edge_marginals[1], laplace.edge_marginals[2]
    assert forward > backward + 0.3
    with pytest.raises(ValueError, match="noise"):
        _native.discover_order_mcmc(["x", "y"], [x, y], noise="cauchy")


def test_a_pag_posterior_merges_markov_equivalent_ancestral_graphs():
    rng = np.random.default_rng(8)
    x = rng.normal(size=1500)
    z = rng.normal(size=1500)
    y = x + z + 0.5 * rng.normal(size=1500)
    result = _native.discover_exact_pag_posterior(["x", "y", "z"], [x, y, z], seed=1)
    _assert_contract(result, "bayesian")
    assert result.atom_kind == "Pag"
    assert sum(result.weights) == pytest.approx(1.0)
    # The uniform prior over graphs gives a class the mass of every graph it contains, and the
    # saturated classes hold many graphs, so the collider class is the mode without being
    # near-certain: it is x o-> y <-o z (bit i * 2 + (j if j < i else j - 1) is i -> j).
    top = max(range(result.n_graphs), key=result.weights.__getitem__)
    assert _edges(result.adjacency[top], 3) == {(0, 1), (2, 1)}
    assert _edges(result.mark_masks[top], 3) == {(0, 1), (2, 1)}
    assert result.weights[top] > 0.5
    with pytest.raises(Exception):  # noqa: B017, PT011 - five variables are not enumerated
        _native.discover_exact_pag_posterior(list("abcde"), [x[:50]] * 5)


def test_a_multi_environment_search_is_resampled_environment_by_environment():
    from antecedent.discovery import JPCMCIPlus

    names, first = _series_rows(n=150)
    _, second = _series_rows(seed=9, n=150)
    result = _native.discover_jpcmci_plus_stability(
        names, [first, second], replicates=6, max_lag=1, seed=1
    )
    assert result.replicates_requested == 6
    assert result.structure_class == "temporal_cpdag"
    wrapped = BootstrapStability(JPCMCIPlus(max_lag=1), replicates=6).run(
        (names, [first, second]), seed=1
    )
    assert wrapped.replicates_requested == 6


def test_a_regime_search_is_resampled_with_its_labels_one_table_per_regime():
    from antecedent.discovery import RPCMCI, two_regime_half_split

    names, cols = _series_rows(n=240)
    regimes = two_regime_half_split(240)
    result = _native.discover_rpcmci_stability(
        names, cols, regimes=regimes, replicates=6, max_lag=1, min_regime_len=40, seed=1
    )
    assert sorted(result.regimes) == [0, 1]
    for table in result.tables:
        assert table.replicates_requested == 6
        assert table.structure_class == "temporal_cpdag"
    assert result.table(0).replicates_requested == 6
    data = dict(zip(names, cols, strict=True))
    wrapped = BootstrapStability(RPCMCI(max_lag=1), replicates=6).run(data, regimes=regimes, seed=1)
    assert len(wrapped.tables) == 2
    with pytest.raises(CausalValueError):
        BootstrapStability(RPCMCI(max_lag=1), replicates=6).run(data)


def test_a_pag_sampler_runs_beyond_four_variables_and_finds_the_exact_top_class():
    rng = np.random.default_rng(12)
    a = rng.normal(size=800)
    b = 0.9 * a + 0.5 * rng.normal(size=800)
    c = 0.9 * b + 0.5 * rng.normal(size=800)
    d = rng.normal(size=800)
    e = rng.normal(size=800)
    sampled = _native.discover_pag_mcmc(
        list("abcde"), [a, b, c, d, e], n_chains=2, n_warmup=100, n_draws=400, seed=1
    )
    _assert_contract(sampled, "bayesian")
    assert sampled.atom_kind == "Pag"
    assert sum(sampled.weights) == pytest.approx(1.0)
    small = _native.discover_pag_mcmc(
        ["a", "b", "c"], [a, b, c], n_chains=2, n_warmup=200, n_draws=1500, seed=1
    )
    exact = _native.discover_exact_pag_posterior(["a", "b", "c"], [a, b, c], seed=1)
    top = lambda post: post.adjacency[max(range(post.n_graphs), key=post.weights.__getitem__)]  # noqa: E731
    assert top(small) == top(exact)
