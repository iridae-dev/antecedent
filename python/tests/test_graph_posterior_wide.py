"""Graph posteriors above eight variables: wide adjacency, ten-variable MCMC workflows."""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import _native
from antecedent.discovery import (
    GraphPosterior,
    OrderMcmc,
    StructureMcmc,
    graph_posterior_map_dag,
    graph_posterior_map_edges,
)

from known_truth import set_edge

#: A ring of colliders: ``x0..x4`` are roots and ``y_j = coef * (x_j + x_{j+1}) + noise``. Ten
#: variables whose every edge sits in an unshielded collider, and whose colliders interact (each
#: ``x_j`` is a parent of two ``y``), the case single-edge samplers get trapped on.
_K = 5
_NAMES = [f"x{j}" for j in range(_K)] + [f"y{j}" for j in range(_K)]
_TRUE_EDGES = sorted(
    {(f"x{j}", f"y{j}") for j in range(_K)} | {(f"x{(j + 1) % _K}", f"y{j}") for j in range(_K)}
)


def _ring(coef: float, rows: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    roots = rng.uniform(-1.0, 1.0, size=(rows, _K))
    data = {f"x{j}": roots[:, j].copy() for j in range(_K)}
    for j in range(_K):
        noise = rng.uniform(-1.0, 1.0, size=rows)
        data[f"y{j}"] = coef * (roots[:, j] + roots[:, (j + 1) % _K]) + noise
    return data


def _mask(edges: list[tuple[str, str]], names: list[str]) -> int:
    index = {name: i for i, name in enumerate(names)}
    mask = 0
    for src, dst in edges:
        mask = set_edge(mask, len(names), index[src], index[dst])
    return mask


def test_wide_adjacency_round_trips_as_python_integers():
    names = [f"v{i}" for i in range(12)]
    chain = [(f"v{i}", f"v{i + 1}") for i in range(11)]
    fork = [("v0", f"v{i}") for i in range(1, 12)]
    chain_mask, fork_mask = _mask(chain, names), _mask(fork, names)
    assert chain_mask >= 2**64, "twelve variables need more than one 64-bit word"

    post = GraphPosterior.from_atoms(names, [0.6, 0.4], [chain_mask, fork_mask])
    assert post.adjacency == [chain_mask, fork_mask]
    assert post.n_vars == 12 and post.n_graphs == 2
    matrix = post.edge_marginal_matrix()
    assert matrix[0][1] == pytest.approx(1.0)
    assert matrix[11][10] == 0.0
    assert matrix[0][11] == pytest.approx(0.4)
    assert matrix[10][11] == pytest.approx(0.6)

    assert graph_posterior_map_edges(post) == chain
    assert sorted(graph_posterior_map_dag(post).edges()) == sorted(chain)
    keys = post.to_weighted_samples()["graph_keys"]
    assert len(keys) == 2 and keys[0] != keys[1]


def test_wide_adjacency_is_validated():
    names = [f"v{i}" for i in range(12)]
    cycle = _mask([("v0", "v1"), ("v1", "v2"), ("v2", "v0")], names)
    with pytest.raises(ValueError, match="not a DAG"):
        GraphPosterior.from_atoms(names, [1.0], [cycle])
    # Bit 132 is the first past the 132 edge bits of twelve variables.
    with pytest.raises(ValueError, match="beyond 12 variables"):
        GraphPosterior.from_atoms(names, [1.0], [1 << 132])
    with pytest.raises(ValueError, match="beyond 12 variables"):
        GraphPosterior.from_atoms(names, [1.0], [1 << 256])
    with pytest.raises(ValueError, match="non-negative"):
        GraphPosterior.from_atoms(names, [1.0], [-1])
    sixteen = [f"v{i}" for i in range(16)]
    top_edge = _mask([("v15", "v14")], sixteen)
    assert top_edge == 1 << 239
    assert GraphPosterior.from_atoms(sixteen, [1.0], [top_edge]).adjacency == [top_edge]
    # An edge set has no word limit: seventeen variables need 272 bits.
    seventeen = [f"v{i}" for i in range(17)]
    wider_edge = _mask([("v16", "v15")], seventeen)
    assert wider_edge == 1 << 271
    assert GraphPosterior.from_atoms(seventeen, [1.0], [wider_edge]).adjacency == [wider_edge]
    with pytest.raises(ValueError, match="beyond 17 variables"):
        GraphPosterior.from_atoms(seventeen, [1.0], [1 << 272])


def test_ten_variable_posteriors_recover_the_colliders():
    data = _ring(coef=0.5, rows=300, seed=41)
    truth = _mask(_TRUE_EDGES, _NAMES)
    assert 0 < truth < 2 ** (len(_NAMES) * (len(_NAMES) - 1))
    runs = {
        "structure": StructureMcmc(
            n_chains=4, n_warmup=3000, n_draws=12_000, thin=2, max_parents=3
        ).run(data, seed=5),
        "order": OrderMcmc(n_chains=4, n_warmup=3000, n_draws=12_000, thin=2, max_parents=3).run(
            data, seed=6
        ),
    }
    for label, post in runs.items():
        assert post.n_vars == 10 and post.names == _NAMES, label
        n = post.n_vars
        matrix = post.edge_marginal_matrix()
        index = {name: i for i, name in enumerate(_NAMES)}
        # BIC keeps a share of the mass on the saturated triangle of each collider (six
        # score-equivalent DAGs), so adjacency is the sharp claim, not the direction.
        for src, dst in _TRUE_EDGES:
            forward, backward = matrix[index[src]][index[dst]], matrix[index[dst]][index[src]]
            assert forward + backward >= 0.95, (label, src, dst)
        assert all(0 <= mask < 2 ** (n * (n - 1)) for mask in post.adjacency), label
        # Chain diagnostics per edge indicator, aligned with ``edge_marginals``.
        assert post.edge_rhat is not None and len(post.edge_rhat) == n * n, label
        assert post.edge_ess is not None and len(post.edge_ess) == n * n, label
        assert post.edge_mcse is not None and len(post.edge_mcse) == n * n, label
        assert all(np.isfinite(post.edge_ess)) and min(post.edge_ess) > 0.0, label
        assert all(value >= 0.0 for value in post.edge_mcse), label


def test_published_ten_variable_conditions_clear_the_publication_gate():
    """Four chains, 3 000 warmup and 12 000 retained draws thinned by two, at most three parents."""
    data = _ring(coef=0.5, rows=300, seed=41)
    columns = [data[name] for name in _NAMES]
    for sampler in (_native.discover_structure_mcmc, _native.discover_order_mcmc):
        post = sampler(
            _NAMES,
            columns,
            n_chains=4,
            n_warmup=3000,
            n_draws=12_000,
            thin=2,
            max_parents=3,
            seed=21,
        )
        assert post.converged, sampler.__name__
        n = post.n_vars
        for mask in post.adjacency:
            for node in range(n):
                parents = [
                    parent
                    for parent in range(n)
                    if parent != node
                    and (mask >> (parent * (n - 1) + (node if node < parent else node - 1))) & 1
                ]
                assert len(parents) <= 3, (sampler.__name__, node, parents)


def test_short_schedule_is_refused_by_the_gate_on_ten_variables():
    data = _ring(coef=0.5, rows=300, seed=41)
    columns = [data[name] for name in _NAMES]
    with pytest.raises(Exception, match="diagnostics gate"):
        _native.discover_structure_mcmc(_NAMES, columns, n_chains=2, n_warmup=20, n_draws=40)
    with pytest.raises(Exception, match="diagnostics gate"):
        _native.discover_order_mcmc(_NAMES, columns, n_chains=2, n_warmup=20, n_draws=40)


def test_more_than_sixteen_variables_reach_the_diagnostics_gate_not_a_size_ceiling():
    rng = np.random.default_rng(1)
    names = [f"v{i}" for i in range(17)]
    columns = [rng.normal(size=60) for _ in names]
    with pytest.raises(Exception, match="diagnostics gate"):
        _native.discover_structure_mcmc(names, columns, n_chains=2, n_warmup=10, n_draws=20)
    with pytest.raises(Exception, match="diagnostics gate"):
        _native.discover_order_mcmc(names, columns, n_chains=2, n_warmup=10, n_draws=20)
