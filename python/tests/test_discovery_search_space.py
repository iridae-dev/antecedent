"""The lag window, contemporaneous admission and CI resolution every discovery run starts from.

One series has a strong same-time effect ``x[t] -> y[t]`` plus lagged autoregression, so the
searches differ observably: lagged-only PCMCI may never name a lag-0 link and never one beyond
its window, while the contemporaneous-capable searches name the same-time ``x - y`` adjacency.
The static search has no time axis at all.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalCompileError

DISCOVERY = antecedent.discovery


def _driven_pair(n: int = 1200, seed: int = 21):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal()
        y[t] = 0.9 * x[t] + 0.4 * y[t - 1] + 0.5 * rng.normal()
    return ["x", "y"], [x, y]


def _same_time_adjacency(result) -> bool:
    return any(
        {edge.source, edge.target} == {"x", "y"} and edge.source_lag == 0 and edge.target_lag == 0
        for edge in result.graph_edges
    )


def test_pcmci_searches_only_the_declared_lagged_window():
    names, cols = _driven_pair()
    result = DISCOVERY.PCMCI(max_lag=2, alpha=0.01, fdr=False).run(
        (names, cols), execution=antecedent.Execution(seed=3)
    )
    recovered = {
        (link.source, link.source_lag, link.target, link.target_lag) for link in result.links
    }
    assert recovered, "the autoregression is strong enough to be found"
    # Lagged-only: every link ends at the present, starts between one and max_lag steps back.
    assert all(
        target_lag == 0 and 1 <= source_lag <= 2 for _, source_lag, _, target_lag in recovered
    ), recovered
    assert ("y", 1, "y", 0) in recovered
    assert ("x", 1, "x", 0) in recovered


def test_a_wider_lag_window_is_searched_to_its_declared_edge_and_no_further():
    rng = np.random.default_rng(8)
    n = 1500
    x = rng.normal(size=n)
    y = np.zeros(n)
    y[3:] = 0.9 * x[:-3] + 0.3 * rng.normal(size=n - 3)
    window_one = DISCOVERY.PCMCI(max_lag=1, alpha=0.01, fdr=False).run(
        (["x", "y"], [x, y]), execution=antecedent.Execution(seed=2)
    )
    window_three = DISCOVERY.PCMCI(max_lag=3, alpha=0.01, fdr=False).run(
        (["x", "y"], [x, y]), execution=antecedent.Execution(seed=2)
    )
    assert all(link.source_lag <= 1 for link in window_one.links)
    assert not any(
        (link.source, link.source_lag, link.target) == ("x", 3, "y") for link in window_one.links
    )
    found = {(link.source, link.source_lag, link.target) for link in window_three.links}
    assert ("x", 3, "y") in found
    assert all(1 <= link.source_lag <= 3 for link in window_three.links)


@pytest.mark.parametrize("method", [DISCOVERY.PCMCIPlus, DISCOVERY.LPCMCI])
def test_contemporaneous_searches_admit_the_same_time_adjacency(method):
    names, cols = _driven_pair()
    result = method(max_lag=1, alpha=0.01, fdr=False).run(
        (names, cols), execution=antecedent.Execution(seed=3)
    )
    assert _same_time_adjacency(result), result.graph_edges
    assert all(edge.source_lag <= 1 and edge.target_lag <= 1 for edge in result.graph_edges)


def test_lpcmci_counts_exactly_the_definitely_directed_edges_of_its_graph():
    names, cols = _driven_pair()
    result = DISCOVERY.LPCMCI(max_lag=1, alpha=0.01, fdr=False).run(
        (names, cols), execution=antecedent.Execution(seed=3)
    )
    definite = sum(
        1 for edge in result.graph_edges if {edge.at_source, edge.at_target} == {"tail", "arrow"}
    )
    # Time orders every lagged edge, so at least one strong autoregression is definitely directed.
    assert definite >= 1
    assert result.cpdag_directed_edges == definite
    marked = {"tail", "arrow", "circle"}
    assert all({edge.at_source, edge.at_target} <= marked for edge in result.graph_edges)


def test_the_multi_environment_search_admits_the_same_time_adjacency():
    environments = []
    for seed in (21, 22):
        names, cols = _driven_pair(n=800, seed=seed)
        environments.append(cols)
    result = DISCOVERY.JPCMCIPlus(max_lag=1, alpha=0.01, fdr=False).run(
        ["x", "y"], environments, execution=antecedent.Execution(seed=3)
    )
    assert result.algorithm_id == "jpcmci_plus"
    assert _same_time_adjacency(result), result.graph_edges


def test_the_static_search_has_no_time_axis_and_keeps_only_the_collider_skeleton():
    rng = np.random.default_rng(7)
    n = 1500
    t = rng.normal(size=n)
    w = rng.normal(size=n)
    y = 1.5 * t + 1.0 * w + 0.3 * rng.normal(size=n)
    result = DISCOVERY.PC(alpha=0.001, fdr=False, max_cond_size=2).run(
        {"t": t, "y": y, "w": w}, execution=antecedent.Execution(seed=1)
    )
    assert result.graph_edges
    assert all(edge.source_lag == 0 and edge.target_lag == 0 for edge in result.graph_edges)
    adjacent = {frozenset((edge.source, edge.target)) for edge in result.graph_edges}
    assert adjacent == {frozenset(("t", "y")), frozenset(("w", "y"))}


def test_conditional_independence_resolves_by_name_and_refuses_unknown_or_misused_weights():
    names, cols = _driven_pair(n=300)
    result = DISCOVERY.PCMCI(max_lag=1, alpha=0.05, fdr=False, ci="parcorr").run(
        (names, cols), execution=antecedent.Execution(seed=1)
    )
    assert result.ci_name == "parcorr"
    weighted = DISCOVERY.PCMCI(max_lag=1, alpha=0.05, fdr=False, ci="weighted_parcorr").run(
        (names, cols), execution=antecedent.Execution(seed=1), weights=[1.0] * 300
    )
    assert weighted.ci_name == "weighted_parcorr"

    with pytest.raises(CausalCompileError, match=r"backend error: unknown CI test name: no_such_test"):
        DISCOVERY.PCMCI(max_lag=1, ci="no_such_test").run(
            (names, cols), execution=antecedent.Execution(seed=1)
        )
    # Weights belong to the weighted test alone, and the weighted test cannot run without them.
    with pytest.raises(CausalCompileError, match="weights"):
        DISCOVERY.PCMCI(max_lag=1, ci="weighted_parcorr").run(
            (names, cols), execution=antecedent.Execution(seed=1)
        )
    with pytest.raises(CausalCompileError, match="weighted_parcorr"):
        DISCOVERY.PCMCI(max_lag=1, ci="parcorr").run(
            (names, cols), execution=antecedent.Execution(seed=1), weights=[1.0] * 300
        )


def test_a_compile_refusal_names_the_family_of_its_typed_cause():
    names, cols = _driven_pair(n=300)
    # An unknown test name is a statistics-layer refusal: the typed cause survives the boundary.
    with pytest.raises(CausalCompileError) as unknown:
        DISCOVERY.PCMCI(max_lag=1, ci="no_such_test").run(
            (names, cols), execution=antecedent.Execution(seed=1)
        )
    assert unknown.value.cause_kind == "stats"
    # A message-only compile refusal (weights without the weighted test) carries no typed cause.
    with pytest.raises(CausalCompileError, match="weights") as untyped:
        DISCOVERY.PCMCI(max_lag=1, ci="weighted_parcorr").run(
            (names, cols), execution=antecedent.Execution(seed=1)
        )
    assert getattr(untyped.value, "cause_kind", None) is None
