"""Conditioning-set provenance (`DiscoveredLink.conditioning_set`) and the PCMCI-family
`max_cond_size` cap.

Two things are covered here:

1. `conditioning_set` on `DiscoveredLink`. Semantics are algorithm-dependent (see the
   field docstring in `python/src/discovery_api.rs`): for PC/GES/FCI/RFCI/PCMCI+, a
   link that survives into `result.links` can never carry a non-empty conditioning set
   (recording one is mutually exclusive with retention in those algorithms' skeleton
   phase). Plain PCMCI is the exception — its PC1 phase records conditioning sets purely
   to prune MCI's conditioning candidates, decoupled from the final per-pair MCI test,
   so a *retained* plain-PCMCI link can legitimately carry a non-empty conditioning set.
   That is exercised directly below rather than asserted on a trivial 3-node chain,
   where the retain/record paths for most algorithms never overlap.

2. `max_cond_size` on `PCMCI` / `PCMCIPlus` / `LPCMCI` (and `JPCMCIPlus` / `RPCMCI`),
   newly threaded through to match `PC`/`GES`/`FCI`/`RFCI`/`LiNGAM`/`NOTEARS`. The cap is
   asserted to actually change discovery behavior (`ci_tests` count), not merely to be
   accepted.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent
from antecedent.discovery import (
    LPCMCI,
    PC,
    PCMCI,
    RPCMCI,
    JPCMCIPlus,
    PCMCIPlus,
    run_temporal_discovery,
)
from antecedent.query import PulseEffect


def _chain_series(n: int = 400):
    """Lag-1 chain X -> Y -> Z; no direct X -> Z edge."""
    t = np.arange(n, dtype=np.float64)
    x = np.sin(t * 0.01)
    y = np.zeros(n, dtype=np.float64)
    z = np.zeros(n, dtype=np.float64)
    y[1:] = 0.8 * x[:-1] + 0.01 * np.cos(t[1:] * 0.03)
    z[1:] = 0.8 * y[:-1] + 0.01 * np.sin(t[1:] * 0.05)
    return ["x", "y", "z"], [x, y, z]


def _pruned_conditioning_series(seed: int = 21, n: int = 300):
    """5-variable lag-1 system whose PCMCI PC1 phase prunes a conditioning candidate for
    v2's self-lag while MCI still retains that exact (v2, lag 1) -> (v2, lag 0) pair —
    the plain-PCMCI decoupling documented on `DiscoveredLink.conditioning_set`. Fixed
    seed, single-threaded, analytic CI test: deterministic across runs/platforms.
    """
    rng = np.random.default_rng(seed)
    p = 5
    cols = [rng.normal(size=n) for _ in range(p)]
    names = [f"v{i}" for i in range(p)]
    for t in range(1, n):
        cols[1][t] = 0.5 * cols[0][t - 1] + 0.3 * rng.normal()
        cols[2][t] = 0.5 * cols[1][t - 1] + 0.2 * cols[0][t - 1] + 0.3 * rng.normal()
        cols[3][t] = 0.5 * cols[2][t - 1] + 0.3 * rng.normal()
        cols[4][t] = 0.3 * cols[1][t - 1] + 0.3 * cols[3][t - 1] + 0.3 * rng.normal()
    return names, cols


def _assert_conditioning_set_shape(links, names):
    for link in links:
        assert isinstance(link.conditioning_set, list)
        for entry in link.conditioning_set:
            var_name, lag = entry
            assert var_name in names
            assert isinstance(lag, int)
            assert lag >= 0


def test_conditioning_set_present_and_well_typed_across_family():
    names, cols = _chain_series()
    pc = antecedent.discover_pc(names, cols, alpha=0.1, fdr=False, seed=9)
    pcmci = antecedent.discover_pcmci(names, cols, max_lag=2, alpha=0.1, fdr=False, seed=9)
    pcmci_plus = antecedent.discover_pcmci_plus(
        names, cols, max_lag=2, alpha=0.1, fdr=False, seed=9
    )
    lpcmci = antecedent.discover_lpcmci(names, cols, max_lag=2, alpha=0.1, fdr=False, seed=9)
    for result in (pc, pcmci, pcmci_plus, lpcmci):
        assert result.links, "expected at least one retained link for this chain"
        _assert_conditioning_set_shape(result.links, names)


def test_conditioning_set_empty_on_retained_pc_links():
    """PC's skeleton phase is mutually exclusive: recording a separating set for a pair
    means it was removed, so a link that survives into `result.links` always reports an
    empty conditioning set — this is the documented invariant, not an omission.
    """
    names, cols = _chain_series()
    result = antecedent.discover_pc(names, cols, alpha=0.1, fdr=False, seed=9)
    assert result.links
    assert all(link.conditioning_set == [] for link in result.links)


def test_conditioning_set_nonempty_and_correct_for_retained_pcmci_link():
    """Plain PCMCI's PC1 phase can prune a conditioning candidate for one target while
    MCI independently retains that same pair — the one case where a retained link
    legitimately carries a non-empty conditioning set (see module docstring).
    """
    names, cols = _pruned_conditioning_series()
    result = antecedent.discover_pcmci(names, cols, max_lag=1, alpha=0.05, fdr=True, seed=1)
    by_key = {
        (link.source, link.source_lag, link.target, link.target_lag): link for link in result.links
    }
    target_key = ("v2", 1, "v2", 0)
    assert target_key in by_key, sorted(by_key)
    link = by_key[target_key]
    assert link.conditioning_set == [("v1", 1)], link.conditioning_set


def test_lpcmci_links_all_correspond_to_final_pag_edges():
    """LPCMCI's scored-link accumulator is reconciled against the final PAG before it is
    returned (`reconcile_evidence_with_pag` in the Rust discovery phases), so every
    entry in `result.links` must name a pair that is actually an edge in
    `result.graph_edges`. Before that reconciliation existed, `links` could retain
    entries from an earlier preliminary iteration that a later iteration's fresh PAG
    (built by `init_complete_pag`, carrying over only parent memory) no longer connected
    at all — this test pins the invariant that no longer happens.

    Edge presence is checked as an unordered pair: a PAG edge for (a, b) may be reported
    with either endpoint as `source`, so a link's `(source, source_lag)` /
    `(target, target_lag)` pair is compared against `graph_edges` regardless of
    orientation.
    """
    names, cols = _chain_series()
    result = antecedent.discover_lpcmci(names, cols, max_lag=2, alpha=0.1, fdr=False, seed=9)
    assert result.links, "expected at least one retained link for this chain"

    edge_pairs = {
        frozenset({(edge.source, edge.source_lag), (edge.target, edge.target_lag)})
        for edge in result.graph_edges
    }
    for link in result.links:
        link_pair = frozenset({(link.source, link.source_lag), (link.target, link.target_lag)})
        assert link_pair in edge_pairs, (
            f"link {link.source}[{link.source_lag}] -> {link.target}[{link.target_lag}] "
            "has no corresponding edge in graph_edges"
        )


def test_max_cond_size_default_matches_pc():
    assert PCMCI().max_cond_size == PC().max_cond_size == 2
    assert PCMCIPlus().max_cond_size == 2
    assert LPCMCI().max_cond_size == 2
    assert JPCMCIPlus().max_cond_size == 2
    assert RPCMCI().max_cond_size == 2


@pytest.mark.parametrize(
    "discover_fn",
    [antecedent.discover_pcmci, antecedent.discover_pcmci_plus, antecedent.discover_lpcmci],
)
def test_max_cond_size_honored_by_native_functions(discover_fn):
    names, cols = _pruned_conditioning_series()
    capped = discover_fn(names, cols, max_lag=1, alpha=0.05, max_cond_size=0)
    default = discover_fn(names, cols, max_lag=1, alpha=0.05, max_cond_size=2)
    assert capped.ci_tests != default.ci_tests, (
        f"{discover_fn.__name__}: max_cond_size cap had no effect on ci_tests"
    )


def test_max_cond_size_flows_through_dataclass_dispatch():
    """PCMCI(max_cond_size=...) reaches the native call via run_temporal_discovery, the
    same dispatcher `analyze(discovery=PCMCI(...))` and GCM compose helpers use.
    """
    names, cols = _pruned_conditioning_series()
    data = dict(zip(names, cols, strict=True))
    default_result, kind = run_temporal_discovery(data, PCMCI(max_lag=1, alpha=0.05))
    capped_result, _ = run_temporal_discovery(data, PCMCI(max_lag=1, alpha=0.05, max_cond_size=0))
    assert kind == "pcmci"
    assert capped_result.ci_tests != default_result.ci_tests


def test_max_cond_size_reaches_analyze_temporal_discover_path():
    """`analyze(discovery=PCMCIPlus(max_cond_size=...))` must not silently drop the cap.

    `run_temporal_discovery` (tested above) is the free-function dispatcher used by
    `AcceptedGraph` and GCM compose helpers; `analyze(..., discovery=...)` for
    PulseEffect/SustainedEffect goes through a completely separate call path
    (`_handle_series_discover` -> `_native.analyze_temporal_discover`), so it needs its
    own coverage — a prior bug had exactly this path silently ignoring `max_cond_size`
    while `run_temporal_discovery` already forwarded it correctly.

    On the 5-variable system from `_pruned_conditioning_series`, capping PCMCI+'s
    skeleton-phase conditioning set to 0 (v1 -> v2, lag 1) changes which lagged parents
    survive discovery enough to make the temporal backdoor unidentifiable within the
    default history window (`CausalIdentifyError`), while the same query with
    `max_cond_size=2` (the library default) discovers a graph that *is* identified and
    returns a finite ATE. Neither `ci_tests` nor the discovered-link list is on
    `analyze()`'s result surface for the temporal path (`AnalysisResult` exposes
    `identification`/`estimate`/`plan`/`diagnostics`/`provenance`, none of which carry a
    raw discovery-performance record), so identifiability flipping is the observable
    signal here — success-vs-error is as much a "different result" as a different ATE
    value, and this is the sharpest such signal this data/query pair produces.
    """
    names, cols = _pruned_conditioning_series()
    data = dict(zip(names, cols, strict=True))
    query = PulseEffect(treatment="v1", outcome="v2", treatment_lag=1, horizon_steps=1)

    with pytest.raises(antecedent.CausalIdentifyError, match="history cap"):
        antecedent.analyze(
            data,
            query=query,
            discovery=PCMCIPlus(max_lag=1, alpha=0.05, max_cond_size=0),
            seed=1,
            threads=1,
        )

    capped_result = antecedent.analyze(
        data,
        query=query,
        discovery=PCMCIPlus(max_lag=1, alpha=0.05, max_cond_size=2),
        seed=1,
        threads=1,
    )
    assert np.isfinite(capped_result.estimate.ate)

    # The library default (max_cond_size=2) must behave identically to an explicit 2 —
    # this is a plumbing fix, not a behavior change for existing callers.
    default_result = antecedent.analyze(
        data,
        query=query,
        discovery=PCMCIPlus(max_lag=1, alpha=0.05),
        seed=1,
        threads=1,
    )
    assert default_result.estimate.ate == capped_result.estimate.ate
