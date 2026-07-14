"""discover_pcmci schema stability and Exact lag-1 recovery."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("causal")
import causal

SCHEMA_FIELDS = {
    "links",
    "algorithm_id",
    "algorithm_config",
    "ci_tests",
    "links_retained",
    "pending_edge_count",
    "lagged_frame_bytes",
    "worker_threads",
    "ci_name",
    "cpdag_nodes",
    "cpdag_directed_edges",
    "cpdag_undirected_edges",
}

LINK_FIELDS = {
    "source",
    "source_lag",
    "target",
    "target_lag",
    "statistic",
    "p_value",
}


def _lag1_series(n: int = 400):
    t = np.arange(n, dtype=np.float64)
    x = np.sin(t * 0.01)
    y = np.zeros(n, dtype=np.float64)
    y[1:] = 0.8 * x[:-1] + 0.01 * np.cos(t[1:] * 0.03)
    return ["x", "y"], [x, y]


def test_discover_pcmci_schema_fields():
    names, cols = _lag1_series()
    result = causal.discover_pcmci(names, cols, max_lag=2, alpha=0.05, fdr=False, seed=9)
    for name in SCHEMA_FIELDS:
        assert hasattr(result, name), name
    assert result.algorithm_id == "pcmci"
    assert "fdr=false" in result.algorithm_config
    assert result.ci_tests > 0
    assert result.lagged_frame_bytes > 0
    assert result.worker_threads >= 1
    assert result.pending_edge_count == result.links_retained
    if result.links:
        link = result.links[0]
        for name in LINK_FIELDS:
            assert hasattr(link, name), name


def test_discover_pcmci_recovers_lag1_parent():
    names, cols = _lag1_series()
    result = causal.discover_pcmci(names, cols, max_lag=2, alpha=0.05, fdr=False, seed=9)
    recovered = {
        (link.source, link.source_lag, link.target, link.target_lag) for link in result.links
    }
    assert ("x", 1, "y", 0) in recovered, recovered
