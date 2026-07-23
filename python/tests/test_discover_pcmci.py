"""discover_pcmci schema stability and Exact lag-1 recovery.

Lag-1 dual: recovers the same parent set as Rust conformance
`discovery_pcmci_lag1_exact_parents` (`conformance/discovery/pcmci_lag1`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent

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

# Shared Exact edge set with Rust `discovery_pcmci_lag1_exact_parents`.
_PCMCI_LAG1_FIXTURE = (
    Path(__file__).resolve().parents[2] / "conformance" / "discovery" / "pcmci_lag1"
)
_TRUE_LAG1_PARENTS = {("x", 1, "y", 0)}


def _lag1_series(n: int = 400):
    t = np.arange(n, dtype=np.float64)
    x = np.sin(t * 0.01)
    y = np.zeros(n, dtype=np.float64)
    y[1:] = 0.8 * x[:-1] + 0.01 * np.cos(t[1:] * 0.03)
    return ["x", "y"], [x, y]


def _conformance_lag1_series():
    csv = _PCMCI_LAG1_FIXTURE / "data.csv"
    raw = np.loadtxt(csv, delimiter=",", skiprows=1)
    x = np.asarray(raw[:, 0], dtype=np.float64)
    y = np.asarray(raw[:, 1], dtype=np.float64)
    return ["x", "y"], [x, y]


def test_discover_pcmci_schema_fields():
    names, cols = _lag1_series()
    result = causal.discover_pcmci(names, cols, max_lag=2, alpha=0.05, fdr=False, seed=9)
    for name in SCHEMA_FIELDS:
        assert hasattr(result, name), name
    assert result.algorithm_id == "pcmci"
    assert "fdr=" in result.algorithm_config
    assert "fdr=BH" not in result.algorithm_config  # fdr=False → no BH adjustment
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


def test_discover_pcmci_conformance_exact_lag1_dual():
    """Python recovers the Exact parent set of the Rust pcmci_lag1 conformance fixture."""
    names, cols = _conformance_lag1_series()
    assert len(cols[0]) == 500
    result = causal.discover_pcmci(names, cols, max_lag=2, alpha=0.05, fdr=False, seed=42)
    recovered = {
        (link.source, link.source_lag, link.target, link.target_lag) for link in result.links
    }
    assert recovered == _TRUE_LAG1_PARENTS, recovered
