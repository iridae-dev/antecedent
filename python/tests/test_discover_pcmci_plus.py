"""Phase 5: discover_pcmci_plus and CI name selection."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("causal")
import causal


def _series(n: int = 400):
    t = np.arange(n, dtype=np.float64)
    x = np.sin(t * 0.01)
    y = np.zeros(n, dtype=np.float64)
    y[1:] = 0.8 * x[:-1] + 0.3 * x[1:] + 0.01 * np.cos(t[1:] * 0.03)
    return ["x", "y"], [x, y]


def test_discover_pcmci_plus_returns_cpdag_summary():
    names, cols = _series()
    result = causal.discover_pcmci_plus(
        names, cols, max_lag=1, alpha=0.05, fdr=False, seed=9, ci="parcorr"
    )
    assert result.algorithm_id == "pcmci_plus"
    assert result.ci_name == "parcorr"
    assert result.cpdag_nodes >= 2
    assert result.cpdag_directed_edges + result.cpdag_undirected_edges >= 1
    recovered = {
        (link.source, link.source_lag, link.target, link.target_lag) for link in result.links
    }
    assert ("x", 1, "y", 0) in recovered or ("x", 0, "y", 0) in recovered, recovered


def test_discover_pcmci_ci_name_oracle():
    names, cols = _series()
    result = causal.discover_pcmci(
        names, cols, max_lag=1, alpha=0.05, fdr=False, seed=1, ci="oracle"
    )
    assert result.ci_name == "oracle"
    assert result.links_retained == 0
