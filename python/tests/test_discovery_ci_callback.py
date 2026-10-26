"""Discovery dispatch preserves caller CI evidence and rejects malformed results."""

from __future__ import annotations

import numpy as np
import pytest
from antecedent.discovery import PC, PCMCI, run_static_discovery, run_temporal_discovery
from antecedent.errors import CausalError


def _data():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    z = rng.normal(size=n)
    return ["x", "y", "z"], [x, y, z]


def _counting_ci(calls):
    def ci(_columns, queries):
        calls["n"] += 1
        return [(0.0, 1.0) for _ in queries]

    return ci


def test_run_forwards_callable_ci_to_native():
    calls = {"n": 0}
    result = PC(alpha=0.05, fdr=False, ci=_counting_ci(calls), max_cond_size=1).run(_data(), seed=1)
    assert result.ci_name == "python.callback"
    assert calls["n"] > 0
    assert result.graph_edges == []


def test_dispatcher_preserves_callable_ci():
    calls = {"n": 0}
    result, algorithm_id = run_static_discovery(
        _data(), PC(alpha=0.05, fdr=False, ci=_counting_ci(calls), max_cond_size=1), seed=1
    )
    assert algorithm_id == "pc"
    assert result.ci_name == "python.callback"
    assert calls["n"] > 0
    assert result.graph_edges == []


def test_temporal_dispatcher_preserves_callable_ci_and_serial_callback_policy():
    calls = {"n": 0}
    result, algorithm_id = run_temporal_discovery(
        _data(), PCMCI(max_lag=1, fdr=False, ci=_counting_ci(calls)), seed=1, threads=4
    )
    assert algorithm_id == "pcmci"
    assert result.ci_name == "python.callback"
    assert result.worker_threads == 1
    assert calls["n"] > 0
    assert result.links == []


def test_string_ci_is_unaffected_on_both_paths():
    assert PC(alpha=0.05, fdr=False, max_cond_size=1).run(_data(), seed=1).ci_name == "parcorr"
    result, _ = run_static_discovery(_data(), PC(alpha=0.05, fdr=False, max_cond_size=1), seed=1)
    assert result.ci_name == "parcorr"
    assert result.graph_edges, "the supplied independent callback must change the discovered graph"


@pytest.mark.parametrize(
    "statistic, probability",
    [
        (np.nan, 0.5),
        (np.inf, 0.5),
        (0.0, np.nan),
        (0.0, -0.1),
        (0.0, 1.1),
    ],
)
def test_callback_invalid_evidence_is_refused_before_graph_search(statistic, probability):
    def ci(_columns, queries):
        return [(statistic, probability) for _ in queries]

    with pytest.raises(CausalError, match="finite statistic.*p_value"):
        run_static_discovery(_data(), PC(fdr=False, ci=ci, max_cond_size=1), seed=1)
