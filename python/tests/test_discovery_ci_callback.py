"""A callable ``ci`` must survive ``Config.run()`` and be collapsed by the dispatchers.

``Config.run()`` replaces the deleted free ``discover_*`` functions, which forwarded a
Python callable CI test to native unchanged. The ``run_static_discovery`` /
``run_temporal_discovery`` dispatchers behind ``analyze`` and
``AcceptedGraph.rediscover`` have always collapsed it to a string instead; both
behaviors are pinned here so neither drifts into the other.
"""

from __future__ import annotations

import numpy as np
from antecedent.discovery import PC, run_static_discovery


def _data():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    z = rng.normal(size=n)
    return ["x", "y", "z"], [x, y, z]


def _counting_ci(calls):
    def ci(*_a, **_k):
        calls["n"] += 1
        return [(0.0, 1.0)]

    return ci


def test_run_forwards_callable_ci_to_native():
    calls = {"n": 0}
    result = PC(alpha=0.05, fdr=False, ci=_counting_ci(calls), max_cond_size=1).run(_data(), seed=1)
    assert result.ci_name == "python.callback"
    assert calls["n"] > 0


def test_dispatcher_collapses_callable_ci():
    calls = {"n": 0}
    result, algorithm_id = run_static_discovery(
        _data(), PC(alpha=0.05, fdr=False, ci=_counting_ci(calls), max_cond_size=1), seed=1
    )
    assert algorithm_id == "pc"
    assert result.ci_name != "python.callback"
    assert calls["n"] == 0


def test_string_ci_is_unaffected_on_both_paths():
    assert PC(alpha=0.05, fdr=False, max_cond_size=1).run(_data(), seed=1).ci_name == "parcorr"
    result, _ = run_static_discovery(_data(), PC(alpha=0.05, fdr=False, max_cond_size=1), seed=1)
    assert result.ci_name == "parcorr"
