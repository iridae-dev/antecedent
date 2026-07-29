"""`latency=` must reach the typed-graph analyze paths (Cpdag / Pag / Admg).

`analyze_ate` has always accepted `latency=`, but the four typed-graph entry
points did not, so `analyze(graph=<Cpdag|Pag|Admg>, latency=...)` raised
``TypeError: unexpected keyword argument 'latency'``. That made every
``AcceptedGraph.analyze()`` on a non-DAG structure fail outright, since
``AcceptedGraph.analyze`` defaults to ``latency="interactive"``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent
from antecedent.graph import Admg, Cpdag


def _binary_treatment_scm(seed: int = 5, n: int = 300):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (0.8 * z + rng.normal(size=n) * 0.4 > 0).astype(float)
    y = 2.0 * t + 1.2 * z + rng.normal(size=n) * 0.3
    return {"z": z, "t": t, "y": y}


_EDGES = [("z", "t"), ("z", "y"), ("t", "y")]


@pytest.mark.parametrize("mode", ["interactive", "standard", "report"])
def test_cpdag_analyze_honours_every_latency_tier(mode):
    data = _binary_treatment_scm()
    cpdag = Cpdag.from_directed_undirected(["z", "t", "y"], _EDGES, [])
    result = antecedent.analyze(
        data,
        graph=cpdag,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        latency=mode,
        seed=1,
    )
    assert result.performance.latency_mode == mode


def test_admg_analyze_honours_latency():
    data = _binary_treatment_scm()
    admg = Admg.from_edges(["z", "t", "y"], _EDGES, [])
    result = antecedent.analyze(
        data,
        graph=admg,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        latency="report",
        seed=1,
    )
    assert result.performance.latency_mode == "report"


def test_accepted_graph_analyze_on_cpdag_runs():
    """`AcceptedGraph.analyze` defaults to latency="interactive" — it must not raise."""
    data = _binary_treatment_scm()
    cpdag = Cpdag.from_directed_undirected(["z", "t", "y"], _EDGES, [])
    accepted = antecedent.AcceptedGraph.asserted(cpdag, algorithm_id="pc")
    result = accepted.analyze(
        data, query=antecedent.AverageEffect(treatment="t", outcome="y"), seed=1
    )
    assert result.performance.latency_mode == "interactive"
    assert abs(result.ate - 2.0) < 0.5
