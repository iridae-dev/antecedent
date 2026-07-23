"""PathSpecific / Interventional queries with discovery= and CPDAG graph=."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _discrete_chain():
    """Binary chain t → m → y (matches Rust path-specific fixture)."""
    t_vals: list[float] = []
    m_vals: list[float] = []
    y_vals: list[float] = []
    for t in (0.0, 1.0):
        for _ in range(50):
            t_vals.append(t)
            m_vals.append(t)
            y_vals.append(t)
    return {
        "t": np.asarray(t_vals, dtype=np.float64),
        "m": np.asarray(m_vals, dtype=np.float64),
        "y": np.asarray(y_vals, dtype=np.float64),
    }


def _binary_pair(n: int = 80, seed: int = 4):
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=n) > 0).astype(np.float64)
    y = ((0.7 * x + rng.normal(size=n) * 0.3) > 0.5).astype(np.float64)
    return {"x": x, "y": y}


def test_path_specific_lingam_discovery_smoke():
    # Path/distribution functionals require discrete levels; LiNGAM orients the chain.
    data = _discrete_chain()
    result = causal.analyze(
        data,
        discovery=causal.LiNGAM(),
        query=causal.PathSpecificEffect("t", "y", path_nodes=["m"]),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert abs(result.ate - 1.0) < 0.1


def test_interventional_lingam_discovery_smoke():
    data = _discrete_chain()
    result = causal.analyze(
        data,
        discovery=causal.LiNGAM(),
        query=causal.InterventionalDistribution("y", interventions={"t": 1.0}),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert np.isfinite(result.ate)


def test_path_specific_exact_dag_posterior_map():
    data = _binary_pair()
    result = causal.analyze(
        data,
        discovery=causal.ExactDagPosterior(),
        query=causal.PathSpecificEffect("x", "y"),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert np.isfinite(result.ate)


def test_path_specific_fci_rejected():
    n = 80
    rng = np.random.default_rng(5)
    z = rng.normal(size=n)
    t = z + rng.normal(size=n) * 0.3
    y = t + z + rng.normal(size=n) * 0.3
    data = {"z": z, "t": t, "y": y}
    with pytest.raises(ValueError, match="oriented DAG|PAG|cannot invent"):
        causal.analyze(
            data,
            discovery=causal.FCI(alpha=0.2, fdr=False),
            query=causal.PathSpecificEffect("t", "y"),
            accept_discovered=True,
            refute=False,
            bootstrap=0,
            seed=1,
        )


def test_path_specific_incomplete_pc_rejected():
    n = 80
    rng = np.random.default_rng(5)
    z = rng.normal(size=n)
    t = z + rng.normal(size=n) * 0.3
    y = t + z + rng.normal(size=n) * 0.3
    data = {"z": z, "t": t, "y": y}
    with pytest.raises(ValueError, match="incomplete|orient|cannot invent|cannot coerce"):
        causal.analyze(
            data,
            discovery=causal.PC(alpha=0.5, fdr=False, max_cond_size=0),
            query=causal.PathSpecificEffect("t", "y"),
            accept_discovered=True,
            refute=False,
            bootstrap=0,
            seed=1,
        )


def test_path_specific_accept_discovered_false_review_attrs():
    n = 80
    rng = np.random.default_rng(11)
    z = rng.normal(size=n)
    t = z + rng.normal(size=n) * 0.3
    y = 1.2 * t + z + rng.normal(size=n) * 0.3
    with pytest.raises(causal.CausalReviewError) as ei:
        causal.analyze(
            {"t": t, "y": y, "z": z},
            discovery=causal.PC(alpha=0.5, fdr=False, max_cond_size=0),
            query=causal.PathSpecificEffect("t", "y"),
            accept_discovered=False,
            refute=False,
            bootstrap=0,
            seed=1,
        )
    err = ei.value
    assert getattr(err, "kind", None) == "static_cpdag"
    assert getattr(err, "algorithm", None) == "pc"
    assert isinstance(getattr(err, "pending_edge_count", None), int)
    assert getattr(err, "hint", None)


def test_analyze_path_specific_graph_cpdag_fully_oriented():
    data = _discrete_chain()
    cpdag = causal.Cpdag.from_directed_undirected(
        ["t", "m", "y"],
        directed=[("t", "m"), ("m", "y")],
        undirected=[],
    )
    result = causal.analyze(
        data,
        graph=cpdag,
        query=causal.PathSpecificEffect("t", "y", path_nodes=["m"]),
        refute=False,
        bootstrap=0,
    )
    assert abs(result.ate - 1.0) < 0.1


def test_analyze_path_specific_graph_cpdag_incomplete():
    data = _discrete_chain()
    cpdag = causal.Cpdag.from_directed_undirected(
        ["t", "m", "y"],
        directed=[("t", "m")],
        undirected=[("m", "y")],
    )
    with pytest.raises(ValueError, match="undirected|orient|not fully oriented"):
        causal.analyze(
            data,
            graph=cpdag,
            query=causal.PathSpecificEffect("t", "y", path_nodes=["m"]),
            refute=False,
            bootstrap=0,
        )
