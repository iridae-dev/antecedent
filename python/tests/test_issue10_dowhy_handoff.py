"""Issue #10: DoWhy ↔ Antecedent handoff on a backdoor ATE toy SCM."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from _repo_text import REPO_ROOT

_EXAMPLE = REPO_ROOT / "examples" / "python" / "dowhy_handoff.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("dowhy_handoff_example", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hajek_ate(data) -> float:
    """Normalised inverse-propensity contrast with a logistic propensity on z fitted by Newton."""
    z, t, y = data["z"], data["t"], data["y"]
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.zeros(2)
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-design @ beta))
        hessian = design.T @ (design * (p * (1.0 - p))[:, None])
        beta += np.linalg.solve(hessian, design.T @ (t - p))
    p = 1.0 / (1.0 + np.exp(-design @ beta))
    treated = np.sum(t * y / p) / np.sum(t / p)
    control = np.sum((1 - t) * y / (1 - p)) / np.sum((1 - t) / (1 - p))
    return float(treated - control)


def test_antecedent_backdoor_adjustment_and_positive_ate():
    ex = _load_example()
    data = ex.confounded_scm(n=600, seed=19)
    ant = ex.antecedent_side(data)
    assert ant["adjustment_set"] == ex.EXPECTED_ADJUSTMENT
    assert ant["ate"] > 0.0
    assert abs(ant["ate"] - ex.TRUE_ATE) < 0.1  # observed 0.019
    # The estimator is the normalised inverse-propensity contrast of a fitted logistic score.
    assert ant["ate"] == pytest.approx(_hajek_ate(data), abs=1e-9)


def test_dowhy_side_matches_when_available():
    ex = _load_example()
    data = ex.confounded_scm(n=600, seed=19)
    ant = ex.antecedent_side(data)
    dowhy = ex.dowhy_side(data, ant["dot"])
    if dowhy is None:
        pytest.skip("dowhy not installed")
    assert dowhy["adjustment_set"] == ant["adjustment_set"] == ex.EXPECTED_ADJUSTMENT
    assert dowhy["round_trip_adjustment_set"] == ex.EXPECTED_ADJUSTMENT
    assert dowhy["ate"] > 0.0 and ant["ate"] > 0.0
    assert abs(dowhy["ate"] - ex.TRUE_ATE) < 0.6
    assert ant["ate"] == pytest.approx(_hajek_ate(data), abs=1e-9)


def test_example_main_runs_antecedent_half():
    """``main`` must succeed without dowhy; with dowhy it asserts both halves."""
    import runpy

    runpy.run_path(str(_EXAMPLE), run_name="__main__")
