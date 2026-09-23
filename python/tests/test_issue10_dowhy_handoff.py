"""Issue #10: DoWhy ↔ Antecedent handoff on a backdoor ATE toy SCM."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("antecedent")

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "python" / "dowhy_handoff.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("dowhy_handoff_example", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_antecedent_backdoor_adjustment_and_positive_ate():
    ex = _load_example()
    data = ex.confounded_scm(n=600, seed=19)
    ant = ex.antecedent_side(data)
    assert ant["adjustment_set"] == ex.EXPECTED_ADJUSTMENT
    assert ant["ate"] > 0.0
    assert abs(ant["ate"] - ex.TRUE_ATE) < 0.6


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


def test_example_main_runs_antecedent_half():
    """``main`` must succeed without dowhy; with dowhy it asserts both halves."""
    import runpy

    runpy.run_path(str(_EXAMPLE), run_name="__main__")
