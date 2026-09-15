"""Panel / event / multi-env frames prepare through PyO3."""

from __future__ import annotations

import numpy as np

import antecedent as ant
from antecedent import PulseEffect
from antecedent.data import event, multi_env, panel
from antecedent.estimation import PreparedAnalysis


def _series(n: int = 40, seed: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = rng.normal(size=n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * t[i - 1]
    return {"t": t, "y": y}


def test_panel_frame_prepares_through_pyo3():
    series = _series()
    frame = panel([series, series])
    prepared = PreparedAnalysis.prepare(
        frame,
        graph=[("t", 1, "y", 0)],
        query=PulseEffect("t", "y"),
        bootstrap=0,
        refute="none",
    )
    result = prepared.estimate()
    encoded = result.export()
    loaded = ant.artifacts.loads(encoded)
    assert loaded.payload_kind == "analysis_result"


def test_event_frame_prepares_through_pyo3():
    series = _series()
    frame = event(
        series, event_times_ns=np.arange(len(series["t"])) * 1_000_000, align_interval_ns=1_000_000
    )
    prepared = PreparedAnalysis.prepare(
        frame,
        graph=[("t", 1, "y", 0)],
        query=PulseEffect("t", "y"),
        bootstrap=0,
        refute="none",
    )
    result = prepared.estimate()
    assert result.export()


def test_multi_env_frame_prepares_through_pyo3():
    series = _series()
    frame = multi_env([series, series])
    prepared = PreparedAnalysis.prepare(
        frame,
        graph=[("t", 1, "y", 0)],
        query=PulseEffect("t", "y"),
        bootstrap=0,
        refute="none",
    )
    result = prepared.estimate()
    encoded = result.export()
    loaded = ant.artifacts.loads(encoded)
    assert loaded.payload_kind == "analysis_result"
