"""Panel / event / multi-env frames prepare through PyO3."""

from __future__ import annotations

import antecedent as ant
import numpy as np
from antecedent import PulseEffect
from antecedent.data import event, multi_env, panel
from antecedent.outcomes import require_success


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
    prepared = ant.prepare(
        frame,
        graph=[("t", 1, "y", 0)],
        query=PulseEffect("t", "y"),
        refute="none",
        inference=ant.Frequentist(bootstrap=0),
    )
    result = require_success(prepared.estimate())
    encoded = bytes(result.export())
    loaded = ant.artifacts.loads(encoded)
    assert loaded.payload_kind == "analysis_result"


def test_event_frame_prepares_through_pyo3():
    series = _series()
    frame = event(
        series, event_times_ns=np.arange(len(series["t"])) * 1_000_000, align_interval_ns=1_000_000
    )
    prepared = ant.prepare(
        frame,
        graph=[("t", 1, "y", 0)],
        query=PulseEffect("t", "y"),
        refute="none",
        inference=ant.Frequentist(bootstrap=0),
    )
    result = require_success(prepared.estimate())
    assert bytes(result.export())


def test_multi_env_frame_prepares_through_pyo3():
    series = _series()
    frame = multi_env([series, series])
    prepared = ant.prepare(
        frame,
        graph=[("t", 1, "y", 0)],
        query=PulseEffect("t", "y"),
        refute="none",
        inference=ant.Frequentist(bootstrap=0),
    )
    result = require_success(prepared.estimate())
    encoded = bytes(result.export())
    loaded = ant.artifacts.loads(encoded)
    assert loaded.payload_kind == "analysis_result"
