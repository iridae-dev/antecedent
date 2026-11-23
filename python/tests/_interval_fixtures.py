"""Small temporal series, one per circular-block resampling family."""

from __future__ import annotations

import antecedent as ant
import numpy as np

from known_truth import temporal_mediation_series, white_noise_pulse_series

BOOTSTRAP = 40


def single_window_data():
    rng = np.random.default_rng(37)
    n = 200
    t = rng.normal(size=n)
    y = np.ones(n)
    y[1:] += 2 * t[:-1] + 0.3 * rng.normal(size=n - 1)
    graph = ant.graph.TemporalDag.from_lagged_edges(["t", "y"], [("t", 1, "y", 0)])
    return {"t": t, "y": y}, graph


def sequential_data():
    rng = np.random.default_rng(11)
    n = 240
    t = rng.normal(size=n)
    y = np.zeros(n)
    for i in range(2, n):
        y[i] = 1 + 2 * t[i - 1] + t[i - 2] + 0.3 * rng.normal()
    graph = ant.graph.TemporalDag.from_lagged_edges(
        ["t", "y"], [("t", 1, "y", 0), ("t", 2, "y", 0)]
    )
    return {"t": t, "y": y}, graph


def mediation_data():
    graph = [("t", 1, "m", 0), ("t", 1, "y", 0), ("m", 0, "y", 0)]
    return temporal_mediation_series(160), graph


def mixture_data(n: int, seed: int):
    return white_noise_pulse_series(n, seed)


def row_block_data():
    rng = np.random.default_rng(11)
    n = 300
    x = rng.integers(0, 2, n).astype(float)
    w = rng.integers(0, 2, n).astype(float)
    y = np.zeros(n)
    y[1:] = np.where(w[:-1] == 1.0, x[:-1], 0.0)
    for t in (9, 33, 71, 120):
        y[t] = 1.0 - y[t]
    graph = ant.graph.TemporalAdmg.from_lagged_edges(
        ["x", "y", "w"], [("x", 1, "y", 0), ("w", 1, "y", 0)], None
    )
    return {"x": x, "y": y, "w": w}, graph
