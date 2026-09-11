"""Shared known-truth mixture fixtures for the 1.1 honesty-pass tests."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import antecedent
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
PIN = json.loads((_ROOT / "conformance/bayesian/known_truth_mixtures/expected.json").read_text())
STATIC = PIN["static_average_effect"]
TEMPORAL = PIN["temporal_effect"]
TEMPORAL_MULTI = PIN["temporal_sustained_multistep"]
TEMPORAL_MEDIATION = PIN["temporal_mediation"]
BAYES = antecedent.Bayesian(backend="conjugate", n_draws=256, prior_scale=1_000_000.0)
FREQ = antecedent.Frequentist()


def set_edge(mask: int, n: int, src: int, dst: int) -> int:
    bit = src * (n - 1) + (dst if dst < src else dst - 1)
    return mask | (1 << bit)


def static_data(n: int) -> dict[str, np.ndarray]:
    assert n % 16 == 0
    treatment: list[float] = []
    outcome: list[float] = []
    confounder: list[float] = []
    for _ in range(n // 16):
        for z, t, count in ((0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)):
            for row in range(count):
                epsilon = -0.2 if row % 2 == 0 else 0.2
                treatment.append(t)
                confounder.append(z)
                outcome.append(2.0 * t + 2.0 * z + epsilon)
    return {
        "t": np.asarray(treatment, dtype=np.float64),
        "y": np.asarray(outcome, dtype=np.float64),
        "z": np.asarray(confounder, dtype=np.float64),
    }


def white_noise_pulse_series(n: int, seed: int) -> dict[str, np.ndarray]:
    pressure = np.empty(n, dtype=np.float64)
    defect = np.zeros(n, dtype=np.float64)
    state = seed
    for t in range(n):
        state = (state * 6_364_136_223_846_793_005 + 1) & 0xFFFFFFFFFFFFFFFF
        u = (state >> 33) / float(1 << 31)
        pressure[t] = u * 2.0 - 1.0
        if t > 0:
            defect[t] = 0.9 * pressure[t - 1]
    return {"pressure": pressure, "defect": defect}


def static_posterior() -> Any:
    weights = [float(w) for w in STATIC["posterior_weights"]]
    direct = set_edge(0, 3, 0, 1)
    adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1), 3, 2, 0), 3, 2, 1)
    unidentified = set_edge(0, 3, 1, 0)
    return antecedent.discovery.GraphPosterior.from_atoms(
        ["t", "y", "z"],
        weights,
        [direct, adjusted, unidentified],
    )


def temporal_mediation_series(n: int) -> dict[str, np.ndarray]:
    t = np.empty(n, dtype=np.float64)
    m = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        t[i] = np.sin(0.071 * i) + 0.35 * np.cos(0.137 * i)
        if i > 0:
            m[i] = 0.8 * t[i - 1] + 0.12 * np.sin(0.43 * i)
            y[i] = 0.25 * t[i - 1] + 0.55 * m[i] + 0.09 * np.cos(0.29 * i)
    return {"t": t, "m": m, "y": y}


def temporal_mediation_posterior() -> Any:
    identified = TEMPORAL_MEDIATION["identified_atom"]
    unidentified = TEMPORAL_MEDIATION["unidentified_atom"]
    return antecedent.discovery.GraphPosterior.from_atoms(
        ["t", "m", "y"],
        [float(w) for w in TEMPORAL_MEDIATION["posterior_weights"]],
        [
            int(identified["contemporaneous_mask"]),
            int(unidentified["contemporaneous_mask"]),
        ],
        lagged_edge_marginals=[float(v) for v in TEMPORAL_MEDIATION["lagged_edge_marginals"]],
        lag_masks=[int(identified["lag_mask"]), int(unidentified["lag_mask"])],
        max_lag=1,
    )


def temporal_posterior() -> Any:
    identified = TEMPORAL["identified_atom"]
    unidentified = TEMPORAL["unidentified_atom"]
    return antecedent.discovery.GraphPosterior.from_atoms(
        ["pressure", "defect"],
        [float(w) for w in TEMPORAL["posterior_weights"]],
        [int(identified["contemporaneous_mask"]), int(unidentified["contemporaneous_mask"])],
        lagged_edge_marginals=[float(v) for v in TEMPORAL["lagged_edge_marginals"]],
        lag_masks=[int(identified["lag_mask"]), int(unidentified["lag_mask"])],
        max_lag=1,
    )
