"""Batch multi-query helper (BACKLOG E)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("causal")
import causal


def _two_treatment_scm(n: int = 500, seed: int = 9):
    rng = random.Random(seed)
    z = np.empty(n, dtype=np.float64)
    t1 = np.empty(n, dtype=np.float64)
    t2 = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = rng.gauss(0.0, 1.0)
        p1 = 1.0 / (1.0 + math.exp(-(-0.3 + 0.8 * zi)))
        p2 = 1.0 / (1.0 + math.exp(-(-0.2 + 0.7 * zi)))
        a = 1.0 if rng.random() < p1 else 0.0
        b = 1.0 if rng.random() < p2 else 0.0
        z[i] = zi
        t1[i] = a
        t2[i] = b
        y[i] = 2.0 * a + 1.5 * b + zi + 0.4 * rng.gauss(0.0, 1.0)
    data = {"t1": t1, "t2": t2, "y": y, "z": z}
    edges = [("z", "t1"), ("z", "t2"), ("z", "y"), ("t1", "y"), ("t2", "y")]
    return data, edges


def test_analyze_many_matches_solo():
    data, edges = _two_treatment_scm()
    q1 = causal.AverageEffect(treatment="t1", outcome="y")
    q2 = causal.AverageEffect(treatment="t2", outcome="y")
    batch = causal.analyze_many(
        data,
        graph=edges,
        queries=[q1, q2],
        refute=False,
        bootstrap=0,
        seed=3,
    )
    assert len(batch) == 2
    solo1 = causal.analyze(
        data, graph=edges, query=q1, refute=False, bootstrap=0, seed=3
    )
    solo2 = causal.analyze(
        data, graph=edges, query=q2, refute=False, bootstrap=0, seed=3
    )
    assert abs(batch[0].ate - solo1.ate) < 1e-12
    assert abs(batch[1].ate - solo2.ate) < 1e-12
    assert abs(batch[0].ate - 2.0) < 0.45
    assert abs(batch[1].ate - 1.5) < 0.45
