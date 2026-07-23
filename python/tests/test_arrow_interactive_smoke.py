"""gate: Arrow CDI is the interactive estimate ingest path (BACKLOG E).

Requires pyarrow. Skipped when unavailable.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("antecedent")

import pyarrow as pa

import antecedent


def _confounded_scm(n: int = 600, seed: int = 7):
    rng = random.Random(seed)
    z = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-(-0.4 + 0.9 * zi)))
        ti = 1.0 if rng.random() < p else 0.0
        yi = 2.0 * ti + zi + rng.gauss(0.0, 0.4)
        z[i] = zi
        t[i] = ti
        y[i] = yi
    return {"t": t, "y": y, "z": z}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_arrow_interactive_zero_copy_and_estimate():
    data, edges = _confounded_scm()
    table = pa.table(
        {
            "t": pa.array(data["t"], type=pa.float64()),
            "y": pa.array(data["y"], type=pa.float64()),
            "z": pa.array(data["z"], type=pa.float64()),
        }
    )
    n_rows = table.num_rows
    n_cols = table.num_columns

    info = causal.load_float64_arrow_c_columns(
        list(table.column_names),
        [table.column(i).combine_chunks() for i in range(n_cols)],
    )
    assert info.bytes_borrowed >= n_rows * n_cols * 8

    arrow_result = causal.analyze(
        table,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        latency="interactive",
        seed=1,
    )
    assert math.isfinite(arrow_result.ate)
    assert abs(arrow_result.ate - 2.0) < 0.5
    assert arrow_result.identification.status
    assert arrow_result.performance.latency_mode == "interactive"
    assert arrow_result.performance.bootstrap_replicates_requested == 0

    # Pandas / dict twin remains correct; CDI must not diverge silently.
    dict_result = causal.analyze(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        latency="interactive",
        seed=1,
    )
    assert abs(arrow_result.ate - dict_result.ate) < 1e-9


def test_arrow_interactive_cancel_and_progress():
    data, edges = _confounded_scm(n=400, seed=11)
    table = pa.table(
        {
            "t": pa.array(data["t"], type=pa.float64()),
            "y": pa.array(data["y"], type=pa.float64()),
            "z": pa.array(data["z"], type=pa.float64()),
        }
    )
    requested = 80
    token = causal.CancellationToken()
    seen = {"bootstrap": False}

    def on_progress(fraction: float, stage: str) -> None:
        if stage == "bootstrap" and not seen["bootstrap"]:
            seen["bootstrap"] = True
            token.cancel()

    partial = causal.analyze(
        table,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        bootstrap=requested,
        refute=False,
        seed=3,
        cancel=token,
        on_progress=on_progress,
    )
    assert math.isfinite(partial.ate)
    assert partial.performance.cancelled
    ok = partial.performance.bootstrap_replicates_ok or 0
    assert ok < requested
