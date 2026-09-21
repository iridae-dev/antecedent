"""Arrow CDI vs NumPy ingest parity for analyze and PreparedAnalysis.

Requires pyarrow. Skipped when unavailable.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("pyarrow")

import antecedent
import pyarrow as pa


def _confounded_scm(n: int = 500, seed: int = 19):
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


def _as_table(data: dict[str, np.ndarray]) -> pa.Table:
    return pa.table({k: pa.array(v, type=pa.float64()) for k, v in data.items()})


def test_analyze_ate_dict_vs_arrow_dense_parity():
    data, edges = _confounded_scm()
    table = _as_table(data)
    q = antecedent.AverageEffect(treatment="t", outcome="y")
    dict_r = antecedent.analyze(data, graph=edges, query=q, latency="interactive", seed=1)
    arrow_r = antecedent.analyze(table, graph=edges, query=q, latency="interactive", seed=1)
    assert abs(dict_r.ate - arrow_r.ate) < 1e-12


def test_prepared_prepare_arrow_estimate_dict_parity():
    """Prepare on Arrow and estimate on dict (same dense float64) must match."""
    data, edges = _confounded_scm()
    table = _as_table(data)
    q = antecedent.AverageEffect(treatment="t", outcome="y")
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        table, graph=edges, query=q, latency="interactive", seed=1
    )
    from_arrow = prepared.estimate(table, seed=1)
    from_dict = prepared.estimate(data, seed=1)
    fresh = antecedent.analyze(data, graph=edges, query=q, latency="interactive", seed=1)
    assert abs(from_arrow.ate - from_dict.ate) < 1e-12
    assert abs(from_dict.ate - fresh.ate) < 1e-12


def test_prepared_prepare_dict_estimate_arrow_parity():
    data, edges = _confounded_scm(n=400, seed=11)
    table = _as_table(data)
    q = antecedent.AverageEffect(treatment="t", outcome="y")
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data, graph=edges, query=q, latency="interactive", seed=1
    )
    from_dict = prepared.estimate(data, seed=1)
    from_arrow = prepared.estimate(table, seed=1)
    assert abs(from_dict.ate - from_arrow.ate) < 1e-12


def test_conditional_effect_dict_vs_arrow_parity():
    # Separate modifier W from confounder Z so the design is full rank.
    n, seed = 450, 23
    rng = random.Random(seed)
    z = np.empty(n, dtype=np.float64)
    w = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = rng.gauss(0.0, 1.0)
        wi = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-(-0.4 + 0.9 * zi)))
        ti = 1.0 if rng.random() < p else 0.0
        yi = 2.0 * ti + zi + 0.5 * wi * ti + rng.gauss(0.0, 0.4)
        z[i] = zi
        w[i] = wi
        t[i] = ti
        y[i] = yi
    data = {"t": t, "y": y, "z": z, "w": w}
    edges = [("z", "t"), ("z", "y"), ("w", "y"), ("t", "y")]
    table = _as_table(data)
    q = antecedent.ConditionalEffect(treatment="t", outcome="y", modifier="w")
    dict_r = antecedent.analyze(data, graph=edges, query=q, latency="interactive", seed=1)
    arrow_r = antecedent.analyze(table, graph=edges, query=q, latency="interactive", seed=1)
    assert abs(dict_r.ate - arrow_r.ate) < 1e-12


def test_arrow_nulls_are_invalid_rows_not_zero():
    """Nullable Arrow must not silently treat null as 0.0."""
    data, edges = _confounded_scm(n=400, seed=3)
    y = data["y"].copy()
    null_idx = np.arange(0, len(y), 10)
    mask = np.zeros(len(y), dtype=bool)
    mask[null_idx] = True
    y_arrow = pa.array(y, type=pa.float64(), mask=mask)
    table = pa.table(
        {
            "t": pa.array(data["t"], type=pa.float64()),
            "y": y_arrow,
            "z": pa.array(data["z"], type=pa.float64()),
        }
    )
    y_zero = y.copy()
    y_zero[null_idx] = 0.0
    zero_filled = {"t": data["t"], "y": y_zero, "z": data["z"]}
    q = antecedent.AverageEffect(treatment="t", outcome="y")
    arrow_r = antecedent.analyze(table, graph=edges, query=q, latency="interactive", seed=1)
    zero_r = antecedent.analyze(zero_filled, graph=edges, query=q, latency="interactive", seed=1)
    assert math.isfinite(arrow_r.ate)
    assert abs(arrow_r.ate - zero_r.ate) > 1e-3


def _response_curve_scm(n: int = 300, seed: int = 29) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    a = 0.7 * x + rng.normal(size=n)
    y = 2.0 * a + x + rng.normal(scale=0.2, size=n)
    return {"x": x, "a": a, "y": y}


def _response_curve_values(data) -> list[float]:
    result = antecedent.analyze(
        data,
        query=antecedent.ResponseCurve("a", "y", grid=[-0.5, 0.0, 0.5]),
        graph=[("x", "a"), ("x", "y"), ("a", "y")],
        seed=1,
    )
    return [float(v) for row in result.response.values for v in row]


@pytest.mark.parametrize("column", ["x", "a", "y"])
@pytest.mark.parametrize("source", ["numpy_nan", "pandas_nan", "arrow_null", "arrow_nan"])
def test_missing_cell_matches_dropped_row_for_response_curve(source, column):
    """A NaN / null cell is excluded like a dropped row, never read as 0.0."""
    data = _response_curve_scm()
    missing = 37
    dropped = {k: np.delete(v, missing) for k, v in data.items()}
    if source == "arrow_nan":
        # A non-null Arrow NaN value is a missing cell too, not an observation.
        holed = {k: v.copy() for k, v in data.items()}
        holed[column][missing] = np.nan
        holed = pa.table({k: pa.array(v, type=pa.float64()) for k, v in holed.items()})
        assert holed.column(column).null_count == 0
    elif source == "arrow_null":
        mask = np.zeros(len(data[column]), dtype=bool)
        mask[missing] = True
        holed = pa.table(
            {
                k: pa.array(v, type=pa.float64(), mask=mask if k == column else None)
                for k, v in data.items()
            }
        )
    else:
        holed = {k: v.copy() for k, v in data.items()}
        holed[column][missing] = np.nan
        if source == "pandas_nan":
            pd = pytest.importorskip("pandas")
            holed = pd.DataFrame(holed)
    assert _response_curve_values(holed) == pytest.approx(
        _response_curve_values(dropped), rel=1e-9, abs=1e-12
    )


@pytest.mark.parametrize("column", ["t", "y", "z"])
def test_arrow_nan_value_matches_dropped_row_for_ate(column):
    """An Arrow NaN value (null_count 0) is dropped like a null, for every role."""
    data, edges = _confounded_scm(n=400, seed=5)
    missing = 37
    dropped = {k: np.delete(v, missing) for k, v in data.items()}
    holed = {k: v.copy() for k, v in data.items()}
    holed[column][missing] = np.nan
    table = _as_table(holed)
    q = antecedent.AverageEffect(treatment="t", outcome="y")
    arrow_r = antecedent.analyze(table, graph=edges, query=q, latency="interactive", seed=1)
    numpy_r = antecedent.analyze(holed, graph=edges, query=q, latency="interactive", seed=1)
    dropped_r = antecedent.analyze(dropped, graph=edges, query=q, latency="interactive", seed=1)
    assert math.isfinite(arrow_r.ate)
    assert arrow_r.ate == pytest.approx(dropped_r.ate, rel=1e-9, abs=1e-12)
    assert arrow_r.ate == pytest.approx(numpy_r.ate, rel=1e-9, abs=1e-12)
