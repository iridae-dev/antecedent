"""The mediation summary runs the compiled temporal mediation route.

``antecedent.estimation.mediation_effects_summary`` reads the lag-one treatment acting on the
outcome directly and through the contemporaneous mediator. Its components are the decomposition
``antecedent.analyze`` publishes for ``TemporalMediationEffect`` over the temporal DAG of that
structure, number for number, and the least-squares path products of the lag-aligned rows.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.outcomes import require_success


def _series(n: int = 2000, seed: int = 21) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = rng.normal(size=n)
    m = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        m[i] = 0.8 * t[i - 1] + rng.normal(scale=0.2)
        y[i] = 0.25 * t[i - 1] + 0.55 * m[i] + rng.normal(scale=0.1)
    return {"t": t, "m": m, "y": y}


def _ols(columns: list[np.ndarray], response: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(response)), *columns])
    return np.linalg.lstsq(design, response, rcond=None)[0]


def test_the_summary_is_the_compiled_temporal_mediation_decomposition() -> None:
    data = _series()
    summary = antecedent.estimation.mediation_effects_summary(
        list(data), list(data.values()), "t", "m", "y", execution=antecedent.Execution(seed=1)
    )
    result = require_success(
        antecedent.analyze(
            data,
            graph=[("t", 1, "m", 0), ("t", 1, "y", 0), ("m", 0, "y", 0)],
            query=antecedent.TemporalMediationEffect("t", "m", "y", contrast="total"),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    mediation = result.mediation
    assert mediation is not None
    assert summary.total == mediation.total
    assert summary.direct == mediation.direct
    assert summary.mediated == mediation.mediated
    # The independent reading: path products of plain least squares on the lag-aligned rows.
    t, m, y = data["t"], data["m"], data["y"]
    a = _ols([t[:-1]], m[1:])[1]
    outcome = _ols([t[:-1], m[1:]], y[1:])
    assert summary.mediated == pytest.approx(a * outcome[2], abs=1e-8)
    assert summary.direct == pytest.approx(outcome[1], abs=1e-8)
    assert summary.total == pytest.approx(_ols([t[:-1]], y[1:])[1], abs=1e-8)


def test_a_column_the_summary_does_not_name_leaves_it_unchanged() -> None:
    data = _series()
    plain = antecedent.estimation.mediation_effects_summary(
        list(data), list(data.values()), "t", "m", "y", execution=antecedent.Execution(seed=1)
    )
    wider = {**data, "z": np.random.default_rng(5).normal(size=len(data["t"]))}
    widened = antecedent.estimation.mediation_effects_summary(
        list(wider), list(wider.values()), "t", "m", "y", execution=antecedent.Execution(seed=1)
    )
    assert (widened.total, widened.direct, widened.mediated) == (
        plain.total,
        plain.direct,
        plain.mediated,
    )
