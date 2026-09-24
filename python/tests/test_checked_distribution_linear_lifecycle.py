"""Public Python checks for the first two sealed static route families."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest


def _distribution_table() -> dict[str, np.ndarray]:
    cells = [
        (0.0, 0.0, 0.0, 80), (0.0, 0.0, 1.0, 20),
        (0.0, 1.0, 0.0, 20), (0.0, 1.0, 1.0, 80),
        (1.0, 0.0, 0.0, 60), (1.0, 0.0, 1.0, 40),
        (1.0, 1.0, 0.0, 40), (1.0, 1.0, 1.0, 60),
    ]
    return {
        name: np.concatenate([np.full(cell[3], cell[column]) for cell in cells])
        for name, column in (("z", 0), ("t", 1), ("y", 2))
    }


def test_distribution_program_refresh_and_independent_replay() -> None:
    data = _distribution_table()
    prepared = ant.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.InterventionalDistribution("y", interventions={"t": 1.0}),
        refute="none",
        bootstrap=0,
    )
    first = prepared.estimate(data)
    assert first.effect == pytest.approx(0.7, abs=1e-12)
    accepted = ant.artifacts.accept(first.export())
    assert accepted["accepts_as_verified_program"] == "true"

    changed = {**data, "y": 1.0 - data["y"]}
    refreshed = prepared.refresh(changed)
    assert refreshed.effect == pytest.approx(0.3, abs=1e-12)
    accepted_refresh = ant.artifacts.accept(refreshed.export())
    assert accepted_refresh["accepts_as_verified_program"] == "true"
    assert accepted_refresh["program"] == accepted["program"]
    assert refreshed.data_snapshot_id != first.data_snapshot_id


def test_linear_lowering_refresh_reports_exact_numeric_dependency() -> None:
    rng = np.random.default_rng(73)
    z = rng.normal(size=512)
    t = (rng.uniform(size=512) < 1 / (1 + np.exp(-z))).astype(float)
    data = {"z": z, "t": t, "y": 2 * t + z + rng.normal(size=512)}
    prepared = ant.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        estimator="linear.adjustment.ate",
        refute="none",
        bootstrap=0,
    )
    first = prepared.estimate(data)
    assert first.effect == pytest.approx(2.0, abs=0.2)
    changed = {**data, "y": data["y"] + 0.5 * t}
    refreshed = prepared.refresh(changed)
    assert refreshed.effect == pytest.approx(first.effect + 0.5, abs=1e-10)
    consumed = ant.artifacts.accept(refreshed.export())
    assert consumed["accepts_as_verified_program"] == "false"
    assert "dependencies.linear_fit_sufficient_statistics" in consumed["unresolved"]
