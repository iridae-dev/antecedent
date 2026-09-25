"""Public lifecycle checks for checked discrete scalar effect programs."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
import json
from pathlib import Path


def _path_table() -> dict[str, np.ndarray]:
    cells = [
        (0.0, 0.0, 0.0, 40), (0.0, 0.0, 1.0, 10),
        (0.0, 1.0, 0.0, 10), (0.0, 1.0, 1.0, 40),
        (1.0, 0.0, 0.0, 10), (1.0, 0.0, 1.0, 10),
        (1.0, 1.0, 0.0, 10), (1.0, 1.0, 1.0, 70),
    ]
    return {
        name: np.concatenate([np.full(cell[3], cell[column]) for cell in cells])
        for name, column in (("t", 0), ("m", 1), ("y", 2))
    }


def test_path_specific_program_refreshes_and_replays_independently() -> None:
    data = _path_table()
    prepared = ant.prepare(
        data,
        graph=[("t", "m"), ("m", "y")],
        query=ant.PathSpecificEffect("t", "y", path_nodes=["m"]),
        refute="none",
        bootstrap=0,
    )
    first = prepared.estimate(data)
    assert np.isfinite(first.effect)
    first_replay = ant.artifacts.accept(first.export())
    assert first_replay["accepts_as_verified_program"] == "true"

    changed = {**data, "y": 1.0 - data["y"]}
    refreshed = prepared.refresh(changed)
    assert refreshed.effect == pytest.approx(-first.effect, abs=1e-12)
    replay = ant.artifacts.accept(refreshed.export())
    assert replay["accepts_as_verified_program"] == "true"
    assert replay["program"] == first_replay["program"]
    assert refreshed.data_snapshot_id != first.data_snapshot_id


def test_admg_response_grid_replays_each_member_after_refresh() -> None:
    pin = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "conformance/estimate/admg_frontdoor_functional/expected.json"
        ).read_text(encoding="utf-8")
    )
    data = {
        name: np.concatenate(
            [np.full(int(cell["count"]), float(cell[name])) for cell in pin["contingency_table"]]
        )
        for name in pin["columns"]
    }
    graph = ant.Admg.from_edges(
        pin["columns"],
        [tuple(edge) for edge in pin["graph"]["directed_edges"]],
        bidirected=[tuple(edge) for edge in pin["graph"]["bidirected_edges"]],
    )
    prepared = ant.prepare(
        data,
        graph=graph,
        query=ant.ResponseCurve("t", "y", grid=[0.0, 1.0]),
        identifier="general.id",
        estimator="functional.effect",
        refute="none",
        bootstrap=0,
    )
    result = prepared.estimate(data)
    assert np.asarray(result.response.values).flatten() == pytest.approx([0.314, 0.596], abs=1e-12)
    refreshed = prepared.refresh(data)
    assert np.asarray(refreshed.response.values).flatten() == pytest.approx(
        [0.314, 0.596], abs=1e-12
    )
    receipt = ant.artifacts.accept(refreshed.export())
    assert receipt["accepts_as_verified_program"] == "true"
    assert receipt["unresolved"] == ""
