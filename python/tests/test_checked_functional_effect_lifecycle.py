"""Public lifecycle checks for checked discrete scalar effect programs."""

from __future__ import annotations

import json

import antecedent as ant
import numpy as np
import pytest
from antecedent.outcomes import require_success

from _repo_text import REPO_ROOT


def _path_table() -> dict[str, np.ndarray]:
    cells = [
        (0.0, 0.0, 0.0, 40),
        (0.0, 0.0, 1.0, 10),
        (0.0, 1.0, 0.0, 10),
        (0.0, 1.0, 1.0, 40),
        (1.0, 0.0, 0.0, 10),
        (1.0, 0.0, 1.0, 10),
        (1.0, 1.0, 0.0, 10),
        (1.0, 1.0, 1.0, 70),
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
        inference=ant.Frequentist(bootstrap=0),
    )
    original_inspection = prepared.inspect()
    first = require_success(prepared.estimate())
    # The chain has no direct edge, so the all-path natural effect is the total effect: the
    # difference of the observed outcome means (0.8 - 0.5), unconfounded in this table.
    treated = data["t"] == 1.0
    assert first.effect == pytest.approx(
        data["y"][treated].mean() - data["y"][~treated].mean(), abs=1e-12
    )
    assert first.effect == pytest.approx(0.3, abs=1e-12)
    first_replay = ant.artifacts.accept(bytes(first.export()))
    assert first_replay["accepts_as_verified_program"] == "true"

    changed = {**data, "y": 1.0 - data["y"]}
    refreshed = require_success(prepared.rebind(data=changed).estimate())
    assert refreshed.effect == pytest.approx(-first.effect, abs=1e-12)
    replay = ant.artifacts.accept(bytes(refreshed.export()))
    assert replay["accepts_as_verified_program"] == "true"
    assert replay["program"] == first_replay["program"]
    assert refreshed.data_snapshot_id != first.data_snapshot_id
    assert prepared.inspect() == original_inspection
    assert first.study.inspect().data_snapshot_id == first.data_snapshot_id
    assert require_success(prepared.estimate()).effect == pytest.approx(first.effect)


def test_admg_response_grid_replays_each_member_after_refresh() -> None:
    pin = json.loads(
        (REPO_ROOT / "conformance/estimate/admg_frontdoor_functional/expected.json").read_text(
            encoding="utf-8"
        )
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
        inference=ant.Frequentist(bootstrap=0),
    )
    original_inspection = prepared.inspect()
    result = require_success(prepared.estimate())
    assert np.asarray(result.response.values).flatten() == pytest.approx([0.314, 0.596], abs=1e-12)
    refreshed = require_success(prepared.rebind(data=data).estimate())
    assert np.asarray(refreshed.response.values).flatten() == pytest.approx(
        [0.314, 0.596], abs=1e-12
    )
    receipt = ant.artifacts.accept(bytes(refreshed.export()))
    assert receipt["accepts_as_verified_program"] == "true"
    assert receipt["unresolved"] == ""
    assert prepared.inspect() == original_inspection
    assert result.study.inspect().data_snapshot_id == result.data_snapshot_id
    assert require_success(prepared.estimate()).response.values == result.response.values
