"""``PulseEffect`` / ``SustainedEffect`` ``max_history_lag`` bounds the temporal unfolding."""

from __future__ import annotations

from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalIdentifyError

# ------------------------------------------------------------ max_history_lag


def _autoregressive(seed: int, n: int = 1000) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = np.zeros(n)
    y = np.zeros(n)
    for i in range(2, n):
        t[i] = 0.6 * t[i - 1] + 0.8 * z[i - 1] + 0.5 * rng.normal()
        y[i] = 0.8 * t[i - 1] + 0.9 * z[i - 2] + 0.5 * rng.normal()
    return {"t": t, "y": y, "z": z}


AUTOREGRESSIVE_DAG = [("t", 1, "t", 0), ("z", 1, "t", 0), ("z", 2, "y", 0), ("t", 1, "y", 0)]


@pytest.mark.parametrize("query_type", [ant.PulseEffect, ant.SustainedEffect])
def test_max_history_lag_is_a_query_field(query_type: Any) -> None:
    assert query_type("t", "y").max_history_lag is None
    assert query_type("t", "y", max_history_lag=3).max_history_lag == 3
    with pytest.raises(ValueError):
        query_type("t", "y", max_history_lag=-1)


def test_max_history_lag_reaches_rust_and_bounds_the_unfolding() -> None:
    data = _autoregressive(1)
    unbounded = ant.analyze(
        data, graph=AUTOREGRESSIVE_DAG, query=ant.PulseEffect("t", "y", treatment_lag=1)
    )
    # The treatment's parents sit two steps back: a one-step cap excludes them,
    # and the refusal names the field that lifts it.
    with pytest.raises(CausalIdentifyError) as caught:
        ant.analyze(
            data,
            graph=AUTOREGRESSIVE_DAG,
            query=ant.PulseEffect("t", "y", treatment_lag=1, max_history_lag=1),
        )
    assert "raise max_history_lag" in str(caught.value)
    capped = ant.analyze(
        data,
        graph=AUTOREGRESSIVE_DAG,
        query=ant.PulseEffect("t", "y", treatment_lag=1, max_history_lag=2),
    )
    assert capped.answer == unbounded.answer
    capped_report = capped.inspect().to_dict()
    assert capped_report["target_id"] != unbounded.inspect().to_dict()["target_id"]
    offsets = [
        c["offset"] for c in capped_report["identification"]["payload"]["adjustment_coordinates"]
    ]
    assert offsets and all(offset >= -2 for offset in offsets)
    loaded = ant.load(capped.export())
    assert loaded.inspect().to_dict()["target_id"] == capped_report["target_id"]


def test_identify_takes_max_history_lag() -> None:
    graph = ant.TemporalDag.from_lagged_edges(["t", "y", "z"], AUTOREGRESSIVE_DAG)
    identified = ant.identify(graph=graph, query=ant.PulseEffect("t", "y", treatment_lag=1))
    assert identified.adjustment_set == ["t", "z"]
    with pytest.raises(CausalIdentifyError) as caught:
        ant.identify(
            graph=graph, query=ant.PulseEffect("t", "y", treatment_lag=1, max_history_lag=1)
        )
    assert "max_history_lag" in str(caught.value)
