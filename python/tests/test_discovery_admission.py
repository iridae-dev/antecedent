"""A finished discovery is admitted. A live search config still searches."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.accepted_graph import AcceptedGraph
from antecedent.discovery import LiNGAM, PC
from antecedent.estimation import _PreparedEngine


def _chain(n: int = 120, seed: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=n)
    t = 0.8 * z + rng.uniform(-1.0, 1.0, size=n)
    y = 1.5 * t + 0.6 * z + rng.uniform(-1.0, 1.0, size=n)
    return {"z": z, "t": t, "y": y}


def _prepare(data, discovery):
    return antecedent.prepare(
        data,
        query=antecedent.AverageEffect("t", "y"),
        discovery=discovery,
        inference=antecedent.Frequentist(bootstrap=0),
        refute=False,
        accept_discovered=True,
        execution=antecedent.Execution(seed=1),
    )


def _program(study) -> str:
    contract = study._engine._native.contract()
    assert contract["structure_source"] == "accepted"
    assert contract["program"]
    return contract["program"]


def test_a_second_prepare_admits_the_finished_result(monkeypatch):
    data = _chain()
    result = LiNGAM().run(data, execution=antecedent.Execution(seed=1))
    assert result.route is not None
    identity = result.route.program_identity
    calls = {"n": 0}
    original = antecedent.discovery._discover_lingam

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(antecedent.discovery, "_discover_lingam", wrapped)
    first = _prepare(data, result)
    second = _prepare(data, result)
    via_route = _prepare(data, result.route)
    assert calls["n"] == 0
    assert result.route.program_identity == identity
    assert _program(first) == _program(second) == _program(via_route)
    _graph, provenance = result.route.accepted_graph(accept_discovered=True)
    contract = first._engine._native.contract()
    assert contract["discovery_algorithm"] == provenance["method"] == "direct_lingam"

    held = AcceptedGraph.from_discovery(result, algorithm_id=result.algorithm_id)
    assert held.algorithm_id == result.algorithm_id.lower().replace("pcmci_plus", "pcmci+")
    assert held.background_knowledge == result.background_knowledge
    supplied = antecedent.prepare(
        data,
        query=antecedent.AverageEffect("t", "y"),
        graph=held,
        inference=antecedent.Frequentist(bootstrap=0),
        refute=False,
        execution=antecedent.Execution(seed=1),
    )
    assert supplied._engine._native.contract()["program"] != _program(first)


def test_a_different_table_is_refused():
    data = _chain()
    result = LiNGAM().run(data, execution=antecedent.Execution(seed=1))
    with pytest.raises(antecedent.CausalError, match="different table") as caught:
        _prepare(_chain(seed=9), result)
    assert caught.value.reason_code == "invalid_argument"


def test_a_live_config_still_searches(monkeypatch):
    calls = {"n": 0}
    original = antecedent.discovery._discover_pc

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(antecedent.discovery, "_discover_pc", wrapped)
    try:
        _PreparedEngine.prepare(
            _chain(n=40),
            query=antecedent.AverageEffect("t", "y"),
            discovery=PC(max_cond_size=0),
            inference=antecedent.Frequentist(bootstrap=0),
            refute=False,
            accept_discovered=True,
            seed=1,
        )
    except Exception:
        pass
    assert calls["n"] == 1
