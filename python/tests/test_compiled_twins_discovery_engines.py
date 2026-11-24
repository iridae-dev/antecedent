"""The direct discovered-transport entries warn and equal their compiled programs.

Each direct entry runs the static selective search and the transport kernels outside the
compiler. Its replacement runs the same search as the compiled ``DiscoverStatic`` program and the
same kernels inside the ``DiscoveredClassTransport`` program, so the envelope, the estimate, the
trial-compatible estimate, the confidence set and its replay are identical.

Fixtures are those of ``test_discovered_transport.py`` and ``test_trial_compatible_transport.py``:
a linear-Gaussian chain, a binary collider whose target mechanism of ``z`` differs (the target
response is ``sum_z P*(z) P(y | do(x), z)``, 0.24 and 0.645), and a completely randomized binary
trial of 200 rows whose control and treated means are 0.2 and 0.8.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from antecedent import selective_discovery as sd
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.transport import discovered
from antecedent.transport import trial_compatible as tc

N = 4000


def _chain(seed: int, n: int = N) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 0.8 * x + 0.6 * rng.normal(size=n)
    z = 0.8 * y + 0.6 * rng.normal(size=n)
    return {"x": x, "y": y, "z": z}


def _binary_collider(rows: int, z_rate: float, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = (rng.random(rows) < 0.5).astype(np.float64)
    z = (rng.random(rows) < z_rate).astype(np.float64)
    y = (rng.random(rows) < 0.1 + 0.3 * x + 0.2 * z + 0.15 * x * z).astype(np.float64)
    return {"x": x, "y": y, "z": z}


def _contract(**kwargs: object) -> sd.StaticContract:
    return sd.StaticContract(variables=("x", "y", "z"), alpha=0.001, fdr=None, seed=3, **kwargs)  # type: ignore[arg-type]


QUERY = {
    "outcome": "z",
    "treatment": "x",
    "grid": [0.0, 1.0],
    "selection_targets": ["y"],
    "source_population": "trial",
    "target_population": "clinic",
    "source_experiments": ["x"],
}

ROWS = 200
TREATMENT = np.array([float(i >= 100) for i in range(ROWS)])
OUTCOME = np.array([float(i < 20 or 100 <= i < 180) for i in range(ROWS)])
SOURCE = {"x": TREATMENT, "y": OUTCOME}
INDEPENDENT = {"x": TREATMENT, "y": np.array([float(i % 2 == 0) for i in range(ROWS)])}
TRIAL_CONTRACT = sd.StaticContract(
    variables=("x", "y"), algorithm="fci", ci_test="gsquared", fdr=None, seed=71
)
TRIAL = {
    "outcome": "y",
    "treatment": "x",
    "selection_targets": [],
    "source_population": "trial",
    "target_population": "target",
}
SET = {
    **TRIAL,
    "alpha": 0.01,
    "support_lower": 0.0,
    "support_upper": 1.0,
    "bootstrap_replicates": 64,
}


def test_discover_and_transport_warns_and_equals_the_compiled_envelope() -> None:
    for contract in (_contract(), _contract(algorithm="fci")):
        with pytest.warns(AntecedentDeprecationWarning, match="discover_and_transport"):
            direct = discovered.discover_and_transport(contract, _chain(1), **QUERY)  # type: ignore[arg-type]
        compiled = discovered.analyze_discovered_transport(contract, _chain(1), **QUERY)  # type: ignore[arg-type]
        assert compiled == direct
        assert compiled.envelope.identity == direct.envelope.identity
    # The chain's class: the three DAGs without a collider.
    assert (
        len(
            discovered.analyze_discovered_transport(
                _contract(), _chain(1), **QUERY
            ).envelope.members
        )
        == 3
    )  # type: ignore[arg-type]


def test_a_refused_transport_request_is_the_same_refusal_with_the_discovery_attached() -> None:
    with (
        pytest.warns(AntecedentDeprecationWarning),
        pytest.raises(discovered.DiscoveredTransportRefusal) as direct,
    ):
        discovered.discover_and_transport(_contract(), _chain(1), **QUERY, member_bound=0)  # type: ignore[arg-type]
    with pytest.raises(discovered.DiscoveredTransportRefusal) as compiled:
        discovered.analyze_discovered_transport(_contract(), _chain(1), **QUERY, member_bound=0)  # type: ignore[arg-type]
    assert compiled.value.detail == direct.value.detail
    assert compiled.value.code == direct.value.code
    assert compiled.value.outcome == direct.value.outcome
    assert compiled.value.report == direct.value.report


def test_estimate_discovered_transport_warns_and_equals_the_compiled_estimate() -> None:
    source = _binary_collider(20_000, 0.3, 5)
    target = _binary_collider(20_000, 0.7, 6)
    arguments: dict[str, Any] = {
        "outcome": "y",
        "treatment": "x",
        "grid": [0.0, 1.0],
        "selection_targets": ["z"],
        "source_population": "trial",
        "target_population": "clinic",
    }
    declared = _contract(ci_test="gsquared")
    with pytest.warns(AntecedentDeprecationWarning, match="estimate_discovered_transport"):
        direct = discovered.estimate_discovered_transport(declared, source, target, **arguments)
    compiled = discovered.analyze_discovered_transport_estimate(
        declared, source, target, **arguments
    )
    assert compiled == direct
    assert bytes(compiled.export()) == bytes(direct.export())
    # Independent check: the target response 0.24 and 0.645 within four standard errors.
    for (_, low, _), expected in zip(compiled.answer.levels, (0.24, 0.645), strict=True):
        assert low == pytest.approx(expected, abs=0.02)


def test_the_trial_compatible_estimate_warns_and_equals_the_compiled_estimate() -> None:
    arguments = {**TRIAL, "grid": [0.0, 1.0], "alpha": 0.01}
    with pytest.warns(AntecedentDeprecationWarning, match="estimate_trial_compatible"):
        direct = tc.estimate_trial_compatible_discovered_transport(
            TRIAL_CONTRACT, SOURCE, SOURCE, **arguments
        )
    compiled = tc.analyze_trial_compatible_discovered_transport(
        TRIAL_CONTRACT, SOURCE, SOURCE, **arguments
    )
    assert compiled == direct
    assert bytes(compiled.export()) == bytes(direct.export())
    # The hand oracle: control mean 20/100, treated mean 80/100 in every retained member.
    assert compiled.answer.levels[0][1] == pytest.approx(0.2, abs=1e-12)
    assert compiled.answer.levels[1][1] == pytest.approx(0.8, abs=1e-12)


@pytest.mark.parametrize("discovery", [None, INDEPENDENT])
def test_the_confidence_set_and_its_replay_warn_and_equal_the_compiled_set(discovery) -> None:
    with pytest.warns(AntecedentDeprecationWarning, match="confidence_set"):
        direct = tc.estimate_trial_compatible_transport_confidence_set(
            TRIAL_CONTRACT, SOURCE, SOURCE, **SET, discovery=discovery
        )
    compiled = tc.analyze_trial_compatible_transport_confidence_set(
        TRIAL_CONTRACT, SOURCE, SOURCE, **SET, discovery=discovery
    )
    assert compiled == direct
    assert (compiled.lower, compiled.upper) == (direct.lower, direct.upper)
    assert compiled.calibration == direct.calibration == "not_bound_to_this_execution"
    receipt = bytes(direct.export())
    with pytest.warns(AntecedentDeprecationWarning, match="confidence_set_against"):
        direct_replay = tc.verify_trial_compatible_transport_confidence_set_against(
            receipt, TRIAL_CONTRACT, SOURCE, SOURCE, **SET, discovery=discovery
        )
    compiled_replay = tc.replay_trial_compatible_transport_confidence_set(
        receipt, TRIAL_CONTRACT, SOURCE, SOURCE, **SET, discovery=discovery
    )
    assert compiled_replay == direct_replay
    if discovery is not None:
        # The trial rejects the class discovered on independent columns: the logical set.
        assert compiled.kind == "logical_support_bound"
        assert (compiled.lower, compiled.upper) == (-1.0, 1.0)


def test_a_receipt_that_does_not_replay_is_the_same_refusal_through_the_program() -> None:
    receipt = bytearray(
        bytes(
            tc.analyze_trial_compatible_transport_confidence_set(
                TRIAL_CONTRACT, SOURCE, SOURCE, **SET
            ).export()
        )
    )
    receipt[-1] ^= 1
    with (
        pytest.warns(AntecedentDeprecationWarning),
        pytest.raises(Exception) as direct,  # noqa: PT011 - the same class is asserted below
    ):
        tc.verify_trial_compatible_transport_confidence_set_against(
            bytes(receipt), TRIAL_CONTRACT, SOURCE, SOURCE, **SET
        )
    with pytest.raises(Exception) as compiled:  # noqa: PT011 - the same class is asserted below
        tc.replay_trial_compatible_transport_confidence_set(
            bytes(receipt), TRIAL_CONTRACT, SOURCE, SOURCE, **SET
        )
    assert type(compiled.value) is type(direct.value)
    assert str(compiled.value) == str(direct.value)


POPULATION_CONTRACT = sd.StaticContract(variables=("x", "y", "z"), alpha=0.001, fdr=None, seed=5)


def _collider(seed: int, n: int = N) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 0.8 * x + 0.8 * z + 0.5 * rng.normal(size=n)
    return {"x": x, "y": y, "z": z}


def test_the_compiled_per_population_run_equals_the_driver_run() -> None:
    populations = {"south": _collider(2), "north": _chain(1)}
    direct = sd.PerPopulationDiscovery().run(POPULATION_CONTRACT, populations)
    driver = sd.PerPopulationDiscovery()
    compiled = driver.compiled_run(POPULATION_CONTRACT, populations)
    assert compiled == direct
    assert driver.report is not None and driver.report.identity == direct.identity
    # The hand oracle: both edges in both populations, marked differently; the collider
    # compels x -> y <- z.
    assert len(compiled.differing_edges) == 2
    assert compiled.edges[0].marks["south"] == ("tail", "arrow")
    assert compiled.edges[1].marks["south"] == ("arrow", "tail")
    assert driver.outcome("north") is not None


def test_a_pooled_request_is_the_same_refusal_through_the_compiled_program() -> None:
    rows = _chain(1)
    driver = sd.PerPopulationDiscovery()
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as direct:
        sd.PerPopulationDiscovery().run(POPULATION_CONTRACT, {"north": rows, "south": rows})
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as compiled:
        driver.compiled_run(POPULATION_CONTRACT, {"north": rows, "south": rows})
    assert compiled.value.detail == direct.value.detail
    assert compiled.value.detail == "selective_discovery.pooled_population_data"
    assert driver.report is None
