"""Explicit temporal limits cross the execution boundary and govern public requests."""

from dataclasses import FrozenInstanceError, replace

import antecedent as ant
import pytest
from antecedent import TemporalMaterializationLimits
from antecedent.errors import CausalResourceError, CausalTypeError, CausalValueError
from antecedent.intervention import Sequence, Set
from antecedent.outcomes import require_success

from test_temporal_response_api import _EDGES, _fixture_data


@pytest.mark.parametrize(
    "name", ["max_offsets", "max_work", "max_bytes", "max_graph_nodes", "max_graph_edges"]
)
@pytest.mark.parametrize(
    "value, error",
    [
        (None, CausalTypeError),
        (True, CausalTypeError),
        (1.0, CausalTypeError),
        (-1, CausalValueError),
        (2**64, CausalValueError),
    ],
)
def test_temporal_limits_require_finite_u64_integers(name, value, error):
    with pytest.raises(error):
        TemporalMaterializationLimits(**{name: value})


def test_temporal_limits_are_immutable_and_allow_zero():
    limits = TemporalMaterializationLimits(max_offsets=0)
    assert limits.max_offsets == 0
    with pytest.raises(FrozenInstanceError):
        limits.max_offsets = 1
    with pytest.raises(CausalTypeError):
        ant.Execution(temporal_materialization={})


def test_python_passes_each_limit_and_default_to_native(monkeypatch):
    import antecedent.execution as execution_module

    captured = []
    monkeypatch.setattr(
        execution_module._native, "_Execution", lambda **kwargs: captured.append(kwargs)
    )
    ant.Execution()
    ant.Execution(temporal_materialization=TemporalMaterializationLimits(1, 2, 3, 4, 5))
    assert captured[0]["temporal_materialization_limits"] is None
    assert captured[1]["temporal_materialization_limits"] == (1, 2, 3, 4, 5)
    assert captured[1]["seed"] == 1 and captured[1]["threads"] is None


def test_temporal_public_request_respects_explicit_limits():
    query = ant.InterventionResponse(
        "y",
        intervention=Sequence([Set("t", 1.0), Set("t", 1.0)]),
        horizons=[1, 2],
        policy="pulse",
        treatment_lag=1,
    )

    def run(limits):
        return require_success(
            ant.analyze(
                _fixture_data(),
                graph=_EDGES,
                query=query,
                refute=False,
                inference=ant.Frequentist(bootstrap=0),
                execution=ant.Execution(seed=21, temporal_materialization=limits),
            )
        )

    limits = TemporalMaterializationLimits()
    with pytest.raises(CausalResourceError):
        run(replace(limits, max_offsets=0))
    result = run(limits)
    assert result.response is not None
    assert len(result.response.values) == 2


@pytest.mark.parametrize("treatment_lag,horizon_steps", [(2_000_000_000, 1), (1, 2_000_000_000)])
def test_public_temporal_identification_refuses_huge_windows_before_materializing(
    treatment_lag, horizon_steps
):
    from antecedent import _native
    from antecedent.execution import ResourceLimits

    graph = ant.TemporalDag.from_lagged_edges(["t", "y"], [("t", 1, "t", 0), ("t", 0, "y", 0)])
    execution = ant.Execution(
        limits=ResourceLimits(memory_bytes=128 * 1024),
        temporal_materialization=TemporalMaterializationLimits(max_graph_nodes=32, max_work=10_000),
    )
    available = execution._handle.available_memory_bytes
    with pytest.raises(CausalResourceError):
        execution._call(
            _native._identify_structure,
            graph,
            "pulse",
            "t",
            "y",
            policy="pulse",
            treatment_lag=treatment_lag,
            horizon_steps=horizon_steps,
            include_details=False,
        )
    assert execution._handle.available_memory_bytes == available
