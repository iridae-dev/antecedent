# SPDX-License-Identifier: MIT OR Apache-2.0
"""Executable cyclic preparation, ownership, selection and independent consumption."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import antecedent as ant
import pytest
from antecedent import _native
from antecedent.cyclic import (
    Equation,
    EquilibriumModel,
    EvidenceSchema,
    EvidenceVariable,
    LinearCyclicModel,
    World,
    WorldSelector,
)
from antecedent.errors import CausalCancelledError, CausalResourceError
from antecedent.execution import Execution, ResourceLimits
from antecedent.intervention import Set
from antecedent.outcomes import Outcome, require_success


def _fitted():
    model = EquilibriumModel(
        (Equation("A", 0.0, (0.0, 0.4), "identity"), Equation("B", 0.0, (0.5, 0.0), "identity")),
        (1.0, 0.0, 0.0, 1.0),
    )
    rows = [{"values": [x, 0.7 * x], "clamps": [{"variable": 0, "value": x}]} for x in (-1.0, 1.0)]
    return model.prepare(
        [[]], [[0.0, 0.0]], parameters=[{"kind": "weight", "equation": 1, "parent": 0}], rows=rows
    )


@pytest.mark.parametrize("family_first", [False, True])
def test_system_family_views_share_native_fit_in_both_directions(family_first):
    source = _fitted()
    family = source.to_family()
    assert source.inspect()["fits_performed"] == 0
    assert family.inspect()["preparation_evaluations"] == 0
    first, second = (family, source) if family_first else (source, family)
    one = require_success(first.execute())
    two = require_success(second.execute())
    assert one.execution_facts.observed_model_fits == 1
    assert two.execution_facts.observed_model_fits == 0
    branch = source.reconfigure([[{"variable": 0, "value": 2.0}]], [[0.0, 0.0]])
    changed = require_success(branch.execute())
    assert changed.execution_facts.observed_model_fits == 0
    assert changed.to_dict()["worlds"][0][0]["values"][1] == pytest.approx(1.4, abs=1e-4)
    rebound = source.rebind(
        [{"values": [x, 0.6 * x], "clamps": [{"variable": 0, "value": x}]} for x in (-1.0, 1.0)]
    )
    assert require_success(rebound.execute()).execution_facts.observed_model_fits == 1
    assert source.inspect()["fits_performed"] == 1


def test_typed_world_branch_selects_units_population_and_replays_without_live_source(tmp_path):
    model = LinearCyclicModel(("A", "B"), (0.0, 0.4, 0.5, 0.0), (1.0, 0.0, 0.0, 1.0))
    original = model.worlds(
        [World(1, Set("A", 1.0))],
        component_id="loop",
        source_id="specified",
        certifying_evidence=["analytic"],
    )
    branch = original.reconfigure(worlds=[World(2, Set("A", 2.0))])
    assert original.inspect()["worlds"] == [1]
    assert branch.inspect()["worlds"] == [2]
    schema = EvidenceSchema(
        "feedback",
        "specified population",
        (EvidenceVariable("A", "meter"), EvidenceVariable("B", "meter")),
    )
    outcome = branch.evidence(schema, [WorldSelector(2, "B")])
    assert isinstance(outcome, Outcome) and outcome.accepted
    evidence = require_success(outcome)
    column = evidence.to_dict()["columns"][0]
    # World 2 hard-clamps A at 2, so B = 0.5 * 2 = 1 with unit structural noise variance.
    assert column["samples"] == pytest.approx([1.0])
    assert column["diagnostics"]["variance"] == pytest.approx(1.0)
    assert evidence.quantities[0].units == "meter"
    assert evidence.quantities[0].population_id == "specified population"
    expected = evidence.identity
    path = tmp_path / "evidence.ant"
    path.write_bytes(bytes(evidence.export()))
    del evidence, branch, original, model, outcome
    script = f"""import json,pathlib
from antecedent import load
from antecedent.outcomes import require_success
value=require_success(load(pathlib.Path({str(path)!r}).read_bytes(),expected={expected!r}))
assert value.acceptance.verified and value.acceptance.identity_bound
assert value.acceptance.replayable and not value.acceptance.resumable
print(json.dumps(value.to_dict()['columns'][0]['samples']))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == pytest.approx([1.0])


def test_execution_controls_reach_native_preparation_and_execution():
    source = _fitted()
    with pytest.raises(CausalResourceError):
        source.execute(execution=Execution(limits=ResourceLimits(1)))
    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError):
        source.to_family(execution=Execution(cancel=token))
    with pytest.raises(CausalCancelledError):
        source.execute(execution=Execution(cancel=token))
    assert source.inspect()["fits_performed"] == 0


def _cyclic_routes(source, family, result, artifact, identity, schema):
    """Every cyclic native entry that takes the execution's byte limit and token."""
    from antecedent.cyclic import consume_artifact

    return {
        "execute": lambda run: source.execute(execution=run),
        "to_family": lambda run: source.to_family(execution=run),
        "rebind": lambda run: source.rebind(
            [{"values": [1.0, 0.7], "clamps": [{"variable": 0, "value": 1.0}]}], execution=run
        ),
        "family_execute": lambda run: family.execute(execution=run),
        "family_evidence": lambda run: family.evidence(
            schema, [WorldSelector(1, "B")], execution=run
        ),
        "result_export": lambda run: result.export(execution=run),
        "consume": lambda run: consume_artifact(
            artifact, expected_identity=identity, execution=run
        ),
    }


@pytest.mark.parametrize(
    ("limits", "token_fired", "error"),
    [
        (ResourceLimits(memory_bytes=1), False, CausalResourceError),
        (ResourceLimits(memory_bytes=1 << 30), True, CausalCancelledError),
    ],
    ids=["tiny_memory_limit", "pre_cancelled_token"],
)
def test_cyclic_routes_honour_the_execution_limit_and_token_and_return_their_leases(
    limits, token_fired, error
):
    import gc

    model = LinearCyclicModel(("A", "B"), (0.0, 0.4, 0.5, 0.0), (1.0, 0.0, 0.0, 1.0))
    family = model.worlds(
        [World(1, Set("A", 1.0))],
        component_id="loop",
        source_id="specified",
        certifying_evidence=["analytic"],
    )
    schema = EvidenceSchema(
        "feedback",
        "specified population",
        (EvidenceVariable("A", "meter"), EvidenceVariable("B", "meter")),
    )
    source = _fitted()
    result = require_success(source.execute())
    routes = _cyclic_routes(source, family, result, bytes(result.export()), result.identity, schema)
    token = _native.CancellationToken()
    if token_fired:
        token.cancel()
    run = Execution(cancel=token, limits=limits)
    start = run._handle.available_memory_bytes
    for name, route in routes.items():
        if name in {"rebind", "to_family"} and error is CausalResourceError:
            continue  # a one-byte pool still admits their small preparation copies
        with pytest.raises(error):
            outcome = route(run)
            require_success(outcome)
        gc.collect()
        assert run._handle.available_memory_bytes == start, name


def test_common_operations_delegate_to_the_same_immutable_cyclic_lifecycle():
    source = _fitted()
    branch = ant.reconfigure(source, worlds=[[{"variable": 0, "value": 2.0}]])
    changed = require_success(ant.estimate(branch))
    original = require_success(ant.estimate(source))
    assert changed.execution_facts.observed_model_fits == 1
    assert original.execution_facts.observed_model_fits == 0
    assert changed.to_dict()["worlds"][0][0]["values"][1] == pytest.approx(1.4, abs=1e-4)
    assert original.to_dict()["worlds"][0][0]["values"][0] == pytest.approx(0.0)
    rebound = ant.rebind(
        source,
        data=[
            {"values": [x, 0.6 * x], "clamps": [{"variable": 0, "value": x}]} for x in (-1.0, 1.0)
        ],
    )
    assert require_success(ant.estimate(rebound)).execution_facts.observed_model_fits == 1
    assert branch.inspect()["fits_performed"] == 1
    assert source.inspect()["fits_performed"] == 0
