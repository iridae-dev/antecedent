"""Preparation retains native scientific refusals; operational faults remain exceptions."""

import traceback

import antecedent as ant
import pytest
from antecedent import _native
from antecedent.cyclic import (
    AsymptoticNormal,
    InitialState,
    PreparedCyclicFamily,
    RecoveryBootstrap,
    Set,
    Shift,
    Soft,
    World,
)
from antecedent.errors import (
    CausalCancelledError,
    CausalResourceError,
    CausalValueError,
    StructuredRefusal,
)
from antecedent.execution import Execution, ResourceLimits
from antecedent.outcomes import require_success

from test_cyclic import _feedback_linear, _recovery


def _worlds(worlds, *, execution=None):
    return _feedback_linear().worlds(
        worlds,
        component_id="feedback",
        source_id="specification",
        certifying_evidence=["independent-bench-test"],
        execution=execution,
    )


@pytest.mark.parametrize("loaded", [False, True])
def test_recovered_unsupported_world_retains_refusal_and_corrective_branch(monkeypatch, loaded):
    source = require_success(_recovery(AsymptoticNormal(4)).execute())
    if loaded:
        artifact = source.export()
        source = require_success(ant.load(artifact, expected=artifact.expectation))
    law = source.law
    before = law.to_dict()
    refused = law.prepare([World(48, Shift("A", 1.0))], variables=["A", "B"], outcome="B")
    assert isinstance(refused, PreparedCyclicFamily)
    refusal = refused.refusal
    assert refusal.code == "route_not_supported"
    assert refusal.detail == "cyclic_recovery.hard_clamp_required"
    assert refusal.remedy and refusal.cause is None

    def forbidden(*args, **kwargs):
        pytest.fail("refused inspection/execution attempted compilation or source serialization")

    with monkeypatch.context() as bounded:
        bounded.setattr(type(law), "to_dict", forbidden)
        bounded.setattr(_native, "prepare_cyclic_family", forbidden)
        inspection = refused.inspect()
        assert inspection["identity"] is None
        assert inspection["identity_unavailable"] == "native_program_not_prepared"
        assert inspection["refusal"] is refusal
        assert refused.contract()["available"] is False
        assert repr(refused) == "PreparedCyclicFamily()"
        assert ant.estimate(refused).refusal._error is refusal._error
        assert refused.interventions().refusal._error is refusal._error
    corrected = refused.reconfigure(worlds=[World(49, Set("A", 3.0))])
    assert corrected.refusal is None
    answer = require_success(ant.estimate(corrected))
    assert answer.law.to_dict() == before
    values = require_success(answer.interventions())[49].draw_values(0)
    assert values is not None and values[0] == pytest.approx(3.0)
    assert refused.refusal is refusal
    assert ant.estimate(refused).refusal._error is refusal._error


def test_native_family_capability_refusal_is_shared_by_direct_and_typed_factories():
    prepared = _worlds([World(1, initial_states=InitialState("A", 2.0))])
    direct = PreparedCyclicFamily(prepared.request_json, prepared.variable_names)
    for refused in (prepared, direct):
        assert refused.refusal.code == "route_not_supported"
        assert ant.estimate(refused).refusal.code == "route_not_supported"
        corrected = refused.reconfigure(worlds=[World(2, Set("A", 2.0))])
        worlds = require_success(require_success(ant.estimate(corrected)).interventions())
        assert worlds[2].mean("A") == pytest.approx(2.0)
        assert worlds[2].mean("B") == pytest.approx(1.0)
        assert refused.refusal is not None


def test_native_sampling_design_refusal_retains_declarations_for_correction():
    refused = _recovery(AsymptoticNormal(4), sampling_design="dependent_time_series")
    assert refused.refusal.code == "estimator_inference_mismatch"
    assert refused.refusal.detail == "cyclic_recovery.row_independence_unlicensed"
    assert ant.estimate(refused).refusal.code == "estimator_inference_mismatch"
    corrected = refused.reconfigure(uncertainty=RecoveryBootstrap(4, block_length=8))
    assert corrected.refusal is None
    result = require_success(ant.estimate(corrected))
    assert result.law.inspect().details["law_meaning"] == "sampling_law"
    assert refused.refusal is not None


@pytest.mark.parametrize(
    ("declare", "message"),
    [
        (lambda: [World(1), World(1)], "a world id is requested twice"),
        (lambda: [World(1, Set("undeclared", 2.0))], "variable 'undeclared' is not declared"),
        (
            lambda: [World(1, Soft("A", "linear_mechanism", parameters=[1.0]))],
            "soft family `linear_mechanism` takes 3 parameters and 1 were supplied",
        ),
        # World validates its own interventions, so the invalid declaration raises as built.
        (lambda: [World(1, object())], "interventions must be Set, Shift or Soft values"),
    ],
    ids=["duplicate-id", "undeclared-variable", "wrong-soft-parameters", "foreign-intervention"],
)
def test_invalid_world_declarations_raise_argument_errors_even_with_next_attempt_text(
    declare, message
):
    with pytest.raises(CausalValueError, match=message):
        _worlds(declare())


@pytest.mark.parametrize("entry", ["ordinary", "retained"])
def test_resource_and_cancel_failures_are_operational(entry):
    if entry == "retained":
        law = require_success(_recovery(AsymptoticNormal(4)).execute()).law

        def prepare(execution):
            return law.prepare(
                [World(1, Set("A", 3.0))], variables=["A", "B"], outcome="B", execution=execution
            )
    else:

        def prepare(execution):
            return _worlds([World(1)], execution=execution)

    with pytest.raises(CausalResourceError):
        prepare(Execution(limits=ResourceLimits(0)))
    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError):
        prepare(Execution(cancel=token))


def test_marked_callback_error_with_scientific_code_preserves_original_traceback(monkeypatch):
    error = StructuredRefusal(
        {
            "code": "route_not_supported",
            "detail": "callback.sentinel",
            "message": "callback sentinel",
        }
    )
    error._antecedent_operational_callback = True

    def original_callback():
        raise error

    def native_boundary(*args, **kwargs):
        original_callback()

    monkeypatch.setattr(_native, "prepare_cyclic_family", native_boundary)
    with pytest.raises(StructuredRefusal) as raised:
        _worlds([World(1)])
    assert raised.value is error
    assert any(
        frame.name == "original_callback" for frame in traceback.extract_tb(error.__traceback__)
    )


def test_missing_mandatory_bootstrap_design_is_an_argument_error():
    with pytest.raises(CausalValueError, match="sampling_design"):
        _recovery(RecoveryBootstrap(4))
