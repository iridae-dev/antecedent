"""A transported claim is handed to a callback input and the record of the connection is
re-derived by a fresh consumer through the Python surface.

The offer is a point of ``recovery`` in ``target`` carried from two trials; the callback is a law
that reads it as its input ``effect`` (the Python spelling of
``crates/antecedent-io-core/tests/transport_handoff_artifact.rs``).
"""

from __future__ import annotations

import dataclasses

import pytest
from antecedent import provider_callbacks as pc
from antecedent import transport_handoff as th
from antecedent.errors import CausalTypeError
from antecedent.joint_distribution import ScientificQuantity

from _refusal import assert_registered_refusal
from _transport_fixtures import digest


def recovery(population: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id="schema:recovery",
        variable_name="recovery",
        role="outcome",
        units="days",
        population_id=population,
        regime_id="do(a=1)",
        horizon=0,
        functional_id="outcome",
    )


def offer(population: str = "target") -> th.TransportedClaimOffer:
    return th.TransportedClaimOffer(
        component="transport_result",
        port="effect",
        quantity=recovery(population),
        semantic_identity="estimand:recovery_under_treatment",
        support=pc.Support.real(),
        operations=("mean",),
        transport=th.TransportIdentity(
            source_populations=("trial_a", "trial_b"),
            target_population="target",
            selection_diagram=digest(7),
            assumptions=("recovery mechanism invariant across populations",),
        ),
        evidence=(digest(5),),
    )


def callback(version: str = "beta=0.5") -> pc.CallbackDeclaration:
    return pc.CallbackDeclaration(
        "downstream_model",
        "law",
        pc.ProviderIdentity("lab", "downstream", "1.0", "snap-1", "req-1"),
        recovery("target"),
        "estimand:downstream_outcome",
        "interventional_predictive",
        pc.Support.real(),
        operations=("mean",),
        dependencies=pc.Dependencies.complete([pc.Dependency("model", "parameters", version)]),
        replay=pc.Replay.deterministic_pure(),
        trust=pc.Trust.externally_attested("lab"),
        evidence=(),
        inputs=(
            pc.Input(
                "effect",
                recovery("target"),
                "estimand:recovery_under_treatment",
                pc.Support.real(),
                "mean",
            ),
        ),
    )


def refusal_of(call) -> pc.CallbackRefusal:
    with pytest.raises(pc.CallbackRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    return caught.value


def test_a_sealed_handoff_round_trips_and_a_fresh_consumer_rederives_it() -> None:
    handoff = th.hand_to_callback(offer(), callback(), "effect")
    assert handoff.standing == "transported_claim"
    assert handoff.native_target_data is False
    assert handoff.calibration == "point_only"
    assert handoff.transport.source_populations == ("trial_a", "trial_b")
    assert handoff.transport.target_population == "target"
    assert handoff.transport.selection_diagram == digest(7)
    assert handoff.inspect().family == "TransportHandoff"

    consumed = th.consume_for_callback(
        handoff.export(), callback(), expected_identity=handoff.identity
    )
    assert consumed.identity == handoff.identity
    assert consumed.transport == handoff.transport
    assert consumed.standing == "transported_claim"

    # Another declaration (a changed dependency version at unchanged shape).
    other = refusal_of(
        lambda: th.consume_for_callback(
            handoff.export(), callback("beta=0.6"), expected_identity=handoff.identity
        )
    )
    assert other.detail.startswith("provider_transport.")


def test_a_handoff_that_the_library_refuses_does_not_seal() -> None:
    # The consumer needs native target data.
    native = refusal_of(
        lambda: th.hand_to_callback(offer(), callback(), "effect", accepts_transported=False)
    )
    assert native.detail == "provider_transport.native_target_required"

    # The claim is not in the target population of its own transport identity.
    elsewhere = refusal_of(lambda: th.hand_to_callback(offer("trial_a"), callback(), "effect"))
    assert elsewhere.detail == "provider_transport.claim_not_in_target"


def test_an_edited_handoff_is_refused() -> None:
    handoff = th.hand_to_callback(offer(), callback(), "effect")
    data = bytearray(handoff.export())
    data[-8] ^= 0x01
    refusal_of(lambda: th.consume_for_callback(bytes(data), callback()))
    # A consumer that retained another identity refuses the bytes.
    refusal_of(
        lambda: th.consume_for_callback(handoff.export(), callback(), expected_identity=digest(9))
    )
    with pytest.raises(CausalTypeError):
        th.consume_for_callback("bytes", callback())  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        th.hand_to_callback(dataclasses.replace(offer(), port="x"), object(), "effect")  # type: ignore[arg-type]
