"""Check and persist a declared transported-claim connection to a callback input.

This checks the handoff contract; no callback executes and no transport theorem
is proved by naming an offer. The claim retains external transported standing.
"""

from hashlib import sha256

from antecedent import provider_callbacks as pc
from antecedent import transport_handoff as th
from antecedent.joint_distribution import ScientificQuantity


def digest(value):
    return sha256(str(value).encode()).hexdigest()


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


def main():
    declaration = callback()
    handoff = th.hand_to_callback(offer(), declaration, "effect")
    assert handoff.standing == "transported_claim"
    assert not handoff.native_target_data
    consumed = th.consume_for_callback(
        handoff.export(), declaration, expected_identity=handoff.identity
    )
    assert consumed.identity == handoff.identity
    print("Handoff standing:", handoff.standing, "calibration:", handoff.calibration)
    try:
        th.hand_to_callback(offer(), declaration, "effect", accepts_transported=False)
    except pc.CallbackRefusal as error:
        assert error.detail == "provider_transport.native_target_required"
        print("Native-only consumer refused:", error.reason_code, error.remedy)
    else:
        raise AssertionError("Native-only consumer accepted a transported claim")


if __name__ == "__main__":
    main()
