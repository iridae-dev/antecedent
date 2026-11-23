"""A mechanism, a scientific bridge and a measurement model take part in the one connection check.

Reference. Each adapter projects a declaration to an interface contract, so every expectation is
read off the declaration it projects: the mechanism ``Y`` with parents ``X`` and ``M`` has one
produced port and exactly two consumed ports named by the parent variables; the bridge from a
surrogate effect to a survival effect consumes the first and produces the second; the measurement
model corrects the recorded ``schema:flag`` into ``construct:disease`` under its own identity,
which the produced port carries and a consumer can require. Whether two components connect is then
decided by the shared connection check on those ports, so a population, units or model that
differs is refused with that check's own ``interface_contract`` detail. Nothing is numerical and
no tolerance applies: the assertions are equalities of declared coordinates and identities.
"""

from __future__ import annotations

import pytest
from antecedent import _reason_details as reason_details
from antecedent import interface_contract as ic
from antecedent import measurement_models as mm
from antecedent.errors import CausalTypeError, CausalValueError

from _refusal import assert_registered_refusal
from test_interface_contract import SOURCE_A, refusal_of
from test_mechanism_composition import declaration
from test_scientific_bridge import clinic_model, prevalence, surrogate_contract

OUTPUT = "output"


def test_a_mechanism_is_one_produced_port_and_one_consumed_port_per_parent():
    mechanism = ic.mechanism_contract(
        declaration("Y", ["X", "M"]), origin="fitted", evidence=(SOURCE_A,)
    )
    assert mechanism.component == "mechanism:Y"
    assert [p.name for p in mechanism.produces] == [OUTPUT]
    assert sorted(c.name for c in mechanism.consumes) == ["M", "X"]
    produced = mechanism.produces[0]
    assert (produced.quantity.variable_id, produced.quantity.population_id) == ("Y", "pop")
    assert produced.operations == ("mean",)
    assert produced.calibration.status == "point_only"
    assert produced.evidence == (SOURCE_A,)
    # The origin decides the provenance stage the output supplies.
    assert produced.provenance == ("evidence",)
    declared = ic.mechanism_contract(declaration("Y", ["X"]), origin="declared").produces[0]
    assert declared.provenance == ()
    bound = ic.mechanism_contract(declaration("Y", ["X"]), origin="external_binding").produces[0]
    assert bound.provenance == ("external_provider",)
    # An offering that carries uncertainty is not measured unless a record says so.
    law = ic.mechanism_contract(
        declaration("Y", ["X"]),
        origin="fitted",
        output=ic.PortFacts(uncertainty="parameter_posterior"),
        offered=("mean", "law"),
    ).produces[0]
    assert law.calibration.status == "not_measured"
    assert law.uncertainty == "parameter_posterior"


def test_mechanisms_connect_through_the_one_check_and_a_mismatch_is_refused():
    # Parents are consumed with declared evidence, so the producer declares its ledger source.
    upstream = ic.mechanism_contract(declaration("X"), origin="fitted", evidence=(SOURCE_A,))
    downstream = ic.mechanism_contract(declaration("Y", ["X"]), origin="fitted")
    connection = ic.check_connection(upstream, OUTPUT, downstream, "X")
    assert (connection.producer, connection.consumer) == ("mechanism:X", "mechanism:Y")
    assert connection.calibration.status == "point_only"
    assert connection.evidence == (SOURCE_A,)

    # A parent stated in other units is another quantity: refused, never admitted.
    other_units = ic.mechanism_contract(declaration("Y", ["X"], units="kg"), origin="fitted")
    mismatch = refusal_of(lambda: ic.check_connection(upstream, OUTPUT, other_units, "X"))
    assert mismatch.detail.startswith("interface_contract.")

    # The order of a feed-forward chain is the producers first.
    chain = ic.check_composition(
        [downstream, upstream], [("mechanism:X", OUTPUT, "mechanism:Y", "X")]
    )
    assert chain.order.index("mechanism:X") < chain.order.index("mechanism:Y")


def test_disjoint_parent_evidence_is_decided_by_the_ledger_rows():
    from antecedent import evidence_ledger as el

    ledger = el.EvidenceLedger()
    left = ledger.add_source(
        el.SourceDeclaration("left", "observation_set", observations=el.rows("s", range(0, 50)))
    )
    right = ledger.add_source(
        el.SourceDeclaration("right", "observation_set", observations=el.rows("s", range(25, 75)))
    )

    def parent(variable, source):
        return ic.mechanism_contract(declaration(variable), origin="fitted", evidence=(source,))

    pooled = ic.mechanism_contract(
        declaration("Y", ["X", "M"]), origin="fitted", disjoint_parent_evidence=True
    )
    wires = [
        ("mechanism:X", OUTPUT, "mechanism:Y", "X"),
        ("mechanism:M", OUTPUT, "mechanism:Y", "M"),
    ]
    error = refusal_of(
        lambda: ic.check_composition_with_ledger(
            [parent("X", left), parent("M", right), pooled], wires, ledger
        )
    )
    assert error.detail == reason_details.interface_contract.SHARED_EVIDENCE


def test_a_bridge_consumes_its_source_and_produces_its_target():
    contract = surrogate_contract()
    bridge = ic.bridge_contract(contract, "surrogate_bridge")
    assert bridge.component == "surrogate_bridge"
    (consumed,) = bridge.consumes
    (produced,) = bridge.produces
    assert (consumed.name, produced.name) == ("source", "target")
    assert consumed.quantity == contract.source
    assert produced.quantity == contract.target
    assert consumed.operations == ("mean",) and produced.operations == ("mean",)
    assert "transformation" in produced.provenance

    def surrogate_producer(population):
        quantity = contract.source
        if population != quantity.population_id:
            from dataclasses import replace

            quantity = replace(quantity, population_id=population)
        return ic.InterfaceContract(
            "trial",
            produces=[
                ic.ProducedPort(
                    "effect",
                    quantity,
                    quantity.variable_id,
                    ic.SupportDomain.real(),
                    operations=("mean",),
                )
            ],
        )

    fed = ic.check_connection(surrogate_producer("adults"), "effect", bridge, "source")
    # The producer here declared no calibration, and a connection keeps the producer's label.
    assert fed.calibration.status == "none"
    # A surrogate effect from another population does not feed the bridge.
    refused = refusal_of(
        lambda: ic.check_connection(surrogate_producer("children"), "effect", bridge, "source")
    )
    assert refused.detail.startswith("interface_contract.")

    survival = ic.InterfaceContract(
        "decision",
        consumes=[
            ic.ConsumedPort(
                "survival",
                contract.target,
                contract.target.variable_id,
                ic.SupportDomain.real(),
                operations=("mean",),
            )
        ],
    )
    assert ic.check_connection(bridge, "target", survival, "survival").producer == (
        "surrogate_bridge"
    )
    with pytest.raises(CausalTypeError):
        ic.bridge_contract(object(), "surrogate_bridge")


def test_a_measurement_model_is_produced_under_its_identity_and_a_consumer_states_it():
    model = clinic_model()
    identity = mm.model_identities(model).model
    indicator, construct = prevalence("schema:flag"), prevalence("construct:disease")
    correction = ic.measurement_contract("correction", model, indicator, construct)
    (consumed,) = correction.consumes
    (produced,) = correction.produces
    assert (consumed.name, produced.name) == ("indicator", "construct")
    assert consumed.quantity == indicator and produced.quantity == construct
    assert produced.measurement_models == (identity,)
    assert consumed.measurement_models == ()
    assert "transformation" in produced.provenance

    def consumer(models):
        return ic.InterfaceContract(
            "report",
            consumes=[
                ic.ConsumedPort(
                    "disease",
                    construct,
                    construct.variable_id,
                    ic.SupportDomain.real(),
                    operations=("mean",),
                    measurement_models=models,
                )
            ],
        )

    assert ic.check_connection(correction, "construct", consumer((identity,)), "disease")
    other = refusal_of(
        lambda: ic.check_connection(correction, "construct", consumer(("cd" * 32,)), "disease")
    )
    assert other.detail == reason_details.interface_contract.MEASUREMENT_ASSUMPTION_MISMATCH


def test_a_measurement_model_that_does_not_bind_the_quantities_is_refused():
    model = clinic_model()
    indicator, construct = prevalence("schema:flag"), prevalence("construct:disease")
    detail = reason_details.measurement_model

    wrong_indicator = prevalence("schema:other")
    error = refusal_of(
        lambda: ic.measurement_contract("correction", model, wrong_indicator, construct)
    )
    assert error.detail == detail.INDICATOR_MISMATCH
    wrong_construct = prevalence("construct:other")
    error = refusal_of(
        lambda: ic.measurement_contract("correction", model, indicator, wrong_construct)
    )
    assert error.detail == detail.CONSTRUCT_MISMATCH
    elsewhere = prevalence("construct:disease", population="clinic-b")
    error = refusal_of(lambda: ic.measurement_contract("correction", model, indicator, elsewhere))
    assert error.detail == detail.POPULATION_MISMATCH
    with pytest.raises(CausalTypeError):
        ic.measurement_contract("correction", object(), indicator, construct)
    with pytest.raises(CausalValueError):
        ic.measurement_contract(
            "correction", model, indicator, construct, operations=("no_such_operation",)
        )
    assert_registered_refusal(error)
