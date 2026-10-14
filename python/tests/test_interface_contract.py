"""Scientific interface contracts through the Python facade.

Every expectation is enumerated from the declarations: which dimensions of two quantities differ,
which operations a producer lacks, which path closes a cycle. Identities are cross-checked by the
clean-room consumer ``spec/consumer/validate.py``, which shares no code with the library.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest
from antecedent import interface_contract as ic
from antecedent.errors import CausalValueError
from antecedent.interface_contract import InterfaceContractRefusal
from antecedent.joint_distribution import ScientificQuantity

from _refusal import assert_registered_refusal

VALIDATOR = pathlib.Path(__file__).resolve().parents[2] / "spec" / "consumer" / "validate.py"
MODEL = "ab" * 32
SOURCE_A = "01" * 32
SOURCE_B = "02" * 32


def quantity(population="adults", units="mg/dL", regime="do(statin=1)"):
    return ScientificQuantity.outcome(
        "ldl_change", units=units, population=population, regime=regime
    )


def producer(population="adults", operations=("mean",), **kwargs):
    port = ic.ProducedPort(
        "effect",
        quantity(population),
        "estimand:ate",
        ic.SupportDomain.real(),
        operations=operations,
        **kwargs,
    )
    return ic.InterfaceContract("estimator", produces=[port])


def consumer(population="adults", operations=("mean",), **kwargs):
    port = ic.ConsumedPort(
        "effect",
        quantity(population),
        "estimand:ate",
        ic.SupportDomain.real(),
        operations=operations,
        **kwargs,
    )
    return ic.InterfaceContract("decision", consumes=[port])


def refusal_of(call):
    with pytest.raises(InterfaceContractRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    return caught.value


# --------------------------------------------------------------------------- connections


def test_a_compatible_pair_connects_and_keeps_the_producers_label():
    produced = producer(
        operations=("mean", "interval", "posterior"),
        uncertainty="causal_functional_posterior",
        calibration=ic.Calibration.measured("cal-0001"),
        measurement_models=(MODEL,),
        evidence=(SOURCE_A, SOURCE_B),
    )
    wanted = consumer(
        operations=("mean", "posterior"),
        uncertainty="causal_functional_posterior",
        calibration="measured",
        measurement_models=(MODEL,),
        dependence="declared_evidence",
    )
    connection = ic.check_connection(produced, "effect", wanted, "effect")
    assert connection.operations == ("mean", "posterior")
    assert connection.calibration == ic.Calibration.measured("cal-0001")
    assert connection.uncertainty_paradigm == "posterior"
    assert connection.evidence == (SOURCE_A, SOURCE_B)
    assert connection == ic.check_connection(produced, "effect", wanted, "effect")


def test_a_pair_that_matches_syntactically_fails_scientifically():
    # Same variable name, units and shape: another population, and a mean offered for a law.
    error = refusal_of(
        lambda: ic.check_connection(
            producer(population="trial_cohort"),
            "effect",
            consumer(operations=("law",), uncertainty="interventional_predictive"),
            "effect",
        )
    )
    assert [r.detail for r in error.refusals] == [
        "interface_contract.population_mismatch",
        "interface_contract.operation_unavailable",
        "interface_contract.uncertainty_meaning_missing",
    ]
    assert error.reason_code == "interface_contract_quantity_mismatch"
    population = error.refusals[0]
    assert (population.expected, population.supplied) == ("adults", "trial_cohort")
    assert all(r.remedy for r in error.refusals)


def test_a_mean_does_not_supply_a_law_and_a_sampling_interval_is_not_a_posterior():
    error = refusal_of(
        lambda: ic.check_connection(
            producer(operations=("mean",)),
            "effect",
            consumer(operations=("law",), uncertainty="interventional_predictive"),
            "effect",
        )
    )
    assert "interface_contract.operation_unavailable" in [r.detail for r in error.refusals]
    error = refusal_of(
        lambda: ic.check_connection(
            producer(operations=("interval",), uncertainty="estimator_sampling"),
            "effect",
            consumer(operations=("posterior",), uncertainty="causal_functional_posterior"),
            "effect",
        )
    )
    details = [r.detail for r in error.refusals]
    assert "interface_contract.uncertainty_paradigm_mismatch" in details


def test_measured_calibration_is_required_not_assumed():
    wanted = consumer(
        operations=("interval",), uncertainty="estimator_sampling", calibration="measured"
    )
    for label in (ic.Calibration.none(), ic.Calibration.unmeasured()):
        error = refusal_of(
            lambda label=label: ic.check_connection(
                producer(
                    operations=("interval",), uncertainty="estimator_sampling", calibration=label
                ),
                "effect",
                wanted,
                "effect",
            )
        )
        assert error.detail == "interface_contract.calibration_not_held"
        assert error.reason_code == "interface_contract_calibration_not_held"


def test_a_malformed_declaration_is_refused_alone():
    error = refusal_of(
        lambda: ic.check_connection(
            producer(operations=("interval",)), "effect", consumer(), "effect"
        )
    )
    assert error.detail == "interface_contract.uncertainty_meaning_required"
    assert error.reason_code == "interface_contract_invalid"
    with pytest.raises(CausalValueError):
        ic.SupportDomain.interval(float("nan"), 1.0)


# --------------------------------------------------------------------------- composition


def stage(name):
    return ic.InterfaceContract(
        name,
        produces=[
            ic.ProducedPort(
                "out", quantity(), "estimand:ate", ic.SupportDomain.real(), operations=("mean",)
            )
        ],
        consumes=[
            ic.ConsumedPort(
                "in", quantity(), "estimand:ate", ic.SupportDomain.real(), operations=("mean",)
            )
        ],
    )


def test_a_feed_forward_composition_is_ordered_and_a_cycle_is_refused_with_its_path():
    parts = [stage("c"), stage("a"), stage("b")]
    composed = ic.check_composition(parts, [("a", "out", "b", "in"), ("b", "out", "c", "in")])
    assert composed.order == ("a", "b", "c")
    shuffled = ic.check_composition(
        [parts[2], parts[0], parts[1]], [("b", "out", "c", "in"), ("a", "out", "b", "in")]
    )
    assert shuffled.identity == composed.identity
    error = refusal_of(
        lambda: ic.check_composition(
            parts, [("a", "out", "b", "in"), ("b", "out", "c", "in"), ("c", "out", "a", "in")]
        )
    )
    assert error.reason_code == "interface_contract_cyclic_composition"
    assert error.detail == "interface_contract.cyclic_composition"
    assert error.offending == "a -> b -> c -> a"
    assert error.remedy


def test_inputs_required_disjoint_cannot_share_evidence():
    def source(name, evidence):
        return ic.InterfaceContract(
            name,
            produces=[
                ic.ProducedPort(
                    "out",
                    quantity(),
                    "estimand:ate",
                    ic.SupportDomain.real(),
                    operations=("mean",),
                    evidence=evidence,
                )
            ],
        )

    pool = ic.InterfaceContract(
        "pool",
        consumes=[
            ic.ConsumedPort(
                side,
                quantity(),
                "estimand:ate",
                ic.SupportDomain.real(),
                operations=("mean",),
                dependence="disjoint_evidence",
            )
            for side in ("left", "right")
        ],
    )
    wires = [("s1", "out", "pool", "left"), ("s2", "out", "pool", "right")]
    assert ic.check_composition([source("s1", (SOURCE_A,)), source("s2", (SOURCE_B,)), pool], wires)
    error = refusal_of(
        lambda: ic.check_composition(
            [source("s1", (SOURCE_A,)), source("s2", (SOURCE_A,)), pool], wires
        )
    )
    assert error.detail == "interface_contract.shared_evidence"


# --------------------------------------------------------------------------- artifact


def rich_contract():
    return ic.InterfaceContract(
        "estimator",
        produces=[
            ic.ProducedPort(
                "effect",
                quantity(),
                "estimand:ate",
                ic.SupportDomain.interval(-0.0, 1.0),
                operations=("mean", "draws", "interval", "posterior"),
                uncertainty="causal_functional_posterior",
                calibration=ic.Calibration.measured("cal-0001"),
                measurement_models=(MODEL,),
                evidence=(SOURCE_A,),
                provenance=("causal_contract", "claim"),
            ),
            ic.ProducedPort(
                "selection",
                quantity(),
                "estimand:enrolment",
                ic.SupportDomain.categorical(["no", "yes"]),
                operations=("point",),
                calibration=ic.Calibration.point_only(),
            ),
        ],
        consumes=[
            ic.ConsumedPort(
                "data",
                quantity(),
                "estimand:observed",
                ic.SupportDomain.non_negative(),
                operations=("mean",),
                dependence="declared_evidence",
            )
        ],
    )


def test_an_exported_contract_round_trips_and_is_inspectable():
    contract = rich_contract()
    exported = ic.export_interface_contract(contract, "estimator-contract")
    assert exported.contract_identity == ic.contract_identity(contract)
    consumed = ic.consume_interface_contract_artifact(exported.artifact, exported.identity)
    assert consumed.identity == exported.identity
    assert [p.name for p in consumed.contract.produces] == ["effect", "selection"]
    assert consumed.contract.produces[0].support == ic.SupportDomain.interval(-0.0, 1.0)
    assert (
        ic.export_interface_contract(contract, "estimator-contract").artifact == exported.artifact
    )
    meta = ic.inspect_interface_contract_artifact(exported.artifact)
    assert meta["feature"] == "interface_contract_v1"
    assert meta["identity"] == exported.identity
    assert meta["contract"]["produces"][0]["name"] == "effect"
    error = refusal_of(lambda: ic.consume_interface_contract_artifact(exported.artifact, "0" * 64))
    assert error.detail == "interface_contract.identity_mismatch"


def test_the_clean_room_consumer_accepts_an_exported_artifact_and_rejects_mutations(tmp_path):
    exported = ic.export_interface_contract(rich_contract(), "estimator-contract")
    path = tmp_path / "contract.artifact"
    path.write_bytes(exported.artifact)

    def run(file, *extra):
        return subprocess.run(
            [
                sys.executable,
                "-I",
                str(VALIDATOR),
                "--kind",
                "interface_contract_v1",
                str(file),
                *extra,
            ],
            capture_output=True,
            text=True,
        )

    accepted = run(path, "--retained-identity", exported.identity)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert exported.identity in accepted.stdout
    assert run(path, "--retained-identity", "0" * 64).returncode != 0
    # A flipped payload byte is a checksum failure; a flipped byte of the identity text, once
    # the container is rewritten consistently, is an identity failure.
    mutated = bytearray(exported.artifact)
    mutated[-1] ^= 0x01
    bad = tmp_path / "corrupt.artifact"
    bad.write_bytes(bytes(mutated))
    rejected = run(bad)
    assert rejected.returncode != 0
    assert "REJECT checksum" in rejected.stdout


def test_a_fresh_process_consumes_the_artifact_against_a_retained_identity(tmp_path):
    exported = ic.export_interface_contract(rich_contract(), "estimator-contract")
    path = tmp_path / "contract.artifact"
    path.write_bytes(exported.artifact)
    script = (
        "import sys, json, pathlib\n"
        "from antecedent import interface_contract as ic\n"
        "c = ic.consume_interface_contract_artifact(pathlib.Path(sys.argv[1]).read_bytes(),"
        " sys.argv[2])\n"
        "print(json.dumps({'identity': c.identity, 'ports': [p.name for p in c.contract.produces]}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(path), exported.identity],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )
    wire = json.loads(done.stdout)
    assert wire == {"identity": exported.identity, "ports": ["effect", "selection"]}
    refused = subprocess.run(
        [sys.executable, "-c", script, str(path), "0" * 64],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert refused.returncode != 0
    assert "interface_contract.identity_mismatch" in refused.stderr
