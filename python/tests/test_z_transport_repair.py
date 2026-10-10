"""Public z-transport repair consumes actual checked failure snapshots."""

import json

import pytest
from antecedent import Admg, repair
from antecedent.errors import CausalSerializationError
from antecedent.transport import advanced as transport


def contract():
    names = ["w", "z", "x", "y"]
    graph = Admg.from_edges(
        names,
        [("w", "z"), ("z", "x"), ("x", "y"), ("w", "y")],
        [("w", "y"), ("z", "y"), ("z", "x")],
    )
    query = transport.ZTransportQuery(
        transport.SelectionDiagram("source", "target", []),
        outcomes=["y"],
        treatments=["x"],
        controllable=["z"],
        experiment_assignment={"z": 0.0},
    )
    stage = transport.identify_z_transport(graph=graph, query=query)
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    catalog = transport.EvidenceCatalog(
        environments=(
            transport.Environment("source", coordinates),
            transport.Environment("target", coordinates),
        )
    )
    return repair.ZTransportContract.from_identification(stage, catalog=catalog, names=names)


def experiment(label, *, population="source", level=0.0, joint=True):
    return repair.StudyCandidate.experiment(
        label,
        population=population,
        interventions=["z"],
        measured=["w", "x", "y"],
        joint=joint,
        sample_size=200,
        recruitment="consecutive",
        timing="baseline",
        unit="patient",
        cost=10,
        cost_unit="USD",
        evidence=(
            repair.ExpectedEvidence(
                population=population,
                interventions=["z"],
                levels={"z": level},
                measured=["w", "x", "y"],
                joint=joint,
            ),
        ),
    )


def test_z_transport_contract_obligations_drive_verified_repair_and_typed_refusals():
    failed = contract()
    obligations = repair.obligations(failed)
    assert failed.family == "z_transport"
    assert obligations
    assert all(o.family == "z_transport" and o.proof_step.startswith("leaf:") for o in obligations)
    result = repair.repair(
        failed,
        [
            experiment("correct"),
            experiment("wrong-pop", population="target"),
            experiment("wrong-level", level=1.0),
            experiment("marginals", joint=False),
        ],
        limits=repair.RepairLimits(max_depth=1),
    )
    assert result.best.labels == ("correct",)
    assert result.best.derivation.verified
    assert result.best.derivation.checker == "z_transport.catalog"
    assert all(o.proof_step in result.best.derivation.steps for o in obligations)
    assert all(o.classification == "insufficient" for o in result.table if o.labels != ("correct",))
    with pytest.raises(repair.RepairRefusal, match="identification_repair.invalid_request"):
        edited = json.loads(failed.failure_snapshot)
        edited["catalog_digest"] = "changed"
        repair.ZTransportContract(names=failed.names, failure_snapshot=json.dumps(edited).encode())


def test_z_transport_repair_artifact_replays_and_preserves_budget_stop():
    failed = contract()
    studies = [experiment("correct"), experiment("wrong", level=1.0)]
    for operations in (1, 100):
        result = repair.repair(
            failed, studies, limits=repair.RepairLimits(max_operations=operations, max_depth=1)
        )
        exported = result.export()
        replay = repair.consume(exported)
        assert replay.contract_id == result.contract_id
        assert replay.obligations == result.obligations
        assert replay.table == result.table
        assert replay.receipt == result.receipt
        if operations == 1:
            assert replay.receipt.stop == "search.operations"
            assert replay.outcome == ("exhausted" if replay.best is None else "repaired")
            assert replay.receipt.unevaluated_total == 1
        else:
            assert replay.best.labels == ("correct",)
        with pytest.raises((repair.RepairArtifactRefusal, CausalSerializationError)):
            repair.consume(exported[:-1])
