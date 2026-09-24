"""Bounded zTR search returns checked, point-only transport results."""

import json

import pytest
from antecedent import Admg
from antecedent.transport import advanced as transport


def fixture(empirical=False):
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
    assert stage.outcome == "identified"

    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    regimes = (
        transport.EvidenceRegime(
            "do_z_0",
            "source",
            kind="experimental",
            interventions=["z"],
            intervention_values={"z": 0.0},
            measured=names,
        ),
        transport.EvidenceRegime(
            "do_z_1",
            "source",
            kind="experimental",
            interventions=["z"],
            intervention_values={"z": 1.0},
            measured=names,
        ),
    )
    catalog = transport.EvidenceCatalog(
        environments=(transport.Environment("source", coordinates),),
        regimes=regimes,
        bindings=(
            transport.RegimeBinding(
                "do_z_0",
                "snapshot_z0",
                schema_names=names,
                sampling="independent",
                dependence="independent_studies",
            ),
            transport.RegimeBinding(
                "do_z_1",
                "snapshot_z1",
                schema_names=names,
                sampling="independent",
                dependence="independent_studies",
            ),
        ),
    )
    probabilities = []
    counts = []
    for w in (0, 1):
        for x in (0, 1):
            for y in (0, 1):
                p = (0.25 if w else 0.75) * (0.35 if x else 0.65) * (0.8 if y == x else 0.2)
                probabilities.append(p)
                counts.append(round(p * 10_000))
    if empirical:
        total = sum(counts)
        probabilities = [count / total for count in counts]
    law = transport.ExactDiscreteLaw(
        "source",
        "do_z_0",
        (("w", (0.0, 1.0)), ("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
        tuple(probabilities),
        "snapshot_z0",
        interventions=(("z", 0.0),),
        empirical_counts=tuple(counts) if empirical else None,
    )
    prepared = (
        stage.prepare_empirical(catalog, (law,), {"x": 0.0})
        if empirical
        else stage.prepare_exact(catalog, (law,), {"x": 0.0})
    )
    return graph, stage, catalog, prepared, (law,)


@pytest.mark.parametrize("empirical", [False, True])
def test_z_transport_prepared_native_route_matches_independent_truth(empirical):
    _, _, _, prepared, laws = fixture(empirical)
    result = json.loads(prepared.estimate())
    assert result["status"] == "available"
    assert result["scope"] == "bounded_single_source_z_transport_sound_incomplete"
    assert result["interval"] == {"available": False, "reason": "no_interval_reported"}
    true_mass = sum(
        p for atom, p in zip(result["atoms"], result["probabilities"], strict=True) if atom == [1.0]
    )
    assert true_mass == pytest.approx(0.2, abs=0.002 if empirical else 1e-12)
    artifact = prepared.export()
    consumed = json.loads(transport.consume_z_transport_artifact(artifact))
    assert consumed["probabilities"] == pytest.approx(result["probabilities"])
    assert consumed["interval"] == {"available": False, "reason": "no_interval_reported"}
    prepared.refresh(laws)
    refreshed = json.loads(prepared.estimate())
    assert refreshed["probabilities"] == pytest.approx(result["probabilities"])
    sensitivity = prepared.mechanism_sensitivity(0.2, 0.4)
    assert sensitivity["baseline"] == pytest.approx(0.6, abs=0.002 if empirical else 1e-12)
    assert sensitivity["delta_domain"] == [0.0, 0.2]
    assert sensitivity["baseline_binding"]["artifact_digest"]
    sensitivity_artifact = prepared.export_sensitivity(0.2, 0.4)
    replayed_sensitivity = transport.consume_z_transport_sensitivity_artifact(sensitivity_artifact)
    assert replayed_sensitivity["baseline"] == pytest.approx(sensitivity["baseline"])
    assert replayed_sensitivity["assumption_range"] == sensitivity["assumption_range"]
    tampered_sensitivity = bytearray(sensitivity_artifact)
    tampered_sensitivity[-1] ^= 0x01
    with pytest.raises(ValueError, match=r".+"):
        transport.consume_z_transport_sensitivity_artifact(bytes(tampered_sensitivity))


def test_z_transport_estimate_refresh_executes_retained_program_after_builder_disposal():
    """The zTR handle keeps its proof and provider binding through refresh and replay."""
    _, builder, catalog, prepared, laws = fixture(False)
    program = builder.inspect_proof(catalog)
    assert program["rules"]
    del builder

    first = json.loads(prepared.estimate())
    retained_plan = prepared.export()
    independently_consumed = json.loads(transport.consume_z_transport_artifact(retained_plan))
    assert independently_consumed["probabilities"] == pytest.approx(first["probabilities"])
    assert independently_consumed["interval"] == {"available": False, "reason": "no_interval_reported"}

    prepared.refresh(laws)
    refreshed = json.loads(prepared.estimate())
    assert refreshed["probabilities"] == pytest.approx(first["probabilities"])
    refreshed_consumer = json.loads(transport.consume_z_transport_artifact(prepared.export()))
    assert refreshed_consumer["probabilities"] == pytest.approx(refreshed["probabilities"])


def test_z_transport_stage_refuses_unregistered_selection_graph():
    graph = Admg.from_edges(
        ["w", "z", "x", "y"],
        [("w", "z"), ("z", "x"), ("x", "y"), ("w", "y")],
        [("w", "y"), ("z", "y"), ("z", "x")],
    )
    query = transport.ZTransportQuery(
        transport.SelectionDiagram("source", "target", ["y"]),
        outcomes=["y"],
        treatments=["x"],
        controllable=["z"],
        experiment_assignment={"z": 0.0},
    )
    stage = transport.identify_z_transport(graph=graph, query=query)
    assert stage.outcome == "not_certified"
    assert stage.reason == "z_transport.no_checked_recursive_formula"


def test_z_transport_failure_snapshot_plan_and_actual_arrival():
    names = ["w", "z", "x", "y"]
    graph = Admg.from_edges(
        names,
        [("w", "z"), ("z", "x"), ("x", "y"), ("w", "y")],
        [("w", "y"), ("z", "y"), ("z", "x")],
    )
    stage = transport.identify_z_transport(
        graph=graph,
        query=transport.ZTransportQuery(
            transport.SelectionDiagram("source", "target", []),
            outcomes=["y"],
            treatments=["x"],
            controllable=["z"],
            experiment_assignment={"z": 0.0},
        ),
    )
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    high = transport.EvidenceRegime(
        "do_z_1", "source", kind="experimental", interventions=["z"],
        intervention_values={"z": 1.0}, measured=names,
    )
    base = transport.EvidenceCatalog(
        environments=(transport.Environment("source", coordinates),),
        regimes=(high,),
        bindings=(transport.RegimeBinding("do_z_1", "snapshot_z1", schema_names=names,
            sampling="independent", dependence="independent_studies"),),
    )
    low_proposed = transport.EvidenceRegime(
        "do_z_0", "source", kind="experimental", evidence_kind="proposed",
        interventions=["z"], intervention_values={"z": 0.0}, measured=names,
    )
    hypothetical = transport.EvidenceCatalog(
        environments=base.environments,
        regimes=(high, low_proposed),
        bindings=base.bindings,
    )
    candidate = transport.ZTransportCandidate(
        "joint_z0", hypothetical, "intervene", targets=["z"], cost=1.0,
        sample_budget=100, recruitment_sampling="randomized source study",
        feasibility_constraints=["z is manipulable"],
    )
    snapshot = stage.failure_snapshot(base)
    snapshot_wire = json.loads(snapshot)
    assert snapshot_wire["version"] == 1
    assert snapshot_wire["status"] == "missing_evidence"
    assert snapshot_wire["proof_graph"]["rules"]
    assert any(factor["failure"] for factor in snapshot_wire["proof_graph"]["factors"])
    inspection = stage.inspect_proof(base)
    assert inspection["rules"]
    assert any(factor["failure"] for factor in inspection["factors"])
    summary, proposals = transport.plan_z_transport_evidence(
        stage, base, [candidate], failure_snapshot=snapshot
    )
    assert summary["ranked_sufficient"] == ["joint_z0"]
    assert len(proposals) == 1
    proposal = proposals[0]
    proposal.replay()
    portable_proposal = proposal.export()
    transport.replay_z_transport_proposal(portable_proposal)
    tampered_proposal = bytearray(portable_proposal)
    tampered_proposal[-1] ^= 0x01
    with pytest.raises(ValueError, match=r".+"):
        transport.replay_z_transport_proposal(bytes(tampered_proposal))

    low_available = transport.EvidenceRegime(
        "do_z_0", "source", kind="experimental", evidence_kind="available",
        interventions=["z"], intervention_values={"z": 0.0}, measured=names,
    )
    actual = transport.EvidenceCatalog(
        environments=base.environments,
        regimes=(high, low_available),
        bindings=(
            base.bindings[0],
            transport.RegimeBinding("do_z_0", "snapshot_z0_arrival", schema_names=names,
                sampling="independent", dependence="independent_studies"),
        ),
    )
    probabilities = []
    for w in (0, 1):
        for x in (0, 1):
            for y in (0, 1):
                probabilities.append((0.25 if w else 0.75) * (0.35 if x else 0.65) * (0.8 if y == x else 0.2))
    law = transport.ExactDiscreteLaw(
        "source", "do_z_0", (("w", (0.0, 1.0)), ("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
        tuple(probabilities), "snapshot_z0_arrival", interventions=(("z", 0.0),),
    )
    arrived = proposal.receive(actual, (law,), {"x": 0.0}, "snapshot_z0_arrival")
    result = json.loads(arrived.estimate())
    y1 = sum(p for atom, p in zip(result["atoms"], result["probabilities"], strict=True) if atom == [1.0])
    assert y1 == pytest.approx(0.2)
    assert result["interval"] == {"available": False, "reason": "no_interval_reported"}

    with pytest.raises(ValueError, match="provider snapshot"):
        proposal.receive(actual, (law,), {"x": 0.0}, "snapshot_z1")
