"""Bounded zTR search returns checked, point-only transport results."""

import json
import struct

import pytest
from antecedent import Admg
from antecedent.errors import (
    CausalCancelledError,
    CausalResourceError,
    CausalSerializationError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.transport import advanced as transport

Z_PREFIX = b"ANTECEDENT-Z-TRANSPORT\x01"


def _cbor_f64(value: float) -> bytes:
    return b"\xfb" + struct.pack(">d", value)


def _cbor_byte_array(raw: bytes) -> bytes:
    """``raw`` as serde-CBOR spells a ``Vec<u8>``: an array of small unsigned ints."""
    return b"".join(bytes([b]) if b < 24 else b"\x18" + bytes([b]) for b in raw)


def _rewrite_f64(blob: bytes, value: float, replacement: float, *, last: bool = False) -> bytes:
    """Re-encode one CBOR float64 field with another value, whether the field
    sits at the top level of the wire or inside an embedded byte vector."""
    for encode in (bytes, _cbor_byte_array):
        pattern = encode(_cbor_f64(value))
        if pattern in blob:
            index = blob.rfind(pattern) if last else blob.find(pattern)
            return blob[:index] + encode(_cbor_f64(replacement)) + blob[index + len(pattern) :]
    raise AssertionError("the semantic field is not encoded as a CBOR float64")


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
    assert result["scope"] == "single_source_z_transport_cited_joints_sound_incomplete"
    true_mass = sum(
        p for atom, p in zip(result["atoms"], result["probabilities"], strict=True) if atom == [1.0]
    )
    if empirical:
        interval = result["interval"]
        assert interval["available"] is True
        assert interval["method"] == "percentile_bootstrap"
        assert interval["reason"] == "estimator_grid_not_measured"
        mean = interval["mean_intervals"][0]
        assert mean["lower"] <= true_mass <= mean["upper"]
    else:
        assert result["interval"] == {"available": False, "reason": "no_interval_reported"}
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
    # Rewrite the recorded baseline contrast: the consumer recomputes the range
    # from the embedded point artifact and refuses the edited claim.
    tampered_sensitivity = _rewrite_f64(
        sensitivity_artifact, sensitivity["baseline"], sensitivity["baseline"] + 0.05, last=True
    )
    with pytest.raises(CausalSerializationError, match="z-transport sensitivity .*mismatch"):
        transport.consume_z_transport_sensitivity_artifact(tampered_sensitivity)
    assert replayed_sensitivity["estimand"] == sensitivity["estimand"]
    assert (
        replayed_sensitivity["baseline_binding"]["query"]
        == sensitivity["baseline_binding"]["query"]
    )
    assert (
        replayed_sensitivity["baseline_binding"]["provider_snapshot"]
        == sensitivity["baseline_binding"]["provider_snapshot"]
    )


def test_z_transport_prepare_exact_compiles_a_checked_plan_after_builder_disposal():
    """Preparation compiles the exact table against the checked proof; the stage may then go."""
    graph_builder, builder, catalog, _, laws = fixture(False)
    program = builder.inspect_proof(catalog)
    assert program["rules"] == ["ztr.surrogate_factorization"]
    prepared = builder.prepare_exact(catalog, laws, {"x": 0.0})
    del builder, graph_builder
    result = json.loads(prepared.estimate())
    assert result["status"] == "available"
    assert result["scope"] == "single_source_z_transport_cited_joints_sound_incomplete"
    true_mass = sum(
        p for atom, p in zip(result["atoms"], result["probabilities"], strict=True) if atom == [1.0]
    )
    # Independent arithmetic: P(y=1 | do(x=0)) = 0.2 under the fixture's Bernoulli law.
    assert true_mass == pytest.approx(0.2, abs=1e-12)
    # An exact law stays point-only; no interval is attached at estimate.
    assert result["interval"] == {"available": False, "reason": "no_interval_reported"}
    consumed = json.loads(transport.consume_z_transport_artifact(prepared.export()))
    assert consumed["proof"]["rules"] == program["rules"]
    assert consumed["probabilities"] == pytest.approx(result["probabilities"])
    prepared.refresh(laws)
    assert json.loads(prepared.estimate())["probabilities"] == pytest.approx(
        result["probabilities"]
    )


@pytest.mark.parametrize(
    "estimator",
    ["empirical_support_bayesian_bootstrap", "state_space_dirichlet"],
)
def test_bayesian_z_transport_publishes_an_unmeasured_posterior(estimator):
    _, _, _, prepared, _ = fixture(True)
    plugin = json.loads(prepared.estimate())
    posterior = json.loads(prepared.estimate(estimator=estimator, posterior_draws=40))
    assert posterior["probabilities"] == pytest.approx(plugin["probabilities"])
    interval = posterior["interval"]
    assert interval["available"] is True
    assert interval["method"] == "posterior_equal_tail"
    assert interval["reason"] == "estimator_grid_not_measured"
    mean = interval["mean_intervals"][0]
    true_mass = sum(
        p
        for atom, p in zip(posterior["atoms"], posterior["probabilities"], strict=True)
        if atom == [1.0]
    )
    assert mean["lower"] <= true_mass <= mean["upper"]
    assert prepared.interval_type == "posterior_equal_tail"


@pytest.mark.parametrize(
    "estimator",
    ["empirical_support_bayesian_bootstrap", "state_space_dirichlet"],
)
def test_bayesian_z_transport_posterior_executes_retained_plan_after_builder_disposal(estimator):
    """The posterior interval is drawn around the retained plug-in point with no stage alive."""
    graph_builder, builder, catalog, prepared, _ = fixture(True)
    program = builder.inspect_proof(catalog)
    assert program["rules"]
    del builder, graph_builder
    plugin = json.loads(prepared.estimate())
    posterior = json.loads(prepared.estimate(estimator=estimator, posterior_draws=40))
    # The point stays the empirical plug-in; only the interval changes.
    assert posterior["probabilities"] == pytest.approx(plugin["probabilities"])
    interval = posterior["interval"]
    assert interval["available"] is True
    assert interval["method"] == "posterior_equal_tail"
    assert interval["reason"] == "estimator_grid_not_measured"
    mean = interval["mean_intervals"][0]
    true_mass = sum(
        p
        for atom, p in zip(posterior["atoms"], posterior["probabilities"], strict=True)
        if atom == [1.0]
    )
    assert mean["lower"] <= true_mass <= mean["upper"]
    assert prepared.interval_type == "posterior_equal_tail"
    # Portable artifacts carry the point result only; the posterior is never replayed.
    consumed = json.loads(transport.consume_z_transport_artifact(prepared.export()))
    assert consumed["probabilities"] == pytest.approx(posterior["probabilities"])
    assert consumed["interval"] == {"available": False, "reason": "no_interval_reported"}


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
    assert independently_consumed["interval"] == {
        "available": False,
        "reason": "no_interval_reported",
    }

    prepared.refresh(laws)
    refreshed = json.loads(prepared.estimate())
    assert refreshed["probabilities"] == pytest.approx(first["probabilities"])
    refreshed_consumer = json.loads(transport.consume_z_transport_artifact(prepared.export()))
    assert refreshed_consumer["probabilities"] == pytest.approx(refreshed["probabilities"])


def test_z_transport_identification_proof_and_binding_survive_query_disposal():
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
    builder = stage
    del graph, query

    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    regimes = tuple(
        transport.EvidenceRegime(
            f"do_z_{value}",
            "source",
            kind="experimental",
            interventions=["z"],
            intervention_values={"z": float(value)},
            measured=names,
        )
        for value in (0, 1)
    )
    catalog = transport.EvidenceCatalog(
        environments=(transport.Environment("source", coordinates),),
        regimes=regimes,
        bindings=tuple(
            transport.RegimeBinding(
                regime.id,
                f"snapshot_{regime.id}",
                schema_names=names,
                sampling="independent",
                dependence="independent_studies",
            )
            for regime in regimes
        ),
    )
    plan = builder.inspect_proof(catalog)
    assert plan["rules"]
    assert plan["factors"]
    assert all(factor["supplied_by"] is not None for factor in plan["factors"])

    # Keep only the identified stage, checked catalog, and laws after this point.
    laws = (
        transport.ExactDiscreteLaw(
            "source",
            "do_z_0",
            (("w", (0.0, 1.0)), ("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
            (0.15, 0.10, 0.10, 0.15, 0.15, 0.10, 0.10, 0.15),
            "snapshot_do_z_0",
            interventions=(("z", 0.0),),
        ),
    )
    prepared = builder.prepare_exact(catalog, laws, {"x": 0.0})
    del builder, stage
    result = json.loads(prepared.estimate())
    assert result["status"] == "available"
    assert sum(
        p for atom, p in zip(result["atoms"], result["probabilities"], strict=True) if atom == [1.0]
    ) == pytest.approx(0.4)
    prepared.refresh(laws)
    assert json.loads(prepared.estimate())["probabilities"] == pytest.approx(
        result["probabilities"]
    )
    replay = json.loads(transport.consume_z_transport_artifact(prepared.export()))
    assert replay["probabilities"] == pytest.approx(result["probabilities"])


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


def test_restricted_experiment_obstruction_is_structural_and_replays_snapshot():
    names = ["x", "y", "z"]
    graph = Admg.from_edges(names, [("x", "y")], [("x", "y")])
    query = transport.ZTransportQuery(
        transport.SelectionDiagram("source", "target", []),
        outcomes=["y"],
        treatments=["x"],
        controllable=["z"],
        experiment_assignment={},
    )
    stage = transport.identify_z_transport(graph=graph, query=query)
    assert stage.outcome == "not_certified"
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    target = transport.EvidenceRegime(
        "target_joint", "target", kind="observational", measured=names
    )
    source_zero = transport.EvidenceRegime(
        "do_z_0",
        "source",
        kind="experimental",
        interventions=["z"],
        intervention_values={"z": 0.0},
        measured=names,
    )
    source_one = transport.EvidenceRegime(
        "do_z_1",
        "source",
        kind="experimental",
        interventions=["z"],
        intervention_values={"z": 1.0},
        measured=names,
    )
    full = transport.EvidenceCatalog(
        environments=(
            transport.Environment("source", coordinates),
            transport.Environment("target", coordinates),
        ),
        regimes=(target, source_zero, source_one),
        bindings=tuple(
            transport.RegimeBinding(
                regime,
                f"snapshot_{regime}",
                schema_names=names,
                sampling="independent",
                dependence="independent_studies",
            )
            for regime in ("target_joint", "do_z_0", "do_z_1")
        ),
    )
    decision = stage.decide(full)
    assert decision["outcome"] == "proven_non_transportable"
    assert decision["obstruction"]["terminal"]["treatments"]
    snapshot = stage.failure_snapshot(full)
    wire = json.loads(snapshot)
    assert wire["version"] == 2
    assert wire["status"] == "proof_obstruction"
    assert wire["z_obstruction"] == decision["obstruction"]
    summary, proposals = transport.plan_z_transport_evidence(
        stage, full, [], failure_snapshot=snapshot
    )
    assert not summary["ranked_sufficient"] and not proposals

    tampered = json.loads(snapshot)
    tampered["z_obstruction"]["terminal"]["treatments"] = []
    with pytest.raises(CausalSerializationError, match="z_transport.obstruction_record_mismatch"):
        transport.plan_z_transport_evidence(
            stage, full, [], failure_snapshot=json.dumps(tampered).encode()
        )
    with pytest.raises(CausalSerializationError, match="does not match this stage and catalog"):
        transport.plan_z_transport_evidence(
            stage,
            full,
            [],
            failure_snapshot=stage.failure_snapshot(
                full.__class__(
                    environments=full.environments,
                    regimes=full.regimes[:2],
                    bindings=full.bindings[:2],
                )
            ),
        )
    partial = transport.EvidenceCatalog(
        environments=full.environments,
        regimes=(target, source_zero),
        bindings=full.bindings[:2],
    )
    partial_decision = stage.decide(partial)
    assert partial_decision["outcome"] == "proven_non_transportable"
    assert partial_decision["obstruction"]["terminal"] == decision["obstruction"]["terminal"]
    assert json.loads(stage.failure_snapshot(partial))["status"] == "proof_obstruction"


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
        "do_z_1",
        "source",
        kind="experimental",
        interventions=["z"],
        intervention_values={"z": 1.0},
        measured=names,
    )
    base = transport.EvidenceCatalog(
        environments=(transport.Environment("source", coordinates),),
        regimes=(high,),
        bindings=(
            transport.RegimeBinding(
                "do_z_1",
                "snapshot_z1",
                schema_names=names,
                sampling="independent",
                dependence="independent_studies",
            ),
        ),
    )
    low_proposed = transport.EvidenceRegime(
        "do_z_0",
        "source",
        kind="experimental",
        evidence_kind="proposed",
        interventions=["z"],
        intervention_values={"z": 0.0},
        measured=names,
    )
    hypothetical = transport.EvidenceCatalog(
        environments=base.environments,
        regimes=(high, low_proposed),
        bindings=base.bindings,
    )
    candidate = transport.ZTransportCandidate(
        "joint_z0",
        hypothetical,
        "intervene",
        targets=["z"],
        cost=1.0,
        sample_budget=100,
        recruitment_sampling="randomized source study",
        feasibility_constraints=["z is manipulable"],
    )
    decision = stage.decide(base)
    assert decision["outcome"] == "missing_evidence"
    assert decision["reason"] == "z_transport.missing_evidence"
    assert decision["missing"]["kind"] in {"cited_factor", "unassigned_controllable"}
    assert "VariableId" not in json.dumps(decision)
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
    # Rename a bound snapshot inside the frozen failure catalog: the proposal's
    # catalog digest no longer matches, and replay refuses by name.
    edited = json.loads(portable_proposal)
    assert "snapshot_z1" in json.dumps(edited["snapshot"]["catalog"])
    edited["snapshot"]["catalog"] = json.loads(
        json.dumps(edited["snapshot"]["catalog"]).replace("snapshot_z1", "snapshot_z1_edited")
    )
    with pytest.raises(CausalSerializationError, match="z-transport planning digest mismatch"):
        transport.replay_z_transport_proposal(json.dumps(edited).encode())

    low_available = transport.EvidenceRegime(
        "do_z_0",
        "source",
        kind="experimental",
        evidence_kind="available",
        interventions=["z"],
        intervention_values={"z": 0.0},
        measured=names,
    )
    actual = transport.EvidenceCatalog(
        environments=base.environments,
        regimes=(high, low_available),
        bindings=(
            base.bindings[0],
            transport.RegimeBinding(
                "do_z_0",
                "snapshot_z0_arrival",
                schema_names=names,
                sampling="independent",
                dependence="independent_studies",
            ),
        ),
    )
    probabilities = []
    for w in (0, 1):
        for x in (0, 1):
            for y in (0, 1):
                probabilities.append(
                    (0.25 if w else 0.75) * (0.35 if x else 0.65) * (0.8 if y == x else 0.2)
                )
    law = transport.ExactDiscreteLaw(
        "source",
        "do_z_0",
        (("w", (0.0, 1.0)), ("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
        tuple(probabilities),
        "snapshot_z0_arrival",
        interventions=(("z", 0.0),),
    )
    with pytest.raises(CausalUnsupportedError, match="empirical_counts_required") as refused:
        proposal.receive(actual, (law,), {"x": 0.0}, "snapshot_z0_arrival", empirical=True)
    assert refused.value.reason_code == "transport_missing_provider"
    arrived = proposal.receive(actual, (law,), {"x": 0.0}, "snapshot_z0_arrival")
    result = json.loads(arrived.estimate())
    y1 = sum(
        p for atom, p in zip(result["atoms"], result["probabilities"], strict=True) if atom == [1.0]
    )
    assert y1 == pytest.approx(0.2)
    assert result["interval"] == {"available": False, "reason": "no_interval_reported"}

    with pytest.raises(CausalValueError, match="provider snapshot"):
        proposal.receive(actual, (law,), {"x": 0.0}, "snapshot_z1")


def test_z_transport_decide_accepts_cited_margin_without_the_experiment_family():
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
    assert stage.outcome == "identified"
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    regime = transport.EvidenceRegime(
        "do_z_0",
        "source",
        kind="experimental",
        interventions=["z"],
        intervention_values={"z": 0.0},
        measured=["w", "x", "y"],
    )
    catalog = transport.EvidenceCatalog(
        environments=(transport.Environment("source", coordinates),),
        regimes=(regime,),
        bindings=(
            transport.RegimeBinding(
                "do_z_0",
                "snapshot_z0",
                schema_names=["w", "x", "y"],
                sampling="independent",
                dependence="independent_studies",
            ),
        ),
    )
    decision = stage.decide(catalog)
    assert decision["outcome"] == "identified"
    assert decision["proof"]["rules"] == ["ztr.surrogate_factorization"]
    probabilities = []
    for w in (0, 1):
        for x in (0, 1):
            for y in (0, 1):
                probabilities.append(
                    (0.25 if w else 0.75) * (0.35 if x else 0.65) * (0.8 if y == x else 0.2)
                )
    law = transport.ExactDiscreteLaw(
        "source",
        "do_z_0",
        (("w", (0.0, 1.0)), ("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
        tuple(probabilities),
        "snapshot_z0",
        interventions=(("z", 0.0),),
    )
    prepared = stage.prepare_exact(catalog, (law,), {"x": 0.0})
    result = json.loads(prepared.estimate())
    assert result["scope"] == "single_source_z_transport_cited_joints_sound_incomplete"
    true_mass = sum(
        p for atom, p in zip(result["atoms"], result["probabilities"], strict=True) if atom == [1.0]
    )
    assert true_mass == pytest.approx(0.2)


def test_z_transport_artifact_tamper_is_refused_by_the_consumer():
    """The consumer recomputes the point from the embedded proof and laws."""
    _, _, _, prepared, laws = fixture(False)
    result = json.loads(prepared.estimate())
    artifact = prepared.export()
    assert artifact.startswith(Z_PREFIX)
    consumed = json.loads(transport.consume_z_transport_artifact(artifact))
    assert consumed["outcomes"] == ["y"]
    assert consumed["proof"]["rules"] == ["ztr.surrogate_factorization"]
    # An edited point result no longer matches the recomputed one.
    moved = _rewrite_f64(artifact, result["probabilities"][1], result["probabilities"][1] + 0.05)
    with pytest.raises(CausalSerializationError, match="point result does not replay"):
        transport.consume_z_transport_artifact(moved)
    # An edited embedded law changes the recomputed point (or fails law validation).
    edited_law = _rewrite_f64(artifact, laws[0].probabilities[0], laws[0].probabilities[0] + 0.01)
    with pytest.raises((CausalSerializationError, CausalValueError)):
        transport.consume_z_transport_artifact(edited_law)
    # Foreign framing is refused before any byte is trusted.
    with pytest.raises(CausalSerializationError, match="invalid z-transport artifact format"):
        transport.consume_z_transport_artifact(
            b"ANTECEDENT-EXACT-TRANSPORT\x01" + artifact[len(Z_PREFIX) :]
        )


def test_z_transport_consumer_rechecks_the_embedded_point_after_builder_disposal():
    """The consumer recomputes the point from the embedded proof and laws with no producer alive."""
    graph_builder, builder, catalog, prepared, laws = fixture(False)
    program = builder.inspect_proof(catalog)
    assert program["rules"] == ["ztr.surrogate_factorization"]
    result = json.loads(prepared.estimate())
    artifact = prepared.export()
    assert artifact.startswith(Z_PREFIX)
    del builder, graph_builder, prepared
    consumed = json.loads(transport.consume_z_transport_artifact(artifact))
    assert consumed["outcomes"] == ["y"]
    assert consumed["proof"]["rules"] == program["rules"]
    assert consumed["probabilities"] == pytest.approx(result["probabilities"])
    # Loaded artifacts stay point-only: no bootstrap interval is replayed.
    assert consumed["interval"] == {"available": False, "reason": "no_interval_reported"}
    # An edited point result no longer matches the point recomputed from the embedded law.
    moved = _rewrite_f64(artifact, result["probabilities"][1], result["probabilities"][1] + 0.05)
    with pytest.raises(CausalSerializationError, match="point result does not replay"):
        transport.consume_z_transport_artifact(moved)
    # An edited embedded law changes the recomputed point or fails law validation.
    edited_law = _rewrite_f64(artifact, laws[0].probabilities[0], laws[0].probabilities[0] + 0.01)
    with pytest.raises((CausalSerializationError, CausalValueError)):
        transport.consume_z_transport_artifact(edited_law)


def test_z_transport_refusals_carry_typed_classes_and_registered_codes():
    graph, stage, catalog, prepared, laws = fixture(False)
    law = laws[0]
    with pytest.raises(CausalUnsupportedError, match="empirical_counts_required") as refused:
        stage.prepare_empirical(catalog, (law,), {"x": 0.0})
    assert refused.value.reason_code == "transport_missing_provider"
    with pytest.raises(CausalValueError, match="unknown z-transport provider"):
        prepared.estimate(estimator="not_a_provider")
    with pytest.raises(CausalValueError, match="posterior_draws requires"):
        prepared.estimate(posterior_draws=5)
    fresh = stage.prepare_exact(catalog, (law,), {"x": 0.0})
    for call in (
        lambda: fresh.mechanism_sensitivity(0.2),
        fresh.export,
        lambda: fresh.export_sensitivity(0.2),
    ):
        with pytest.raises(CausalUnsupportedError, match="no_execution_claim") as refused:
            call()
        assert refused.value.reason_code == "not_executed"
    from antecedent.state import CancellationToken

    token = CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError, match="cancel"):
        fresh.estimate(cancel=token)
    with pytest.raises(CausalResourceError, match="memory|budget"):
        stage.prepare_exact(catalog, (law,), {"x": 0.0}, memory_bytes=1)
    with pytest.raises(CausalValueError, match="non-negative"):
        transport.ExactDiscreteLaw(
            law.population,
            law.regime,
            law.axes,
            law.probabilities,
            law.snapshot_identity,
            interventions=law.interventions,
            empirical_counts=(-1,) * len(law.probabilities),
        )
    with pytest.raises(CausalValueError, match="memory_bytes"):
        transport.identify_z_transport(
            graph=graph,
            query=transport.ZTransportQuery(
                transport.SelectionDiagram("source", "target", []),
                outcomes=["y"],
                treatments=["x"],
                controllable=["z"],
                experiment_assignment={"z": 0.0},
            ),
            memory_bytes=-1,
        )
    from antecedent.transport import TransportControls

    with pytest.raises(CausalValueError, match="non-negative"):
        TransportControls(max_depth=-1)
    refused_stage = transport.identify_z_transport(
        graph=graph,
        query=transport.ZTransportQuery(
            transport.SelectionDiagram("source", "target", ["y"]),
            outcomes=["y"],
            treatments=["x"],
            controllable=["z"],
            experiment_assignment={"z": 0.0},
        ),
    )
    assert refused_stage.outcome == "not_certified"
    with pytest.raises(CausalUnsupportedError, match="no_checked_recursive_formula") as refused:
        refused_stage.inspect_proof(catalog)
    assert refused.value.reason_code == "transport_not_certified"
    with pytest.raises(CausalValueError, match="design_kind"):
        transport.ZTransportCandidate("bad", catalog, "observe")


def test_z_transport_plan_evidence_refuses_a_candidate_that_rewrites_the_failure_catalog():
    names = ["w", "z", "x", "y"]
    graph, stage, catalog, _prepared, _laws = fixture(False)
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    base = transport.EvidenceCatalog(
        environments=(transport.Environment("source", coordinates),),
        regimes=catalog.regimes[1:],
        bindings=catalog.bindings[1:],
    )
    rewritten = transport.EvidenceCatalog(
        environments=base.environments,
        regimes=(
            *catalog.regimes[1:],
            transport.EvidenceRegime(
                "do_z_0",
                "source",
                kind="experimental",
                evidence_kind="proposed",
                interventions=["z"],
                intervention_values={"z": 0.0},
                measured=names,
            ),
        ),
        bindings=base.bindings,
        target_sampling="representative_sample",
    )
    candidate = transport.ZTransportCandidate(
        "rewrite", rewritten, "intervene", targets=["z"], cost=1.0, sample_budget=10
    )
    with pytest.raises(CausalValueError, match="must preserve the failure catalog"):
        transport.plan_z_transport_evidence(stage, base, [candidate])
    del graph


def test_z_transport_estimate_reports_its_seed_and_limits_are_inherited():
    _, stage, catalog, _prepared, laws = fixture(True)
    prepared = stage.prepare_empirical(catalog, laws, {"x": 0.0}, seed=7)
    assert prepared.seed == 7
    first = json.loads(prepared.estimate())
    assert first["seed"] == 7 and first["interval"]["seed"] == 7
    again = json.loads(prepared.estimate(seed=11))
    assert again["seed"] == 11 and prepared.seed == 11
    assert again["probabilities"] == pytest.approx(first["probabilities"])
    bounded = transport.identify_z_transport(
        graph=Admg.from_edges(
            ["w", "z", "x", "y"],
            [("w", "z"), ("z", "x"), ("x", "y"), ("w", "y")],
            [("w", "y"), ("z", "y"), ("z", "x")],
        ),
        query=transport.ZTransportQuery(
            transport.SelectionDiagram("source", "target", []),
            outcomes=["y"],
            treatments=["x"],
            controllable=["z"],
            experiment_assignment={"z": 0.0},
        ),
        memory_bytes=1,
    )
    # The stage's limits are inherited by every preparation made from it.
    with pytest.raises(CausalResourceError, match="memory|budget"):
        bounded.prepare_exact(catalog, laws, {"x": 0.0})


def _rewrite_version(blob: bytes, current: int, replacement: int) -> bytes:
    """Re-encode the first top-level CBOR ``version`` field of a wire, whether
    the wire is bare or embedded as a byte vector inside the Python frame."""
    assert 0 <= current < 24 and 0 <= replacement < 24
    for encode in (bytes, _cbor_byte_array):
        pattern = encode(b"\x67version" + bytes([current]))
        if pattern in blob:
            index = blob.find(pattern)
            return (
                blob[:index]
                + encode(b"\x67version" + bytes([replacement]))
                + blob[index + len(pattern) :]
            )
    raise AssertionError("the wire does not carry a small-integer version field")


def test_z_transport_version_one_artifacts_are_refused_by_the_consumers():
    """Every z-transport wire is version 2; version 1 bytes are refused by their
    typed version check before any embedded field is trusted."""
    from antecedent import _native

    _, stage, catalog, prepared, _ = fixture(False)
    prepared.estimate()
    point = prepared.export()
    assert _rewrite_version(point, 2, 2) == point
    with pytest.raises(CausalSerializationError, match="unsupported container version 1"):
        transport.consume_z_transport_artifact(_rewrite_version(point, 2, 1))
    sensitivity = prepared.export_sensitivity(0.2, 0.4)
    with pytest.raises(CausalSerializationError, match="unsupported container version 1"):
        transport.consume_z_transport_sensitivity_artifact(_rewrite_version(sensitivity, 2, 1))
    missing_low = transport.EvidenceCatalog(
        environments=catalog.environments,
        regimes=catalog.regimes[1:],
        bindings=catalog.bindings[1:],
    )
    snapshot = json.loads(stage.failure_snapshot(missing_low))
    snapshot["version"] = 7
    with pytest.raises(CausalSerializationError, match="unsupported z-transport planning wire"):
        _native.consume_z_transport_failure_snapshot(json.dumps(snapshot).encode())


def test_z_transport_consumer_limits_below_the_artifact_declared_limits_refuse():
    """Consumer limits are the consumer's own: an artifact that recorded a larger
    evaluation budget, or stores more laws than allowed, is refused as a resource
    refusal rather than replayed under its own limits."""
    _, _, _, prepared, _ = fixture(False)
    prepared.estimate()
    artifact = prepared.export()
    consumed = json.loads(transport.consume_z_transport_artifact(artifact, max_laws=1))
    assert consumed["premises_digest"]
    with pytest.raises(CausalResourceError, match="consumer limit exceeded"):
        transport.consume_z_transport_artifact(artifact, max_operations=1)
    with pytest.raises(CausalResourceError, match="consumer limit exceeded"):
        transport.consume_z_transport_artifact(artifact, max_laws=0)
    with pytest.raises(CausalResourceError, match="consumer limit exceeded"):
        transport.consume_z_transport_artifact(artifact, max_law_cells=1)
    sensitivity = prepared.export_sensitivity(0.2, 0.4)
    replayed = transport.consume_z_transport_sensitivity_artifact(sensitivity)
    assert replayed["baseline_binding"]["premises_digest"] == consumed["premises_digest"]
    with pytest.raises(CausalResourceError, match="consumer limit exceeded"):
        transport.consume_z_transport_sensitivity_artifact(sensitivity, max_operations=1)
    with pytest.raises(CausalValueError, match="max_laws"):
        transport.consume_z_transport_artifact(artifact, max_laws=-1)


def test_z_transport_estimate_reports_interval_method_and_reason_separately():
    _, _, _, prepared, _ = fixture(True)
    assert prepared.interval_method is None
    assert prepared.interval_reason == "no_interval_reported"
    result = json.loads(prepared.estimate())
    assert result["interval"]["method"] == "percentile_bootstrap"
    assert result["interval"]["reason"] == "estimator_grid_not_measured"
    assert prepared.interval_method == "percentile_bootstrap"
    assert prepared.interval_reason == "estimator_grid_not_measured"
    assert prepared.interval_type == "percentile_bootstrap"


def test_z_transport_refresh_on_an_empirical_handle_requires_counts():
    """An empirical plug-in handle keeps its contract through refresh: laws
    without counts are refused with the typed missing-provider code."""
    _, _, _, prepared, laws = fixture(True)
    law = laws[0]
    count_free = transport.ExactDiscreteLaw(
        law.population,
        law.regime,
        law.axes,
        law.probabilities,
        law.snapshot_identity,
        interventions=law.interventions,
    )
    before = json.loads(prepared.estimate())
    with pytest.raises(CausalUnsupportedError, match="empirical_counts_required") as refused:
        prepared.refresh((count_free,))
    assert refused.value.reason_code == "transport_missing_provider"
    # The previous prepared state survives the refused refresh.
    assert json.loads(prepared.estimate())["probabilities"] == pytest.approx(
        before["probabilities"]
    )


def test_z_transport_unreconciled_empirical_counts_are_refused():
    """Counts whose frequencies disagree with the table are refused as invalid
    input (the exact-law error class), never accepted as an empirical table."""
    _, stage, catalog, _, laws = fixture(True)
    law = laws[0]
    assert law.empirical_counts is not None
    reversed_counts = tuple(reversed(law.empirical_counts))
    assert reversed_counts != law.empirical_counts
    unreconciled = transport.ExactDiscreteLaw(
        law.population,
        law.regime,
        law.axes,
        law.probabilities,
        law.snapshot_identity,
        interventions=law.interventions,
        empirical_counts=reversed_counts,
    )
    with pytest.raises(CausalValueError, match="unreconciled_empirical_counts"):
        stage.prepare_empirical(catalog, (unreconciled,), {"x": 0.0})
    with pytest.raises(CausalValueError, match="unreconciled_empirical_counts"):
        stage.prepare_exact(catalog, (unreconciled,), {"x": 0.0})


class _BinaryScm:
    """Binary structural causal model enumerated exactly from its equations.

    ``mechanisms[i](values, exogenous)`` gives node ``i`` from earlier nodes and
    the independent exogenous bits; ``exogenous_p`` is the chance each bit is one.
    """

    def __init__(self, exogenous_p, mechanisms):
        self.exogenous_p = tuple(exogenous_p)
        self.mechanisms = tuple(mechanisms)

    def law(self, do, measured):
        """Exact joint law over ``measured`` (first variable most significant)."""
        out = [0.0] * (1 << len(measured))
        m = len(self.exogenous_p)
        for mask in range(1 << m):
            exogenous = [(mask >> bit) & 1 for bit in range(m)]
            weight = 1.0
            for bit, p in enumerate(self.exogenous_p):
                weight *= p if exogenous[bit] else 1.0 - p
            values = [0] * len(self.mechanisms)
            for i, mechanism in enumerate(self.mechanisms):
                values[i] = do[i] if i in do else mechanism(values, exogenous)
            index = 0
            for v in measured:
                index = (index << 1) | values[v]
            out[index] += weight
        return out

    def risk(self, do, outcome):
        return self.law(do, [outcome])[1]


def _two_exchange_fixture():
    """The asymmetric two-exchange model of the Rust known-truth test: X→Z→Y with
    X↔Z, X↔Y, Z↔V and an isolated R, whose P(Y=1 | do(x, z)) varies with z."""
    names = ["x", "z", "v", "y", "r"]
    scm = _BinaryScm(
        [0.3, 0.6, 0.5, 0.5, 0.4, 0.7, 0.25],
        [
            lambda _v, e: int(e[0] == 1) ^ int(e[1] == 1 and e[4] == 1),
            lambda v, e: int(v[0] == 1 and e[5] == 1) ^ int(e[0] == 1),
            lambda v, e: int(v[1] == 1) ^ int(e[2] == 1),
            lambda v, e: int((v[1] == 1 and e[1] == 1) or e[6] == 1),
            lambda _v, e: e[3],
        ],
    )
    graph = Admg.from_edges(names, [("x", "z"), ("z", "y")], [("x", "z"), ("x", "y"), ("z", "v")])
    # The full source experiment family over {x, z} plus both observational laws.
    specs = [("target", {}), ("source", {})]
    for variables in ([0], [1], [0, 1]):
        for levels in range(1 << len(variables)):
            specs.append(("source", {v: (levels >> bit) & 1 for bit, v in enumerate(variables)}))
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    regimes, bindings, laws = [], [], []
    for k, (population, assignments) in enumerate(specs):
        regime_id = f"{population}-{k}"
        measured = [names[i] for i in range(len(names)) if i not in assignments]
        regimes.append(
            transport.EvidenceRegime(
                regime_id,
                population,
                kind="experimental" if assignments else "observational",
                interventions=[names[i] for i in assignments],
                intervention_values={names[i]: float(level) for i, level in assignments.items()},
                measured=measured,
            )
        )
        bindings.append(
            transport.RegimeBinding(
                regime_id,
                f"snapshot_{regime_id}",
                schema_names=measured,
                sampling="independent",
                dependence="independent_studies",
            )
        )
        laws.append(
            transport.ExactDiscreteLaw(
                population,
                regime_id,
                tuple((name, (0.0, 1.0)) for name in measured),
                tuple(scm.law(assignments, [names.index(name) for name in measured])),
                f"snapshot_{regime_id}",
                interventions=tuple((names[i], float(level)) for i, level in assignments.items()),
            )
        )
    catalog = transport.EvidenceCatalog(
        environments=(
            transport.Environment("source", coordinates),
            transport.Environment("target", coordinates),
        ),
        regimes=tuple(regimes),
        bindings=tuple(bindings),
    )
    query = transport.ZTransportQuery(
        transport.SelectionDiagram("source", "target", []),
        outcomes=["y"],
        treatments=["x"],
        controllable=["x", "z"],
        experiment_assignment={"x": 0.0, "z": 0.0},
    )
    return scm, graph, query, catalog, tuple(laws)


def _y_risk(result):
    return sum(
        p for atom, p in zip(result["atoms"], result["probabilities"], strict=True) if atom == [1.0]
    )


def test_z_transport_exchanged_treatment_binds_symbolically_to_the_request():
    """The recursive derivation exchanges two source factors; the treatment is
    bound by the request and the summed exchange coordinate by its summation,
    so requesting x=1 answers from the do(x=1) experiments, never from the
    declared x=0 level."""
    scm, graph, query, catalog, laws = _two_exchange_fixture()
    truth = [scm.risk({0: 0}, 3), scm.risk({0: 1}, 3)]
    assert truth[0] == pytest.approx(0.385, abs=5e-4)
    assert truth[1] == pytest.approx(0.511, abs=5e-4)
    stage = transport.identify_z_transport(graph=graph, query=query)
    assert stage.outcome == "identified"
    decision = stage.decide(catalog)
    assert decision["outcome"] == "identified"
    proof = decision["proof"]
    assert proof.get("surrogate") is None and proof.get("confounder") is None
    exchanges = [rule for rule in proof["rules"] if "line10.source_exchange" in rule]
    assert len(exchanges) == 2
    # Symbolic exchange coordinates are spelled as unbound (`None`) levels.
    assert any("(0, None)" in rule for rule in proof["rules"]), proof["rules"]
    risks = []
    for level, expected in ((0.0, truth[0]), (1.0, truth[1])):
        prepared = stage.prepare_exact(catalog, laws, {"x": level})
        result = json.loads(prepared.estimate())
        risk = _y_risk(result)
        assert risk == pytest.approx(expected, abs=1e-12)
        consumed = json.loads(transport.consume_z_transport_artifact(prepared.export()))
        assert consumed["proof"].get("surrogate") is None
        assert _y_risk(consumed) == pytest.approx(risk, abs=1e-12)
        risks.append(risk)
    assert abs(risks[0] - risks[1]) > 0.05


def test_z_transport_treatment_request_outside_the_cited_regimes_is_refused():
    """A treatment level no cited source experiment supplies is a typed
    refusal, never a number computed from another level's law."""
    _scm, graph, query, catalog, laws = _two_exchange_fixture()
    stage = transport.identify_z_transport(graph=graph, query=query)
    assert stage.outcome == "identified"
    with pytest.raises(CausalUnsupportedError, match="no cited source experiment") as refused:
        stage.prepare_exact(catalog, laws, {"x": 7.0}).estimate()
    assert refused.value.reason_code == "transport_missing_provider"


def test_two_single_family_line11_terminals_are_not_certified_across_sources():
    """Two sources whose obstructions come from different experiment families
    are never combined into one impossibility claim: the decision is the named
    refusal to search the multi-source combination, not proven_non_transportable."""
    from antecedent import _native

    names = ["a", "b", "c", "d"]
    graph = Admg.from_edges(names, [("a", "b"), ("c", "d")], [("a", "b"), ("c", "d")])
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    empty = transport.EvidenceCatalog(
        environments=tuple(
            transport.Environment(population, coordinates)
            for population in ("alpha", "beta", "target")
        ),
        regimes=(),
        bindings=(),
    )
    decision = _native.decide_two_source_z_transport_stage(
        graph,
        "target",
        ["b", "d"],
        ["a", "c"],
        [("alpha", ["a"], {"a": 0.0}, []), ("beta", ["c"], {"c": 0.0}, [])],
        [empty, empty],
    )
    assert decision["outcome"] == "not_certified"
    assert decision["reason"] == "z_transport.multi_source_combination_not_searched"


def test_z_transport_plan_report_carries_the_evaluation_budget_receipt():
    """Planning under ``max_evaluated`` evaluates the first candidates only and
    names the rest as unevaluated; the receipt says the search was truncated."""
    names = ["w", "z", "x", "y"]
    graph, stage, catalog, _prepared, _laws = fixture(False)
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    base = transport.EvidenceCatalog(
        environments=(transport.Environment("source", coordinates),),
        regimes=catalog.regimes[1:],
        bindings=catalog.bindings[1:],
    )
    proposed = transport.EvidenceRegime(
        "do_z_0",
        "source",
        kind="experimental",
        evidence_kind="proposed",
        interventions=["z"],
        intervention_values={"z": 0.0},
        measured=names,
    )
    hypothetical = transport.EvidenceCatalog(
        environments=base.environments,
        regimes=(*base.regimes, proposed),
        bindings=base.bindings,
    )
    candidates = [
        transport.ZTransportCandidate(
            candidate_id,
            hypothetical,
            "intervene",
            targets=["z"],
            cost=cost,
            sample_budget=100,
        )
        for candidate_id, cost in (("cheap", 1.0), ("costly", 5.0))
    ]
    full, proposals = transport.plan_z_transport_evidence(stage, base, candidates)
    assert full["ranked_sufficient"] == ["cheap", "costly"]
    assert full["candidate_universe_size"] == 2 and full["search_limit"] == 2
    assert full["evaluated"] == ["cheap", "costly"] and full["unevaluated"] == []
    assert full["truncated"] is False and len(proposals) == 2
    truncated, proposals = transport.plan_z_transport_evidence(
        stage, base, candidates, max_evaluated=1
    )
    assert truncated["ranked_sufficient"] == ["cheap"]
    assert [item["id"] for item in truncated["assessments"]] == ["cheap"]
    assert truncated["candidate_universe_size"] == 2 and truncated["search_limit"] == 1
    assert truncated["evaluated"] == ["cheap"] and truncated["unevaluated"] == ["costly"]
    assert truncated["truncated"] is True and len(proposals) == 1
    with pytest.raises(CausalValueError, match="max_evaluated"):
        transport.plan_z_transport_evidence(stage, base, candidates, max_evaluated=0)
    del graph
