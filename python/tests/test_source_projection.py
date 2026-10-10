"""Actual original-source replay, scoped affine utilities and opaque execution resolution."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from antecedent import decision, program_claims
from antecedent.composition_bundle import Bundle, Failed, consume_bundle
from antecedent.source_projection import SourceProjectionArtifact, consume_source_projection

from test_program_claims import _native_program, _native_view, _response, _spec


def _truth():
    return json.loads(
        (
            Path(__file__).parents[2] / "conformance/composition/source_projection/expected.json"
        ).read_text(encoding="utf-8")
    )


def _claim():
    view = _native_view()
    return program_claims.native_claim(view, _native_program(view))


def _contract(quantities):
    return decision.Contract(
        actions=(
            decision.Action("A", (quantities[0],), 2 * decision.x(0) - 1),
            decision.Action("B", (quantities[1],), 0.5 * decision.x(0)),
        ),
        utility_units=_truth()["utility_units"],
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )


def _bundle(source, quantities, *, external_source=False):
    builder = Bundle.builder()
    for name, projection in [
        ("source", "attestation" if external_source else "causal_contract"),
        ("coordinates", "quantity_coordinates"),
        ("utility", "transformation"),
    ]:
        artifact = SourceProjectionArtifact.produce(
            source,
            projection,
            contract=_contract(quantities) if projection == "transformation" else None,
        )
        builder.add_artifact(artifact.kind, artifact.export(), node_id=name)
    builder.connect("source", "coordinates").connect("coordinates", "utility")
    return builder.build()


def test_source_projection_native_original_engine_affine_truth_and_unresolved_import():
    claim = _claim()
    artifact = SourceProjectionArtifact.produce(
        claim.source_evidence, "transformation", contract=_contract(claim.coordinates)
    )
    report = consume_source_projection(
        artifact.export(), expected_identity=artifact.identity
    ).report
    assert [action["expected_utility"] for action in report["output"]["actions"]] == pytest.approx(
        [5, 2.5], abs=_truth()["native_tolerance"] * 2
    )
    assert all(action["standard_error"] is None for action in report["output"]["actions"])
    assert report["output"]["calibration"] == _truth()["calibration"]
    assert report["unresolved"] == ["dependencies.checked_response_grid_operation"]
    bundle = _bundle(claim.source_evidence, claim.coordinates)
    imported = consume_bundle(bundle.export(), expected_identity=bundle.identity)
    assert not imported.all_verified
    assert isinstance(imported.node("source").status, Failed)
    assert imported.node("source").status.reason == "composition_bundle.callback_unavailable"
    resolved = consume_bundle(
        bundle.export(),
        expected_identity=bundle.identity,
        actual_claims={claim.source_evidence.source_artifact_digest: claim},
    )
    resolved.require_verified()
    assert resolved.value("utility", "A.expected_utility") == pytest.approx(
        _truth()["utilities"]["A"], abs=2e-4
    )
    assert all(node.facts["trust"] == "unverified" for node in resolved.nodes)
    assert all(
        node.facts["native_execution_authority_issued"] == "false" for node in resolved.nodes
    )


def test_source_projection_external_attestation_coordinates_and_affine_point_only_truth():
    claim = _spec().bind(replace(_response(), support=("supported", "supported")))
    bundle = _bundle(claim.export(), claim.quantities, external_source=True)
    consumed = consume_bundle(bundle.export(), expected_identity=bundle.identity)
    consumed.require_verified()
    assert consumed.value("utility", "A.expected_utility") == _truth()["utilities"]["A"]
    assert consumed.value("utility", "B.expected_utility") == _truth()["utilities"]["B"]
    assert all(node.facts["trust"] == "externally_attested" for node in consumed.nodes)
    assert all(node.facts["source_authentication_issued"] == "false" for node in consumed.nodes)


def test_source_projection_refuses_false_resolution_units_nonlinear_and_wrong_identity():
    claim = _claim()
    bundle = _bundle(claim.source_evidence, claim.coordinates)
    with pytest.raises(TypeError, match="issued NativeClaim"):
        consume_bundle(
            bundle.export(),
            expected_identity=bundle.identity,
            actual_claims={claim.source_evidence.source_artifact_digest: {"verified": True}},
        )
    with pytest.raises(ValueError, match="source_projection.source_binding_mismatch") as failure:
        consume_bundle(
            bundle.export(),
            expected_identity=bundle.identity,
            actual_claims={"changed-source": claim},
        )
    assert failure.value.reason_code == "invalid_argument"
    changed_view = _native_view(bayesian=True)
    changed_claim = program_claims.native_claim(changed_view, _native_program(changed_view))
    with pytest.raises(ValueError, match="source_projection.source_binding_mismatch"):
        consume_bundle(
            bundle.export(),
            expected_identity=bundle.identity,
            actual_claims={claim.source_evidence.source_artifact_digest: changed_claim},
        )
    missing_support = _spec().bind(_response())
    with pytest.raises(ValueError, match="source_projection.unsupported_coordinate"):
        SourceProjectionArtifact.produce(
            missing_support.export(),
            "transformation",
            contract=_contract(missing_support.quantities),
        )
    contract = _contract(claim.coordinates)
    wrong = replace(claim.coordinates[0], units="unconverted_units")
    with pytest.raises(ValueError, match="source_projection.coordinate_mismatch"):
        SourceProjectionArtifact.produce(
            claim.source_evidence,
            "transformation",
            contract=replace(
                contract,
                actions=(replace(contract.actions[0], inputs=(wrong,)), contract.actions[1]),
            ),
        )
    with pytest.raises(ValueError, match="source_projection.affine_contract_refused"):
        SourceProjectionArtifact.produce(
            claim.source_evidence,
            "transformation",
            contract=replace(
                contract,
                actions=(
                    replace(contract.actions[0], utility=decision.x(0) * decision.x(0)),
                    contract.actions[1],
                ),
            ),
        )
    artifact = SourceProjectionArtifact.produce(claim.source_evidence, "quantity_coordinates")
    with pytest.raises(ValueError, match="source_projection.identity_mismatch"):
        consume_source_projection(artifact.export(), expected_identity="changed-identity")


def test_source_projection_fresh_process_requires_actual_original_raw_data_execution(tmp_path):
    claim = _claim()
    bundle = _bundle(claim.source_evidence, claim.coordinates)
    path = tmp_path / "source-projection.art"
    path.write_bytes(bundle.export())
    script = """
import json,sys
from pathlib import Path
import numpy as np
import antecedent as ac
from antecedent import program_claims
from antecedent.composition_bundle import consume_bundle
raw=Path(sys.argv[1]).read_bytes()
assert not consume_bundle(raw,expected_identity=sys.argv[2]).all_verified
a=np.tile(np.array([0.,1.,2.,3.]),80); x=np.repeat(np.linspace(-1.,1.,80),4); y=1.+2.*a+.2*x
view=ac.analyze({'x':x,'a':a,'y':y},graph=[('x','a'),('x','y'),('a','y')],query=ac.ResponseCurve('a','y',grid=[1.,2.]),refute='none',bootstrap=0)
actual=program_claims.native_claim(view,program_claims.ProgramBinding.from_response(view,outcome_units='mmHg',dose_units='mg'))
resolved=consume_bundle(raw,expected_identity=sys.argv[2],actual_claims={sys.argv[3]:actual})
resolved.require_verified()
print(json.dumps({'A':resolved.value('utility','A.expected_utility'),'B':resolved.value('utility','B.expected_utility'),'trust':[n.facts['trust'] for n in resolved.nodes]}))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(path),
            bundle.identity,
            claim.source_evidence.source_artifact_digest,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["A"] == pytest.approx(_truth()["utilities"]["A"], abs=2e-4)
    assert report["B"] == pytest.approx(_truth()["utilities"]["B"], abs=2e-4)
    assert report["trust"] == ["unverified"] * 3


def test_source_projection_bounds_refuse_before_copy_or_actual_source_execution():
    claim = _claim()
    bundle = _bundle(claim.source_evidence, claim.coordinates)
    with pytest.raises(ValueError, match="source resolver count exceeds"):
        consume_bundle(
            bundle.export(),
            expected_identity=bundle.identity,
            actual_claims={str(index): claim for index in range(65)},
        )
    with pytest.raises(ValueError, match="source_projection.limits_exceeded") as failure:
        consume_source_projection(b"x" * (18 * 1024 * 1024 + 1), expected_identity="oversized")
    assert failure.value.reason_code == "invalid_argument"
