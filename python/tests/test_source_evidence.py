"""Original diagnostic scopes and explicit matching native-operation resolution."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from antecedent import decision, program_claims
from antecedent.source_evidence import SourceEvidence

from test_program_claims import _contract, _native_program, _native_view


def _claim():
    view = _native_view()
    return program_claims.native_claim(view, _native_program(view))


def test_source_evidence_preserves_original_scopes_and_semantic_coordinates():
    claim = _claim()
    evidence = claim.source_evidence
    assert claim.means == pytest.approx((3.0, 5.0), abs=1e-4)
    assert evidence.coordinates == claim.coordinates
    assert evidence.diagnostics
    assert evidence.resolution is None
    assert evidence.original_acceptance["unresolved_dependencies"] == [
        "dependencies.checked_response_grid_operation"
    ]
    for index, quantity in enumerate(evidence.coordinates):
        scoped = evidence.diagnostics_at(quantity)
        assert len(scoped) == len(evidence.diagnostics)
        for original, local in zip(evidence.diagnostics, scoped, strict=True):
            assert local.source_coordinate == quantity
            if original["scope"] == "per_coordinate":
                assert local.local_value == original["values"][index]
                assert local.global_values == ()
            elif original["scope"] == "global":
                assert local.local_value is None
                assert local.global_values == tuple(original["values"])
    loaded = SourceEvidence.consume(evidence.export())
    assert loaded.diagnostics == evidence.diagnostics
    assert loaded.coordinates == evidence.coordinates
    assert loaded.resolution is None


def test_source_evidence_projection_preserves_contributors_and_resolution_requires_issuer():
    claim = _claim()
    evidence = claim.source_evidence
    projected = evidence.project(_contract(decision.Criterion.expected_utility()), ["B"])
    assert projected.action_contributors == {"B": (claim.coordinates[1],)}
    assert projected.diagnostics == evidence.diagnostics
    resolved = SourceEvidence.consume(projected.export()).resolve_with(claim)
    assert resolved.resolution["kind"] == "original_native_response_reexecution"
    assert resolved.resolution["native_authority_issued"] is False
    assert resolved.resolution["calibration_license_issued"] is False
    assert resolved.original_acceptance == projected.original_acceptance
    assert SourceEvidence.consume(resolved.export()).resolution is None
    with pytest.raises(TypeError, match="issued native claim"):
        projected.resolve_with({"verified": True})


def test_source_evidence_rejects_changed_native_source_and_corrupted_envelope():
    claim = _claim()
    evidence = claim.source_evidence
    changed = _native_view(bayesian=True)
    changed_claim = program_claims.native_claim(changed, _native_program(changed))
    with pytest.raises(ValueError, match="source_evidence.source_binding_mismatch") as changed_error:
        evidence.resolve_with(changed_claim)
    assert changed_error.value.reason_code == "invalid_argument"
    with pytest.raises(ValueError, match="source_evidence.invalid_artifact") as invalid_error:
        SourceEvidence.consume(b"not an evidence artifact")
    assert invalid_error.value.reason_code == "invalid_argument"


def test_source_evidence_fresh_process_requires_actual_reanalysis(tmp_path):
    claim = _claim()
    path = tmp_path / "source.art"
    path.write_bytes(claim.source_evidence.export())
    script = """
import json,sys
import antecedent as ac
import numpy as np
from antecedent import program_claims
from antecedent.source_evidence import SourceEvidence
a=np.tile(np.array([0.,1.,2.,3.]),80)
x=np.repeat(np.linspace(-1.,1.,80),4)
y=1.+2.*a+.2*x
view=ac.analyze({'x':x,'a':a,'y':y},graph=[('x','a'),('x','y'),('a','y')],query=ac.ResponseCurve('a','y',grid=[1.,2.]),refute='none',bootstrap=0)
program=program_claims.ProgramBinding.from_response(view,outcome_units='mmHg',dose_units='mg')
actual=program_claims.native_claim(view,program)
original=SourceEvidence.consume(open(sys.argv[1],'rb').read())
assert original.resolution is None
resolved=original.resolve_with(actual)
print(json.dumps({'means':actual.means,'coordinates':len(resolved.coordinates),'dependencies':resolved.resolution['resolved_dependencies'],'native_authority_issued':resolved.resolution['native_authority_issued']}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)], check=True, capture_output=True, text=True
    )
    report = json.loads(completed.stdout)
    assert report["means"] == pytest.approx([3.0, 5.0], abs=1e-4)
    assert report["coordinates"] == 2
    assert report["dependencies"] == ["dependencies.checked_response_grid_operation"]
    assert report["native_authority_issued"] is False


def test_source_evidence_equal_global_values_remain_distinct_semantic_coordinates():
    evidence = _claim().source_evidence
    left, right = (evidence.diagnostics_at(q) for q in evidence.coordinates)
    globals_seen = 0
    for a, b in zip(left, right, strict=True):
        if a.scope == "global":
            globals_seen += 1
            assert a.global_values == b.global_values
            assert a.local_value is b.local_value is None
            assert a.source_coordinate != b.source_coordinate
    assert globals_seen > 0


def test_source_evidence_survives_native_decision_and_original_inverse_artifact():
    from antecedent import composition as comp
    from antecedent import inverse_query as iq

    claim = _claim()
    contract = _contract(decision.Criterion.expected_utility())
    actual = comp.DecisionInput.from_native_claim("actual", claim, contract=contract)
    decided = comp.evaluate_with_support(contract, [actual])
    assert decided.verdict.selected == "B"
    assert len(decided.source_evidence) == 1
    assert decided.source_evidence[0].action_contributors == {
        "A": (claim.coordinates[0],),
        "B": (claim.coordinates[1],),
    }
    inverse = iq.InverseQuery(contract, ("A", "B"), (iq.target_mean(3.2),)).evaluate(claim)
    assert inverse.selected == "B"
    assert inverse.source_evidence[0].diagnostics == claim.source_evidence.diagnostics
    loaded = iq.InverseResult.consume(inverse.export())
    assert loaded.selected == "B"
    assert (
        loaded.source_evidence[0].source_artifact_digest
        == claim.source_evidence.source_artifact_digest
    )
    assert loaded.source_evidence[0].resolution is None
