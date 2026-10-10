"""Original diagnostics and complete source requests survive functional handoffs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from antecedent import composition as comp
from antecedent import decision, program_claims
from antecedent import design as dr
from antecedent import sensitivity_decision as sd
from antecedent.functional_source import LawFunctionalArtifact
from antecedent.source_projection import consume_source_projection
from antecedent.transport import advanced as transport

from test_composition import action, contract, joint
from test_program_claims import _contract, _native_program, _native_view
from test_rollout import fixture as rollout_fixture
from test_sensitivity_decision import _prepared_stage, _quantity


def _native_functional():
    view = _native_view()
    claim = program_claims.native_claim(view, _native_program(view))
    declaration = _contract(decision.Criterion.expected_utility())
    source = comp.DecisionInput.from_native_claim("original", claim, contract=declaration)
    return claim, comp.evaluate_functional(declaration, "B", comp.Functional.expectation(), source)


def test_native_functional_keeps_original_semantic_diagnostics_and_typed_lineage():
    claim, value = _native_functional()
    assert value.value == pytest.approx(3.5, abs=1e-4)
    assert value.standard_error is None
    evidence = value.source_evidence[0]
    assert evidence.action_contributors == {"B": (claim.coordinates[1],)}
    assert evidence.diagnostics == claim.source_evidence.diagnostics
    assert evidence.stages_behind() == frozenset(
        {"causal_contract", "data", "evidence", "decision_contract", "transformation", "claim"}
    )
    assert claim.stages_behind() == frozenset({"causal_contract", "data", "evidence", "claim"})
    artifact = value.source_artifact
    loaded = consume_source_projection(artifact.export(), expected_identity=artifact.identity)
    assert loaded.report["unresolved"] == ["dependencies.checked_response_grid_operation"]
    assert loaded.report["output"]["action_id"] == "B"
    assert loaded.report["output"]["actions"] == [
        {
            "id": "B",
            "expected_utility": pytest.approx(value.value),
            "criterion_value": pytest.approx(value.value),
            "standard_error": None,
        }
    ]


@pytest.mark.parametrize(
    "functional,truth",
    [
        (comp.Functional.expectation(), 3.0),
        (comp.Functional.variance(), 5.0),
        (comp.Functional.probability(2.0, "upper"), 0.75),
        (comp.Functional.quantile(0.5), 2.0),
        (comp.Functional.tail_expectation(0.5, "upper"), 5.0),
    ],
)
def test_law_functional_full_original_request_replays_truth_without_trust_upgrade(
    functional, truth
):
    declaration = contract(
        [
            action("wait", "do(a=0)", "outcome", decision.x(0)),
            action("treat", "do(a=1)", "outcome", decision.x(0)),
        ]
    )
    source = comp.DecisionInput.from_distribution("law", joint())
    result = comp.evaluate_functional(declaration, "treat", functional, source)
    assert result.value == pytest.approx(truth)
    original = result.source_artifact
    loaded = LawFunctionalArtifact.consume(original.export())
    report = loaded.report
    assert report["value"] == pytest.approx(truth)
    assert report["standard_error"] is None
    assert report["source_metadata"]["identity"]["alignment"] == "joint"
    assert report["source_metadata"]["identity"]["snapshot_id"] == "snap-n"
    assert report["native_authority_issued"] is False
    assert report["calibration_license_issued"] is False
    assert {row["stage"] for row in report["lineage"]} == {
        "causal_contract",
        "data",
        "distribution_artifact",
        "decision_contract",
        "transformation",
        "claim",
    }


def _sensitivity():
    stage = _prepared_stage()
    return sd.SensitivityArtifact.from_joint_sensitivity(
        stage,
        transport.JointDeviation({"outcome_kernel": 0.2}),
        effect=_quantity("effect", "outcome_units"),
        actions=[
            sd.SensitivityAction("adopt", sd.quantity("effect")),
            sd.SensitivityAction("status_quo", sd.const(0.45)),
        ],
        causal_contract_id="checked-contract",
    )


def test_sensitivity_full_original_v3_source_replays_surface_and_preserves_withheld_sampling():
    artifact = _sensitivity()
    assert artifact.source_evidence["source_verified"] is True
    assert artifact.source_evidence["native_authority_issued"] is False
    assert artifact.original_source()
    loaded = sd.SensitivityArtifact.consume_with_source(artifact.export_with_source())
    assert loaded.identity == artifact.identity
    assert loaded.quantities == artifact.quantities
    assert loaded.uncertainty == artifact.uncertainty
    assert loaded.source_evidence == artifact.source_evidence
    assert loaded.lineage == artifact.lineage
    assert {
        "causal_contract",
        "data",
        "evidence",
        "sensitivity_input",
        "claim",
    } <= loaded.stages_behind()
    historical = sd.SensitivityArtifact.consume(artifact.export())
    assert historical.source_evidence is None
    with pytest.raises(ValueError, match="sensitivity_source.source_unavailable") as missing:
        historical.export_with_source()
    assert missing.value.reason_code == "invalid_argument"


def test_rollout_ranking_handoff_consumes_original_source_and_retains_named_state_standing():
    source, ranked, rollout = rollout_fixture()
    loaded = dr.consume_rollout(rollout.export(), expected_identity=rollout.expectation())
    assert loaded.source_evidence["source_verified"] is True
    assert loaded.source_evidence["unresolved_source_digests"] == []
    assert loaded.source_trust == source.trust == "external_attested"
    assert loaded.calibration == source.calibration == "unmeasured"
    assert loaded.ranking.ranking_identity == ranked.ranking_identity
    assert loaded.ranking.entries[0].evsi == pytest.approx(ranked.candidates[0].evsi)
    assert loaded.lineage == rollout.lineage
    assert {
        "causal_contract",
        "data",
        "distribution_artifact",
        "transformation",
        "decision_contract",
        "study_ranking_provider",
        "claim",
    } <= {link.stage for link in loaded.lineage}


def test_full_sensitivity_and_law_sources_replay_in_fresh_process(tmp_path):
    sensitivity = _sensitivity()
    declaration = contract(
        [
            action("wait", "do(a=0)", "outcome", decision.x(0)),
            action("treat", "do(a=1)", "outcome", decision.x(0)),
        ]
    )
    result = comp.evaluate_functional(
        declaration,
        "treat",
        comp.Functional.probability(2.0, "upper"),
        comp.DecisionInput.from_distribution("law", joint()),
    )
    sensitivity_path = tmp_path / "sensitivity.art"
    law_path = tmp_path / "law.art"
    sensitivity_path.write_bytes(sensitivity.export_with_source())
    law_path.write_bytes(result.source_artifact.export())
    script = """
import json,sys
from antecedent.sensitivity_decision import SensitivityArtifact
from antecedent.functional_source import LawFunctionalArtifact
s=SensitivityArtifact.consume_with_source(open(sys.argv[1],'rb').read())
l=LawFunctionalArtifact.consume(open(sys.argv[2],'rb').read()).report
print(json.dumps({'lower':s.quantities[0].lower[-1],'upper':s.quantities[0].upper[-1],'sensitivity':s.source_evidence['source_verified'],'probability':l['value'],'native':l['native_authority_issued'],'calibration':l['calibration_license_issued']}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(sensitivity_path), str(law_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["lower"] == pytest.approx(sensitivity.quantities[0].lower[-1])
    assert report["upper"] == pytest.approx(sensitivity.quantities[0].upper[-1])
    assert report["sensitivity"] is True
    assert report["probability"] == pytest.approx(0.75)
    assert report["native"] is False and report["calibration"] is False


def test_law_functional_changed_complete_request_identity_and_invalid_artifact_refuse():
    declaration = contract(
        [
            action("wait", "do(a=0)", "outcome", decision.x(0)),
            action("treat", "do(a=1)", "outcome", decision.x(0)),
        ]
    )
    source = comp.DecisionInput.from_distribution("law", joint())
    original = comp.evaluate_functional(
        declaration, "treat", comp.Functional.expectation(), source
    ).source_artifact
    changed = comp.evaluate_functional(
        declaration, "treat", comp.Functional.probability(2.0, "upper"), source
    ).source_artifact
    with pytest.raises(
        ValueError, match="functional_source.expected_identity_mismatch"
    ) as mismatch:
        LawFunctionalArtifact.consume(changed.export(), expected_identity=original.identity)
    assert mismatch.value.reason_code == "invalid_argument"
    with pytest.raises(ValueError, match="functional_source.invalid_artifact") as corrupt:
        LawFunctionalArtifact.consume(b"not a source artifact")
    assert corrupt.value.reason_code == "invalid_argument"


def _issued_joint():
    view = _native_view(bayesian=True)
    claim = program_claims.native_claim(view, _native_program(view))
    declaration = decision.Contract(
        actions=(
            decision.Action(
                "product", inputs=claim.coordinates, utility=decision.x(0) * decision.x(1)
            ),
            decision.Action("reference", inputs=claim.coordinates, utility=decision.x(0)),
        ),
        utility_units="util",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    return claim, declaration, claim.as_decision_source(declaration).source


def test_issued_native_law_alias_retains_diagnostics_but_import_does_not_issue_authority():
    from antecedent.joint_distribution import JointDistributionArtifact

    claim, declaration, law = _issued_joint()
    source = comp.DecisionInput.from_distribution("issued", law, requirement="native")
    assert source.provenance.native
    assert law.source_evidence.diagnostics == claim.source_evidence.diagnostics
    value = comp.evaluate_functional(declaration, "product", comp.Functional.expectation(), source)
    assert value.value == pytest.approx(law.joint_product_expectation(0, 1))
    assert value.source_evidence[0].diagnostics == claim.source_evidence.diagnostics
    imported = JointDistributionArtifact.load(
        law.export("native-law-alias"), expected_identity=law.identity
    )
    assert imported.source_evidence is None
    with pytest.raises(comp.UnverifiedTrustRefusal):
        comp.DecisionInput.from_distribution("imported", imported, requirement="native")


def test_native_law_functional_portable_diagnostics_require_actual_row_origin_resolution():
    claim, declaration, law = _issued_joint()
    source = comp.DecisionInput.from_distribution("issued", law, requirement="native")
    original = comp.evaluate_functional(
        declaration, "product", comp.Functional.expectation(), source
    ).source_artifact
    loaded = LawFunctionalArtifact.consume(original.export(), expected_identity=original.identity)
    assert (
        loaded.report["source_evidence"]["diagnostics"]
        == original.report["source_evidence"]["diagnostics"]
    )
    assert loaded.report["native_row_origin_verified"] is False
    resolved = loaded.resolve_with(claim)
    assert resolved.report["native_row_origin_verified"] is True
    assert resolved.report["native_authority_issued"] is False
    assert resolved.report["calibration_license_issued"] is False
    assert (
        LawFunctionalArtifact.consume(resolved.export()).report["native_row_origin_verified"]
        is False
    )
    with pytest.raises(TypeError, match="issued native claim"):
        loaded.resolve_with(law)


def test_native_law_functional_fresh_process_resolves_only_actual_reissued_source(tmp_path):
    claim, declaration, law = _issued_joint()
    source = comp.DecisionInput.from_distribution("issued", law, requirement="native")
    value = comp.evaluate_functional(declaration, "product", comp.Functional.expectation(), source)
    path = tmp_path / "native-law-functional.art"
    path.write_bytes(value.source_artifact.export())
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, sys
from antecedent.functional_source import LawFunctionalArtifact
sys.path.insert(0, sys.argv[2])
from test_composition_source_lineage import _issued_joint
artifact = LawFunctionalArtifact.consume(open(sys.argv[1], 'rb').read())
claim, _, _ = _issued_joint()
resolved = artifact.resolve_with(claim)
print(json.dumps({'value':resolved.report['value'], 'rows_verified':resolved.report['native_row_origin_verified'], 'native_authority':resolved.report['native_authority_issued'], 'diagnostics':resolved.report['source_evidence']['diagnostics']}))
""",
            str(path),
            str(Path(__file__).parent),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(child.stdout)
    assert report["value"] == pytest.approx(value.value)
    assert report["rows_verified"] is True
    assert report["native_authority"] is False
    assert report["diagnostics"] == value.source_artifact.report["source_evidence"]["diagnostics"]


def test_native_rollout_retains_original_semantic_diagnostics_in_independent_ranking_handoff():
    import numpy as np

    claim, _, law = _issued_joint()
    state = claim.coordinates[0]
    states = np.asarray(law)[:, 0].tolist()
    problem = dr.DesignDecision(
        "native-state-ranking",
        (dr.ActionUtility("wait", 0.0, 0.0), dr.ActionUtility("act", -2.0, 1.0)),
        dr.StatePrior.draws(states),
        "utility",
    )
    ranked = dr.rank_designs(
        [dr.Candidate("sample", 1, dr.GaussianMeanSignal(1.0), 0.0, "utility")],
        decision=problem,
        signal=dr.SignalSpec(
            law.identity.source_id, state, state, (law.identity.source_id,), rng_seed=3
        ),
        source_digests=(decision.source_digest(law),),
        rng_seed=3,
    )
    rollout = problem.export_rollout(law, state, ranked, interpretation="posterior_state")
    loaded = dr.consume_rollout(rollout.export(), expected_identity=rollout.expectation())
    assert (
        loaded.source_evidence["original_diagnostics"]["diagnostics"]
        == rollout.source_evidence["original_diagnostics"]["diagnostics"]
    )
    assert loaded.source_evidence["native_row_origin_verified"] is False
    assert (
        loaded.source_evidence["original_diagnostics"]["source_artifact_digest"]
        == claim.source_evidence.source_artifact_digest
    )
    assert loaded.ranking.ranking_identity == ranked.ranking_identity
    assert loaded.source_trust == "native_licensed"
    assert loaded.native_execution_authority is False


def test_native_law_row_resolution_refuses_other_actual_source_and_missing_original_citation():
    _, declaration, law = _issued_joint()
    source = comp.DecisionInput.from_distribution("issued", law, requirement="native")
    artifact = comp.evaluate_functional(
        declaration, "product", comp.Functional.expectation(), source
    ).source_artifact
    other_view = _native_view()
    other = program_claims.native_claim(other_view, _native_program(other_view))
    with pytest.raises(ValueError, match="functional_source.source_mismatch") as mismatch:
        artifact.resolve_with(other)
    assert mismatch.value.reason_code == "invalid_argument"
    external_contract = contract(
        [
            action("wait", "do(a=0)", "outcome", decision.x(0)),
            action("treat", "do(a=1)", "outcome", decision.x(0)),
        ]
    )
    external = comp.evaluate_functional(
        external_contract,
        "treat",
        comp.Functional.expectation(),
        comp.DecisionInput.from_distribution("external", joint()),
    ).source_artifact
    with pytest.raises(ValueError, match="functional_source.native_source_unavailable") as missing:
        external.resolve_with(other)
    assert missing.value.reason_code == "invalid_argument"


@pytest.mark.parametrize("slot", ["interval_region", "identified_set", "scenarios"])
def test_inverse_all_original_native_evidence_slots_retain_distinct_source_diagnostics(slot):
    import antecedent as ac
    import numpy as np
    from antecedent import inverse_query as iq

    first, declaration, first_law = _issued_joint()
    treatment = np.tile(np.array([0.0, 1.0, 2.0, 3.0]), 80)
    covariate = np.repeat(np.linspace(-1.0, 1.0, 80), 4)
    view = ac.analyze(
        {"x": covariate, "a": treatment, "y": 2.0 + 2.0 * treatment + 0.2 * covariate},
        graph=[("x", "a"), ("x", "y"), ("a", "y")],
        query=ac.ResponseCurve("a", "y", grid=[1.0, 2.0]),
        inference=ac.Bayesian(backend="conjugate", n_draws=256),
        refute="none",
    )
    second = program_claims.native_claim(view, _native_program(view))
    second_law = second.as_decision_source(declaration).source
    members = (
        iq.Scenario.evaluated("first", first_law),
        iq.Scenario.evaluated("second", second_law),
    )
    supplied = {
        "interval_region": iq.IntervalRegion(first_law, second_law, True),
        "identified_set": iq.IdentifiedSet(members, True),
        "scenarios": members,
    }
    result = iq.InverseQuery(
        declaration, ("product", "reference"), (iq.target_mean(0.0),)
    ).evaluate(**{slot: supplied[slot]})
    assert result.selected is None
    expected = {
        first.source_evidence.source_artifact_digest,
        second.source_evidence.source_artifact_digest,
    }
    assert len(expected) == 2
    assert {source.source_artifact_digest for source in result.source_evidence} == expected
    loaded = iq.InverseResult.consume(result.export())
    assert {source.source_artifact_digest for source in loaded.source_evidence} == expected
    assert all(source.resolution is None for source in loaded.source_evidence)
    assert all(
        source.original_acceptance["unresolved_dependencies"]
        == ["dependencies.checked_response_grid_operation"]
        for source in loaded.source_evidence
    )
    assert loaded.action("product") == result.action("product")
    assert len(result.source_evidence) == 2


def test_issued_native_aligned_law_functionals_and_evsi_match_independent_finite_arithmetic():
    import numpy as np
    from antecedent.joint_distribution import JointDistributionArtifact

    claim, declaration, law = _issued_joint()
    rows = np.asarray(law)
    product = rows[:, 0] * rows[:, 1]
    source = comp.DecisionInput.from_distribution("native-joint", law, requirement="native")
    assert law.covariance(0, 1) == pytest.approx(
        float(np.mean((rows[:, 0] - rows[:, 0].mean()) * (rows[:, 1] - rows[:, 1].mean()))),
        abs=1e-12,
    )
    threshold = float(np.median(product))
    probability = comp.evaluate_functional(
        declaration, "product", comp.Functional.probability(threshold, "lower"), source
    )
    quantile = comp.evaluate_functional(
        declaration, "product", comp.Functional.quantile(0.75), source
    )
    expectation = comp.evaluate_functional(
        declaration, "product", comp.Functional.expectation(), source
    )
    assert probability.value == pytest.approx(float(np.mean(product <= threshold)), abs=1e-12)
    assert quantile.value == float(sorted(product)[int(np.ceil(0.75 * len(product))) - 1])
    assert expectation.value == pytest.approx(float(product.mean()), abs=1e-12)
    assert (
        probability.standard_error is quantile.standard_error is expectation.standard_error is None
    )
    imported = JointDistributionArtifact.load(
        law.export("issued-law-consumer"), expected_identity=law.identity
    )
    assert np.array_equal(np.asarray(imported), rows)
    assert imported.source_evidence is None
    replayed = LawFunctionalArtifact.consume(probability.source_artifact.export())
    assert replayed.report["value"] == probability.value
    assert replayed.resolve_with(claim).report["native_row_origin_verified"] is True

    states = rows[:, 0]
    center = float(states.mean())
    scale = float(states.std())
    standardized = (states - center) / scale
    problem = dr.DesignDecision(
        "issued-law-conditional-study",
        (dr.ActionUtility("wait", 0.0, 0.0), dr.ActionUtility("act", -center / scale, 1.0 / scale)),
        dr.StatePrior.draws(states.tolist()),
        "utility",
    )
    ranked = dr.rank_designs(
        [dr.Candidate("one-observation", 1, dr.GaussianMeanSignal(scale * scale), 0.0, "utility")],
        decision=problem,
        signal=dr.SignalSpec(
            law.identity.source_id,
            claim.coordinates[0],
            claim.coordinates[0],
            (law.identity.source_id,),
            rng_seed=71,
        ),
        source_digests=(decision.source_digest(law),),
        rng_seed=71,
        monte_carlo=dr.MonteCarlo(64, 64, 64, 0.0),
        mc_error_tolerance=0.05,
    )

    def integrate_value(steps):
        # For U(wait)=0 and U(act)=z, integrate max(0,E[z phi(y-z)]) directly.
        observations = np.linspace(
            float(standardized.min()) - 8.0, float(standardized.max()) + 8.0, steps + 1
        )
        densities = np.exp(-0.5 * (observations[:, None] - standardized[None, :]) ** 2) / np.sqrt(
            2.0 * np.pi
        )
        integrand = np.maximum(0.0, (densities * standardized).mean(axis=1))
        spacing = float(observations[1] - observations[0])
        return spacing / 3.0 * (
            integrand[0]
            + integrand[-1]
            + 4.0 * integrand[1:-1:2].sum()
            + 2.0 * integrand[2:-1:2].sum()
        ) - max(0.0, float(standardized.mean()))

    oracle = float(integrate_value(8192))
    assert oracle > 0.1
    assert abs(oracle - integrate_value(4096)) < 2e-5
    candidate = ranked.candidates[0]
    assert candidate.claim == "monte_carlo_estimate"
    assert candidate.integration.method == "monte_carlo"
    assert candidate.evsi == pytest.approx(oracle, abs=5.0 * candidate.integration.stderr + 2e-4)
    evpi = float(np.maximum(0.0, standardized).mean()) - max(0.0, float(standardized.mean()))
    assert ranked.evpi == pytest.approx(evpi, abs=1e-11)
    rollout = problem.export_rollout(
        law, claim.coordinates[0], ranked, interpretation="posterior_state"
    )
    consumed = dr.consume_rollout(rollout.export(), expected_identity=rollout.expectation())
    assert consumed.ranking.entries[0].evsi == pytest.approx(
        oracle, abs=5.0 * candidate.integration.stderr + 2e-4
    )
    assert consumed.source_trust == "native_licensed"
    assert consumed.calibration == "unmeasured"
    assert consumed.native_execution_authority is False


@pytest.mark.parametrize("forward", ["bound", "mean_adapter"])
def test_external_original_source_survives_functional_supported_decision_and_inverse_artifacts(
    forward,
    tmp_path,
):
    from dataclasses import replace

    from antecedent import inverse_query as iq
    from antecedent.source_evidence import SourceEvidence

    from test_program_claims import _response, _spec
    from test_source_projection import _contract as external_contract

    claim = _spec().bind(replace(_response(), support=("supported", "weak_overlap")))
    declaration = external_contract(claim.quantities)
    source = comp.DecisionInput.from_claim("foreign", claim)
    evidence = source.source_evidence
    assert evidence is not None
    assert evidence.identities == claim.identity_fields
    assert evidence.diagnostics == ()
    assert (
        evidence._summary["diagnostic_availability"]
        == "not_retained_by_original_external_claim_format"
    )
    assert evidence._summary["point_status"] == ["supported", "weak_overlap"]
    assert evidence._summary["trust"] == "externally_attested"
    imported = SourceEvidence.consume(evidence.export())
    assert imported.identities == claim.identity_fields
    assert imported.lineage == evidence.lineage
    assert imported._summary["native_authority_issued"] is False
    value = comp.evaluate_functional(declaration, "A", comp.Functional.expectation(), source)
    assert value.value == 5.0
    with pytest.raises(comp.SupportRefusal) as unsupported:
        comp.evaluate_functional(declaration, "B", comp.Functional.expectation(), source)
    assert unsupported.value.detail == "composition_boundary.unsupported_action_not_comparable"
    assert unsupported.value.reason_code == "decision_contract_unsatisfied"
    assert value.source_evidence[0].action_contributors == {"A": (claim.quantities[0],)}
    original = consume_source_projection(
        value.source_artifact.export(), expected_identity=value.source_artifact.identity
    )
    assert original.report["trust"] == "externally_attested"
    result = comp.evaluate_with_support(declaration, [source])
    assert result.source_evidence[0].identities == claim.identity_fields
    query = iq.InverseQuery(
        declaration,
        grid=("A", "B"),
        constraints=[iq.target_mean(2)],
        selection="first_in_grid_order",
    )
    point = claim if forward == "bound" else iq.MeanClaim.from_external(claim)
    inverse = query.evaluate(point=point)
    assert inverse.selected == "A"
    loaded = iq.InverseResult.consume(inverse.export(), expected_identity=inverse.identity)
    assert loaded.source_evidence[0].identities == claim.identity_fields
    assert loaded.source_evidence[0].diagnostics == ()
    assert loaded.source_evidence[0]._summary["native_authority_issued"] is False
    if forward == "mean_adapter":
        with pytest.raises(ValueError, match="source_evidence.point_binding_mismatch"):
            query.evaluate(point=replace(point, means=(30.0, 50.0)))
    decision_result = declaration.evaluate(claim)
    assert decision_result.source_evidence[0].identities == claim.identity_fields
    (tmp_path / "external_evidence").write_bytes(value.source_evidence[0].export())
    (tmp_path / "inverse").write_bytes(inverse.export())
    script = """
import json,sys
from pathlib import Path
from antecedent.source_evidence import SourceEvidence
from antecedent.inverse_query import InverseResult
p=Path(sys.argv[1]); evidence=SourceEvidence.consume((p/'external_evidence').read_bytes())
result=InverseResult.consume((p/'inverse').read_bytes())
print(json.dumps({'selected':result.selected,'identity':evidence.identities,
 'authority':evidence._summary['native_authority_issued'],
 'inverse_identity':result.source_evidence[0].identities,'diagnostics':evidence.diagnostics}))
"""
    wire = json.loads(
        subprocess.check_output([sys.executable, "-c", script, str(tmp_path)], text=True)
    )
    assert wire == {
        "selected": "A",
        "identity": claim.identity_fields,
        "authority": False,
        "inverse_identity": claim.identity_fields,
        "diagnostics": [],
    }


@pytest.mark.parametrize("bayesian", [False, True])
def test_named_native_mean_adapters_preserve_original_source_and_remain_mean_only(
    bayesian, tmp_path
):
    from dataclasses import replace

    from antecedent import inverse_query as iq
    from antecedent.errors import CausalUnsupportedError

    view = _native_view(bayesian=bayesian)
    claim = program_claims.native_claim(view, _native_program(view))
    declaration = _contract(decision.Criterion.expected_utility())
    adapter = claim.as_decision_source(declaration)
    assert isinstance(adapter.source, decision.MeanSource)
    result = declaration.evaluate(adapter.source)
    assert result.source_evidence[0].diagnostics == claim.source_evidence.diagnostics
    assert result.source_evidence[0].identities == claim.source_evidence.identities
    mean = adapter.mean_claim()
    query = iq.InverseQuery(declaration, ("A", "B"), (iq.target_mean(3.25),))
    inverse = query.evaluate(point=mean)
    assert inverse.selected == "B"
    imported = iq.InverseResult.consume(inverse.export(), expected_identity=inverse.identity)
    assert imported.source_evidence[0].diagnostics == claim.source_evidence.diagnostics
    assert imported.source_evidence[0].resolution is None
    assert imported.source_evidence[0].original_acceptance["unresolved_dependencies"] == [
        "dependencies.checked_response_grid_operation"
    ]
    (tmp_path / "native_mean_inverse").write_bytes(inverse.export())
    script = """
import json,sys
from pathlib import Path
from antecedent.inverse_query import InverseResult
result=InverseResult.consume(Path(sys.argv[1]).read_bytes())
evidence=result.source_evidence[0]
print(json.dumps({'selected':result.selected,'identities':evidence.identities,
 'diagnostics':evidence.diagnostics,'resolution':evidence.resolution,
 'unresolved':evidence.original_acceptance['unresolved_dependencies']}))
"""
    replay = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", script, str(tmp_path / "native_mean_inverse")], text=True
        )
    )
    assert replay == {
        "selected": "B",
        "identities": claim.source_evidence.identities,
        "diagnostics": json.loads(json.dumps(claim.source_evidence.diagnostics)),
        "resolution": None,
        "unresolved": ["dependencies.checked_response_grid_operation"],
    }
    with pytest.raises(CausalUnsupportedError) as quantile_refusal:
        iq.InverseQuery(declaration, ("A", "B"), (iq.target_quantile(0.5, 0),)).evaluate(point=mean)
    assert quantile_refusal.value.reason_code == "decision_contract_unsatisfied"
    assert quantile_refusal.value.detail == "decision_evaluation.mean_source_insufficient"
    with pytest.raises(ValueError, match="source_evidence.point_binding_mismatch") as changed:
        query.evaluate(point=replace(mean, means=tuple(value + 1 for value in mean.means)))
    assert changed.value.reason_code == "invalid_argument"
    with pytest.raises(ValueError, match="source_evidence.point_binding_mismatch"):
        declaration.evaluate(replace(adapter.source, snapshot_id="substituted-snapshot"))
