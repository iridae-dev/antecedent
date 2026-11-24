"""Every Python artifact loader admits bytes only as the artifact its consumer retained.

The check is the native ``bind_retained_artifact`` (the Rust ``bind`` the ``*_expecting``
loaders use). A loader called with ``retained_identity`` of other bytes refuses with
``external_binding_mismatch`` before it reads them; with the retained identity of the same bytes
it returns what it returned before and warns about nothing; without one, a loader that binds no
other retained identity warns with ``AntecedentDeprecationWarning``.
"""

from __future__ import annotations

import inspect
import warnings
from importlib import import_module
from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent import _native
from antecedent import joint_law_propagation as jl
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.artifacts import artifact_identity
from antecedent.errors import CausalError
from antecedent.outcomes import Outcome, require_success

from test_analysis_workflow import GRAPH, QUERY, sample
from test_joint_law_propagation import _artifact, _draws

MISMATCH = "external_binding_mismatch"
GARBAGE = b"not the retained artifact"
OTHER = artifact_identity(b"some other artifact")

# Every loader that gained ``retained_identity``, and whether it warns when called with no
# retained identity at all (True: it binds no other expectation the caller retained).
LOADERS: tuple[tuple[str, bool], ...] = (
    ("antecedent._compiled_robust.CompiledRobustDecision.consume", True),
    ("antecedent._compiled_robust.consume_compiled_robust_decision", False),
    ("antecedent._compiled_robust.consume_joint_scenarios", False),
    (
        "antecedent._joint_panel_designs.JointDesignResult.consume_population_reference_current",
        True,
    ),
    (
        "antecedent._joint_panel_designs.JointDesignResult.consume_reference_decomposition_current",
        True,
    ),
    ("antecedent._joint_panel_designs.PreparedJointDesign.consume_current", True),
    ("antecedent._joint_panel_designs.PreparedNativeJointDesign.consume_current", True),
    ("antecedent._measured_inference.MeasuredInference.load", False),
    ("antecedent._posterior_artifact.consume_posterior", False),
    ("antecedent.adaptive_inference.RepeatedMeasuresProcedure.load_posterior", True),
    ("antecedent.adaptive_inference.consume_adaptive_inference_result_artifact", False),
    ("antecedent.adaptive_inference.consume_adaptive_monitoring_contract_artifact", False),
    ("antecedent.adaptive_inference.consume_repeated_measures_posterior_artifact", False),
    ("antecedent.artifacts.accept", True),
    ("antecedent.artifacts.loads", True),
    ("antecedent.bridge_transport_decision.consume_bridge_bundle_recomputation", False),
    ("antecedent.bridge_transport_decision.consume_bridge_transport_decision", False),
    ("antecedent.bridge_transport_decision.consume_premise_assessment", False),
    (
        "antecedent.categorical_treatment.PreparedCategoricalPosteriorCandidate.consume_source",
        False,
    ),
    ("antecedent.categorical_treatment.consume_categorical_effects", False),
    ("antecedent.choice_inference.ChoiceInferenceSession.consume_current", False),
    ("antecedent.choice_inference.consume_choice_inference_result_artifact", False),
    ("antecedent.cohort_decision.CohortDecisionSource.consume", False),
    ("antecedent.cohort_decision.consume_cohort_decision_source", False),
    ("antecedent.compact_export.CompactExport.consume", False),
    ("antecedent.composition_bundle.consume_bundle", False),
    ("antecedent.composition_bundle.consume_bundle_recomputation", False),
    ("antecedent.composition_bundle.consume_portable_sequential_reliability_node_bundle", False),
    ("antecedent.composition_verification.consume_composition_verification_artifact", False),
    ("antecedent.counterfactual_id.consume_counterfactual_id_artifact", False),
    ("antecedent.cross_world.consume_cross_world_artifact", False),
    ("antecedent.cross_world_execution.consume_cross_world_execution_artifact", False),
    ("antecedent.cyclic.consume_artifact", False),
    ("antecedent.cyclic.consume_evidence", False),
    ("antecedent.cyclic.consume_family_artifact", False),
    ("antecedent.cyclic_decisions.consume_bundle_recomputation", False),
    ("antecedent.cyclic_decisions.consume_bundle_refitting", False),
    ("antecedent.cyclic_decisions.consume_decision", False),
    ("antecedent.decision.Contract.load", False),
    ("antecedent.decision.replay", False),
    ("antecedent.decision_learning.load", False),
    ("antecedent.decision_reliability.consume_decision_reliability_artifact", False),
    ("antecedent.decision_reliability.consume_derived_uncertainty", False),
    ("antecedent.decision_reliability.consume_sequential_policy_reliability", False),
    ("antecedent.decision_robust.AdmissibleContract.load", False),
    ("antecedent.design._declarations.consume", False),
    ("antecedent.design._declarations.consume_rollout", False),
    ("antecedent.design.evidence.consume_plan", True),
    ("antecedent.design.ranking.DesignRankingResult.consume", False),
    ("antecedent.discretized_composition.consume_discretized_composition", False),
    ("antecedent.dose_grid.consume_dose_grid_artifact", True),
    ("antecedent.dose_grid.consume_dose_grid_posterior_artifact", True),
    ("antecedent.drift.consume_baseline", False),
    ("antecedent.evidence_ledger.consume_evidence_ledger_artifact", False),
    ("antecedent.exposure_measurement.consume_exposure_draws", False),
    ("antecedent.extensibility.ProviderVerificationReceipt.load", False),
    ("antecedent.external.ExternalSpec.load", False),
    ("antecedent.external_mechanisms.consume_artifact", False),
    ("antecedent.external_mechanisms.load_artifact", False),
    ("antecedent.functional_source.LawFunctionalArtifact.consume", False),
    ("antecedent.graph_imputation_decision.consume_graph_imputation_decision", False),
    ("antecedent.graph_imputation_decision.consume_graph_imputation_fit_bundle", False),
    ("antecedent.graph_imputation_decision.consume_graph_imputation_transport_bundle", False),
    ("antecedent.interface_contract.consume_interface_contract_artifact", False),
    ("antecedent.intervention_callbacks.decode_programs", True),
    ("antecedent.intervention_programs.consume_policy_program", False),
    ("antecedent.intervention_programs.decode_policy_program", True),
    ("antecedent.inverse_query.InverseResult.consume", False),
    ("antecedent.inverse_query.consume", False),
    ("antecedent.joint_distribution.JointDistributionArtifact.load", False),
    ("antecedent.joint_law_propagation.consume_joint_law_propagation_artifact", False),
    ("antecedent.joint_panel.PreparedJointResponse.consume_current", True),
    ("antecedent.joint_panel.consume", True),
    ("antecedent.joint_panel.consume_instrumented_selection", True),
    ("antecedent.joint_panel.consume_program_certificate", True),
    ("antecedent.joint_panel.consume_randomized_selection", True),
    ("antecedent.latent_class.consume_latent_class_artifact", False),
    ("antecedent.measurement_models.consume_measurement_model_artifact", False),
    ("antecedent.mechanism_composition.consume_mechanism_composition_artifact", False),
    ("antecedent.mechanism_discrepancy.MechanismDiscrepancyResult.consume", False),
    ("antecedent.missing.consume_compiled_graph_imputation", True),
    ("antecedent.mixed_decision.MixedDecisionWorkflow.load", False),
    ("antecedent.nonlinear_mediation.consume_mediation_artifact", False),
    ("antecedent.nuisance_scores.load_artifact", False),
    ("antecedent.observation_query.consume_observation_query_artifact", False),
    ("antecedent.observational_policy.consume_compiled_observational_policy", False),
    ("antecedent.observational_policy.consume_policy_subsampling", False),
    ("antecedent.prediction.FittedEffectModel.load", False),
    ("antecedent.predictive_assessment.consume_predictive_assessment_artifact", False),
    ("antecedent.propagation_chain.consume_chain_decision", False),
    ("antecedent.propagation_chain.consume_chain_robust_decision", False),
    ("antecedent.propagation_chain.consume_propagation_chain", False),
    ("antecedent.proposal_arrival.consume_arrival", False),
    ("antecedent.provider_callbacks.consume_callback_result", False),
    ("antecedent.recalc.RecalcReceipt.consume", False),
    ("antecedent.recalc.consume", False),
    ("antecedent.recalc_cell.FrozenScores.load", False),
    ("antecedent.recalc_composite.ConditionalStudyRanking.consume", False),
    ("antecedent.recalc_design.consume_design_result", True),
    ("antecedent.recalc_temporal.CheckedTemporalIntervalCandidate.load", False),
    ("antecedent.recalc_temporal.consume_temporal_result", True),
    ("antecedent.recovery_chain.consume_recovery_chain_artifact", True),
    ("antecedent.release_migration.consume_previous_release_artifact", True),
    ("antecedent.release_migration.consume_previous_route_records", True),
    ("antecedent.repair.consume", True),
    ("antecedent.sampling_assumption_union.consume_sampling_assumption_union", False),
    ("antecedent.scientific_bridge.consume_scientific_bridge_artifact", False),
    ("antecedent.selective_discovery.consume_selective_discovery_artifact", False),
    ("antecedent.sensitivity_decision.SensitivityArtifact.consume", False),
    ("antecedent.sensitivity_decision.SensitivityArtifact.consume_with_source", True),
    ("antecedent.sequential_evaluation.consume_artifact", False),
    ("antecedent.source_evidence.SourceEvidence.consume", True),
    ("antecedent.source_projection.consume_source_projection", False),
    ("antecedent.temporal.consume_effect_constancy_artifact", False),
    ("antecedent.temporal_counterfactual.consume_temporal_counterfactual_artifact", False),
    ("antecedent.temporal_counterfactual.consume_temporal_worlds_artifact", False),
    ("antecedent.transport._admg_conditional.consume_admg_conditional_obstruction_artifact", True),
    ("antecedent.transport._admg_conditional.consume_admg_conditional_transport_artifact", True),
    ("antecedent.transport._counterfactual.consume_counterfactual_transport_artifact", False),
    (
        "antecedent.transport._counterfactual.consume_sampled_counterfactual_transport_artifact",
        False,
    ),
    ("antecedent.transport._cpdag_scenarios.consume_cpdag_scenarios_artifact", True),
    ("antecedent.transport._cpdag_scenarios.consume_scenario_covariance_artifact", True),
    ("antecedent.transport._impl.exact.consume_exact", True),
    ("antecedent.transport._impl.exact.consume_identification", True),
    ("antecedent.transport._impl.grid.consume_response_grid", True),
    ("antecedent.transport._impl.identify.consume_z_transport_artifact", True),
    ("antecedent.transport._impl.identify.consume_z_transport_sensitivity_artifact", True),
    ("antecedent.transport._impl.statistical.consume_statistical", True),
    ("antecedent.transport._joint_interval_artifact.consume_joint_sensitivity_interval", True),
    ("antecedent.transport._joint_posterior.consume_joint_transport_posterior", False),
    ("antecedent.transport._joint_sensitivity.consume_joint_mechanism_sensitivity_artifact", True),
    ("antecedent.transport._learned_continuous.consume_learned_continuous", True),
    ("antecedent.transport._mixed_source.consume_mixed_source_artifact", True),
    ("antecedent.transport._multi_source.consume_multi_source_z_transport_artifact", True),
    ("antecedent.transport._nested_fisher.NestedFisherCandidate.load", False),
    ("antecedent.transport._nested_posterior.NestedMarkovPosteriorCandidate.load", False),
    ("antecedent.transport._period_refresh.PeriodRefreshSummary.consume", True),
    ("antecedent.transport._period_refresh.consume_period_refresh_artifact", True),
    ("antecedent.transport._recovery.consume_observation_recovery_artifact", True),
    ("antecedent.transport._sampled_candidate.SampledRecoveryCandidate.load", False),
    ("antecedent.transport._scenarios.consume_transport_scenarios_artifact", True),
    ("antecedent.transport._smoothed_dose.consume_smoothed_dose", True),
    ("antecedent.transport._smoothed_dose.consume_smoothed_dose_posterior", True),
    ("antecedent.transport._temporal.consume_temporal_transport_artifact", True),
    ("antecedent.transport._temporal_extensions.TemporalIntervalCandidate.load", False),
    ("antecedent.transport._temporal_extensions.consume_temporal_initial_state_artifact", True),
    ("antecedent.transport._temporal_extensions.consume_temporal_refresh_artifact", True),
    ("antecedent.transport.decisions.consume_fit_bundle", False),
    ("antecedent.transport.decisions.consume_node_bundle", False),
    ("antecedent.transport.decisions.consume_transport_decision", False),
    ("antecedent.transport.temporal_paths.consume_temporal_path_transport", False),
    ("antecedent.transport_handoff.consume_for_callback", False),
    ("antecedent.transported_counterfactual.TransportedPathSpecificEffect.consume", False),
    ("antecedent.transported_counterfactual.consume_transported_counterfactual_artifact", False),
    ("antecedent.transported_response.consume_transported_decision", False),
    ("antecedent.vector_treatment.consume_joint_effects", False),
    ("antecedent.model.decode_model_bundle", True),
    ("antecedent.priors.PriorSourceMeta.from_cbor", True),
    ("antecedent.prepared.reexecute_retarget", True),
    ("antecedent.estimation._PreparedEngine.reexecute_retarget", True),
)


def _resolve(qual: str) -> tuple[Any, bool]:
    """The loader, and whether it can be called without an instance."""
    parts = qual.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            owner: Any = import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for name in parts[split:-1]:
            owner = getattr(owner, name)
        static = inspect.getattr_static(owner, parts[-1])
        without_instance = split == len(parts) - 1 or isinstance(
            static, (classmethod, staticmethod)
        )
        return getattr(owner, parts[-1]), without_instance
    raise AssertionError(f"cannot import {qual}")


def _call_mismatched(loader: Any, retained_identity: str | None) -> Any:
    """Call ``loader`` on garbage bytes, every other required argument ``None``."""
    parameters = inspect.signature(loader).parameters
    args: list[Any] = []
    kwargs: dict[str, Any] = {"retained_identity": retained_identity}
    data_given = False
    for name, parameter in parameters.items():
        if (
            name in ("self", "cls", "retained_identity")
            or parameter.default is not inspect.Parameter.empty
        ):
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        value: Any = None
        if not data_given and parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            annotation = str(parameter.annotation)
            if "bytes" in annotation or name in ("data", "artifact", "source", "encoded"):
                value = GARBAGE
                data_given = True
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[name] = value
        else:
            args.append(value)
    assert data_given, "no artifact parameter found"
    return loader(*args, **kwargs)


def _refusal_text(run: Any) -> str:
    try:
        result = run()
    except Exception as error:  # the loader's own refusal, whatever its class
        return f"{getattr(error, 'reason_code', '')} {error}"
    if isinstance(result, Outcome) and result.refusal is not None:
        return f"{result.refusal.code} {result.refusal.message}"
    return ""


def test_native_identity_is_the_digest_of_the_bytes() -> None:
    identity = artifact_identity(b"artifact")
    assert len(identity) == 64 and identity == identity.lower()
    assert identity == _native.retained_artifact_identity(b"artifact")
    assert identity != artifact_identity(b"artifacT")
    _native.bind_retained_artifact(b"artifact", identity)
    with pytest.raises(CausalError, match=MISMATCH):
        _native.bind_retained_artifact(b"artifacT", identity)
    with pytest.raises(CausalError, match="invalid_argument"):
        _native.bind_retained_artifact(b"artifact", "00")


@pytest.mark.parametrize(("qual", "warns_unbound"), LOADERS)
def test_every_loader_takes_a_retained_identity_and_binds_it_first(
    qual: str, warns_unbound: bool
) -> None:
    loader, callable_without_instance = _resolve(qual)
    parameter = inspect.signature(loader).parameters["retained_identity"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    if not callable_without_instance:
        return
    # The binding runs before the loader reads anything: other bytes than the retained ones are
    # refused as a binding mismatch, never as a decode failure of the garbage.
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntecedentDeprecationWarning)
        assert MISMATCH in _refusal_text(lambda: _call_mismatched(loader, OTHER))
    if warns_unbound:
        with pytest.warns(AntecedentDeprecationWarning, match="retained"):
            _refusal_text(lambda: _call_mismatched(loader, None))


def _joint_law_claim() -> tuple[Any, bytes]:
    artifact = _artifact("retained-identity", np.array([0.0, 1.0]), np.array([2.0, 4.0]))
    x, y = _draws(artifact)
    claim = require_success(jl.aligned_functional("product", x, y, "product"))
    return claim, bytes(claim.export())


def test_family_consumer_admits_the_retained_artifact_only() -> None:
    claim, data = _joint_law_claim()
    with pytest.warns(AntecedentDeprecationWarning, match="retained"):
        before = require_success(jl.consume_joint_law_propagation_artifact(data))
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntecedentDeprecationWarning)
        bound = require_success(
            jl.consume_joint_law_propagation_artifact(
                data, retained_identity=artifact_identity(data)
            )
        )
        # A semantic identity the caller retained is still a binding: no warning.
        require_success(
            jl.consume_joint_law_propagation_artifact(data, expected_identity=claim.identity)
        )
    assert bound.identity == before.identity == claim.identity
    assert bound.values == before.values
    assert bound.acceptance.identity_bound
    with pytest.raises(CausalError, match=MISMATCH):
        require_success(jl.consume_joint_law_propagation_artifact(data, retained_identity=OTHER))
    mutated = bytearray(data)
    mutated[len(mutated) // 2] ^= 0x01
    with pytest.raises(CausalError, match=MISMATCH):
        require_success(
            jl.consume_joint_law_propagation_artifact(
                bytes(mutated), retained_identity=artifact_identity(data)
            )
        )


def test_load_admits_the_retained_artifact_only() -> None:
    claim, data = _joint_law_claim()
    with pytest.warns(AntecedentDeprecationWarning, match="antecedent.load"):
        before = ant.load(data)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntecedentDeprecationWarning)
        loaded = ant.load(data, retained_identity=artifact_identity(data))
    assert loaded.answer == before.answer
    assert require_success(loaded).acceptance.identity_bound
    with pytest.raises(CausalError, match=MISMATCH):
        require_success(ant.load(data, retained_identity=OTHER))


def test_analysis_result_readers_admit_the_retained_artifact_only() -> None:
    result = require_success(
        ant.analyze(
            sample(),
            graph=GRAPH,
            query=QUERY,
            execution=ant.Execution(seed=19),
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
        )
    )
    data = bytes(result.export())
    retained = artifact_identity(data)
    with pytest.warns(AntecedentDeprecationWarning, match="artifacts.loads"):
        before = ant.artifacts.loads(data)
    with pytest.warns(AntecedentDeprecationWarning, match="artifacts.accept"):
        receipt = ant.artifacts.accept(data)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntecedentDeprecationWarning)
        assert ant.artifacts.loads(data, retained_identity=retained) == before
        assert ant.artifacts.accept(data, retained_identity=retained) == receipt
        loaded = require_success(ant.load(data, retained_identity=retained))
    assert loaded.acceptance.identity_bound
    for reader in (ant.artifacts.loads, ant.artifacts.accept):
        with pytest.raises(CausalError, match=MISMATCH):
            reader(data, retained_identity=OTHER)
