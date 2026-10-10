"""The deliberate root namespace: ``antecedent.__all__`` is an explicit contract.

``__init__.py`` keeps the root namespace deliberately small. Causal-response
queries sit on the root while their configuration and result helpers stay on
stage modules. This test spells out the resulting contract so future changes
remain conscious. Previously nothing
enforced that claim — ``test_notebook_api_surface.py`` only checked names the
example notebooks happened to use. That gap is exactly how
``antecedent.estimators`` went missing from the deliberate-but-unlisted
import block while every sibling stage module resolved fine (see the report
for this change): nothing asserted the unlisted-but-reachable set either.

This test hardcodes both sets in full so a future change to either one must
consciously edit this file rather than silently drift.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import antecedent
import numpy as np
import pytest

from _repo_text import read_text

# --- 1. The root `__all__` contract, spelled out in full. -------------------------

_EXPECTED_ALL = {
    # Verbs
    "analyze",
    "prepare",
    "load",
    "identify",
    "estimate",
    # Structure and results
    "AcceptedGraph",
    "Identification",
    "Analysis",
    "AnalysisResult",
    # Queries
    "AnomalyAttribution",
    "AnomalyReference",
    "AverageDerivative",
    "AverageEffect",
    "ChangeAttribution",
    "ConditionalEffect",
    "Counterfactual",
    "DirectionalDerivative",
    "Elasticity",
    "InterferenceQuery",
    "InterventionalDistribution",
    "InterventionResponse",
    "MediationEffect",
    "NestedCounterfactual",
    "PathSpecificEffect",
    "PulseEffect",
    "PointDerivative",
    "ResponseCurve",
    "ResponseJacobian",
    "SemiElasticity",
    "SustainedEffect",
    "TemporalMediationEffect",
    # Graphs (five graph classes)
    "Dag",
    "Cpdag",
    "Pag",
    "Admg",
    "TemporalDag",
    # Selectors
    "Frequentist",
    "Bayesian",
    "ClassPrior",
    "Identifier",
    "Estimator",
    "Latency",
    "Refute",
    # Errors
    "CausalError",
    "ReviewRequired",
    # Stage modules (eighteen)
    "attribution",
    "data",
    "design",
    "discovery",
    "errors",
    "experiment",
    "estimation",
    "extensibility",
    "factorial",
    "gcm",
    "graph",
    "policy",
    "quasi",
    "regimes",
    "survival",
    "priors",
    "state",
    "validation",
    # Version
    "__version__",
}

# --- 2. Reachable as `antecedent.<name>` but deliberately outside `__all__`. ------
#
# Their public content is re-exported on the root surface above (queries,
# inference selectors) or belongs to a narrower stage surface. `estimators`
# (the typed `estimator_config=` front-end) is included here as of this fix —
# see the module docstring above. `artifacts` and `intervention` were both
# missing from this set until this fix: `intervention` is load-bearing for
# `InterventionResponse` (`_analyze.py`) and documented in
# `docs/causal-responses.md`; `artifacts` backs the documented
# `antecedent.artifacts.dumps`/`.loads` surface (`docs/artifacts.md`). Neither
# omission was caught because nothing checked this set for completeness --
# see ``test_unlisted_but_reachable_set_matches_init_py`` below, which now
# derives the deliberate-import block from ``__init__.py`` itself so a future
# stage module cannot be silently added there without this set changing too.

_EXPECTED_UNLISTED_BUT_REACHABLE = {
    "accepted_graph",
    "artifacts",
    "counterfactual",
    "decision",
    "derived",
    "estimators",
    "external",
    "handoff",
    "ids",
    "inference",
    "interference",
    "intervention",
    "learners",
    "matched",
    "model",
    "observation",
    "population",
    "prediction",
    "query",
    "results",
    "transport",
}


# The root-exported stage modules are public surfaces too. Freezing only
# the package root would still let a refactor silently add or remove names from
# ``antecedent.discovery`` (or any sibling) while the root freeze continued
# to pass.  Keep these lists literal: changing one is an API decision.
_EXPECTED_STAGE_ALL = {
    "inference": {
        "MeasuredInference",
        "MeasuredInferenceIdentity",
        "MeasuredScalar",
        "Bayesian",
        "ClassPrior",
        "Frequentist",
        "PosteriorArtifact",
        "decode_posterior_artifact",
        "encode_posterior_artifact",
    },
    "attribution": {
        "AnomalyScores",
        "ChangeAttributionResult",
        "Contribution",
        "FeatureRelevance",
        "MechanismChangeDetection",
        "anomaly_attribution",
        "attribute_distribution_change",
        "attribute_distribution_change_robust",
        "attribute_feature_relevance",
        "attribute_path_specific",
        "attribute_paths",
        "attribute_structure_change",
        "attribute_unit_change",
        "mechanism_change_detection",
        "rank_root_causes",
    },
    "data": {
        "ArrowLoadInfo",
        "EventFrame",
        "MultiEnvFrame",
        "PanelFrame",
        "event",
        "load_float64_arrow_c_columns",
        "load_float64_columns",
        "multi_env",
        "panel",
        "to_f64",
    },
    "design": {
        "CheckedPriorSignal",
        "PriorSignalSource",
        "adapt_prior_to_signal",
        "DecisionRegret",
        "DesignInformation",
        "DesignObjective",
        "EffectWidth",
        "EnvironmentInformation",
        "GraphEntropy",
        "MeasurementColumn",
        "ModelDistinction",
        "ObjectiveCandidate",
        "RolloutResult",
        "consume_rollout",
        "ARTIFACT_KIND",
        "CALIBRATION",
        "RESULT_LINK_ID",
        "ActionUtility",
        "Basis",
        "BinomialSignal",
        "Candidate",
        "CandidateValue",
        "ConstraintViolation",
        "ConsumedEntry",
        "ConsumedRanking",
        "CostMap",
        "CostUnitsRefusal",
        "DecisionEvaluation",
        "DesignDecision",
        "DesignPlan",
        "DesignRankingRefusal",
        "DesignRankingResult",
        "DesignSearchReceipt",
        "Environment",
        "Expectation",
        "Experiment",
        "ExternalLaw",
        "ExternalSignal",
        "GateEntry",
        "GaussianMeanSignal",
        "IdentificationCandidate",
        "IdentificationGate",
        "IntegrationReport",
        "Measurement",
        "MonteCarlo",
        "ProviderIdentity",
        "Sampling",
        "SignalProvider",
        "SignalProviderRefusal",
        "SignalSpec",
        "SourceOverlapDiagnostics",
        "SourceOverlapRefusal",
        "StatePrior",
        "StructuralCandidate",
        "StructuralEntry",
        "StructuralRanking",
        "StructurePrior",
        "consume",
        "evaluate_decision",
        "evsi",
        "rank_designs",
        "rank_structural",
    },
    "experiment": {
        "ANCOVAEstimate",
        "ComplierEffect",
        "ComplierEffectEstimate",
        "CUPEDEstimate",
        "ExperimentDesign",
        "FactorialRandomization",
        "FixedCUPED",
        "MultiArmContrast",
        "MultiArmExperimentDesign",
        "MultiArmExperimentEstimate",
        "RandomizedEffect",
        "RandomizedExperimentEstimate",
        "RandomizationTest",
        "StratifiedRandomization",
        "SwitchbackDesign",
        "SwitchbackEffect",
        "SwitchbackEstimate",
        "TreatmentOnTreated",
        "estimate_cuped_effect",
        "estimate_ancova_effect",
        "exact_randomization_test",
    },
    "discovery": {
        "CiScreenedPosterior",
        "DbnPosterior",
        "DiscoveredLink",
        "DiscoveryResult",
        "ExactDagPosterior",
        "FCI",
        "GES",
        "GraphEdge",
        "GraphPosterior",
        "JPCMCIPlus",
        "LPCMCI",
        "LiNGAM",
        "NOTEARS",
        "OrderMcmc",
        "PC",
        "PCMCI",
        "PCMCIPlus",
        "PcmciDiscoveryResult",
        "RFCI",
        "RPCMCI",
        "RpcmciDiscoverySummary",
        "StructureMcmc",
        "cpdag_oriented_edges",
        "discovery_algorithm",
        "discovery_to_dag",
        "graph_posterior_map_dag",
        "graph_posterior_map_edges",
        "run_static_discovery",
        "run_temporal_discovery",
        "two_regime_half_split",
    },
    "errors": {
        "ExternalRefusal",
        "CausalAttributionError",
        "CausalCancelled",
        "CausalCancelledError",
        "CausalCompileError",
        "CausalCounterfactualError",
        "CausalDataError",
        "CausalDesignError",
        "CausalDiscoveryError",
        "CausalEstimateError",
        "CausalError",
        "CausalGraphError",
        "CausalIdentifyError",
        "CausalModelError",
        "CausalResourceError",
        "CausalReviewError",
        "CausalSerializationError",
        "CausalStateError",
        "CausalTypeError",
        "CausalUnsupportedError",
        "CausalValidateError",
        "CausalValueError",
        "CallbackUnavailableRefusal",
        "CompositionBundleRefusal",
        "DecisionRefusal",
        "EdgeDigestMismatchRefusal",
        "ExpectedIdentityMismatchRefusal",
        "GraphOrSnapshotMismatchRefusal",
        "IncompatibleVersionRefusal",
        "MechanismDiscrepancyRefusal",
        "MsmSensitivityRefusal",
        "NodeNotFoundRefusal",
        "OversizedRefusal",
        "ProviderRequestChangedRefusal",
        "RecalcNoLiveState",
        "RecalcReceiptRefusal",
        "RecalcRefusal",
        "RecalcUnavailable",
        "RepairArtifactRefusal",
        "RepairBudgetRefusal",
        "RepairRefusal",
        "ScenarioInvarianceRefusal",
        "ScoreResumeRefusal",
        "ScoreResumeUnavailable",
        "SensitivityRefusal",
        "StructuredRefusal",
        "StudyCandidateRefusal",
        "SwappedEvidenceRefusal",
        "TamperedQuantityRefusal",
        "TransportedCounterfactualRefusal",
        "UnknownNodeKindRefusal",
        "UnsupportedLawRefusal",
        "EffectNotIdentified",
        "PendingEdge",
        "ReviewRequired",
        "build_review_error",
        "named_pending_edges",
        "next_action",
        "pending_edges",
        "resolve_display_name",
    },
    "estimation": {
        "AnalysisResult",
        "ConflictSummaryView",
        "EffectEnvelope",
        "EstimateView",
        "IdentificationView",
        "IdentifyResult",
        "MediationEffectsSummary",
        "MediationView",
        "PerformanceView",
        "PhysicalPlanView",
        "PlanView",
        "PosteriorView",
        "PredictiveCheckReport",
        "PreparedAnalysis",
        "PreparedBatch",
        "BatchRetarget",
        "RetargetClaim",
        "RetargetContrast",
        "RetargetMember",
        "SimultaneousBandMember",
        "SimultaneousInterval",
        "SharedBatchDesign",
        "CandidateScreen",
        "PriorSensitivityReport",
        "RefutationReport",
        "ValidationView",
        "analyze_many",
        "identify",
        "mediation_effects_summary",
    },
    "extensibility": {
        "CiBatchTest",
        "CausalProvider",
        "CausalProviderSpec",
        "EffectValidator",
        "ExecutableProvider",
        "MechanismWrapper",
        "ProviderTrust",
        "ProviderExecution",
        "ProviderRegistry",
        "ProviderResult",
        "ProviderQuery",
        "ProviderVerificationFixture",
        "ProviderVerificationReport",
        "UtilityFn",
        "providers",
    },
    "gcm": {
        "anomaly_attribution_discovered",
        "attribute_distribution_change_discovered",
        "attribute_paths_discovered",
        "fit_gcm_discovered",
    },
    "graph": {
        "Admg",
        "Cpdag",
        "Dag",
        "Pag",
        "TemporalCpdag",
        "TemporalDag",
        "TemporalPag",
        "cpdag_oriented_edges",
        "discovery_to_dag",
        "TieredBackground",
        "WithinTier",
    },
    "policy": {
        "BinaryPolicy",
        "ConditionalDoseResponse",
        "ConditionalDoseResponseEstimate",
        "ConditionalDoseResponsePoint",
        "DoublyRobustPolicyEvaluation",
        "FiniteClassRegretEvaluation",
        "FixedDosePolicyValueEstimate",
        "MultiActionCatePoint",
        "MultiActionPolicy",
        "MultiActionPolicyValue",
        "PolicyEvaluation",
        "PolicyValue",
        "UpliftBin",
        "evaluate_policy_doubly_robust",
        "evaluate_multi_action_policy",
        "evaluate_policy",
        "uplift_by_score",
    },
    "factorial": {"FactorialDesign", "FactorialEstimate", "estimate"},
    "quasi": {
        "DifferenceInDifferences",
        "DifferenceInDifferencesEstimate",
        "FuzzyRegressionDiscontinuity",
        "GroupTimeATT",
        "LocalPolynomialRatioEstimate",
        "PanelDifferenceInDifferences",
        "PanelDifferenceInDifferencesEstimate",
        "RegressionKink",
        "SharpRegressionDiscontinuity",
        "StaggeredAdoption",
        "StaggeredAdoptionEstimate",
        "StaggeredEventStudyEstimate",
        "StaggeredEventTimeEffect",
        "SyntheticControl",
        "SyntheticControlEstimate",
        "SyntheticDifferenceInDifferences",
        "SyntheticDifferenceInDifferencesEstimate",
        "AugmentedPanelDiD",
        "AugmentedPanelDiDEstimate",
        "estimate_did",
        "estimate_group_time_att",
    },
    "regimes": {
        "LongitudinalRegime",
        "LongitudinalRegimeEstimate",
        "DoublyRobustRegimeValue",
        "GFormulaValue",
        "HistoryPolicy",
        "Regime",
        "RegimeValue",
        "evaluate_regime_value",
        "evaluate_sequential_doubly_robust",
        "evaluate_sequential_gformula",
        "MarginalStructuralModelResult",
    },
    "survival": {
        "CompetingRisksOutcome",
        "CumulativeIncidenceEstimate",
        "IPCWCumulativeIncidenceEstimate",
        "IPCWSurvivalEstimate",
        "KnownCensoringSurvival",
        "SurvivalDifferenceBand",
        "SurvivalEstimate",
        "SurvivalOutcome",
        "estimate_cumulative_incidence_ipcw",
        "estimate_survival_ipcw",
    },
    "priors": {
        "BetaHyperparameters",
        "CompatibilityReport",
        "ComposedPrior",
        "ConflictPolicy",
        "DesignVariable",
        "EstimandFingerprint",
        "ExternalPriorSourceSpec",
        "ExternalPriorWeight",
        "GammaHyperparameters",
        "POPULATION_TAG_KEY",
        "PriorCatalog",
        "PriorMapping",
        "PriorSource",
        "PriorSourceMeta",
        "TransportPolicy",
        "beta_from_mean_and_ess",
        "beta_from_moments",
        "compose_external_priors",
        "gamma_from_mean_and_ess",
        "gamma_from_moments",
        "populations_from_prior_sources",
    },
    "state": {"CancellationToken", "CausalState", "antecedent_state_append"},
    "validation": {
        "validate_environment_holdout",
        "validate_pcmci_alpha_sensitivity",
        "validate_pcmci_block_bootstrap",
        "validate_pcmci_ci_sensitivity",
        "validate_pcmci_false_positive",
        "validate_pcmci_lag_sensitivity",
        "validate_pcmci_plus_orientation",
        "validate_regime_stability",
        "validate_synthetic_null_calibration",
    },
    "transport.advanced": {
        "AdjustedContrast",
        "ClassicalTransportIdentification",
        "CompletionCounts",
        "CompletionEvidence",
        "CompletionPoint",
        "CompletionReceipt",
        "ConditionalTransportQuery",
        "CpdagCompletion",
        "CpdagScenarioRefusal",
        "CpdagScenarioResult",
        "DirectFormula",
        "DoseBasis",
        "Environment",
        "EstimatorMenu",
        "EstimatorMenuEntry",
        "EvidenceCatalog",
        "EvidenceCatalogDelta",
        "EvidenceRegime",
        "ExactDiscreteLaw",
        "ExactTransportData",
        "ExactTransportDistribution",
        "ExactTransportQuery",
        "GaussianTransportPrior",
        "IDENTIFICATION_STATUSES",
        "InitialStateLaw",
        "JointDeviation",
        "JointTransportIdentity",
        "JointTransportPosterior",
        "JointTransportPriors",
        "JointTransportSource",
        "JointTransportTarget",
        "LEGACY_IDENTIFICATION_STATUSES",
        "LearnedContinuousEstimate",
        "LearnedContinuousOptions",
        "LearnedTrialEstimate",
        "LinearScore",
        "MeanEnvelope",
        "MissingEvidenceCertificate",
        "MixedSourceQuery",
        "MultiSourceZTransportQuery",
        "NestedFisherCandidate",
        "NestedMarkovPrior",
        "NestedMarkovPosteriorCandidate",
        "NonTransportableCertificate",
        "NotCertifiedCertificate",
        "ObservationRecoveryQuery",
        "PartiallyObservedVariable",
        "PopulationFactor",
        "PreparedLearnedContinuous",
        "PreparedSmoothedDose",
        "RecursiveFactorizationFormula",
        "RegimeBinding",
        "RegimeSample",
        "RowTable",
        "SampledRecoveryCandidate",
        "SampledRecoveryIdentity",
        "ScenarioCovarianceRefusal",
        "ScenarioCovarianceResult",
        "ScenarioEstimator",
        "ScoreTerm",
        "SelectionDiagram",
        "SmoothedDoseData",
        "SmoothedDoseEstimate",
        "SmoothedDoseGridPoint",
        "SmoothedDoseOptions",
        "SmoothedDoseQuery",
        "StandardizationFormula",
        "StatisticalTransportData",
        "StatisticalTransportDistribution",
        "StatisticalTransportQuery",
        "StructuralEnvelope",
        "StudyCandidate",
        "TemporalExtensionRefusal",
        "TemporalInitialStateResult",
        "TemporalIntervalCandidate",
        "TemporalPremises",
        "TemporalRefreshResult",
        "TemporalSequenceSpec",
        "TemporalUnitPanel",
        "TemporalWindow",
        "TransportCertificate",
        "TransportIdentification",
        "TransportQuery",
        "TransportResponseGrid",
        "TransportResponseGridQuery",
        "TransportScenario",
        "TransportScenarioRefusal",
        "TransportScenarioSet",
        "TrialAipwData",
        "TrialAipwQuery",
        "TrialNuisanceDiagnostics",
        "TrialTransportEstimate",
        "VariableCoordinate",
        "ZTransportCandidate",
        "ZTransportQuery",
        "ZTransportSensitivityResult",
        "ZTransportSource",
        "binary_nested_markov",
        "binary_nested_markov_fisher_interval",
        "consume_admg_conditional_obstruction_artifact",
        "consume_admg_conditional_transport_artifact",
        "consume_cpdag_scenarios_artifact",
        "consume_exact",
        "consume_identification",
        "consume_joint_mechanism_sensitivity_artifact",
        "consume_joint_transport_posterior",
        "consume_learned_continuous",
        "consume_mixed_source_artifact",
        "consume_multi_source_z_transport_artifact",
        "consume_observation_recovery_artifact",
        "consume_response_grid",
        "consume_scenario_covariance_artifact",
        "consume_smoothed_dose",
        "consume_statistical",
        "consume_temporal_initial_state_artifact",
        "consume_temporal_refresh_artifact",
        "consume_temporal_transport_artifact",
        "consume_transport_scenarios_artifact",
        "consume_z_transport_artifact",
        "consume_z_transport_sensitivity_artifact",
        "cpdag_completion_scenarios",
        "estimate_trial_effect",
        "estimator_menu",
        "evaluate_exact",
        "evaluate_statistical_grid",
        "export_joint_mechanism_sensitivity",
        "identification_status",
        "identify",
        "identify_admg_conditional_transport",
        "identify_classical",
        "identify_meta",
        "identify_mixed_source_transport",
        "identify_multi_source_z_transport",
        "identify_observation_recovery",
        "identify_z_transport",
        "inspect_catalog",
        "inspect_proof_graph",
        "joint_bayesian_transport",
        "joint_mechanism_sensitivity",
        "joint_mechanism_sensitivity_interval",
        "learned_joint_transport",
        "plan_studies",
        "plan_z_transport_evidence",
        "prepare",
        "prepare_exact",
        "prepare_learned_continuous",
        "prepare_response_grid",
        "prepare_smoothed_dose",
        "prepare_statistical",
        "prepare_temporal_transport_sequence",
        "prepare_transport_scenarios",
        "prepare_trial",
        "reload_lowered_expression",
        "reload_lowered_program",
        "replay_study_plan",
        "replay_z_transport_proposal",
        "restore_lowered_program",
        "sampled_observation_recovery",
        "scenario_shared_covariance",
        "smoothed_dose_estimator_menu",
        "temporal_dependent_interval",
        "temporal_initial_state",
        "temporal_new_period_refresh",
    },
}


def test_all_matches_the_documented_root_set():
    assert set(antecedent.__all__) == _EXPECTED_ALL


def test_all_has_no_duplicates():
    assert len(antecedent.__all__) == len(set(antecedent.__all__))


@pytest.mark.parametrize("name", sorted(_EXPECTED_ALL))
def test_every_root_name_resolves(name):
    assert hasattr(antecedent, name), f"antecedent.{name} is in __all__ but does not resolve"


@pytest.mark.parametrize("name", sorted(_EXPECTED_UNLISTED_BUT_REACHABLE))
def test_every_deliberately_unlisted_name_still_resolves(name):
    """These stay off `__all__` on purpose, but must still be reachable.

    ``antecedent.estimators`` unreachable (defect this test file guards
    against) would fail silently here before this fix: ``import antecedent;
    antecedent.estimators`` raised ``AttributeError`` even though
    ``from antecedent.estimators import LinearAdjustment`` worked fine.
    """
    assert hasattr(antecedent, name), f"antecedent.{name} should be reachable but does not resolve"
    assert name not in antecedent.__all__, f"antecedent.{name} should not be in __all__"


def _deliberate_unlisted_reachable_imports_from_init_py() -> set[str]:
    """Parse ``__init__.py`` for its ``from . import X as X`` block.

    That self-aliased spelling (``as X`` repeating the imported name) is what
    marks a stage-module import as the deliberate "reachable but outside
    ``__all__``" family in this file -- as opposed to plain ``from . import
    (a, b, c)`` (the root-exported stage modules, no alias) or ``from
    ._native import name as name`` (re-exported constants/classes, not this
    package's own submodules). Deriving the set this way means a future
    ``from . import newmod as newmod`` line added to that block, without a
    matching update here, fails this test instead of silently drifting.
    """

    source = read_text(Path(antecedent.__file__))
    tree = ast.parse(source, filename=antecedent.__file__)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is not None or node.level != 1:
            continue  # not a plain `from . import ...`
        for alias in node.names:
            if alias.asname == alias.name:
                names.add(alias.name)
    return names


def test_unlisted_but_reachable_set_matches_init_py():
    """Defects 3 & 4, guarded structurally: this set must track `__init__.py`.

    Before this fix, `_EXPECTED_UNLISTED_BUT_REACHABLE` was a hand-maintained
    list that had already drifted from `__init__.py` twice (missing both
    `intervention` and `artifacts`). This test makes that drift impossible to
    reintroduce silently: it derives the actual deliberate-import block from
    the source rather than trusting a second hand-copied list.
    """
    assert _deliberate_unlisted_reachable_imports_from_init_py() == _EXPECTED_UNLISTED_BUT_REACHABLE


@pytest.mark.parametrize("module_name", sorted(_EXPECTED_STAGE_ALL))
def test_stage_module_all_is_frozen(module_name):
    module = importlib.import_module(f"antecedent.{module_name}")
    actual = tuple(module.__all__)
    assert len(actual) == len(set(actual)), f"antecedent.{module_name}.__all__ has duplicates"
    assert set(actual) == _EXPECTED_STAGE_ALL[module_name]


@pytest.mark.parametrize(
    ("module_name", "name"),
    [
        ("interference", "InterferencePointwiseInterval"),
        ("handoff", "EconMLProviderAdapter"),
    ],
)
def test_public_result_types_are_listed_in_their_module_all(module_name, name):
    """A public type a result hands back is exported by its module's ``__all__``."""
    import importlib

    module = importlib.import_module(f"antecedent.{module_name}")
    assert name in module.__all__
    assert getattr(module, name) is not None


def test_estimators_module_is_reachable_and_not_root_exported():
    """Defect 1, directly: `antecedent.estimators` resolves without a direct import."""
    assert hasattr(antecedent, "estimators")
    assert antecedent.estimators.LinearAdjustment is not None
    assert "estimators" not in antecedent.__all__


# --- 3. Retired-name migration signpost (`__getattr__` in `__init__.py`). ---------


def test_retired_module_rename_has_a_signpost_message():
    with pytest.raises(AttributeError, match="renamed to antecedent.priors"):
        _ = antecedent.prior_bank


def test_retired_discover_functions_have_a_signpost_message():
    with pytest.raises(AttributeError, match="discovery config dataclasses"):
        _ = antecedent.discover_pc


def test_unknown_name_still_raises_plain_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'this_name_was_never_a_thing'"):
        _ = antecedent.this_name_was_never_a_thing


# --- 4. Defect 6 smoke test: GCM discovered-attribution helpers reuse the fit. ----
#
# `attribute_paths_discovered` / `anomaly_attribution_discovered` /
# `attribute_distribution_change_discovered` in `antecedent/gcm.py` used to
# call `fit_gcm_discovered` and discard the fitted model, then re-derive edges
# and call a native `attribute_*` free function that re-fits internally —
# silent double work, and none of the three had a caller or test. This is a
# smoke test for one of them (per the fix, they now call the fitted model's
# own method instead of the free function).


def test_attribute_paths_discovered_runs_end_to_end():
    n = 400
    rng = np.random.default_rng(7)
    # Non-Gaussian noise so LiNGAM can orient (matches test_gcm_discovered.py's pattern).
    z = rng.uniform(-1.0, 1.0, size=n)
    t = 0.8 * z + rng.uniform(-1.0, 1.0, size=n)
    y = 1.5 * t + 0.6 * z + rng.uniform(-1.0, 1.0, size=n)
    data = {"z": z, "t": t, "y": y}
    # A single source: `attribute_paths` rejects sources where one is a
    # directed ancestor of another (z -> t here), so ["z", "t"] would raise
    # CausalAttributionError — a pre-existing native constraint, not part of
    # what this smoke test is checking.
    result, edges = antecedent.gcm.attribute_paths_discovered(
        data,
        discovery=antecedent.discovery.LiNGAM(),
        sources=["z"],
        outcome="y",
        seed=1,
    )
    assert edges
    assert result is not None
    assert hasattr(result, "total_change")
