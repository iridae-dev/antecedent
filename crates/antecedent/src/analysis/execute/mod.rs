//! Unified `Study` facade execution (split by modality for SRP).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::cast_precision_loss,
    clippy::wildcard_imports
)]

pub(super) use std::sync::Arc;
pub(super) use std::time::Instant;

pub(super) use super::latency::{INTERACTIVE_MAX_ENVELOPE_GRAPHS, LatencyMode};
pub(super) use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalResponse, DataClassification, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, ExecutionContext, Intervention, ObservationSpec,
    PopulationRegistry, ProvenanceGraph, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, TemporalEffectQuery, VariableId,
};
pub(super) use antecedent_data::{
    DiscoveryEstimationSplit, PanelData, TableView, TabularData, TemporalIndexer, TimeIndex,
    TimeSeriesData,
};
pub(super) use antecedent_discovery::GraphPosterior;
pub(super) use antecedent_estimate::{
    AnalyticSeKind, BayesianGCompWorkspace, BayesianGComputationAte, BayesianTemporalGcomp,
    CausalPosterior, ConditionalLinearAdjustment, ContinuousResponseEstimator, EffectEstimate,
    EnvelopeOptions, EstimationWorkspace, FunctionalDistribution, FunctionalDistributionWorkspace,
    FunctionalEffect, GraphEffectDraws, LinearAdjustmentAte, ObservationMechanismEstimator,
    OverlapPolicy, PreparedBayesianProblem, RdWorkspace, SharpRegressionDiscontinuity,
    TemporalLinearAdjustment, TemporalMediationEstimate, TemporalMediationEstimator,
    TemporalResponseEstimator, aggregate_effect_envelope, nonidentified_with_prior,
};
pub(super) use antecedent_expr::{
    CausalExprArena, DerivationMeta, DomainRef, EstimandMethod, ExprNode, IdentifiedEstimand,
    OutcomeExprId,
};
pub(super) use antecedent_graph::{Admg, Dag, DenseNodeId, Pag, TemporalDag};
pub(super) use antecedent_identify::{
    DerivationTrace, IdentificationEnvelope, IdentificationPerformanceRecord, IdentificationResult,
    IdentificationStatus, SharpRdConfig, SharpRdIdentifier, TemporalBackdoorIdentifier,
    TemporalMediationIdentifier,
};
pub(super) use antecedent_prob::{
    GraphIdentFlag, InferenceDiagnostics, PriorSet, WeightedGraphSamples,
};
pub(super) use antecedent_validate::{
    BayesianSuiteContext, PosteriorPredictiveCheck, PredictiveCheckReport, PriorPredictiveCheck,
    TemporalRefitContext, ValidationSuite, ValidatorId, stack_panel_tabular, with_conflict_summary,
    with_prior_sensitivity,
};

pub(super) use crate::accepted::{AcceptedGraph, GraphClass};
pub(super) use crate::callback_plan::mark_python_callback_plan;
pub(super) use crate::error::CausalError;
pub(super) use crate::gcm::{
    anomaly_attribution, attribute_distribution_change, attribute_unit_change, counterfactual_ite,
    fit_gcm, mechanism_change_detection,
};
pub(super) use crate::inference::{
    BayesianConfig, InferenceMode, resolve_bayesian_prior, resolve_bayesian_prior_with_conflict,
};
pub(super) use crate::planner::{
    LogicalAnalysisPlan, PhysicalExecutionPlan, StaticAteCompileInput,
    StaticCpdagResponseCompileInput, StaticDistributionCompileInput, StaticPagAteCompileInput,
    StaticPagResponseCompileInput, StaticPathSpecificCompileInput, StaticResponseCompileInput,
    compile_logical_distribution, compile_logical_path_specific, compile_logical_static_ate,
    compile_logical_static_cpdag_ate, compile_logical_static_cpdag_response,
    compile_logical_static_pag_ate, compile_logical_static_pag_response,
    compile_logical_static_response, compile_logical_temporal_class_effect,
    compile_logical_temporal_effect, compile_logical_temporal_effect_classified,
    compile_logical_temporal_response, reject_dag_only_on_pag,
};
pub(super) use crate::result::StudyResult;
pub(super) use crate::strategy_table::{
    DEFAULT_ADMG_ESTIMATOR_ID, DEFAULT_ADMG_IDENTIFIER_ID, DEFAULT_CONDITIONAL_ESTIMATOR_ID,
    DEFAULT_CONDITIONAL_IDENTIFIER_ID, DEFAULT_DISTRIBUTION_ESTIMATOR,
    DEFAULT_DISTRIBUTION_ESTIMATOR_ID, DEFAULT_DISTRIBUTION_IDENTIFIER,
    DEFAULT_DISTRIBUTION_IDENTIFIER_ID, DEFAULT_ESTIMATOR, DEFAULT_ESTIMATOR_ID,
    DEFAULT_IDENTIFIER, DEFAULT_IDENTIFIER_ID, DEFAULT_PAG_ESTIMATOR_ID, DEFAULT_PAG_IDENTIFIER_ID,
    DEFAULT_PATH_ESTIMATOR, DEFAULT_PATH_ESTIMATOR_ID, DEFAULT_PATH_IDENTIFIER,
    DEFAULT_PATH_IDENTIFIER_ID, DEFAULT_RESPONSE_ESTIMATOR, DEFAULT_RESPONSE_IDENTIFIER,
    DEFAULT_RESPONSE_IDENTIFIER_ID, EstimatorId, IdentifierId, StaticEstimateWorkspaces,
    estimate_provenance_step, estimate_static_effect, identify_admg, identify_cpdag, identify_pag,
    identify_provenance_step, identify_static, identify_static_query,
    identify_static_query_with_rd, identify_temporal_cpdag, identify_temporal_pag,
    require_identified, select_estimand, validate_static_pair,
};

pub(super) use super::builder::{DataInput, RdConfig, RefuteSuite};
pub(super) use super::helpers::{
    AssembleArgs, assemble_result, effect_from_posterior, evaluate_bayesian_prior_sensitivity,
    overlap_diagnostic, project_for_ate_estimate, projection_diagnostic, provenance_pair,
    push_conflict_diagnostics, refute_outcomes, run_plugin_level_refuters, run_refuters,
    validator_not_applicable_diagnostic, validator_not_applicable_diagnostics,
};

/// Prepared analysis (static or temporal).
#[derive(Clone)]
pub struct Study {
    pub(crate) data: DataInput,
    pub(crate) graph: AcceptedGraph,
    /// Set instead of a single `graph` atom when [`crate::StudyBuilder::graph_posterior`]
    /// was used; `graph` then holds only a placeholder shape (variable count / modality
    /// only — never consulted for identification). Mutually exclusive with a "real" `graph`.
    pub(crate) graph_posterior: Option<GraphPosterior>,
    /// Matrix structure-source axis recorded at [`crate::StudyBuilder::build`].
    pub(crate) structure_source: crate::support::StructureSource,
    /// Licensed vs allowlisted evidence status recorded at build. `None` when
    /// the query is not on the public matrix axis.
    pub(crate) support_status: Option<crate::support::CellStatus>,
    pub(crate) query: CausalQuery,
    pub(crate) refute: RefuteSuite,
    /// Set at [`crate::StudyBuilder::build`] when the caller did not call
    /// `.refute(..)` explicitly and the default refute suite
    /// (`RefuteSuite::PlaceboAndRcc`) was silently downgraded to
    /// `RefuteSuite::None` because the requested cell was
    /// `NotApplicable`/`Refused` while the `RefuteSuite::None` cell for the
    /// same query was `Licensed`/`Allowlisted`. Holds the suite that was
    /// requested before the downgrade (always some suite other than
    /// `None`); `None` here means no downgrade happened. Surfaced to the
    /// caller as diagnostic `exec.refute.default_suite_unsupported`.
    pub(crate) refute_default_downgrade: Option<RefuteSuite>,
    pub(crate) bootstrap_replicates: u32,
    pub(crate) split: Option<DiscoveryEstimationSplit>,
    pub(crate) identifier: Option<IdentifierId>,
    pub(crate) estimator: Option<EstimatorId>,
    pub(crate) estimator_spec: Option<crate::estimator_spec::EstimatorSpec>,
    pub(crate) response_options: Option<antecedent_estimate::ContinuousResponseOptions>,
    pub(crate) observation_options: antecedent_estimate::ObservationEstimatorOptions,
    pub(crate) observation_delayed_entry: Option<antecedent_core::VariableId>,
    pub(crate) rd: Option<RdConfig>,
    pub(crate) inference: InferenceMode,
    pub(crate) overlap_policy: Option<OverlapPolicy>,
    pub(crate) population_registry: Option<PopulationRegistry>,
    pub(crate) custom_validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    pub(crate) latency_mode: Option<super::latency::LatencyMode>,
    pub(crate) stage_sink: Option<Arc<dyn super::stage::StageResultSink>>,
    /// Prepare-time identification for the static ATE path (including the
    /// general-ID result and functional estimand of a bidirected ADMG), set
    /// only by [`Study::prepare`]. Identification depends solely on
    /// (identifier, graph, query, rd) — all frozen at prepare — so reusing it
    /// per estimate click is exact; `None` (every builder-constructed study)
    /// identifies on each run.
    pub(crate) identification_cache: Option<Arc<super::prepared::CachedStaticIdentification>>,
    /// Frozen graph-derived mediation adjustment, including temporal lags.
    pub(crate) mediation_adjustment_cache: Option<Arc<[antecedent_data::LaggedColumn]>>,
    /// Prepare-time generalized-adjustment envelope for a supplied PAG.
    pub(crate) pag_identification_cache: Option<Arc<super::prepared::CachedPagIdentification>>,
    /// Prepare-time MEC envelope for a supplied CPDAG.
    pub(crate) cpdag_identification_cache: Option<Arc<super::prepared::CachedCpdagIdentification>>,
    /// Prepare-time temporal-backdoor identification + indexer for temporal response.
    pub(crate) temporal_identification_cache:
        Option<Arc<super::prepared::CachedTemporalIdentification>>,
    /// Prepare-time TemporalCpdag/Pag envelope.
    pub(crate) temporal_class_identification_cache:
        Option<Arc<super::prepared::CachedTemporalClassIdentification>>,
    /// Prepare-time per-atom identification and weights for a static graph posterior.
    pub(crate) graph_posterior_identification_cache:
        Option<Arc<super::prepared::CachedGraphPosteriorIdentification>>,
    /// Prepare-time per-atom identification, indexers, and weights for a DBN posterior.
    pub(crate) dbn_posterior_identification_cache:
        Option<Arc<super::prepared::CachedDbnPosteriorIdentification>>,
    /// Optional tier-rule background for O(p) closure certification.
    pub(crate) tiered: Option<antecedent_graph::TieredBackground>,
    /// Optional coarsened continuous coordinate for cell-AIPW.
    pub(crate) continuous_cell: Option<(antecedent_core::VariableId, std::sync::Arc<[f64]>)>,
    /// Shared fold assignment / covariate design when this study is part of a batch.
    pub(crate) shared_batch_design: Option<std::sync::Arc<super::batch::SharedBatchDesign>>,
}

impl std::fmt::Debug for Study {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Study")
            .field("continuous_cell", &self.continuous_cell)
            .field("data", &"<data>")
            .field("graph", &self.graph)
            .field("tiered", &self.tiered)
            .field("graph_posterior", &self.graph_posterior)
            .field("structure_source", &self.structure_source)
            .field("support_status", &self.support_status)
            .field("query", &"<query>")
            .field("refute", &self.refute)
            .field("refute_default_downgrade", &self.refute_default_downgrade)
            .field("bootstrap_replicates", &self.bootstrap_replicates)
            .field("split", &self.split)
            .field("identifier", &self.identifier)
            .field("estimator", &self.estimator)
            .field("estimator_spec", &self.estimator_spec)
            .field("response_options", &self.response_options)
            .field("observation_options", &self.observation_options)
            .field("observation_delayed_entry", &self.observation_delayed_entry)
            .field("rd", &self.rd)
            .field("inference", &self.inference)
            .field("overlap_policy", &self.overlap_policy)
            .field("population_registry", &self.population_registry.as_ref().map(|_| "<registry>"))
            .field("custom_validators", &self.custom_validators.len())
            .field("latency_mode", &self.latency_mode)
            .field("stage_sink_is_some", &self.stage_sink.is_some())
            .field("identification_cache_is_some", &self.identification_cache.is_some())
            .field("mediation_adjustment_cache", &self.mediation_adjustment_cache)
            .field("pag_identification_cache_is_some", &self.pag_identification_cache.is_some())
            .field("cpdag_identification_cache_is_some", &self.cpdag_identification_cache.is_some())
            .field(
                "temporal_identification_cache_is_some",
                &self.temporal_identification_cache.is_some(),
            )
            .field(
                "temporal_class_identification_cache_is_some",
                &self.temporal_class_identification_cache.is_some(),
            )
            .field(
                "graph_posterior_identification_cache_is_some",
                &self.graph_posterior_identification_cache.is_some(),
            )
            .field(
                "dbn_posterior_identification_cache_is_some",
                &self.dbn_posterior_identification_cache.is_some(),
            )
            .field("shared_batch_design_is_some", &self.shared_batch_design.is_some())
            .finish()
    }
}

mod attribution_path;
mod bayesian_path;
mod compile;
mod dispatch;
mod pag_path;
mod panel_path;
mod response_path;
mod static_path;
mod temporal_path;
include!("execute_helpers.rs");

pub(crate) use response_path::{class_aware_response_supported, response_witness_ate};

#[cfg(test)]
mod envelope_se_tests {
    use super::*;

    #[test]
    fn multi_atom_se_requires_joint_sampling_covariance() {
        let se = mix_weighted_analytic_se([(0.5, 2.0), (0.5, 0.0)]);
        assert!(se.is_nan());
        assert!((mix_weighted_analytic_se([(1.0, 2.0)]) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mix_weighted_analytic_se_is_nan_if_any_atom_is_nonfinite() {
        assert!(mix_weighted_analytic_se([(0.5, 0.1), (0.5, f64::NAN)]).is_nan());
        assert!(mix_weighted_analytic_se([(1.0, f64::INFINITY)]).is_nan());
    }
}

#[cfg(test)]
mod support_tests {
    use super::*;

    #[test]
    fn parametric_scm_identification_functional_is_resolvable() {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let anomaly_query = antecedent_core::AnomalyAttributionQuery::new([outcome], 10);
        let query = CausalQuery::AnomalyAttribution(anomaly_query);

        let (identification, estimand) = parametric_scm_identification(query, treatment, outcome);

        // Arena actually has content (the bug was a nil ExprId(0) into an *empty* arena, which
        // ignored `treatment`/`outcome` entirely).
        assert!(!identification.arena.is_empty());

        // `functional` resolves to a real node, not a dangling/out-of-range id.
        let _ = identification.arena.node(estimand.functional);

        // And carries derivation metadata naming the real treatment/outcome.
        let derivation = identification
            .arena
            .derivation(estimand.functional)
            .expect("functional should have derivation metadata");
        assert_eq!(derivation.rule.as_ref(), "gcm.parametric");
        let note = derivation.note.as_deref().unwrap_or_default();
        assert!(note.contains(&format!("{treatment:?}")));
        assert!(note.contains(&format!("{outcome:?}")));

        // The adjustment set stays deliberately empty (GCM doesn't identify via backdoor
        // covariates); this is unchanged behavior, asserted here as a scope guard.
        assert!(estimand.adjustment_set.is_empty());
        assert!(identification.required_assumptions.entries.iter().any(|record| {
            matches!(
                &record.assumption,
                antecedent_core::Assumption::ParametricRestriction(restriction)
                    if restriction.id.as_ref() == "gcm.supplied_structural_mechanisms"
            ) && record.scope == antecedent_core::AssumptionScope::Identification
        }));
    }

    #[test]
    fn graph_envelope_unions_assumptions_from_every_identified_case() {
        fn case_result(
            query: &AverageEffectQuery,
            status: IdentificationStatus,
            assumption_id: &'static str,
        ) -> IdentificationResult {
            let mut assumptions = antecedent_core::AssumptionSet::new();
            assumptions.push(antecedent_core::AssumptionRecord {
                assumption: antecedent_core::Assumption::Custom {
                    id: Arc::from(assumption_id),
                    description: Arc::from("case-specific restriction"),
                },
                source: antecedent_core::AssumptionSource::AlgorithmDefault {
                    algorithm: Arc::from("test.case"),
                },
                scope: antecedent_core::AssumptionScope::Identification,
                status: antecedent_core::AssumptionStatus::Declared,
            });
            IdentificationResult::from_parts(
                status,
                CausalQuery::AverageEffect(query.clone()),
                Vec::new(),
                CausalExprArena::new(),
                DerivationTrace::default(),
                assumptions,
                Vec::new(),
                IdentificationPerformanceRecord::default(),
                None,
            )
        }

        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let envelope = IdentificationEnvelope::from_cases(vec![
            antecedent_identify::GraphIdentificationCase {
                graph: Pag::with_variables(2),
                result: case_result(
                    &query,
                    IdentificationStatus::IdentifiedUnderParametricRestrictions,
                    "case.zero",
                ),
                weight: antecedent_identify::ProbabilityMass(0.5),
            },
            antecedent_identify::GraphIdentificationCase {
                graph: Pag::with_variables(2),
                result: case_result(
                    &query,
                    IdentificationStatus::IdentifiedUnderPriorRestrictions,
                    "case.one",
                ),
                weight: antecedent_identify::ProbabilityMass(0.5),
            },
        ]);
        let result = envelope_to_identification_result(&envelope, &query);
        assert_eq!(result.status, IdentificationStatus::IdentifiedUnderPriorRestrictions);
        for expected in ["case.zero", "case.one"] {
            assert!(result.required_assumptions.entries.iter().any(|record| {
                matches!(&record.assumption, antecedent_core::Assumption::Custom { id, .. } if id.as_ref() == expected)
            }));
        }
    }
}

#[cfg(test)]
mod identify_only_tests {
    use antecedent_core::{
        AverageEffectQuery, CausalQuery, Intervention, InterventionalDistributionQuery, Value,
        VariableId,
    };
    use antecedent_data::TabularData;
    use antecedent_discovery::{mask_is_dag, set_edge};
    use antecedent_graph::{Admg, DenseNodeId, Pag};
    use antecedent_prob::InferenceDiagnostics;

    use super::*;
    use crate::support::SupportRefusal;

    fn toy_data() -> TabularData {
        TabularData::from_f64_columns([("t", &[0.0_f64, 1.0][..]), ("y", &[0.0_f64, 1.0][..])])
            .unwrap()
    }

    fn ate() -> AverageEffectQuery {
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
    }

    fn known_truth_graph_mixture_data(n: usize) -> TabularData {
        // Sixteen rows per block give the exact joint distribution
        // P(T=1 | Z=0)=1/4 and P(T=1 | Z=1)=3/4.  Hence the unadjusted
        // contrast is 2 + 2*(3/4 - 1/4) = 3, while adjustment for Z
        // recovers the structural coefficient 2.  Pairing +/- residuals
        // within every cell keeps both population regressions exact.
        assert_eq!(n % 16, 0);
        let mut treatment = Vec::with_capacity(n);
        let mut outcome = Vec::with_capacity(n);
        let mut confounder = Vec::with_capacity(n);
        for _ in 0..(n / 16) {
            for (z, t, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
                for row in 0..count {
                    let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                    treatment.push(t);
                    confounder.push(z);
                    outcome.push(2.0 * t + 2.0 * z + epsilon);
                }
            }
        }
        TabularData::from_f64_columns([
            ("t", treatment.as_slice()),
            ("y", outcome.as_slice()),
            ("z", confounder.as_slice()),
        ])
        .unwrap()
    }

    /// Records progress labels so the test can prove identification is computed
    /// on fresh runs and prepare only, never on a prepared estimate/refresh click.
    #[derive(Default)]
    struct RecordingProgress(std::sync::Mutex<Vec<String>>);

    impl antecedent_core::ProgressSink for RecordingProgress {
        fn report(&self, _fraction: f64, stage: &str) {
            self.0.lock().unwrap().push(stage.to_owned());
        }
    }

    struct CancelOnEnvelopeIdentify {
        token: antecedent_core::CancellationToken,
    }

    impl antecedent_core::ProgressSink for CancelOnEnvelopeIdentify {
        fn report(&self, _fraction: f64, stage: &str) {
            if stage == "envelope.identify" {
                self.token.cancel();
            }
        }
    }

    fn recording_ctx(seed: u64) -> (ExecutionContext, std::sync::Arc<RecordingProgress>) {
        let sink = std::sync::Arc::new(RecordingProgress::default());
        let mut ctx = ExecutionContext::for_tests(seed);
        ctx.progress =
            Some(std::sync::Arc::clone(&sink) as std::sync::Arc<dyn antecedent_core::ProgressSink>);
        (ctx, sink)
    }

    fn identify_computations(sink: &RecordingProgress) -> usize {
        sink.0
            .lock()
            .unwrap()
            .iter()
            .filter(|stage| stage.as_str() == super::super::stage::PROGRESS_IDENTIFY_COMPUTE)
            .count()
    }

    fn cancel_on_envelope_identify_ctx(seed: u64) -> ExecutionContext {
        let mut ctx = ExecutionContext::for_tests(seed);
        let token = ctx.cancellation.clone();
        ctx.progress = Some(std::sync::Arc::new(CancelOnEnvelopeIdentify { token }));
        ctx
    }

    fn cached_count(result: &StudyResult) -> usize {
        result.diagnostics.iter().filter(|d| d.code.as_ref() == "exec.identify.cached").count()
    }

    fn assert_identify_only_refused(err: &CausalError, message: &'static str) {
        assert!(
            matches!(
                err,
                CausalError::Support { id: SupportRefusal::Refused, message: m } if *m == message
            ),
            "{err:?}"
        );
        assert_eq!(err.to_string(), format!("refused: {message}"));
    }

    #[test]
    fn identify_only_refuses_graph_posterior() {
        // graph_posterior x Frequentist is now closed at `build()`, so reach
        // identify_only's own guard through the still-open Bayesian cell.
        let gp = GraphPosterior::new(
            2,
            vec![1.0],
            vec![0u64],
            vec![0.0; 4],
            vec![0.0; 4],
            1.0,
            InferenceDiagnostics::analytic("test"),
            0,
        )
        .unwrap();
        let err = Study::tabular(toy_data())
            .graph_posterior(gp)
            .query(ate())
            .refute(RefuteSuite::None)
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate()))
            .build()
            .unwrap()
            .identify_only()
            .unwrap_err();
        assert_identify_only_refused(
            &err,
            "identify_only is not a graph-posterior cell; identification \
                          runs per-graph inside execute.",
        );
    }

    #[test]
    fn graph_posterior_ate_known_truth_mixture() {
        // Analytic, independently specified fixture: atom 0 estimates the
        // unadjusted effect 3, atom 1 adjusts for Z and estimates 2, and atom
        // 2 is the reverse-causal Y -> T DAG, for which the licensed
        // backdoor identifier has no admissible adjustment; its 0.2 posterior
        // weight must remain unidentified.
        // The envelope contract reports E[tau | identified] while retaining
        // that 0.2 separately: (0.5*3 + 0.3*2) / 0.8 = 2.625.
        let expected: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../conformance/bayesian/known_truth_mixtures/expected.json"
        ))
        .unwrap();
        let pin = &expected["static_average_effect"];
        let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
        let weights: Vec<f64> = pin["posterior_weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let atom_effects: Vec<f64> = pin["identified_atom_effects"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let identified_mass = pin["identified_mass"].as_f64().unwrap();
        let weighted_sum = pin["identified_weighted_sum"].as_f64().unwrap();
        let mixture_truth = pin["expected_effect_given_identified"].as_f64().unwrap();
        let unidentified_truth = pin["expected_unidentified_mass"].as_f64().unwrap();
        let tolerance = pin["effect_abs_tolerance"].as_f64().unwrap();
        assert!(
            (weights[0] * atom_effects[0] + weights[1] * atom_effects[1] - weighted_sum).abs()
                < 1e-12
        );
        assert!((weighted_sum / identified_mass - mixture_truth).abs() < 1e-12);
        assert!((weights[2] - unidentified_truth).abs() < 1e-12);

        let direct = set_edge(0, 3, 0, 1, true); // T -> Y; no adjustment.
        let adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true); // Z -> T, Z -> Y, T -> Y.
        let unidentified = set_edge(0, 3, 1, 0, true); // Y -> T; no admissible adjustment.
        assert!(mask_is_dag(direct, 3));
        assert!(mask_is_dag(adjusted, 3));
        assert!(mask_is_dag(unidentified, 3));

        // Keep marginals consistent with the three frozen atoms even though
        // effect execution consumes their masks and weights directly.
        let mut marginals = vec![0.0; 9];
        marginals[1] = weights[0] + weights[1]; // T -> Y in both identified atoms.
        marginals[3] = weights[2]; // Y -> T only in the unidentified atom.
        marginals[6] = weights[1]; // Z -> T only in the adjusted atom.
        marginals[7] = weights[1]; // Z -> Y only in the adjusted atom.

        // Cover each licensed validation coordinate so cheap/full support is
        // demonstrated by the same known-truth staged mixture, not inferred
        // from a validation-none run.
        let mut report_counts = Vec::new();
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let gp = GraphPosterior::new(
                3,
                weights.clone(),
                vec![direct, adjusted, unidentified],
                marginals.clone(),
                marginals.clone(),
                1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
                InferenceDiagnostics::analytic("known_truth_mixtures"),
                0,
            )
            .unwrap()
            .with_algorithm("known_truth_fixture");
            let data = known_truth_graph_mixture_data(n);
            let (ctx, sink) = recording_ctx(1);
            let study = Study::tabular(data.clone())
                .graph_posterior(gp)
                .query(ate())
                .refute(suite)
                .inference(InferenceMode::Bayesian(
                    BayesianConfig::conjugate().n_draws(256).prior_scale(1_000_000.0),
                ))
                .build()
                .unwrap();
            let fresh = study.clone().run(&ctx).unwrap();
            assert_eq!(identify_computations(&sink), 1, "a fresh run identifies its atoms once");
            let mut prepared = study.prepare(&ctx).unwrap();
            assert_eq!(identify_computations(&sink), 2, "prepare identifies the atoms once");
            let click = prepared.estimate(&data, &ctx).unwrap();
            let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
            assert_eq!(
                identify_computations(&sink),
                2,
                "prepared estimate and refresh clicks must not re-identify"
            );
            assert!(
                (click.estimate.ate - mixture_truth).abs() < tolerance,
                "{suite:?} mixture mean={} truth={mixture_truth}",
                click.estimate.ate
            );
            assert!((click.estimate.ate - fresh.estimate.ate).abs() < 1e-12);
            assert!((refreshed.estimate.ate - click.estimate.ate).abs() < 1e-12);
            assert_eq!(click.support_status.unwrap().as_str(), "licensed");
            let click_post = click.posterior.as_ref().expect("prepared graph mixture posterior");
            let fresh_post = fresh.posterior.as_ref().expect("fresh graph mixture posterior");
            let refreshed_post =
                refreshed.posterior.as_ref().expect("refreshed graph mixture posterior");
            assert_eq!(click_post.identification, IdentificationStatus::GraphDependent);
            assert!((click_post.unidentified_mass - unidentified_truth).abs() < 1e-12);
            assert!((fresh_post.unidentified_mass - unidentified_truth).abs() < 1e-12);
            assert!((refreshed_post.unidentified_mass - unidentified_truth).abs() < 1e-12);
            assert_eq!(
                cached_count(&fresh),
                0,
                "fresh graph-posterior execution must identify its atoms"
            );
            assert_eq!(
                cached_count(&click),
                1,
                "prepared graph-posterior execution must consume its cache exactly once"
            );
            assert_eq!(cached_count(&refreshed), 1, "same-schema refresh must reuse the cache");
            assert_eq!(click.refutations.len(), fresh.refutations.len());
            assert_eq!(click.predictive_checks.len(), fresh.predictive_checks.len());
            if matches!(suite, RefuteSuite::Full) {
                assert!(
                    click_post.prior_sensitivity.is_some(),
                    "full validation must attach mixture-weighted prior sensitivity"
                );
            } else {
                assert!(click_post.prior_sensitivity.is_none());
            }
            match suite {
                RefuteSuite::None => {
                    assert!(click.predictive_checks.is_empty(), "validation none must not run PPC");
                }
                RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
                    assert!(
                        click
                            .predictive_checks
                            .iter()
                            .any(|c| { c.kind == antecedent_validate::PredictiveCheckKind::Prior }),
                        "graph-posterior {suite:?} must attach mixture-weighted prior PPC"
                    );
                    assert!(
                        click.predictive_checks.iter().any(|c| {
                            c.kind == antecedent_validate::PredictiveCheckKind::Posterior
                        }),
                        "graph-posterior {suite:?} must attach mixture-weighted posterior PPC"
                    );
                    assert!(
                        click
                            .diagnostics
                            .iter()
                            .any(|d| d.code.as_ref() == "refute.envelope.effect_mixture"),
                        "graph-posterior {suite:?} must mix effect refuters across atoms"
                    );
                }
            }
            report_counts.push(click.refutations.len());
        }
        assert_eq!(report_counts[0], 0, "validation none must emit no reports");
        assert!(report_counts[1] > 0, "cheap validation must execute a refuter");
        assert!(
            report_counts[2] > report_counts[1],
            "full validation must add prior-sensitivity reports beyond the cheap suite"
        );
    }

    #[test]
    fn graph_posterior_ate_known_truth_mixture_frequentist() {
        // Same atoms and unidentified-mass rule as the Bayesian pin; the
        // aggregator is mass-weighted linear.adjustment.ate, so the mixture
        // hits the analytic 2.625 exactly rather than a posterior mean.
        let expected: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../conformance/bayesian/known_truth_mixtures/expected.json"
        ))
        .unwrap();
        let pin = &expected["static_average_effect"];
        let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
        let weights: Vec<f64> = pin["posterior_weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let mixture_truth = pin["expected_effect_given_identified"].as_f64().unwrap();
        let unidentified_truth = pin["expected_unidentified_mass"].as_f64().unwrap();

        let direct = set_edge(0, 3, 0, 1, true);
        let adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
        let unidentified = set_edge(0, 3, 1, 0, true);
        let mut marginals = vec![0.0; 9];
        marginals[1] = weights[0] + weights[1];
        marginals[3] = weights[2];
        marginals[6] = weights[1];
        marginals[7] = weights[1];

        let mut report_counts = Vec::new();
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let gp = GraphPosterior::new(
                3,
                weights.clone(),
                vec![direct, adjusted, unidentified],
                marginals.clone(),
                marginals.clone(),
                1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
                InferenceDiagnostics::analytic("known_truth_mixtures"),
                0,
            )
            .unwrap()
            .with_algorithm("known_truth_fixture");
            let data = known_truth_graph_mixture_data(n);
            let (ctx, sink) = recording_ctx(1);
            let study = Study::tabular(data.clone())
                .graph_posterior(gp)
                .query(ate())
                .refute(suite)
                .inference(InferenceMode::Frequentist)
                .build()
                .unwrap();
            let fresh = study.clone().run(&ctx).unwrap();
            assert_eq!(identify_computations(&sink), 1, "a fresh run identifies its atoms once");
            let mut prepared = study.prepare(&ctx).unwrap();
            assert_eq!(identify_computations(&sink), 2, "prepare identifies the atoms once");
            let click = prepared.estimate(&data, &ctx).unwrap();
            let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
            assert_eq!(
                identify_computations(&sink),
                2,
                "prepared estimate and refresh clicks must not re-identify"
            );
            assert!(
                (click.estimate.ate - mixture_truth).abs() < 1e-8,
                "{suite:?} mixture mean={} truth={mixture_truth}",
                click.estimate.ate
            );
            assert!((click.estimate.ate - fresh.estimate.ate).abs() < 1e-12);
            assert!((refreshed.estimate.ate - click.estimate.ate).abs() < 1e-12);
            assert_eq!(click.support_status.unwrap().as_str(), "licensed");
            assert!(click.posterior.is_none(), "Frequentist mixture must not attach a posterior");
            assert_eq!(click.identification.status, IdentificationStatus::GraphDependent);
            assert_eq!(fresh.identification.status, IdentificationStatus::GraphDependent);
            assert!(
                fresh.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "estimate.graph_posterior.envelope"
                        && d.message.contains(&format!("unidentified_mass={unidentified_truth}"))
                }),
                "fresh Frequentist mixture must retain unidentified mass"
            );
            assert!(
                fresh.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "estimate.envelope.se_omits_between_atom_variance"
                }),
                "Frequentist mixture SE must disclose omitted between-atom variance"
            );
            assert_eq!(cached_count(&fresh), 0);
            assert_eq!(cached_count(&click), 1);
            assert_eq!(cached_count(&refreshed), 1);
            assert_eq!(click.refutations.len(), fresh.refutations.len());
            match suite {
                RefuteSuite::None => {
                    assert!(click.refutations.is_empty(), "validation none must emit no reports");
                }
                RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
                    assert!(!click.refutations.is_empty(), "{suite:?} must execute a refuter");
                    assert!(
                        click
                            .diagnostics
                            .iter()
                            .any(|d| d.code.as_ref() == "refute.envelope.effect_mixture"),
                        "Frequentist {suite:?} must mix effect refuters across atoms"
                    );
                }
            }
            report_counts.push(click.refutations.len());
        }
        assert_eq!(report_counts[0], 0, "validation none must emit no reports");
        assert!(report_counts[1] > 0, "cheap validation must execute a refuter");
        assert!(
            report_counts[2] >= report_counts[1],
            "full validation must not drop the cheap refuters"
        );
    }

    fn cheap_overlap_comparison(gp: GraphPosterior, n: usize, frequentist: bool) -> f64 {
        let mut builder = Study::tabular(known_truth_graph_mixture_data(n))
            .graph_posterior(gp)
            .query(ate())
            .refute(RefuteSuite::Cheap);
        builder = if frequentist {
            builder.inference(InferenceMode::Frequentist)
        } else {
            builder.inference(InferenceMode::Bayesian(
                BayesianConfig::conjugate().n_draws(256).prior_scale(1_000_000.0),
            ))
        };
        let result = builder.build().unwrap().run(&ExecutionContext::for_tests(3)).unwrap();
        result
            .refutations
            .iter()
            .find(|r| r.refuter.as_ref() == "overlap.assessment")
            .expect("cheap suite must emit overlap.assessment")
            .comparison
    }

    #[test]
    fn graph_posterior_overlap_mixes_distinct_adjustment_atoms() {
        // Atom 0 is unadjusted T→Y; atom 1 adjusts for Z. First-atom validation
        // would report only the empty-Z overlap. Mixing must move the comparison.
        let n = 64;
        let direct = set_edge(0, 3, 0, 1, true);
        let adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
        let unidentified = set_edge(0, 3, 1, 0, true);
        let mix = GraphPosterior::new(
            3,
            vec![0.5, 0.3, 0.2],
            vec![direct, adjusted, unidentified],
            vec![0.0; 9],
            vec![0.0; 9],
            1.0,
            InferenceDiagnostics::analytic("overlap_mixture"),
            0,
        )
        .unwrap();
        let first_only = GraphPosterior::new(
            3,
            vec![1.0],
            vec![direct],
            vec![0.0; 9],
            vec![0.0; 9],
            1.0,
            InferenceDiagnostics::analytic("overlap_first_atom"),
            0,
        )
        .unwrap();
        let mixed = cheap_overlap_comparison(mix, n, false);
        let first = cheap_overlap_comparison(first_only, n, false);
        assert!(
            (mixed - first).abs() > 1e-9,
            "mixture overlap comparison={mixed} must not equal first-atom comparison={first}"
        );
    }

    #[test]
    fn frequentist_graph_posterior_overlap_mixes_distinct_adjustment_atoms() {
        let n = 64;
        let direct = set_edge(0, 3, 0, 1, true);
        let adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
        let unidentified = set_edge(0, 3, 1, 0, true);
        let mix = GraphPosterior::new(
            3,
            vec![0.5, 0.3, 0.2],
            vec![direct, adjusted, unidentified],
            vec![0.0; 9],
            vec![0.0; 9],
            1.0,
            InferenceDiagnostics::analytic("overlap_mixture"),
            0,
        )
        .unwrap();
        let first_only = GraphPosterior::new(
            3,
            vec![1.0],
            vec![direct],
            vec![0.0; 9],
            vec![0.0; 9],
            1.0,
            InferenceDiagnostics::analytic("overlap_first_atom"),
            0,
        )
        .unwrap();
        let mixed = cheap_overlap_comparison(mix, n, true);
        let first = cheap_overlap_comparison(first_only, n, true);
        assert!(
            (mixed - first).abs() > 1e-9,
            "Frequentist mixture overlap comparison={mixed} must not equal first-atom comparison={first}"
        );
    }

    #[test]
    fn fresh_graph_posterior_cancelled_during_identification_returns_typed_error() {
        let direct = set_edge(0, 3, 0, 1, true);
        let adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
        let gp = GraphPosterior::new(
            3,
            vec![0.5, 0.5],
            vec![direct, adjusted],
            vec![0.0; 9],
            vec![0.0; 9],
            2.0,
            InferenceDiagnostics::analytic("cancel_during_identification"),
            0,
        )
        .unwrap();
        let ctx = cancel_on_envelope_identify_ctx(17);
        let error = Study::tabular(known_truth_graph_mixture_data(16))
            .graph_posterior(gp)
            .query(ate())
            .refute(RefuteSuite::None)
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)))
            .build()
            .unwrap()
            .run(&ctx)
            .expect_err("identification cancellation must not become unidentified mass");

        assert!(
            matches!(error, CausalError::Cancelled { stage } if stage == super::super::stage::STAGE_IDENTIFY),
            "expected identify-stage cancellation, got {error:?}"
        );
    }

    #[test]
    fn bidirected_admg_non_ate_refuses_at_build_with_matrix_id() {
        // Every non-AverageEffect query on an ADMG is now a closed cell, so
        // the refusal fires at `build()` with the stable matrix reason.
        // identify_only's own ADMG guard stays as defense in depth but is no
        // longer reachable through the public builder.
        let mut admg = Admg::with_variables(2);
        admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let query = InterventionalDistributionQuery::new(
            VariableId::from_raw(1),
            [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
        );
        let err = Study::tabular(toy_data())
            .graph(admg)
            .query(CausalQuery::Distribution(query))
            .refute(RefuteSuite::None)
            .build()
            .unwrap_err();
        assert!(
            matches!(&err, CausalError::Support { id: SupportRefusal::Refused, .. }),
            "{err:?}"
        );
        assert!(err.to_string().starts_with("refused:"), "{err}");
    }

    #[test]
    fn identify_only_refuses_non_dag_admg_graph() {
        let mut pag = Pag::with_variables(2);
        pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let err = Study::tabular(toy_data())
            .graph(pag)
            .query(ate())
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .identify_only()
            .unwrap_err();
        assert_identify_only_refused(
            &err,
            "identify_only supports static DAG and ADMG graphs only.",
        );
    }
}
