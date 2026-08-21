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
    AverageEffectQuery, CausalQuery, DataClassification, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, ExecutionContext, Intervention, MediationContrast, ObservationSpec,
    PopulationRegistry, ProvenanceGraph, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, TemporalEffectQuery, VariableId,
};
pub(super) use antecedent_data::{
    DiscoveryEstimationSplit, PanelData, TableView, TabularData, TimeSeriesData,
};
pub(super) use antecedent_discovery::{
    GraphPosterior, dag_from_adjacency_mask, temporal_dag_from_dbn_masks,
};
pub(super) use antecedent_estimate::{
    AnalyticSeKind, BayesianGCompWorkspace, BayesianGComputationAte, BayesianTemporalGcomp,
    CausalPosterior, ConditionalLinearAdjustment, ContinuousResponseEstimator, EffectEstimate,
    EnvelopeOptions, EstimationWorkspace, FunctionalDistribution, FunctionalDistributionWorkspace,
    FunctionalEffect, GraphEffectDraws, LinearAdjustmentAte, ObservationMechanismEstimator,
    OverlapPolicy, RdWorkspace, SharpRegressionDiscontinuity, TemporalLinearAdjustment,
    TemporalMediationEstimate, TemporalMediationEstimator, TemporalResponseEstimator,
    aggregate_effect_envelope, nonidentified_with_prior,
};
pub(super) use antecedent_expr::{
    CausalExprArena, DerivationMeta, DomainRef, ExprNode, IdentifiedEstimand, OutcomeExprId,
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
    BayesianSuiteContext, PosteriorPredictiveCheck, PriorPredictiveCheck, TemporalRefitContext,
    ValidationSuite, ValidatorId, stack_panel_tabular, with_conflict_summary,
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
    StaticDistributionCompileInput, StaticPagAteCompileInput, StaticPathSpecificCompileInput,
    StaticResponseCompileInput, compile_logical_distribution, compile_logical_path_specific,
    compile_logical_static_ate, compile_logical_static_pag_ate, compile_logical_static_response,
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
    estimate_provenance_step, estimate_static_effect, identify_admg, identify_pag,
    identify_provenance_step, identify_static, identify_static_query,
    identify_static_query_with_rd, require_identified, select_estimand, validate_static_pair,
};

pub(super) use super::builder::{DataInput, RdConfig, RefuteSuite};
pub(super) use super::helpers::{
    AssembleArgs, assemble_result, effect_from_posterior, evaluate_bayesian_prior_sensitivity,
    overlap_diagnostic, project_for_ate_estimate, projection_diagnostic, provenance_pair,
    push_conflict_diagnostics, run_refuters,
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
    pub(crate) bootstrap_replicates: u32,
    pub(crate) split: Option<DiscoveryEstimationSplit>,
    pub(crate) identifier: Option<IdentifierId>,
    pub(crate) estimator: Option<EstimatorId>,
    pub(crate) estimator_spec: Option<crate::estimator_spec::EstimatorSpec>,
    pub(crate) response_options: Option<antecedent_estimate::ContinuousResponseOptions>,
    pub(crate) rd: Option<RdConfig>,
    pub(crate) inference: InferenceMode,
    pub(crate) overlap_policy: Option<OverlapPolicy>,
    pub(crate) population_registry: Option<PopulationRegistry>,
    pub(crate) custom_validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    pub(crate) latency_mode: Option<super::latency::LatencyMode>,
    pub(crate) stage_sink: Option<Arc<dyn super::stage::StageResultSink>>,
    /// Prepare-time identification for the static ATE path, set only by
    /// [`Study::prepare`]. Identification depends solely on
    /// (identifier, graph, query, rd) — all frozen at prepare — so reusing it
    /// per estimate click is exact; `None` (every builder-constructed study)
    /// keeps the identify-per-run behavior.
    pub(crate) identification_cache: Option<Arc<super::prepared::CachedStaticIdentification>>,
    /// Prepare-time temporal-backdoor identification + indexer for temporal response.
    pub(crate) temporal_identification_cache:
        Option<Arc<super::prepared::CachedTemporalIdentification>>,
}

impl std::fmt::Debug for Study {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Study")
            .field("data", &"<data>")
            .field("graph", &self.graph)
            .field("graph_posterior", &self.graph_posterior)
            .field("structure_source", &self.structure_source)
            .field("support_status", &self.support_status)
            .field("query", &"<query>")
            .field("refute", &self.refute)
            .field("bootstrap_replicates", &self.bootstrap_replicates)
            .field("split", &self.split)
            .field("identifier", &self.identifier)
            .field("estimator", &self.estimator)
            .field("estimator_spec", &self.estimator_spec)
            .field("response_options", &self.response_options)
            .field("rd", &self.rd)
            .field("inference", &self.inference)
            .field("overlap_policy", &self.overlap_policy)
            .field("population_registry", &self.population_registry.as_ref().map(|_| "<registry>"))
            .field("custom_validators", &self.custom_validators.len())
            .field("latency_mode", &self.latency_mode)
            .field("stage_sink_is_some", &self.stage_sink.is_some())
            .field("identification_cache_is_some", &self.identification_cache.is_some())
            .field(
                "temporal_identification_cache_is_some",
                &self.temporal_identification_cache.is_some(),
            )
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
    }
}

#[cfg(test)]
mod identify_only_tests {
    use antecedent_core::{
        AverageEffectQuery, CausalQuery, Intervention, InterventionalDistributionQuery, Value,
        VariableId,
    };
    use antecedent_data::TabularData;
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
    fn prepare_graph_posterior_ate_runs_identify_per_click() {
        let _pin = include_str!("../../../../../conformance/bayesian/dag_posterior/expected.json");
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
        let data = toy_data();
        let ctx = ExecutionContext::for_tests(1);
        let study = Study::tabular(data.clone())
            .graph_posterior(gp)
            .query(ate())
            .refute(RefuteSuite::None)
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
            .build()
            .unwrap();
        let fresh = study.clone().run(&ctx).unwrap();
        let prepared = study.prepare(&ctx).unwrap();
        let click = prepared.estimate(&data, &ctx).unwrap();
        assert!(click.estimate.ate.is_finite());
        assert!((click.estimate.ate - fresh.estimate.ate).abs() < 1e-12);
        assert_eq!(click.support_status.unwrap().as_str(), "licensed");
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
