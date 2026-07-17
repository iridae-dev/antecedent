//! Unified static/temporal `CausalAnalysis` facade (identify → estimate → refute).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod analysis;
pub mod design;
pub mod discovery;
pub mod discovery_defaults;
pub mod error;
pub mod gcm;
pub mod inference;
pub mod options;
pub mod planner;
pub mod result;
pub mod review;
pub mod state;
pub mod strategy_table;

pub use analysis::{CausalAnalysis, CausalAnalysisBuilder, RefuteSuite};
pub use design::rank_designs;
pub use discovery::{
    DiscoverParams, discover_jpcmci_plus, discover_lpcmci, discover_pcmci, discover_pcmci_plus,
    discover_rpcmci, pag_definite_directed_edge_count,
};
pub use discovery_defaults::{
    DEFAULT_ALPHA, DEFAULT_MAX_COND_SIZE, DEFAULT_RPCMCI_MIN_REGIME_LEN,
    contemporaneous_constraints, jpcmci_constraints, pcmci_constraints, resolve_ci,
};
pub use error::AnalysisError;
pub use options::{DiscoveryAccept, FdrControl};
pub use causal_stats::{FdrAdjustment, MultipleTestingMethod};
pub use gcm::{
    FittedGcm, IteResult, anomaly_attribution, attribute_distribution_change,
    attribute_distribution_change_robust, attribute_feature_relevance, attribute_paths,
    attribute_unit_change, change_attribution_builder, counterfactual_ite, fit_gcm,
    mechanism_change_detection, rank_root_causes, sample_do,
};
pub use inference::{BayesianConfig, InferenceMode};
pub use planner::{
    CompiledAnalysis, GraphInput, LogicalAnalysisPlan, PhysicalExecutionPlan,
    StaticAteCompileInput, compile_logical_static_ate, compile_logical_temporal_effect,
    is_dag_only_identifier, reject_dag_only_on_pag,
};
pub use strategy_table::{
    DEFAULT_ESTIMATOR, DEFAULT_ESTIMATOR_ID, DEFAULT_IDENTIFIER, DEFAULT_IDENTIFIER_ID, EstimatorId,
    IdentifierId, estimate_provenance_step, estimate_static_effect, identify_provenance_step,
    identify_static, validate_static_pair,
};

// PAG / LPCMCI surfaces.
pub use causal_discovery::{
    ContextKind, CpdagDiscoveryResult, DagDiscoveryResult, DiscoveryPerformanceRecord, JpcmciPlus,
    JpcmciNodeRole, Lpcmci, MultiDatasetConstraints, PagDiscoveryResult, RegimeAssignment,
    RegimeGraphCollection, Rpcmci, RpcmciDiscoveryResult, ScoredLink, two_regime_half_split,
};
pub use causal_estimate::{
    ConditionalLinearAdjustment, OverlapPolicy, TemporalEffectSurface, TemporalLinearPredictor,
    TemporalMediationEstimator,
};
pub use causal_graph::{
    Admg, CompletionSampler, Pag, PagCompletion, TemporalPag, TemporalPagReview, latent_project,
};
pub use causal_identify::{
    GeneralizedAdjustmentConfig, GeneralizedAdjustmentIdentifier, GraphIdentificationCase,
    IdentificationEnvelope, ProbabilityMass, TemporalMediationIdentifier,
};
pub use result::CausalAnalysisResult;
pub use review::{
    PendingCpdagReview, PendingGraphReview, compile_review_required, compile_review_required_cpdag,
    compile_review_required_pag, compile_temporal_with_graph, ensure_review_complete,
};
pub use state::{apply_state_event, new_causal_state};

/// Parse a DOT digraph into a [`causal_graph::Dag`].
///
/// # Errors
///
/// [`AnalysisError::Serialization`] on malformed DOT or invalid DAG structure.
pub fn dag_from_dot(dot: &str) -> Result<causal_graph::Dag, AnalysisError> {
    causal_io::dag_from_dot(dot).map_err(AnalysisError::from)
}

/// Serialize a DAG to DOT.
///
/// # Errors
///
/// [`AnalysisError::Serialization`] on conversion failure.
pub fn dag_to_dot(
    dag: &causal_graph::Dag,
    names: Option<&[String]>,
) -> Result<String, AnalysisError> {
    causal_io::dag_to_dot(dag, names).map_err(AnalysisError::from)
}

/// Parse a JSON DAG document into a [`causal_graph::Dag`].
///
/// # Errors
///
/// [`AnalysisError::Serialization`] on malformed JSON or invalid DAG structure.
pub fn dag_from_json(json: &str) -> Result<causal_graph::Dag, AnalysisError> {
    causal_io::dag_from_json(json).map_err(AnalysisError::from)
}

/// Serialize a DAG to JSON.
///
/// # Errors
///
/// [`AnalysisError::Serialization`] on conversion failure.
pub fn dag_to_json(
    dag: &causal_graph::Dag,
    names: Option<&[String]>,
) -> Result<String, AnalysisError> {
    causal_io::dag_to_json(dag, names).map_err(AnalysisError::from)
}

/// Encode a [`causal_estimate::CausalPosterior`] to durable bytes.
///
/// # Errors
///
/// [`AnalysisError::Serialization`] on IO failures.
pub fn encode_causal_posterior_bytes(
    posterior: &causal_estimate::CausalPosterior,
    artifact_id: &str,
) -> Result<Vec<u8>, AnalysisError> {
    causal_io::encode_causal_posterior_bytes(posterior, artifact_id).map_err(AnalysisError::from)
}

/// Encode a [`causal_estimate::CausalPosterior`] to a durable artifact.
///
/// # Errors
///
/// [`AnalysisError::Serialization`] on IO failures.
pub fn encode_causal_posterior(
    posterior: &causal_estimate::CausalPosterior,
    artifact_id: &str,
) -> Result<causal_io::EncodedArtifact, AnalysisError> {
    causal_io::encode_causal_posterior(posterior, artifact_id).map_err(AnalysisError::from)
}

/// Decode posterior wire metadata + draws.
///
/// # Errors
///
/// [`AnalysisError::Serialization`] on IO failures.
pub fn decode_causal_posterior_bytes(
    bytes: &[u8],
) -> Result<(causal_io::CausalPosteriorWire, Vec<f64>), AnalysisError> {
    causal_io::decode_causal_posterior_bytes(bytes).map_err(AnalysisError::from)
}

// GCM / counterfactual / attribution surfaces.
#[allow(deprecated)]
pub use causal_attribution::{
    AnomalyScores, ArrowStrength, AttributionError, ChangeAttribution, ChangeAttributionResult,
    DifferenceMeasure, DistributionChangeOptions, FeatureRelevance, MechanismChangeDetection,
    MechanismChangeMethod, RobustChangeOptions, RootCauseRank, UnitChangeResult, arrow_strengths,
    detect_mechanism_changes, distribution_change, distribution_change_robust, feature_relevance,
    intrinsic_influence, path_decompose, population_do_contrast, root_cause_rank, score_anomalies,
    unit_change,
};
pub use causal_counterfactual::{
    CompiledCounterfactualPlan, CounterfactualEngine, CounterfactualError, CounterfactualResult,
    CounterfactualWorld, ExogenousPosterior, MissingPolicy, NoiseInferenceKind,
    simultaneous_hard_counterfactual, streaming_matches_retained,
};
#[allow(deprecated)]
pub use causal_counterfactual::nested_hard_counterfactual;
pub use causal_model::{
    CompiledCausalModel, DoSampleResult, InvertibleStructuralCausalModel, KdeDoSampler,
    McmcDoSampler, MechanismAssignment, MechanismFamily, MechanismRegistry, MechanismWorkspace,
    ModelCollection, ModelError, ModelEvaluator, ProbabilisticCausalModel, SelectionPolicy,
    StructuralCausalModel, WeightingDoSampler, interventional_mean, sample_interventional,
    sample_observational,
};

// design / incremental state surfaces.
pub use causal_design::{
    CandidateDesign, ConstraintViolation, DecisionConstraint, DecisionEvaluation, DecisionProblem,
    DecisionProblemId, DesignConstraints, DesignCost, DesignError, DesignEvaluationContext,
    DesignObjective, DesignRankConfig, DesignRanker, DesignRanking, EffectWidthContext,
    EnvironmentPlan, ExperimentPlan, InterventionDesignEffect, MeasureColumnSpec, MeasurementPlan, ModelLoglikDraws, RankedCandidate,
    SamplingPlan, Utility, evaluate_decision,
};
pub use causal_prob::{GraphIdentFlag, WeightedGraphSamples};
pub use causal_state::{
    CachedResult, CausalState, ConstraintId, DataBatchRef, DataCatalog, DataVersion,
    GraphConstraintRecord, GraphEvidenceRecord, GraphEvidenceStore, InterventionRecord,
    InvalidationEntry, InvalidationLog, InvalidationTarget, LagIndexCacheEntry, LagIndexCacheKey,
    LinearOlsSuffStats, ModelRecord, ModelStore, QueryRecord, QueryStore, ResultStore,
    RetentionPolicy, StateError, StateEvent, StreamingCovariance, SuffStatStore,
};

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_graph::{Dag, DenseNodeId};
    use causal_kernels::standard_normal;

    use super::*;

    fn scm() -> (TabularData, Dag, AverageEffectQuery) {
        let n = 200usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let z: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64).collect();
        let t: Vec<f64> = (0..n).map(|i| if z[i] > 0.5 { 1.0 } else { 0.0 }).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + 3.0 * z[i]).collect();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let mut dag = Dag::with_variables(3);
        // z -> t, z -> y, t -> y
        let z_id = DenseNodeId::from_raw(2);
        let t_id = DenseNodeId::from_raw(0);
        let y_id = DenseNodeId::from_raw(1);
        dag.insert_directed(z_id, t_id).unwrap();
        dag.insert_directed(z_id, y_id).unwrap();
        dag.insert_directed(t_id, y_id).unwrap();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        (TabularData::new(storage), dag, query)
    }

    #[test]
    fn end_to_end_ate() {
        let (data, graph, query) = scm();
        let analysis = CausalAnalysis::builder()
            .data(data)
            .graph(graph)
            .query(query)
            .refute(RefuteSuite::PlaceboAndRcc)
            .bootstrap_replicates(30)
            .build()
            .unwrap();
        let ctx = ExecutionContext::for_tests(3);
        let result = analysis.run(&ctx).unwrap();
        assert!((result.estimate.ate - 2.0).abs() < 1e-6);
        assert_eq!(result.refutations.len(), 2);
        assert!(!result.provenance.is_empty());
        assert!(!result.identification.derivation.steps.is_empty());

        let trace = result.analysis_trace_wire();
        assert_eq!(&*trace.method, "backdoor.adjustment");
        assert!(!trace.assumptions.is_empty());
        assert!(!trace.derivation.is_empty());
        let bytes = causal_io::to_cbor(&trace).unwrap();
        let round: causal_io::AnalysisTraceWire = causal_io::from_cbor(&bytes).unwrap();
        assert_eq!(round.method, trace.method);
        assert_eq!(round.derivation.len(), trace.derivation.len());
    }

    /// Confounded SCM: `Z ~ N(0,1)`, `T ~ Bernoulli(logit(-0.5 + Z))`, `Y = 2T + Z + noise`.
    /// True ATE = 2; OLS-on-observables is biased here unless `Z` is adjusted for, so this
    /// exercises the propensity path independent of the linear-adjustment default.
    fn confounded_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x1234_u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = standard_normal(&mut rng);
            let logit = -0.5 + zi;
            let p = 1.0 / (1.0 + (-logit).exp());
            let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
            let noise = standard_normal(&mut rng) * 0.5;
            z[i] = zi;
            t[i] = ti;
            y[i] = 2.0 * ti + zi + noise;
        }
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let mut dag = Dag::with_variables(3);
        let z_id = DenseNodeId::from_raw(2);
        let t_id = DenseNodeId::from_raw(0);
        let y_id = DenseNodeId::from_raw(1);
        dag.insert_directed(z_id, t_id).unwrap();
        dag.insert_directed(z_id, y_id).unwrap();
        dag.insert_directed(t_id, y_id).unwrap();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        (TabularData::new(storage), dag, query)
    }

    #[test]
    fn end_to_end_propensity_weighting_recovers_confounded_effect() {
        let (data, graph, query) = confounded_scm(800, 1);
        let analysis = CausalAnalysis::builder()
            .data(data)
            .graph(graph)
            .query(query)
            .identifier("backdoor.adjustment")
            .estimator("propensity.weighting")
            .bootstrap_replicates(30)
            .build()
            .unwrap();
        let ctx = ExecutionContext::for_tests(11);
        let result = analysis.run(&ctx).unwrap();
        assert!((result.estimate.ate - 2.0).abs() < 0.3, "ate={}", result.estimate.ate);
        // Placebo/RCC refuters are hardwired to LinearAdjustmentAte and are skipped for
        // every non-default estimator (see `CausalAnalysis::execute_static`).
        assert!(result.refutations.is_empty());
        assert_eq!(result.logical_plan.estimator.as_deref(), Some("propensity.weighting"));
    }

    /// `Z ∈ {0,1} → T → Y` with `U` confounding `T`–`Y` (unobserved, not in the graph).
    /// True structural effect = 2.0.
    fn iv_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x1E71_u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = (i % 2) as f64;
            let u = standard_normal(&mut rng);
            let ti = 0.5 * zi + u + 0.1 * standard_normal(&mut rng);
            let yi = 2.0 * ti + u + 0.1 * standard_normal(&mut rng);
            z[i] = zi;
            t[i] = ti;
            y[i] = yi;
        }
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let mut dag = Dag::with_variables(3);
        let z_id = DenseNodeId::from_raw(2);
        let t_id = DenseNodeId::from_raw(0);
        let y_id = DenseNodeId::from_raw(1);
        dag.insert_directed(z_id, t_id).unwrap();
        dag.insert_directed(t_id, y_id).unwrap();
        let query = AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            0.0,
            1.0,
        );
        (TabularData::new(storage), dag, query)
    }

    #[test]
    fn end_to_end_iv_two_stage_least_squares() {
        let (data, graph, query) = iv_scm(4000, 5);
        let analysis = CausalAnalysis::builder()
            .data(data)
            .graph(graph)
            .query(query)
            .identifier("iv")
            .estimator("iv.2sls")
            .bootstrap_replicates(30)
            .build()
            .unwrap();
        let ctx = ExecutionContext::for_tests(21);
        let result = analysis.run(&ctx).unwrap();
        assert!((result.estimate.ate - 2.0).abs() < 0.6, "ate={}", result.estimate.ate);
        assert!(result.refutations.is_empty());
    }

    #[test]
    fn end_to_end_temporal_effect() {
        use causal_core::{Lag, TemporalEffectQuery, TemporalPolicy};
        use causal_data::{SamplingRegularity, TimeIndex, TimeSeriesData};
        use causal_graph::{TemporalDag, ensure_lagged};

        let n = 250usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "pressure",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "defect",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = ((t as f64) * 0.05).sin();
            y[t] = 0.75 * x[t - 1];
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let series = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        let mut g = TemporalDag::empty();
        let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(x1, y0).unwrap();
        let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1);
        let analysis = CausalAnalysis::builder()
            .series(series)
            .temporal_graph(g)
            .temporal_query(q)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let ctx = ExecutionContext::for_tests(7);
        let result = analysis.run(&ctx).unwrap();
        assert!((result.estimate.ate - 0.75).abs() < 0.08, "ate={}", result.estimate.ate);
        assert_eq!(&*result.logical_plan.plan_id, "temporal_effect");
        assert!(result.physical_plan.estimated_peak_memory_bytes.is_some());
        assert!(result.physical_plan.estimated_copy_bytes.is_some());
        assert!(!result.physical_plan.task_schedule.is_empty());
        assert!(!result.physical_plan.materializations.is_empty());

        let compiled = analysis.compile(&ctx).unwrap();
        match compiled {
            CompiledAnalysis::Ready(plan) => {
                assert!(plan.temporal_graph().is_some());
                assert_eq!(plan.record.batch_size, Some(250));
            }
            CompiledAnalysis::ReviewRequired(_)
            | CompiledAnalysis::ReviewRequiredCpdag(_)
            | CompiledAnalysis::ReviewRequiredPag(_) => {
                panic!("expected Ready")
            }
        }
    }
}
