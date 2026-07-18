//! GCM workflow helpers (fit → sample → CF → anomaly).
//!
//! Thin facade over `causal-model` / `causal-counterfactual` / `causal-attribution`
//! so planners and Python bind once at the library boundary.
//!
//! # Example
//!
//! ```rust,ignore
//! use causal::gcm::{fit_gcm, sample_do};
//!
//! let fitted = fit_gcm(dag, &data)?;
//! let draws = sample_do(&fitted.model, treatment, do_value, n, &mut rng, &ctx)?;
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::too_many_arguments)]

use std::sync::Arc;

use causal_attribution::{
    AnomalyScores, AttributionError, ChangeAttribution, ChangeAttributionResult,
    DistributionChangeOptions, FeatureRelevance, MechanismChangeMethod, RobustChangeOptions,
    RootCauseRank, StructureChangeOptions, UnitChangeResult, detect_mechanism_changes,
    distribution_change, distribution_change_robust, feature_relevance, path_decompose,
    root_cause_rank, score_anomalies, structure_change, unit_change,
};
use causal_core::{
    AnomalyAttributionQuery, CausalRng, ChangeAttributionQuery, ExecutionContext, Intervention,
    InterventionalDistributionQuery, MechanismChangeQuery, PathSpecificEffectQuery,
    TargetPopulation, UnitChangeQuery, Value, VariableId,
};
use causal_counterfactual::{
    CounterfactualEngine, CounterfactualError, ExogenousPosterior, AbductionMissingPolicy,
    NoiseInferenceKind,
};
use causal_data::TabularData;
use causal_graph::Dag;
use causal_model::{
    CompiledCausalModel, MechanismAssignment, MechanismRegistry, MechanismWorkspace, ModelError,
    SelectionPolicy, ValueBatch, sample_interventional,
};

use crate::error::AnalysisError;

/// Fitted GCM plus per-node assignment records.
#[derive(Clone, Debug)]
pub struct FittedGcm {
    /// Compiled plan with fitted mechanisms.
    pub model: CompiledCausalModel,
    /// Auto-assignment provenance (no silent defaults).
    pub assignments: Vec<MechanismAssignment>,
}

/// Fit a standard mechanism registry to `data` on `graph`.
///
/// # Errors
///
/// Propagates model fit / assignment failures.
pub fn fit_gcm(graph: Dag, data: &TabularData) -> Result<FittedGcm, AnalysisError> {
    let compiled = CompiledCausalModel::compile(graph).map_err(map_model)?;
    let (store, assignments) = MechanismRegistry::standard()
        .assign_and_fit(&compiled, data, SelectionPolicy::BestScore)
        .map_err(map_model)?;
    Ok(FittedGcm { model: compiled.with_mechanisms(store), assignments })
}

/// Interventional ancestral sample under hard `do` values (batch, one GIL/boundary crossing).
///
/// # Errors
///
/// Sampling failures.
pub fn sample_do(
    model: &CompiledCausalModel,
    interventions: &[Intervention],
    n: usize,
    rng: &mut CausalRng,
    ctx: &ExecutionContext,
) -> Result<ValueBatch, AnalysisError> {
    let mut ws = MechanismWorkspace::default();
    sample_interventional(model, interventions, n, rng, &mut ws, ctx).map_err(map_model)
}

/// Abduction once, then unit-level ITE for binary hard interventions on `treatment`.
///
/// # Errors
///
/// Abduction / prediction failures.
pub fn counterfactual_ite(
    model: CompiledCausalModel,
    data: &TabularData,
    treatment: VariableId,
    outcome: VariableId,
    active: f64,
    control: f64,
    ctx: &ExecutionContext,
) -> Result<IteResult, AnalysisError> {
    let engine = CounterfactualEngine::new(model);
    let exo = engine.abduct(data, AbductionMissingPolicy::Error).map_err(map_cf)?;
    let mut ws = MechanismWorkspace::default();
    let ite = engine
        .individual_treatment_effect(
            &exo,
            outcome,
            Intervention::set(treatment, Value::f64(active)),
            Intervention::set(treatment, Value::f64(control)),
            &mut ws,
            ctx,
        )
        .map_err(map_cf)?;
    let n = ite.len().max(1) as f64;
    let mean = ite.iter().sum::<f64>() / n;
    Ok(IteResult { unit_effects: ite, mean_ite: mean, noise_inference: exo.kind, exogenous: exo })
}

/// ITE summary with visible noise-inference kind.
#[derive(Clone, Debug)]
pub struct IteResult {
    /// Per-unit effects.
    pub unit_effects: Arc<[f64]>,
    /// Mean ITE.
    pub mean_ite: f64,
    /// How noise was obtained.
    pub noise_inference: NoiseInferenceKind,
    /// Shared exogenous state.
    pub exogenous: ExogenousPosterior,
}

/// Score anomalies for listed outcome variables.
///
/// # Errors
///
/// Attribution failures.
pub fn anomaly_attribution(
    model: &CompiledCausalModel,
    data: &TabularData,
    outcomes: impl IntoIterator<Item = VariableId>,
    max_units: usize,
) -> Result<Vec<AnomalyScores>, AnalysisError> {
    let targets: Arc<[VariableId]> = outcomes.into_iter().collect::<Vec<_>>().into();
    let q = AnomalyAttributionQuery::new(targets, max_units);
    score_anomalies(model, data, &q).map_err(map_attr)
}

/// Distribution-change attribution (pinned baseline-GCM parity).
///
/// # Errors
///
/// Attribution failures.
pub fn attribute_distribution_change(
    model: &CompiledCausalModel,
    data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &DistributionChangeOptions,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AnalysisError> {
    distribution_change(model, data, query, options, ctx).map_err(map_attr)
}

/// Robust distribution-change attribution.
///
/// # Errors
///
/// Attribution failures.
pub fn attribute_distribution_change_robust(
    model: &CompiledCausalModel,
    data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &RobustChangeOptions,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AnalysisError> {
    distribution_change_robust(model, data, query, options, ctx).map_err(map_attr)
}

/// Structure-change attribution between two DAGs (parent-set Shapley).
///
/// # Errors
///
/// Attribution failures.
pub fn attribute_structure_change(
    baseline_model: &CompiledCausalModel,
    comparison_model: &CompiledCausalModel,
    data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &StructureChangeOptions,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AnalysisError> {
    structure_change(baseline_model, comparison_model, data, query, options, ctx).map_err(map_attr)
}

/// Builder-style change attribution (§34.3).
#[must_use]
pub fn change_attribution_builder() -> ChangeAttribution {
    ChangeAttribution::new()
}

/// Mechanism-change detection (not attribution).
///
/// # Errors
///
/// Detection failures.
pub fn mechanism_change_detection(
    model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    method: MechanismChangeMethod,
    ctx: &ExecutionContext,
) -> Result<Vec<causal_attribution::MechanismChangeDetection>, AnalysisError> {
    detect_mechanism_changes(model, data, query, method, ctx).map_err(map_attr)
}

/// Unit-change attribution.
///
/// # Errors
///
/// Attribution failures.
pub fn attribute_unit_change(
    model: &CompiledCausalModel,
    data: &TabularData,
    query: &UnitChangeQuery,
    ctx: &ExecutionContext,
) -> Result<UnitChangeResult, AnalysisError> {
    unit_change(model, data, query, ctx).map_err(map_attr)
}

/// Sample an interventional distribution under [`InterventionalDistributionQuery`].
///
/// Thin wrapper over [`sample_do`]. Only [`TargetPopulation::AllObserved`] is supported.
///
/// # Errors
///
/// Query validation, unsupported target population, or sampling failures.
pub fn sample_interventional_distribution(
    model: &CompiledCausalModel,
    query: &InterventionalDistributionQuery,
    n: usize,
    rng: &mut CausalRng,
    ctx: &ExecutionContext,
) -> Result<ValueBatch, AnalysisError> {
    query.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
    if query.target_population != TargetPopulation::AllObserved {
        return Err(AnalysisError::Unsupported {
            message:
                "sample_interventional_distribution only supports TargetPopulation::AllObserved",
        });
    }
    sample_do(model, &query.interventions, n, rng, ctx)
}

/// Path-specific contribution via [`PathSpecificEffectQuery`].
///
/// Thin wrapper over [`path_decompose`]. Only [`TargetPopulation::AllObserved`] is supported.
/// When `path_nodes` is non-empty, keeps paths that visit every listed intermediate node.
///
/// # Errors
///
/// Query validation, unsupported target population, or path decomposition failures.
pub fn attribute_path_specific(
    model: &CompiledCausalModel,
    query: &PathSpecificEffectQuery,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AnalysisError> {
    query.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
    if query.target_population != TargetPopulation::AllObserved {
        return Err(AnalysisError::Unsupported {
            message: "attribute_path_specific only supports TargetPopulation::AllObserved",
        });
    }
    let mut result = path_decompose(
        model,
        &[query.treatment],
        query.outcome,
        query.max_paths,
        query.max_len,
        ctx,
    )
    .map_err(map_attr)?;
    if !query.path_nodes.is_empty() {
        let filtered: Vec<_> = result
            .path_breakdown
            .iter()
            .filter(|p| query.path_nodes.iter().all(|n| p.path.iter().any(|v| v == n)))
            .cloned()
            .collect();
        result.total_change = filtered.iter().map(|p| p.contribution).sum();
        result.path_breakdown = Arc::from(filtered);
    }
    Ok(result)
}

/// Path decomposition.
///
/// # Errors
///
/// Path / model failures.
pub fn attribute_paths(
    model: &CompiledCausalModel,
    sources: &[VariableId],
    outcome: VariableId,
    max_paths: usize,
    max_len: usize,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AnalysisError> {
    path_decompose(model, sources, outcome, max_paths, max_len, ctx).map_err(map_attr)
}

/// Feature relevance under interventions.
///
/// # Errors
///
/// Sampling failures.
pub fn attribute_feature_relevance(
    model: &CompiledCausalModel,
    data: &TabularData,
    outcome: VariableId,
    features: &[VariableId],
    delta: f64,
    n_samples: usize,
    max_features: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<FeatureRelevance>, AnalysisError> {
    feature_relevance(model, data, outcome, features, delta, n_samples, max_features, ctx)
        .map_err(map_attr)
}

/// Rank root causes from an attribution result.
///
/// # Errors
///
/// Ranking failures.
pub fn rank_root_causes(
    attribution: &ChangeAttributionResult,
    ctx: &ExecutionContext,
) -> Result<Vec<RootCauseRank>, AnalysisError> {
    root_cause_rank(attribution, None, None, ctx).map_err(map_attr)
}

#[allow(clippy::needless_pass_by_value)] // map_err adapters
fn map_model(e: ModelError) -> AnalysisError {
    AnalysisError::from(e)
}

#[allow(clippy::needless_pass_by_value)] // map_err adapters
fn map_cf(e: CounterfactualError) -> AnalysisError {
    AnalysisError::from(e)
}

#[allow(clippy::needless_pass_by_value)] // map_err adapters
fn map_attr(e: AttributionError) -> AnalysisError {
    AnalysisError::from(e)
}
