//! Phase 7 GCM workflow helpers (fit → sample → CF → anomaly).
//!
//! Thin facade over `causal-model` / `causal-counterfactual` / `causal-attribution`
//! so planners and Python bind once at the library boundary.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_attribution::{AnomalyScores, AttributionError, score_anomalies};
use causal_core::{ExecutionContext, Intervention, Value, VariableId};
use causal_counterfactual::{
    CounterfactualEngine, CounterfactualError, ExogenousPosterior, NoiseInferenceKind,
};
use causal_data::TabularData;
use causal_graph::Dag;
use causal_model::{
    CompiledCausalModel, MechanismAssignment, MechanismRegistry, MechanismWorkspace, ModelError,
    SelectionPolicy, ValueBatch, sample_interventional,
};

use crate::error::AnalysisError;
use causal_core::{AnomalyAttributionQuery, CausalRng};

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
    Ok(FittedGcm {
        model: compiled.with_mechanisms(store),
        assignments,
    })
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
    let exo = engine.abduct(data, false).map_err(map_cf)?;
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
    Ok(IteResult {
        unit_effects: Arc::from(ite),
        mean_ite: mean,
        noise_inference: exo.kind,
        exogenous: exo,
    })
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

fn map_model(e: ModelError) -> AnalysisError {
    AnalysisError::Compile { message: format!("gcm model: {e}") }
}

fn map_cf(e: CounterfactualError) -> AnalysisError {
    AnalysisError::Compile { message: format!("gcm counterfactual: {e}") }
}

fn map_attr(e: AttributionError) -> AnalysisError {
    AnalysisError::Compile { message: format!("gcm attribution: {e}") }
}
