//! GCM workflow helpers (fit → sample → CF → anomaly).
//!
//! Thin facade over `antecedent-model` / `antecedent-counterfactual` / `antecedent-attribution`
//! so planners and Python bind once at the library boundary.
//!
//! # Example
//!
//! ```rust,ignore
//! use antecedent::gcm::{fit_gcm, sample_do};
//!
//! let fitted = fit_gcm(dag, &data)?;
//! let draws = sample_do(&fitted.model, treatment, do_value, n, &mut rng, &ctx)?;
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use antecedent_core::{
    AnomalyAttributionQuery, CausalRng, ChangeAttributionQuery, ExecutionContext, Intervention,
    InterventionalDistributionQuery, MechanismChangeQuery, PathSpecificEffectQuery,
    TargetPopulation, UnitChangeQuery, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::Dag;
use antecedent_model::ValueBatch;

use crate::error::CausalError;

pub use antecedent_attribution::{
    AnomalyScores, ArrowStrength, AttributionError, ChangeAttribution, ChangeAttributionResult,
    DifferenceMeasure, DistributionChangeOptions, FeatureRelevance, MechanismChangeDetection,
    MechanismChangeMethod, PopulationDoContrast, RobustChangeOptions, RootCauseRank,
    StructureChangeOptions, UnitChangeResult, arrow_strengths, detect_mechanism_changes,
    distribution_change, distribution_change_robust, distribution_change_with_row_weights,
    feature_relevance, path_decompose, population_do_contrast, root_cause_rank, score_anomalies,
    score_anomalies_with, structure_change, unit_change,
};
pub use antecedent_counterfactual::{
    AbductionMissingPolicy, CompiledCounterfactualPlan, CounterfactualEngine, CounterfactualError,
    CounterfactualResult, CounterfactualWorld, ExogenousPosterior, NoiseInferenceKind,
    RANK_PRESERVING_ASSUMPTION, nested_counterfactual, nested_hard_counterfactual,
    simultaneous_hard_counterfactual,
};
pub use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, DoSampleResult, DynamicMechanism,
    InvertibleStructuralCausalModel, KdeDoSampler, McmcDoSampler, MechanismAssignment,
    MechanismFamily, MechanismRegistry, MechanismSlot, MechanismTyping, MechanismWorkspace,
    ModelCollection, ModelError, ModelEvaluator, ProbabilisticCausalModel, SelectionPolicy,
    StructuralCausalModel, WeightingDoSampler, interventional_mean, sample_interventional,
    sample_observational,
};

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
pub fn fit_gcm(graph: Dag, data: &TabularData) -> Result<FittedGcm, CausalError> {
    let compiled = CompiledCausalModel::compile(graph).map_err(map_model)?;
    let (store, assignments) = MechanismRegistry::standard()
        .assign_and_fit(&compiled, data, SelectionPolicy::BestScore)
        .map_err(map_model)?;
    Ok(FittedGcm { model: compiled.with_mechanisms(store), assignments })
}

/// Fit the counterfactual path's registry
/// ([`MechanismRegistry::with_heterogeneity_families`]) to `data` on `graph`.
///
/// Differs from [`fit_gcm`] only in the candidate set: the heterogeneity-capable
/// families compete on validation score beside the standard ones, so a per-unit
/// effect that does not vary is a finding about the data wherever such a family
/// could be fit, and a property of the candidate set only where none could.
///
/// # Errors
///
/// Propagates model fit / assignment failures; a mechanism that does not
/// converge is refused with reason `mechanism_fit_not_converged`.
pub fn fit_gcm_counterfactual(graph: Dag, data: &TabularData) -> Result<FittedGcm, CausalError> {
    let compiled = CompiledCausalModel::compile(graph).map_err(map_model)?;
    let (store, assignments) = MechanismRegistry::with_heterogeneity_families()
        .assign_and_fit(&compiled, data, SelectionPolicy::BestScore)
        .map_err(map_mechanism_fit)?;
    Ok(FittedGcm { model: compiled.with_mechanisms(store), assignments })
}

/// Map a mechanism-fit failure, turning non-convergence into a reason-coded
/// refusal that names the remedy instead of surfacing a raw deviance.
///
/// Every parent-conditional categorical fit already runs on standardized parent
/// columns, so a scale mismatch between parents is not the cause; what remains
/// is near-separation or collinearity among the parents, which only the caller
/// can resolve.
pub(crate) fn map_mechanism_fit(e: ModelError) -> CausalError {
    match e {
        ModelError::NotConverged { .. } => crate::unsupported_reason!(
            "mechanism_fit_not_converged",
            "a parent-conditional categorical mechanism did not converge even on standardized \
             parent columns; drop or coarsen parents that nearly separate the categories or \
             duplicate one another, then re-run"
        ),
        other => map_model(other),
    }
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
) -> Result<ValueBatch, CausalError> {
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
) -> Result<IteResult, CausalError> {
    let engine = CounterfactualEngine::new(model);
    let exo = engine.abduct(data, AbductionMissingPolicy::Error, ctx).map_err(map_cf)?;
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
        unit_effects: ite,
        mean_ite: mean,
        noise_inference: exo.kind,
        exogenous: exo,
        unit_effect_intervals: None,
        unit_extrapolative: None,
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
    /// Per-unit posterior intervals, when the inference mode produced per-unit
    /// draws (Bayesian). `None` for Frequentist results, which carry no
    /// per-unit construction.
    pub unit_effect_intervals: Option<UnitEffectIntervals>,
    /// Per-unit extrapolation flags, aligned with [`Self::unit_effects`]: `true`
    /// when the unit's prediction into the arm it did not receive leaves that
    /// arm's observed support (see `gcm.counterfactual.support`).
    pub unit_extrapolative: Option<Arc<[bool]>>,
}

/// Level-tagged per-unit intervals for [`IteResult::unit_effects`].
#[derive(Clone, Debug, PartialEq)]
pub struct UnitEffectIntervals {
    /// Lower bound per unit, aligned with the unit effects.
    pub lower: Arc<[f64]>,
    /// Upper bound per unit.
    pub upper: Arc<[f64]>,
    /// Nominal level the bounds were read at.
    pub level: f64,
    /// Construction id (`unit_posterior_quantile`, `IntervalMethod::as_str`).
    pub method: &'static str,
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
) -> Result<Vec<AnomalyScores>, CausalError> {
    let targets: Arc<[VariableId]> = outcomes.into_iter().collect::<Vec<_>>().into();
    let q = AnomalyAttributionQuery::new(targets, max_units);
    score_anomalies(model, data, &q).map_err(map_attr)
}

/// [`anomaly_attribution`] under an execution context: rows are scored in parallel up to its
/// thread budget and a cancelled context stops the run.
///
/// # Errors
///
/// Attribution failures.
pub fn anomaly_attribution_with(
    model: &CompiledCausalModel,
    data: &TabularData,
    outcomes: impl IntoIterator<Item = VariableId>,
    max_units: usize,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Vec<AnomalyScores>, CausalError> {
    let targets: Arc<[VariableId]> = outcomes.into_iter().collect::<Vec<_>>().into();
    let q = AnomalyAttributionQuery::new(targets, max_units);
    score_anomalies_with(model, data, &q, ctx).map_err(map_attr)
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
) -> Result<ChangeAttributionResult, CausalError> {
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
) -> Result<ChangeAttributionResult, CausalError> {
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
) -> Result<ChangeAttributionResult, CausalError> {
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
) -> Result<Vec<antecedent_attribution::MechanismChangeDetection>, CausalError> {
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
) -> Result<UnitChangeResult, CausalError> {
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
) -> Result<ValueBatch, CausalError> {
    query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    if query.target_population != TargetPopulation::AllObserved {
        return Err(crate::unsupported_reason!(
            "population_not_estimable",
            "sample_interventional_distribution only supports TargetPopulation::AllObserved"
        ));
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
) -> Result<ChangeAttributionResult, CausalError> {
    query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    if query.target_population != TargetPopulation::AllObserved {
        return Err(crate::unsupported_reason!(
            "population_not_estimable",
            "attribute_path_specific only supports TargetPopulation::AllObserved"
        ));
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
) -> Result<ChangeAttributionResult, CausalError> {
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
) -> Result<Vec<FeatureRelevance>, CausalError> {
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
) -> Result<Vec<RootCauseRank>, CausalError> {
    root_cause_rank(attribution, None, None, ctx).map_err(map_attr)
}

#[allow(clippy::needless_pass_by_value)] // map_err adapters
fn map_model(e: ModelError) -> CausalError {
    CausalError::from(e)
}

#[allow(clippy::needless_pass_by_value)] // map_err adapters
fn map_cf(e: CounterfactualError) -> CausalError {
    CausalError::from(e)
}

#[allow(clippy::needless_pass_by_value)] // map_err adapters
fn map_attr(e: AttributionError) -> CausalError {
    CausalError::from(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        AllocationMethod, CausalSchemaBuilder, MeasurementSpec, RoleHint, ShapleyConfig,
        SmallRoleSet, ValueType,
    };
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage};
    use antecedent_graph::DenseNodeId;

    /// A mechanism that still does not converge is a reason-coded refusal naming
    /// the remedy, never a raw deviance in a compile error; every other fit
    /// failure keeps its model error.
    #[test]
    fn non_convergence_maps_to_a_registered_refusal_and_nothing_else_does() {
        let refused = map_mechanism_fit(ModelError::NotConverged {
            message: "multinomial logit did not converge (iters=50, deviance=4044.85)".into(),
        });
        let text = refused.to_string();
        let (code, message) =
            antecedent_core::reason_code::split_prefix(&text).expect("reason-coded refusal");
        assert_eq!(code, "mechanism_fit_not_converged");
        assert!(antecedent_core::reason_code::is_runtime_refusal(code));
        assert!(message.contains("standardized parent columns"), "{message}");
        assert!(!message.contains("deviance"), "{message}");
        assert!(matches!(refused, CausalError::Unsupported { .. }));

        let numerical = map_mechanism_fit(ModelError::Numerical { message: "sigma".into() });
        assert!(matches!(numerical, CausalError::Model(ModelError::Numerical { .. })));
    }

    fn chain_xy(n: usize) -> (CompiledCausalModel, TabularData) {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
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
        let schema = b.build().unwrap();
        let xv: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let yv: Vec<f64> = xv.iter().map(|x| 1.0 + 2.0 * x).collect();
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        (fit_gcm(g, &data).unwrap().model, data)
    }

    /// Facade must surface a typed out-of-range `unit_rows` error (row == n), not panic.
    #[test]
    fn attribute_unit_change_rejects_out_of_range_unit_row() {
        let n = 10usize;
        let (model, data) = chain_xy(n);
        let q = UnitChangeQuery::new(VariableId::from_raw(1), 20)
            .with_unit_rows([n])
            .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let err =
            attribute_unit_change(&model, &data, &q, &ExecutionContext::for_tests(1)).unwrap_err();
        assert!(matches!(
            err,
            CausalError::Attribution(AttributionError::PopulationOutOfRange {
                kind: "row",
                index,
                limit,
            }) if index == n && limit == n
        ));
    }

    /// Facade `score_anomalies` must likewise refuse row == n as typed error.
    #[test]
    fn score_anomalies_rejects_out_of_range_unit_row_through_facade() {
        let n = 10usize;
        let (model, data) = chain_xy(n);
        let q = AnomalyAttributionQuery::new([VariableId::from_raw(1)], 100).with_unit_rows([n]);
        let err = score_anomalies(&model, &data, &q).unwrap_err();
        assert_eq!(err, AttributionError::PopulationOutOfRange { kind: "row", index: n, limit: n });
    }
}
