//! Graphical causal models: fit one mechanism per node of a DAG, then sample, compute
//! counterfactuals and attribute under the fitted model.
//!
//! The module gathers the fitted-model types of `antecedent-model`, `antecedent-counterfactual`
//! and `antecedent-attribution` behind one path.
//!
//! Attribution and simulation on a fitted model are compiled routes: a study with a
//! [`antecedent_core::CausalQuery::Attribute`] or
//! [`antecedent_core::CausalQuery::Simulate`] query on a supplied or accepted DAG
//! fits its mechanisms through a [`crate::MechanismProgram`] (held-out
//! family selection, with the selection record on the result) and returns the
//! engine's answer with a [`crate::CausalContract`] and a claim. The `sample_do`,
//! `policy_value`, `counterfactual_ite`, `anomaly_attribution*`, `attribute_*`,
//! noise-replay and policy functions are the engines those routes run; they are
//! crate-private, and public only under the `calibration-internal` feature for the suites
//! that compare an engine with its route.
//!
//! # Specialized model operations outside the central compiler
//!
//! Some operations on a fitted model have no compiled route and stay public:
//! [`arrow_strengths`], [`population_do_contrast`], [`counterfactual_contrasts`],
//! [`simultaneous_hard_counterfactual`], [`interventional_mean`], the do-samplers
//! ([`KdeDoSampler`], [`McmcDoSampler`], [`WeightingDoSampler`]), the held-out
//! [`ModelEvaluator`], [`rank_root_causes`] (which only reorders an attribution a route
//! returned) and the cyclic, external-mechanism and composition modules. Each takes the
//! caller's [`ExecutionContext`] (or its seed and thread budget) and returns the numbers
//! it computed under the fitted mechanisms. None carries a contract, a claim, a
//! calibration status or an identification statement: the result is a statement about the
//! fitted model, not a checked causal effect of the data. Use a study query for a claim.
//!
//! # Example
//!
//! ```rust,ignore
//! use antecedent::gcm::fit_gcm;
//!
//! let fitted = fit_gcm(dag, &data)?;
//! ```
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AnomalyAttributionQuery, CausalRng, ChangeAttributionQuery, CounterfactualQuery,
    ExecutionContext, Intervention, InterventionalDistributionQuery, MechanismChangeQuery,
    NestedCounterfactualQuery, PathSpecificEffectQuery, TargetPopulation, UnitChangeQuery, Value,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::Dag;
use antecedent_model::{InterventionOverlay, ModelView, NoiseFix, NoiseReplayer, NoiseSource};

use crate::error::CausalError;

/// Declares a model-level engine behind a compiled route.
///
/// A route in the central compiler (`CausalQuery::Simulate`, `CausalQuery::Attribute`) answers
/// every operation declared through this macro, with a contract and a claim; the raw engine is
/// what that route runs, so it is crate-private. It is public only under the
/// `calibration-internal` feature, for the suites that compare an engine's output with the
/// route's.
macro_rules! internal_engine {
    ($(#[$meta:meta])* pub fn $name:ident $($rest:tt)*) => {
        $(#[$meta])*
        #[cfg(feature = "calibration-internal")]
        pub fn $name $($rest)*

        $(#[$meta])*
        #[cfg(not(feature = "calibration-internal"))]
        #[allow(dead_code, reason = "reached only through a compiled route or a calibration suite")]
        pub(crate) fn $name $($rest)*
    };
}

mod attribution_routes;
pub use attribution_routes::rank_root_causes;
#[cfg(feature = "calibration-internal")]
pub use attribution_routes::{
    anomaly_attribution, anomaly_attribution_query_with, attribute_distribution_change,
    attribute_distribution_change_robust, attribute_feature_relevance,
    attribute_feature_relevance_with, attribute_intrinsic_influence, attribute_path_specific,
    attribute_paths, attribute_structure_change, attribute_unit_change, mechanism_change_detection,
    sample_interventional_distribution,
};
#[cfg(not(feature = "calibration-internal"))]
#[allow(unused_imports, reason = "some engines are used only by calibration suites")]
pub(crate) use attribution_routes::{
    anomaly_attribution, anomaly_attribution_query_with, attribute_distribution_change,
    attribute_distribution_change_robust, attribute_feature_relevance,
    attribute_feature_relevance_with, attribute_intrinsic_influence, attribute_path_specific,
    attribute_paths, attribute_structure_change, attribute_unit_change, mechanism_change_detection,
    sample_interventional_distribution,
};

mod policy_value;
#[cfg(feature = "calibration-internal")]
pub use policy_value::policy_value;
#[cfg(not(feature = "calibration-internal"))]
pub(crate) use policy_value::policy_value;
pub use policy_value::{PolicyValue, PreparedPolicy};

mod counterfactual_operations;
#[allow(unused_imports, reason = "preserve the crate-internal non-polled fitting entry point")]
pub(crate) use counterfactual_operations::fit_nested_mechanisms;
#[cfg(test)]
use counterfactual_operations::fit_non_separable_nested_outcome;
pub use counterfactual_operations::{
    CheckedCounterfactualOperation, NestedCounterfactualOperation, NestedOutcomeMechanism,
};
pub(crate) use counterfactual_operations::{
    ContrastTarget, SelectedFamilies, TreatmentLevels, fit_nested_mechanisms_polled, world_levels,
};

mod counterfactual_engine;
#[cfg(feature = "calibration-internal")]
pub use counterfactual_engine::counterfactual_ite;
#[cfg(not(feature = "calibration-internal"))]
pub(crate) use counterfactual_engine::counterfactual_ite;
use counterfactual_engine::{ContrastRequest, contrasts_from_abduction, ite_from_abduction};
pub use counterfactual_engine::{
    ContrastSetResult, IteResult, OutcomeContrast, UnitEffectIntervals, counterfactual_contrasts,
};
pub(crate) use counterfactual_engine::{
    contrast_outcome_nodes, counterfactual_contrasts_with_noise, counterfactual_ite_with_noise,
};

mod counterfactual_contrasts;
pub use counterfactual_contrasts::{ContrastedOutcome, CounterfactualContrasts};

mod adaptive_policy_source;
pub use adaptive_policy_source::{ModelPolicySource, ModelRolloutDesign, StageUtility};

pub use antecedent_attribution::{
    AnomalyScores, ArrowStrength, AttributionError, ChangeAttribution, ChangeAttributionResult,
    ComponentStatus, ComponentUncertainty, DifferenceMeasure, DistributionChangeOptions,
    FeatureRelevance, FitResampling, INTRINSIC_MAX_EXACT_PLAYERS, IntrinsicAllocation,
    IntrinsicBudget, IntrinsicInfluenceQuery, IntrinsicInfluenceResult, IntrinsicMethod,
    MechanismChangeDetection, MechanismChangeMethod, NoiseIndependence, PartialStderr,
    PopulationDoContrast, RobustChangeOptions, RootCauseRank, ShapleyMethod, SourceInfluence,
    StructureChangeOptions, UncertaintyComponent, UnitChangeResult,
};
pub use antecedent_counterfactual::{
    AbductionMissingPolicy, CompiledCounterfactualPlan, CounterfactualEngine, CounterfactualError,
    CounterfactualResult, CounterfactualWorld, ExogenousPosterior, NoiseInferenceKind,
    RANK_PRESERVING_ASSUMPTION,
};
pub use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, DoSampleResult, DynamicMechanism,
    GraphFalsificationConfig, GraphFalsificationOutcome, GraphFalsificationReport,
    GraphUninformativeReason, InvertibleStructuralCausalModel, KdeDoSampler, MarkovImplicationTest,
    McmcDoSampler, MechanismAssignment, MechanismFamily, MechanismRegistry, MechanismSlot,
    MechanismTyping, MechanismWorkspace, ModelError, ModelEvaluator, NodeFamilyConstraint,
    NoiseTable, PolicyRoute, PolicySupport, ProbabilisticCausalModel, RouteDecision,
    SelectionOptions, SelectionPolicy, SelectionProvenance, SelectionRule, TemporalNoise,
    TemporalPath, TemporalPlan, ValidationSplits, WeightingDoSampler, policy_support,
};

/// Distribution-change attribution: the items of `antecedent_attribution::distribution_change`,
/// whose run entries are deprecated wrappers of the kernel functions.
pub mod distribution_change {
    pub use antecedent_attribution::distribution_change::{
        DifferenceMeasure, DistributionChangeOptions, PopulationRefit,
    };

    /// Attribute distributional change between baseline and comparison populations.
    ///
    /// `graph_model` supplies structure; mechanisms are fit separately on each
    /// population subset. Only mechanism components that are ancestors of the
    /// outcome (inclusive) participate, unless `query.components` restricts further.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_attribution::distribution_change`](fn@antecedent_attribution::distribution_change).
    ///
    /// # Errors
    ///
    /// Query validation, fit/sample failures, or Shapley size limits.
    #[deprecated(
        since = "3.1.0",
        note = "use Study with CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::DistributionChange ..)) (route cell GcmAttributeDag); the route fits the standard mechanism program on the study's graph"
    )]
    pub fn distribution_change(
        graph_model: &antecedent_model::compile::CompiledCausalModel,
        data: &antecedent_data::dataset::TabularData,
        query: &antecedent_core::ChangeAttributionQuery,
        options: &antecedent_attribution::DistributionChangeOptions,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_attribution::ChangeAttributionResult,
        antecedent_attribution::AttributionError,
    > {
        antecedent_attribution::distribution_change(graph_model, data, query, options, ctx)
    }

    /// Evaluate a mechanism attribution after refitting both populations under
    /// frequency weights (multinomial bootstrap counts). A count of zero drops that row
    /// from the refit. Mechanism families are selected once on the unweighted
    /// populations and held fixed, as on the Bayesian-bootstrap path.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_attribution::distribution_change::distribution_change_with_frequency_weights`].
    ///
    /// # Errors
    /// Counts of the wrong length, negative or all zero, incompatible selected
    /// mechanisms, cancellation, or attribution failures.
    #[deprecated(
        since = "3.1.0",
        note = "use Study with an Attribute change target and Bayesian inference or bootstrap_replicates (route cell GcmAttributeDag); the route draws the row weights"
    )]
    pub fn distribution_change_with_frequency_weights(
        graph_model: &antecedent_model::compile::CompiledCausalModel,
        data: &antecedent_data::dataset::TabularData,
        query: &antecedent_core::ChangeAttributionQuery,
        options: &antecedent_attribution::DistributionChangeOptions,
        baseline_counts: &[f64],
        comparison_counts: &[f64],
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_attribution::ChangeAttributionResult,
        antecedent_attribution::AttributionError,
    > {
        antecedent_attribution::distribution_change::distribution_change_with_frequency_weights(
            graph_model,
            data,
            query,
            options,
            baseline_counts,
            comparison_counts,
            ctx,
        )
    }

    /// Evaluate a mechanism attribution after refitting both populations under shared
    /// Bayesian-bootstrap row weights. Mechanism families are selected once on the
    /// unweighted populations and held fixed for these refits.
    ///
    /// The two weight vectors correspond to the resolved baseline and comparison rows.
    /// Within each population, one weight is shared by every mechanism factorization.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_attribution::distribution_change_with_row_weights`].
    ///
    /// # Errors
    /// Invalid weights, incompatible selected mechanisms, or attribution failures.
    #[deprecated(
        since = "3.1.0",
        note = "use Study with an Attribute change target and Bayesian inference or bootstrap_replicates (route cell GcmAttributeDag); the route draws the row weights"
    )]
    pub fn distribution_change_with_row_weights(
        graph_model: &antecedent_model::compile::CompiledCausalModel,
        data: &antecedent_data::dataset::TabularData,
        query: &antecedent_core::ChangeAttributionQuery,
        options: &antecedent_attribution::DistributionChangeOptions,
        baseline_weights: &[f64],
        comparison_weights: &[f64],
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_attribution::ChangeAttributionResult,
        antecedent_attribution::AttributionError,
    > {
        antecedent_attribution::distribution_change_with_row_weights(
            graph_model,
            data,
            query,
            options,
            baseline_weights,
            comparison_weights,
            ctx,
        )
    }
}

#[allow(deprecated, reason = "re-exports the deprecated wrappers at their earlier paths")]
pub use distribution_change::{distribution_change, distribution_change_with_row_weights};

/// Structure-change attribution: the items of `antecedent_attribution::structure_change`, whose
/// run entries are deprecated wrappers of the kernel functions.
pub mod structure_change {
    pub use antecedent_attribution::structure_change::StructureChangeOptions;

    /// Attribute outcome-marginal change to differing parent sets between two graphs.
    ///
    /// `baseline_model` and `comparison_model` must share the same variable layout.
    /// Only nodes whose parent sets differ and that are ancestors of the outcome
    /// (in either graph) are Shapley players; other structural diffs are reported
    /// in `ChangeAttributionResult::unidentified`. A player swaps both its parent set and
    /// its comparison-fitted mechanism. `ChangeAttributionResult::total_change` is the change
    /// those players explain; `ChangeAttributionResult::observed_change` is the change measured
    /// on the two populations, so any change outside the players shows up as
    /// `ChangeAttributionResult::unexplained_change`.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_attribution::structure_change`](fn@antecedent_attribution::structure_change).
    ///
    /// # Errors
    ///
    /// Layout mismatch, empty players, cyclic hybrids, fit/sample failures, or
    /// Shapley size limits.
    #[deprecated(
        since = "3.1.0",
        note = "use Study with CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::StructureChange ..)) (route cell GcmAttributeDag); the route fits the standard mechanism program on the study's graph"
    )]
    pub fn structure_change(
        baseline_model: &antecedent_model::compile::CompiledCausalModel,
        comparison_model: &antecedent_model::compile::CompiledCausalModel,
        data: &antecedent_data::dataset::TabularData,
        query: &antecedent_core::ChangeAttributionQuery,
        options: &antecedent_attribution::StructureChangeOptions,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_attribution::ChangeAttributionResult,
        antecedent_attribution::AttributionError,
    > {
        antecedent_attribution::structure_change(
            baseline_model,
            comparison_model,
            data,
            query,
            options,
            ctx,
        )
    }

    /// `structure_change` with both populations' mechanisms refit under `refit`, the families of
    /// each hybrid held at the ones selected on the unweighted populations: one replicate of the
    /// row bootstrap (counts) or one draw of the posterior over the mechanisms (row weights).
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_attribution::structure_change::structure_change_with_refit`].
    ///
    /// # Errors
    ///
    /// As `structure_change`, and weights that do not match the resolved populations.
    #[deprecated(
        since = "3.1.0",
        note = "use Study with an Attribute change target and Bayesian inference or bootstrap_replicates (route cell GcmAttributeDag); the route draws the row weights"
    )]
    pub fn structure_change_with_refit(
        baseline_model: &antecedent_model::compile::CompiledCausalModel,
        comparison_model: &antecedent_model::compile::CompiledCausalModel,
        data: &antecedent_data::dataset::TabularData,
        query: &antecedent_core::ChangeAttributionQuery,
        options: &antecedent_attribution::StructureChangeOptions,
        refit: &antecedent_attribution::PopulationRefit<'_>,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_attribution::ChangeAttributionResult,
        antecedent_attribution::AttributionError,
    > {
        antecedent_attribution::structure_change::structure_change_with_refit(
            baseline_model,
            comparison_model,
            data,
            query,
            options,
            refit,
            ctx,
        )
    }
}

#[allow(deprecated, reason = "re-exports the deprecated wrapper at its earlier path")]
pub use structure_change::structure_change;

/// Sample `n_rows` observational draws from a fitted model.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_model::sample_observational`].
///
/// # Errors
///
/// Unfitted mechanisms or shape errors.
#[deprecated(
    since = "3.1.0",
    note = "use Study with SimulationTarget::InterventionalSample and no intervention (route cell GcmSimulateDag)"
)]
pub fn sample_observational(
    model: &antecedent_model::CompiledCausalModel,
    n_rows: usize,
    rng: &mut antecedent_core::CausalRng,
    ws: &mut antecedent_model::MechanismWorkspace,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_model::ValueBatch, antecedent_model::ModelError> {
    antecedent_model::sample_observational(model, n_rows, rng, ws, ctx)
}

/// Sample under interventions (compiled to an overlay; model is not cloned).
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_model::sample_interventional`].
///
/// # Errors
///
/// Overlay / mechanism failures.
#[deprecated(
    since = "3.1.0",
    note = "use Study with SimulationTarget::InterventionalSample (route cell GcmSimulateDag)"
)]
pub fn sample_interventional(
    model: &antecedent_model::CompiledCausalModel,
    interventions: &[antecedent_core::intervention::Intervention],
    n_rows: usize,
    rng: &mut antecedent_core::CausalRng,
    ws: &mut antecedent_model::MechanismWorkspace,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_model::ValueBatch, antecedent_model::ModelError> {
    antecedent_model::sample_interventional(model, interventions, n_rows, rng, ws, ctx)
}

/// Convenience: interventional mean of a variable from ancestral sampling.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_model::interventional_mean`].
///
/// # Errors
///
/// Sampling failures.
#[deprecated(
    since = "3.1.0",
    note = "use Study with SimulationTarget::InterventionalSample (route cell GcmSimulateDag) and average the outcome column; the route's generator starts at the target's seed"
)]
pub fn interventional_mean(
    model: &antecedent_model::CompiledCausalModel,
    interventions: &[antecedent_core::intervention::Intervention],
    outcome: antecedent_core::ids::VariableId,
    n_draws: usize,
    rng: &mut antecedent_core::CausalRng,
    ws: &mut antecedent_model::MechanismWorkspace,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<f64, antecedent_model::ModelError> {
    antecedent_model::interventional_mean(model, interventions, outcome, n_draws, rng, ws, ctx)
}

/// Simultaneous hard interventions under shared abduced noise.
///
/// The `outer` and `inner` sets are concatenated into one world, so they must
/// target disjoint variables: two interventions on the same variable are
/// contradictory when simultaneous, and picking a winner silently would misreport
/// the query. Nested queries (`Y_{x, M_{x'}}`) go through `nested_counterfactual`,
/// which takes the frozen set explicitly.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_counterfactual::simultaneous_hard_counterfactual`].
///
/// # Errors
///
/// Engine failures, unknown outcome, or overlapping targets.
#[deprecated(
    since = "3.1.0",
    note = "use Study with SimulateQuery::simultaneous_world(outer, inner, outcomes) (route cell GcmSimulateDag); the declaration refuses overlapping sets"
)]
pub fn simultaneous_hard_counterfactual(
    engine: &antecedent_counterfactual::CounterfactualEngine,
    data: &antecedent_data::dataset::TabularData,
    outer: &[antecedent_core::intervention::Intervention],
    inner: &[antecedent_core::intervention::Intervention],
    outcome: antecedent_core::ids::VariableId,
    ws: &mut antecedent_model::batch::MechanismWorkspace,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<f64, antecedent_counterfactual::CounterfactualError> {
    antecedent_counterfactual::simultaneous_hard_counterfactual(
        engine, data, outer, inner, outcome, ws, ctx,
    )
}

/// Alias for `nested_counterfactual` (historical name).
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_counterfactual::nested_hard_counterfactual`].
///
/// # Errors
///
/// As `nested_counterfactual`.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::model_operation(&ModelOperation::NestedCounterfactual(..)) on a study prepared on the DAG and table (program family FittedModelSimulation)"
)]
pub fn nested_hard_counterfactual(
    engine: &antecedent_counterfactual::CounterfactualEngine,
    data: &antecedent_data::dataset::TabularData,
    spec: &antecedent_counterfactual::NestedSpec<'_>,
    ws: &mut antecedent_model::batch::MechanismWorkspace,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<f64, antecedent_counterfactual::CounterfactualError> {
    antecedent_counterfactual::nested_hard_counterfactual(engine, data, spec, ws, ctx)
}

/// `nested_counterfactual` against a caller-supplied `ExogenousPosterior`
/// (abduct once, evaluate many nested queries).
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_counterfactual::nested_counterfactual_with_exo`].
///
/// # Errors
///
/// Unsupported intervention kinds, an invalid frozen set, or predict failures.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::model_operation(&ModelOperation::NestedCounterfactual(..)) on a study prepared on the DAG and table (program family FittedModelSimulation)"
)]
pub fn nested_counterfactual_with_exo(
    engine: &antecedent_counterfactual::CounterfactualEngine,
    exo: &antecedent_counterfactual::ExogenousPosterior,
    spec: &antecedent_counterfactual::NestedSpec<'_>,
    ws: &mut antecedent_model::batch::MechanismWorkspace,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<f64, antecedent_counterfactual::CounterfactualError> {
    antecedent_counterfactual::nested_counterfactual_with_exo(engine, exo, spec, ws, ctx)
}

/// Nested counterfactual (twin-network) under shared abduced noise.
///
/// The query `Y_{inner, F_{outer}}`: the outcome under the `inner` interventions
/// while every variable in `frozen` keeps the value it takes in the `outer` world.
/// For the mediation contrast `Y_{x, M_{x'}}` take `outer = do(X = x')`,
/// `inner = do(X = x)` and `frozen = {M}`.
///
/// 1. Abduct exogenous noise once from factual data.
/// 2. Evaluate the **outer** world.
/// 3. Freeze each node of `frozen` at its outer counterfactual value. Soft/stochastic
///    outer interventions are realized first; freezes store those realized values.
/// 4. Evaluate the **inner** world with those freezes plus `inner`. Every node that is
///    not frozen — including nodes between an inner target and the outcome, and side
///    paths such as a second mediator — is re-evaluated under the inner intervention.
///
/// Freezing more than the mediators changes the estimand (it holds every frozen path
/// at the outer world), which is why the set is an argument rather than inferred.
/// Supports Set, Shift, Soft, and Stochastic interventions.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_counterfactual::nested_counterfactual`].
///
/// # Errors
///
/// Unsupported intervention kinds, a frozen node that is an `inner` target or the
/// outcome, unknown variables, abduction/predict failures.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::model_operation(&ModelOperation::NestedCounterfactual(..)) on a study prepared on the DAG and table (program family FittedModelSimulation)"
)]
pub fn nested_counterfactual(
    engine: &antecedent_counterfactual::CounterfactualEngine,
    data: &antecedent_data::dataset::TabularData,
    spec: &antecedent_counterfactual::NestedSpec<'_>,
    ws: &mut antecedent_model::batch::MechanismWorkspace,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<f64, antecedent_counterfactual::CounterfactualError> {
    antecedent_counterfactual::nested_counterfactual(engine, data, spec, ws, ctx)
}

/// Per-unit change attribution: the items of `antecedent_attribution::unit_change`.
/// The run entry is a deprecated wrapper of the kernel function.
pub mod unit_change {
    /// Attribute per-unit outcome change to input / mechanism components.
    ///
    /// Abduces exogenous noise once, then evaluates the outcome mechanism on
    /// coalition-mixed parent values with that fixed noise (the factual-vs-reference,
    /// counterfactual Shapley decomposition of Budhathoki, Michailidis & Janzing 2022).
    /// Shapley values therefore attribute the real mechanism payoff, not a linear
    /// surrogate.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_attribution::unit_change`](fn@antecedent_attribution::unit_change).
    ///
    /// # Errors
    ///
    /// Size limits, out-of-range `unit_rows`, abduction, or Shapley failures.
    #[deprecated(
        since = "3.1.0",
        note = "use Study with CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::UnitChange ..)) (route cell GcmAttributeDag); the route fits the standard mechanism program on the study's graph"
    )]
    pub fn unit_change(
        model: &antecedent_model::compile::CompiledCausalModel,
        data: &antecedent_data::dataset::TabularData,
        query: &antecedent_core::UnitChangeQuery,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<antecedent_attribution::UnitChangeResult, antecedent_attribution::AttributionError>
    {
        antecedent_attribution::unit_change(model, data, query, ctx)
    }
}

#[allow(deprecated, reason = "re-exports the deprecated wrapper at its earlier path")]
pub use unit_change::unit_change;

/// Rank components by absolute contribution, optionally aggregating across a
/// graph ensemble's contribution draws.
///
/// When `graph_samples` and `contribution_draws` are provided, each draw column
/// is a component contribution; weights come from the graph ensemble. Results
/// include per-rank `graph_std`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_attribution::root_cause_rank`].
///
/// # Errors
///
/// Shape mismatches or empty contributions.
#[deprecated(
    since = "3.1.0",
    note = "use Study with CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::RootCauses ..)) (route cell GcmAttributeDag); the route fits the standard mechanism program on the study's graph"
)]
pub fn root_cause_rank(
    attribution: &antecedent_attribution::ChangeAttributionResult,
    graph_samples: Option<&antecedent_prob::graph_samples::WeightedGraphSamples>,
    contribution_draws: Option<&antecedent_prob::posterior::PosteriorDraws>,
    _ctx: &antecedent_core::ExecutionContext,
) -> Result<Vec<antecedent_attribution::RootCauseRank>, antecedent_attribution::AttributionError> {
    antecedent_attribution::root_cause_rank(attribution, graph_samples, contribution_draws, _ctx)
}

/// Decompose outcome influence into directed-path shares using arrow strengths
/// along each path (product of signed β on linear-Gaussian edges).
///
/// Signed products preserve cancelling paths; `total_change` is the sum of signed
/// path shares (not absolute shares). Non-linear mechanisms are refused.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_attribution::path_decompose`].
///
/// # Errors
///
/// Missing nodes, path enumeration limits, or non-linear mechanisms on a path.
#[deprecated(
    since = "3.1.0",
    note = "use Study with CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::Paths ..)) (route cell GcmAttributeDag); the route fits the standard mechanism program on the study's graph"
)]
pub fn path_decompose(
    model: &antecedent_model::compile::CompiledCausalModel,
    sources: &[antecedent_core::ids::VariableId],
    outcome: antecedent_core::ids::VariableId,
    max_paths: usize,
    max_len: usize,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_attribution::ChangeAttributionResult, antecedent_attribution::AttributionError>
{
    antecedent_attribution::path_decompose(model, sources, outcome, max_paths, max_len, ctx)
}

/// Intrinsic causal influence: the items of `antecedent_attribution::intrinsic_influence`.
/// The run entry is a deprecated wrapper of the kernel function.
pub mod intrinsic_influence {
    pub use antecedent_attribution::intrinsic_influence::{
        ComponentStatus, ComponentUncertainty, FitResampling, INTRINSIC_MAX_EXACT_PLAYERS,
        IntrinsicAllocation, IntrinsicBudget, IntrinsicInfluenceQuery, IntrinsicInfluenceResult,
        IntrinsicMethod, MIN_FIT_BOOTSTRAP_REPLICATES, NoiseIndependence, PartialStderr,
        ShapleyMethod, SourceInfluence, UncertaintyComponent,
        intrinsic_influence_with_temporal_runs,
    };

    /// Attribute the variance of `query.target` to the exogenous noise terms of its ancestors and
    /// itself.
    ///
    /// `noise` states what is known about dependence among the noise terms; `data`, when supplied
    /// and `query.fit_bootstrap_replicates > 0`, is the sample the mechanisms were fitted on and is
    /// resampled to measure mechanism-fitting uncertainty (each refit selects mechanisms with the
    /// standard registry). Without data that component is reported as unquantified.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_attribution::intrinsic_influence`](fn@antecedent_attribution::intrinsic_influence).
    ///
    /// # Errors
    ///
    /// Coded refusals for undeclared or correlated noise (bidirected edges, latent common
    /// causes), an unfitted mechanism or one whose noise is not independent across rows, a
    /// non-continuous target, too many sources or an oversized simulation plan; invalid settings;
    /// model or replay failures; `AttributionError::Cancelled`.
    #[deprecated(
        since = "3.1.0",
        note = "use Study with CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::IntrinsicInfluence ..)) (route cell GcmAttributeDag); the route fits the standard mechanism program on the study's graph"
    )]
    pub fn intrinsic_influence(
        model: &antecedent_model::compile::CompiledCausalModel,
        noise: antecedent_attribution::NoiseIndependence<'_>,
        data: Option<&antecedent_data::dataset::TabularData>,
        query: &antecedent_attribution::IntrinsicInfluenceQuery,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_attribution::IntrinsicInfluenceResult,
        antecedent_attribution::AttributionError,
    > {
        antecedent_attribution::intrinsic_influence(model, noise, data, query, ctx)
    }
}

#[allow(deprecated, reason = "re-exports the deprecated wrapper at its earlier path")]
pub use intrinsic_influence::intrinsic_influence;

/// Feature relevance under interventions: the items of `antecedent_attribution::feature_relevance`.
/// The run entry is a deprecated wrapper of the kernel function.
pub mod feature_relevance {
    pub use antecedent_attribution::feature_relevance::FeatureRelevanceOptions;

    /// Score interventional relevance of each `feature` for `outcome` via Shapley values.
    ///
    /// For coalition `S`, `v(S) = E[Y | do(Xᵢ = μᵢ + δ ∀ i∈S)]` (features outside `S` free under
    /// ancestral sampling). `δ` is the shift from each feature's empirical mean. Shapley values
    /// `φᵢ` attribute the change from the empty intervention to intervening on all listed
    /// features; the reported score is `|φᵢ|`.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_attribution::feature_relevance`](fn@antecedent_attribution::feature_relevance).
    ///
    /// # Errors
    ///
    /// Sampling / size failures.
    #[deprecated(
        since = "3.1.0",
        note = "use Study with CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::FeatureRelevance ..)) (route cell GcmAttributeDag); the route fits the standard mechanism program on the study's graph"
    )]
    pub fn feature_relevance(
        model: &antecedent_model::compile::CompiledCausalModel,
        data: &antecedent_data::dataset::TabularData,
        outcome: antecedent_core::ids::VariableId,
        features: &[antecedent_core::ids::VariableId],
        options: &antecedent_attribution::FeatureRelevanceOptions,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        Vec<antecedent_attribution::FeatureRelevance>,
        antecedent_attribution::AttributionError,
    > {
        antecedent_attribution::feature_relevance(model, data, outcome, features, options, ctx)
    }
}

#[allow(deprecated, reason = "re-exports the deprecated wrapper at its earlier path")]
pub use feature_relevance::feature_relevance;

/// Robust attribution via mechanism hybrids + Shapley.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_attribution::distribution_change_robust`].
///
/// # Errors
///
/// Fit / size / Shapley failures.
#[deprecated(
    since = "3.1.0",
    note = "use Study with CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::DistributionChange ..)) (route cell GcmAttributeDag); the route fits the standard mechanism program on the study's graph"
)]
pub fn distribution_change_robust(
    graph_model: &antecedent_model::compile::CompiledCausalModel,
    data: &antecedent_data::dataset::TabularData,
    query: &antecedent_core::ChangeAttributionQuery,
    options: &antecedent_attribution::RobustChangeOptions,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_attribution::ChangeAttributionResult, antecedent_attribution::AttributionError>
{
    antecedent_attribution::distribution_change_robust(graph_model, data, query, options, ctx)
}

/// Detect which mechanisms differ between baseline and comparison populations.
///
/// This does **not** attribute outcome change — a changed mechanism need not
/// drive the target.
///
/// Applies a Benjamini–Hochberg false-discovery-rate correction across
/// `query.targets` by default (see
/// `detect_mechanism_changes_with_correction` to override or disable this).
/// `query.targets` is bounded by `query.max_targets`, i.e. the API is designed
/// to scan multiple candidate mechanisms in a single call, so judging each
/// target against the nominal `significance_level` independently would let the
/// family-wise false-positive rate grow with the number of targets (≈40% at
/// ten truly-unchanged targets and α=0.05). BH is chosen as the default
/// because it is the same default `antecedent_discovery::PcmciPlus` uses for
/// an analogous multi-hypothesis family (`PcmciPlus::new` sets
/// `fdr: Some(FdrAdjustment::bh())`), and it controls FDR rather than the
/// stricter (and more conservative) family-wise error rate, which is the
/// usual choice when the family is a moderate, human-reviewed set of
/// candidate mechanisms rather than a single make-or-break decision.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_attribution::detect_mechanism_changes`].
///
/// # Errors
///
/// Query / fit / stats failures.
#[deprecated(
    since = "3.1.0",
    note = "use Study with CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::MechanismChange ..)) (route cell GcmAttributeDag); the route fits the standard mechanism program on the study's graph"
)]
pub fn detect_mechanism_changes(
    graph_model: &antecedent_model::compile::CompiledCausalModel,
    data: &antecedent_data::dataset::TabularData,
    query: &antecedent_core::MechanismChangeQuery,
    method: antecedent_attribution::MechanismChangeMethod,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<
    Vec<antecedent_attribution::MechanismChangeDetection>,
    antecedent_attribution::AttributionError,
> {
    antecedent_attribution::detect_mechanism_changes(graph_model, data, query, method, ctx)
}

/// `score_anomalies` under an `ExecutionContext`: rows are independent exact-Shapley
/// problems (deterministic given the row), so they are scored in parallel up to the context's
/// thread budget, and a cancelled context stops the run. The result does not depend on the
/// thread count.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_attribution::score_anomalies_with`].
///
/// # Errors
///
/// As `score_anomalies`; the lowest-index failing row's error when several fail.
#[deprecated(
    since = "3.1.0",
    note = "use Study with CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::Anomaly ..)) (route cell GcmAttributeDag); the route fits the standard mechanism program on the study's graph"
)]
pub fn score_anomalies_with(
    model: &antecedent_model::compile::CompiledCausalModel,
    data: &antecedent_data::dataset::TabularData,
    query: &antecedent_core::AnomalyAttributionQuery,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Vec<antecedent_attribution::AnomalyScores>, antecedent_attribution::AttributionError> {
    antecedent_attribution::score_anomalies_with(model, data, query, ctx)
}

/// Score anomalies and attribute them to ancestor noise terms via Shapley
/// (Budhathoki, Minorics, Bloebaum & Janzing 2022): replace noise coordinates outside the
/// coalition with the point-mass reference `0` and redistribute the target's IT score
/// (see the module docs for why that is a tail probability and not a density, and for the
/// reference distribution).
///
/// Runs serially; see `score_anomalies_with` for a thread budget.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_attribution::score_anomalies`].
///
/// # Errors
///
/// Size limit, out-of-range `unit_rows`, non-finite target or reconstructed values, or
/// data/model failures.
#[deprecated(
    since = "3.1.0",
    note = "use Study with CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::Anomaly ..)) (route cell GcmAttributeDag); the route fits the standard mechanism program on the study's graph"
)]
pub fn score_anomalies(
    model: &antecedent_model::compile::CompiledCausalModel,
    data: &antecedent_data::dataset::TabularData,
    query: &antecedent_core::AnomalyAttributionQuery,
) -> Result<Vec<antecedent_attribution::AnomalyScores>, antecedent_attribution::AttributionError> {
    antecedent_attribution::score_anomalies(model, data, query)
}

/// Population do-contrast of parent on child: `E[Y|do(X=μ+δ/2)] − E[Y|do(X=μ−δ/2)]`.
///
/// Both interventional arms replay the same random stream (common random numbers), so the
/// exogenous noise cancels in every per-draw difference. Independent streams would make a
/// null edge read `E|N(0, 2σ²/n)| = 2σ/√(πn) > 0` — a phantom influence that shrinks only as
/// `1/√n`.
///
/// This is **not** intrinsic (noise-based) causal influence.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_attribution::population_do_contrast`].
///
/// # Errors
///
/// Size / model failures.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::model_operation(&ModelOperation::PopulationDoContrast(..)) (program family FittedModelAttribution); the draws follow the operation's seed, which equals a direct call under a context seeded with it"
)]
pub fn population_do_contrast(
    model: &antecedent_model::compile::CompiledCausalModel,
    data: &antecedent_data::dataset::TabularData,
    parent: antecedent_core::ids::VariableId,
    child: antecedent_core::ids::VariableId,
    delta: f64,
    max_units: usize,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_attribution::PopulationDoContrast, antecedent_attribution::AttributionError>
{
    antecedent_attribution::population_do_contrast(
        model, data, parent, child, delta, max_units, ctx,
    )
}

/// Compute arrow strengths for all edges in the compiled model: the cut-edge divergence
/// `ArrowStrength::strength` and the variance contribution
/// `ArrowStrength::variance_contribution` of each.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_attribution::arrow_strengths`].
///
/// # Errors
///
/// `AttributionError::InvalidInput` when an edge carries variance into a child with no noise
/// variance of its own (the divergence is infinite) or a variance is not finite.
///
/// `AttributionError::NonLinearGaussianMechanism` when a child with parents is not a
/// linear-family mechanism, or when any node's variance is not available in closed form
/// (the strength of an edge depends on its parent's variance, so an unusable mechanism
/// anywhere upstream makes the answer unavailable rather than approximate).
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::model_operation(&ModelOperation::ArrowStrengths(..)) on a study prepared on the DAG and table (program family FittedModelAttribution); Python: antecedent.analyze(data, graph=edges, query=antecedent.model_operations.ArrowStrengths())"
)]
pub fn arrow_strengths(
    model: &antecedent_model::compile::CompiledCausalModel,
) -> Result<Vec<antecedent_attribution::ArrowStrength>, antecedent_attribution::AttributionError> {
    antecedent_attribution::arrow_strengths(model)
}

/// Fitted GCM plus per-node assignment records.
#[derive(Clone, Debug)]
pub struct FittedGcm {
    /// Compiled plan with fitted mechanisms.
    pub model: CompiledCausalModel,
    /// Auto-assignment provenance (no silent defaults).
    pub assignments: Vec<MechanismAssignment>,
}

/// Falsify `graph` against `data` by graph-node permutation, with no fitted mechanisms.
///
/// Tests the graph's local Markov implications with the configured conditional-independence test
/// and compares the violation count with that of random node relabellings of the same graph.
/// The result is `Rejected`, `NotRejected`, `Uninformative` or `Incomplete`, carrying every
/// implication's test evidence, the seeds, the test budget consumed and the stated limitations.
/// This is not the shuffled-outcome likelihood diagnostic of a fitted model.
///
/// # Errors
///
/// Invalid configuration, a graph that is not a static DAG over numeric observed columns, an
/// unknown or non-frequentist CI test, or a CI-test failure. A budget or cancellation stop is an
/// `Incomplete` report, not an error.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::falsify_graph on a study prepared on the DAG and the table (program family GraphFalsification)"
)]
pub fn evaluate_graph(
    graph: &Dag,
    data: &TabularData,
    config: &GraphFalsificationConfig,
    ctx: &ExecutionContext,
) -> Result<GraphFalsificationReport, CausalError> {
    antecedent_model::falsify_graph(graph, data, config, ctx).map_err(CausalError::from)
}

/// Fit a standard mechanism registry to `data` on `graph`.
///
/// # Errors
///
/// Propagates model fit / assignment failures.
pub fn fit_gcm(graph: Dag, data: &TabularData) -> Result<FittedGcm, CausalError> {
    fit_with_registry(&MechanismRegistry::standard(), Arc::new(graph), data, CausalError::from)
}

/// Compile `graph` and fit `registry` to `data`; `fit_error` maps a failure of the
/// fit itself (compile failures always convert plainly).
fn fit_with_registry(
    registry: &MechanismRegistry,
    graph: Arc<Dag>,
    data: &TabularData,
    fit_error: fn(ModelError) -> CausalError,
) -> Result<FittedGcm, CausalError> {
    let compiled = CompiledCausalModel::compile_shared(graph)?;
    let (store, assignments) =
        registry.assign_and_fit(&compiled, data, SelectionPolicy::BestScore).map_err(fit_error)?;
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
///
/// The graph is shared, not copied: the fitted model keeps a handle to the same frozen
/// graph, so refitting one frozen target costs no graph clone.
pub fn fit_gcm_counterfactual(
    graph: &Arc<Dag>,
    data: &TabularData,
) -> Result<FittedGcm, CausalError> {
    fit_with_registry(
        &MechanismRegistry::with_heterogeneity_families(),
        Arc::clone(graph),
        data,
        map_mechanism_fit,
    )
}

/// Fit a caller-chosen candidate set to `data` on `graph`, with per-node family
/// constraints and a declared validation-split scheme.
///
/// `registry` fixes the candidate families per outcome type (see
/// [`registry_from_candidates`]); `options` narrows or overrides them per node and
/// declares how held-out scores are cut when rows are dependent. With
/// [`SelectionOptions::default`] and [`MechanismRegistry::standard`] this is
/// [`fit_gcm`]. Every node's assignment keeps each candidate's held-out score on
/// the shared folds, the failed candidates with their reasons, and the selection
/// provenance.
///
/// # Errors
///
/// Fit failures; a constraint or split declaration the model or data cannot honour
/// is refused with reason `mechanism_candidates_invalid`, and a required family
/// that could not be scored with reason `mechanism_family_unavailable`.
pub fn fit_gcm_with(
    graph: Dag,
    data: &TabularData,
    registry: &MechanismRegistry,
    options: &SelectionOptions,
) -> Result<FittedGcm, CausalError> {
    let compiled = CompiledCausalModel::compile(graph)?;
    let (store, assignments) = registry
        .assign_and_fit_with(&compiled, data, SelectionPolicy::BestScore, options)
        .map_err(map_mechanism_fit)?;
    Ok(FittedGcm { model: compiled.with_mechanisms(store), assignments })
}

/// Fit the existing registry selection while polling the caller's execution cancellation.
/// Numerical selection and failure codes are the same as [`fit_gcm`] / [`fit_gcm_with`].
///
/// # Errors
/// The existing fitting failures, or cancellation before a node, candidate or validation fold.
pub fn fit_gcm_polled(
    graph: Dag,
    data: &TabularData,
    selection: Option<(&MechanismRegistry, &SelectionOptions)>,
    ctx: &ExecutionContext,
) -> Result<FittedGcm, CausalError> {
    let compiled = CompiledCausalModel::compile(graph)?;
    let standard = MechanismRegistry::standard();
    let defaults = SelectionOptions::default();
    let (registry, options) = selection.unwrap_or((&standard, &defaults));
    let (store, assignments) = registry
        .assign_and_fit_with_in(&compiled, data, SelectionPolicy::BestScore, options, ctx)
        .map_err(|error| match error {
            ModelError::Cancelled => CausalError::Cancelled { stage: "policy_fit" },
            other if selection.is_some() => map_mechanism_fit(other),
            other => CausalError::from(other),
        })?;
    Ok(FittedGcm { model: compiled.with_mechanisms(store), assignments })
}

/// Every family a candidate token may name exactly, by its registry id.
const ALL_FAMILIES: [MechanismFamily; 16] = [
    MechanismFamily::LinearGaussian,
    MechanismFamily::Constant,
    MechanismFamily::Discrete,
    MechanismFamily::LinearInteractions,
    MechanismFamily::LinearSpline,
    MechanismFamily::DiscreteInteractions,
    MechanismFamily::DiscreteSpline,
    MechanismFamily::HierarchicalLinear,
    MechanismFamily::HierarchicalGlm,
    MechanismFamily::Bvar,
    MechanismFamily::LinearGaussianStateSpace,
    MechanismFamily::GaussianProcess,
    MechanismFamily::TreeBoostedAdditive,
    MechanismFamily::TreeForestAdditive,
    MechanismFamily::TreeBoostedCategorical,
    MechanismFamily::TreeForestCategorical,
];

/// Families a candidate token names. Exact family ids name one family; the
/// outcome-agnostic aliases name the member of the family for each outcome type
/// (a continuous node uses the regression member, a categorical node the
/// classification member).
///
/// | token | families |
/// | --- | --- |
/// | `linear` | `linear_gaussian`, `discrete` |
/// | `logistic` | `discrete` |
/// | `additive_spline` | `linear_spline`, `discrete_spline` |
/// | `interactions` | `linear_interactions`, `discrete_interactions` |
/// | `random_forest` | `tree_forest_additive`, `tree_forest_categorical` |
/// | `gradient_boosted_trees` | `tree_boosted_additive`, `tree_boosted_categorical` |
/// | `tree_ensembles` | all four tree families |
/// | `tree_ensemble_classifier` | `tree_boosted_categorical`, `tree_forest_categorical` |
///
/// # Errors
///
/// An unknown token, refused with reason `mechanism_candidates_invalid`.
pub fn candidate_families(token: &str) -> Result<Vec<MechanismFamily>, CausalError> {
    use MechanismFamily as F;
    let families: &[F] = match token {
        "linear" => &[F::LinearGaussian, F::Discrete],
        "logistic" => &[F::Discrete],
        "additive_spline" => &[F::LinearSpline, F::DiscreteSpline],
        "interactions" => &[F::LinearInteractions, F::DiscreteInteractions],
        "random_forest" => &[F::TreeForestAdditive, F::TreeForestCategorical],
        "gradient_boosted_trees" => &[F::TreeBoostedAdditive, F::TreeBoostedCategorical],
        "tree_ensembles" => &[
            F::TreeBoostedAdditive,
            F::TreeForestAdditive,
            F::TreeBoostedCategorical,
            F::TreeForestCategorical,
        ],
        "tree_ensemble_classifier" => &[F::TreeBoostedCategorical, F::TreeForestCategorical],
        exact => {
            return ALL_FAMILIES.iter().copied().find(|f| f.id() == exact).map_or_else(
                || {
                    Err(crate::compile_reason!(
                        "mechanism_candidates_invalid",
                        "unknown mechanism candidate `{exact}`"
                    ))
                },
                |family| Ok(vec![family]),
            );
        }
    };
    Ok(families.to_vec())
}

/// A registry whose candidate lists are exactly the families `tokens` name (see
/// [`candidate_families`]), split by outcome type, in the order given, followed by
/// [`MechanismFamily::Constant`] (admissible only for a degenerate column). An outcome
/// type no token names keeps its standard family (`linear_gaussian`, `discrete`).
///
/// # Errors
///
/// An unknown token or an empty list, refused with reason `mechanism_candidates_invalid`.
pub fn registry_from_candidates(tokens: &[&str]) -> Result<MechanismRegistry, CausalError> {
    if tokens.is_empty() {
        return Err(crate::compile_reason!(
            "mechanism_candidates_invalid",
            "the candidate list is empty"
        ));
    }
    let mut continuous = Vec::new();
    let mut discrete = Vec::new();
    for token in tokens {
        for family in candidate_families(token)? {
            let list = if family.is_categorical() { &mut discrete } else { &mut continuous };
            if family != MechanismFamily::Constant && !list.contains(&family) {
                list.push(family);
            }
        }
    }
    // An outcome type no token names keeps its standard family, so a list of
    // categorical-only candidates does not leave every continuous root without one.
    if continuous.is_empty() {
        continuous.push(MechanismFamily::LinearGaussian);
    }
    if discrete.is_empty() {
        discrete.push(MechanismFamily::Discrete);
    }
    continuous.push(MechanismFamily::Constant);
    discrete.push(MechanismFamily::Constant);
    Ok(MechanismRegistry { continuous: Arc::from(continuous), discrete: Arc::from(discrete) })
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
        ModelError::InvalidSelection { message } => {
            crate::compile_reason!("mechanism_candidates_invalid", "{message}")
        }
        ModelError::RequiredFamilyUnavailable { message } => {
            crate::compile_reason!("mechanism_family_unavailable", "{message}")
        }
        ModelError::NotConverged { .. } => crate::unsupported_reason!(
            "mechanism_fit_not_converged",
            "a parent-conditional categorical mechanism did not converge even on standardized \
             parent columns; drop or coarsen parents that nearly separate the categories or \
             duplicate one another, then re-run"
        ),
        other => CausalError::from(other),
    }
}

internal_engine! {
    /// Interventional ancestral sample under hard `do` values (batch, one GIL/boundary crossing).
    ///
    /// # Errors
    ///
    /// Sampling failures.
    #[doc(hidden)]
    pub fn sample_do(
        model: &CompiledCausalModel,
        interventions: &[Intervention],
        n: usize,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<ValueBatch, CausalError> {
        let mut ws = MechanismWorkspace::default();
        antecedent_model::sample::sample_interventional(model, interventions, n, rng, &mut ws, ctx).map_err(CausalError::from)
    }
}

internal_engine! {
    /// Abduction once, then the factual units replayed under `interventions`, which may carry
    /// value programs (`Intervention::transform` / `Intervention::assign`). The retained noise
    /// goes through the same per-node precedence and program executor as [`sample_do`], so a
    /// policy gives the same values on the factual rows of an invertible model on both routes.
    ///
    /// # Errors
    ///
    /// Abduction / prediction failures, or a refused policy.
    #[doc(hidden)]
    pub fn counterfactual_policy(
        model: CompiledCausalModel,
        data: &TabularData,
        interventions: &[Intervention],
        outcomes: &[VariableId],
        ctx: &ExecutionContext,
    ) -> Result<CounterfactualResult, CausalError> {
        let engine = CounterfactualEngine::new(model);
        let exo = engine.abduct(data, AbductionMissingPolicy::Error, ctx)?;
        let mut ws = MechanismWorkspace::default();
        let world = CounterfactualWorld { unit_rows: None, interventions: Arc::from(interventions) };
        Ok(engine.predict(&exo, &[world], outcomes, false, &mut ws, ctx)?)
    }
}

internal_engine! {
    /// Abduction at every step of a factual panel, then the retained per-step noise replayed through
    /// the model under the plan's scheduled interventions. `steps[t]` holds the factual units at
    /// step `t` (every unit present at every step, in the same order); `plan.n_steps` must equal
    /// `steps.len()` and `plan.n_rows` each table's row count.
    ///
    /// The noise goes through the executor [`counterfactual_policy`] and [`sample_temporal_policy`]
    /// use, so a policy (including a lagged read of earlier realised steps) means the same thing on
    /// all three routes.
    ///
    /// # Errors
    ///
    /// Abduction failures at any step, noise that does not match the model, rows or steps, a policy
    /// reading history that does not exist yet, or a refused program.
    #[doc(hidden)]
    pub fn counterfactual_temporal_policy(
        model: CompiledCausalModel,
        steps: &[TabularData],
        plan: &TemporalPlan<'_>,
        ctx: &ExecutionContext,
    ) -> Result<TemporalPath, CausalError> {
        let engine = CounterfactualEngine::new(model);
        let exo = steps
            .iter()
            .map(|data| engine.abduct(data, AbductionMissingPolicy::Error, ctx))
            .collect::<Result<Vec<_>, _>>()?;
        let mut ws = MechanismWorkspace::default();
        Ok(engine.predict_temporal(&exo, plan, &mut ws, ctx)?)
    }
}

internal_engine! {
    /// Draw `n` rows under `interventions` together with the exogenous noise that produced them.
    /// Replaying the returned table with [`replay_noise`] under the same interventions and salt `0`
    /// reproduces the draws bit for bit.
    ///
    /// # Errors
    ///
    /// An invalid intervention set, an unfitted mechanism that is not hard-set, a table over the
    /// context's memory limit, or cancellation.
    #[doc(hidden)]
    pub fn sample_with_noise(
        model: &CompiledCausalModel,
        interventions: &[Intervention],
        n: usize,
        ctx: &ExecutionContext,
    ) -> Result<(ValueBatch, NoiseTable), CausalError> {
        let overlay = InterventionOverlay::from_interventions(model, interventions)?;
        Ok(NoiseReplayer::new().sample(&ModelView::with_overlay(model, overlay), n, ctx)?)
    }
}

internal_engine! {
    /// A noise table of `n_rows` rows holding the caller's own noise: one column per model node in
    /// dense node order (`values[node * n_rows + row]`). Noise must be finite, and strictly inside
    /// `(0, 1)` for a categorical node.
    ///
    /// # Errors
    ///
    /// A shape mismatch, an invalid noise value, or a table over the context's memory limit.
    #[doc(hidden)]
    pub fn noise_table_from_columns(
        model: &CompiledCausalModel,
        n_rows: usize,
        values: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<NoiseTable, CausalError> {
        let n_nodes = model.n_nodes();
        if n_rows.checked_mul(n_nodes) != Some(values.len()) {
            return Err(CausalError::from(ModelError::Shape {
                message: "noise must hold n_rows values for each of the model's nodes".into(),
            }));
        }
        let mut table = NoiseTable::sample(&ModelView::observational(model), n_rows, ctx)?;
        let nodes: Vec<_> = (0..n_nodes)
            .map(|node| {
                u32::try_from(node).map(antecedent_graph::DenseNodeId::from_raw).map_err(|_| {
                    CausalError::from(ModelError::Shape { message: "node index exceeds u32".into() })
                })
            })
            .collect::<Result<_, _>>()?;
        table.fix(model, &nodes, NoiseFix::Columns(values))?;
        Ok(table)
    }
}

internal_engine! {
    /// Replay retained noise through the model under `interventions`, with the per-node precedence
    /// of forward sampling (a stochastic policy draws from the stream keyed by `policy_salt`).
    ///
    /// # Errors
    ///
    /// An invalid intervention set, a table of another model's width, a mechanism that refuses its
    /// noise, a table over the context's memory limit, or cancellation.
    #[doc(hidden)]
    pub fn replay_noise(
        model: &CompiledCausalModel,
        interventions: &[Intervention],
        noise: &NoiseTable,
        policy_salt: u64,
        ctx: &ExecutionContext,
    ) -> Result<ValueBatch, CausalError> {
        let overlay = InterventionOverlay::from_interventions(model, interventions)?;
        Ok(NoiseReplayer::new().replay(
            &ModelView::with_overlay(model, overlay),
            NoiseSource::Table(noise),
            policy_salt,
            ctx,
        )?)
    }
}

internal_engine! {
    /// Roll the model out over time under a scheduled intervention sequence, drawing fresh noise.
    /// A value program may read earlier realised steps through `plan.window` (see
    /// [`antecedent_model::rollout_temporal`]).
    ///
    /// # Errors
    ///
    /// Rollout failures: shape, a policy reading history that does not exist yet, or a refused
    /// program.
    #[doc(hidden)]
    pub fn sample_temporal_policy(
        model: &CompiledCausalModel,
        plan: &TemporalPlan<'_>,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<TemporalPath, CausalError> {
        let mut ws = MechanismWorkspace::default();
        Ok(antecedent_model::rollout_temporal(model, plan, TemporalNoise::Draw(rng), &mut ws, ctx)?)
    }
}

pub(crate) fn sample_temporal_policy_with_history_availability(
    model: &CompiledCausalModel,
    plan: &TemporalPlan<'_>,
    history_available: &[bool],
    rng: &mut CausalRng,
    ctx: &ExecutionContext,
) -> Result<TemporalPath, CausalError> {
    let mut ws = MechanismWorkspace::default();
    Ok(antecedent_model::rollout_temporal_with_history_availability(
        model,
        plan,
        Some(history_available),
        TemporalNoise::Draw(rng),
        &mut ws,
        ctx,
    )?)
}

// --- Further items the language bindings use, re-exported from their defining crates ---

/// Attribution result types.
pub use antecedent_attribution::{CacheStats, ComponentContribution, ComputeBudget};
/// Compiled topological execution plans.
pub use antecedent_model::standard_normal_log_density;
/// Columnar value / noise batches and mechanism workspaces.
pub use antecedent_model::{ParentBatch, ValueBatch};
/// Composition of independently fitted causal mechanisms.
pub mod compose {
    pub use antecedent_model::compose::{
        ContextCoordinate, ContrastArm, CrossCovarianceBlock, Domain, InvarianceDeclaration,
        MeanContrast, MechanismContext, MechanismFamily, MechanismOrigin, MechanismProvenance,
        MechanismUncertainty, ParameterId, ParameterTerm, SelectionDiagramDeclaration,
        SelectionNodeDeclaration, TabularRow, UncertaintyMeaning,
    };
}
/// Items of `antecedent-model::continuous_cyclic`.
pub mod continuous_cyclic {
    pub use antecedent_model::continuous_cyclic::{OdePolicy, TimedClamp};
}
/// Items of `antecedent-model::cyclic_world`.
pub mod cyclic_world {
    pub use antecedent_model::cyclic_world::{CyclicInterventionRefusal, CyclicRefusalKind};
}
/// Portable form of a discretized composition and its fresh-consumer re-derivation.
pub mod discretized_scm_artifact {
    pub use antecedent_model::discretized_scm_artifact::export_discretized_scm;
}
/// Items of `antecedent-model::external_mechanism`.
pub mod external_mechanism {
    pub use antecedent_model::external_mechanism::{
        ExternalFeedbackSystem, ExternalFitParameter, ExternalMechanismImplementation,
        ExternalMechanismSpec, ExternalOdeSystem, TrustedMechanismResolver,
    };
}
/// Items of `antecedent-model::local_cyclic`.
pub mod local_cyclic {
    pub use antecedent_model::callback_cyclic::CallbackCyclicMechanism;
    pub use antecedent_model::local_cyclic::{
        LocalCyclicMechanism, LocalCyclicModel, LocalCyclicPolicy,
    };
}
/// Items of `antecedent-model::mechanism_expr`.
pub mod mechanism_expr {
    pub use antecedent_model::mechanism_expr::MechanismExpression;
}
/// Items of `antecedent-model::nonlinear_equilibrium`.
pub mod nonlinear_equilibrium {
    pub use antecedent_model::nonlinear_equilibrium::{
        Clamp, EquilibriumError, EquilibriumOutcome, EquilibriumResult, RegimeRow,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        AllocationMethod, CausalSchemaBuilder, InterventionSequence, MeasurementSpec, RoleHint,
        SequencedIntervention, ShapleyConfig, SmallRoleSet, TemporalPolicy, ValueType,
    };
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TableView};
    use antecedent_graph::DenseNodeId;

    #[test]
    fn checked_counterfactual_freezes_worlds_and_rejects_changed_target() {
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let query =
            CounterfactualQuery::new(outcome, vec![Intervention::set(treatment, Value::f64(1.0))]);
        let operation =
            CheckedCounterfactualOperation::compile(graph.clone(), query.clone()).unwrap();
        assert!(operation.matches(&graph, &query));
        assert!(!operation.matches(&graph, &query.clone().with_control_level(-1.0)));
        assert_eq!(operation.query(), &query);

        let invalid = CounterfactualQuery::new(
            outcome,
            vec![Intervention::set(treatment, Value::f64(f64::INFINITY))],
        );
        assert!(CheckedCounterfactualOperation::compile(graph, invalid).is_err());
    }

    #[test]
    fn checked_counterfactual_executes_from_frozen_worlds_against_linear_truth() {
        let (_, data) = chain_xy(50);
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let query = CounterfactualQuery::new(
            VariableId::from_raw(1),
            vec![Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
        );
        let operation = CheckedCounterfactualOperation::compile(graph, query).unwrap();
        let (_, ite) = operation.execute(&data, &ExecutionContext::for_tests(3)).unwrap();
        assert!((ite.mean_ite - 2.0).abs() < 0.1, "mean_ite={}", ite.mean_ite);
        assert_eq!(ite.unit_effects.len(), data.row_count());
    }

    #[test]
    fn a_contrast_set_reads_every_outcome_from_one_abduction() {
        let (_, data) = chain_xy(50);
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let (x, y) = (VariableId::from_raw(0), VariableId::from_raw(1));
        let query = CounterfactualQuery::new(y, vec![Intervention::set(x, Value::f64(1.0))]);
        let operation = CheckedCounterfactualOperation::compile(graph, query).unwrap();
        let ctx = ExecutionContext::for_tests(3);
        let (fitted, single) = operation.execute(&data, &ctx).unwrap();
        let active = [Intervention::set(x, Value::f64(1.0))];
        let control = [Intervention::set(x, Value::f64(0.0))];
        let set = counterfactual_contrasts(fitted.model, &data, &[y, x], (&active, &control), &ctx)
            .unwrap();
        // The outcome shared with the single-contrast route reads the same effects.
        let from_set = set.ite_result(y).unwrap();
        assert_eq!(from_set.unit_effects.as_ref(), single.unit_effects.as_ref());
        assert!((from_set.mean_ite - single.mean_ite).abs() < 1e-12);
        // The intervened variable moves by exactly the contrast between the worlds.
        let own = set.contrast(x).unwrap();
        assert!(own.unit_effects.iter().all(|effect| (effect - 1.0).abs() < 1e-12));
        assert_eq!(set.contrasts.iter().map(|c| c.outcome).collect::<Vec<_>>(), [y, x]);
        assert!(set.contrast(VariableId::from_raw(7)).is_none());
    }

    #[test]
    fn a_contrast_set_refuses_an_empty_request_and_an_unknown_outcome() {
        let (_, data) = chain_xy(20);
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let (x, y) = (VariableId::from_raw(0), VariableId::from_raw(1));
        let ctx = ExecutionContext::for_tests(3);
        let query = CounterfactualQuery::new(y, vec![Intervention::set(x, Value::f64(1.0))]);
        let operation = CheckedCounterfactualOperation::compile(graph, query).unwrap();
        let model = operation.execute(&data, &ctx).unwrap().0.model;
        let active = [Intervention::set(x, Value::f64(1.0))];
        let control = [Intervention::set(x, Value::f64(0.0))];
        let run = |outcomes: &[VariableId], active: &[Intervention]| {
            counterfactual_contrasts(model.clone(), &data, outcomes, (active, &control), &ctx)
        };
        assert!(run(&[], &active).is_err());
        assert!(run(&[y], &[]).is_err());
        assert!(run(&[VariableId::from_raw(9)], &active).is_err());
        assert!(run(&[y], &active).is_ok());
    }

    /// `y = 1 + 2·x1 + 3·x2` exactly, with `x1` and `x2` unrelated: variables `0 = x1`,
    /// `1 = x2`, `2 = y`.
    fn two_cause_data(n: usize) -> (TabularData, Dag) {
        let x1: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let x2: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64 * 0.3).collect();
        let y: Vec<f64> = x1.iter().zip(&x2).map(|(a, b)| 1.0 + 2.0 * a + 3.0 * b).collect();
        let data = TabularData::from_f64_columns([
            ("x1", x1.as_slice()),
            ("x2", x2.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(3);
        for source in [0, 1] {
            graph.insert_directed(DenseNodeId::from_raw(source), DenseNodeId::from_raw(2)).unwrap();
        }
        (data, graph)
    }

    fn joint_query(outcomes: &[VariableId]) -> CounterfactualQuery {
        let (x1, x2) = (VariableId::from_raw(0), VariableId::from_raw(1));
        let mut query = CounterfactualQuery::new(
            outcomes[0],
            vec![Intervention::set(x1, Value::f64(1.0)), Intervention::set(x2, Value::f64(1.0))],
        )
        .with_joint_control([
            Intervention::set(x1, Value::f64(0.0)),
            Intervention::set(x2, Value::f64(0.0)),
        ]);
        query.outcomes = Arc::from(outcomes);
        query
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact equality preserves the declared boundary and identity semantics"
    )]
    fn the_single_form_freezes_no_contrast_target_and_a_joint_query_freezes_one() {
        let (_, graph) = two_cause_data(10);
        let (x1, x2, y) =
            (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        let single = CounterfactualQuery::new(y, vec![Intervention::set(x1, Value::f64(1.0))]);
        let operation = CheckedCounterfactualOperation::compile(graph.clone(), single).unwrap();
        assert!(operation.contrast_target().is_none());

        let operation =
            CheckedCounterfactualOperation::compile(graph.clone(), joint_query(&[y, x2])).unwrap();
        let target = operation.contrast_target().expect("two outcomes and two interventions");
        assert_eq!(target.outcomes.as_ref(), [y, x2]);
        assert_eq!(target.treatments(), vec![x1, x2]);
        assert!(target.levels.iter().all(|l| l.active == 1.0 && l.control == 0.0));
        assert!(!target.allow_nested);
        // One outcome under one `Set` is the single form unless the query asks for nesting.
        let nested_flag = CounterfactualQuery::new(y, vec![Intervention::set(x1, Value::f64(1.0))])
            .with_nested(true);
        let operation = CheckedCounterfactualOperation::compile(graph, nested_flag).unwrap();
        assert!(operation.contrast_target().is_some_and(|t| t.allow_nested));
    }

    #[test]
    fn a_contrast_set_refuses_worlds_that_do_not_intervene_alike() {
        let (_, graph) = two_cause_data(10);
        let (x1, x2, y) =
            (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        // Two interventions against one baseline control: not a joint baseline.
        let unmatched = CounterfactualQuery::new(
            y,
            vec![Intervention::set(x1, Value::f64(1.0)), Intervention::set(x2, Value::f64(1.0))],
        );
        assert!(CheckedCounterfactualOperation::compile(graph.clone(), unmatched).is_err());
        // A one-element baseline is the single control, not a joint one.
        let single_control =
            CounterfactualQuery::new(y, vec![Intervention::set(x1, Value::f64(1.0))])
                .with_joint_control([Intervention::set(x1, Value::f64(0.0))]);
        let operation =
            CheckedCounterfactualOperation::compile(graph.clone(), single_control).unwrap();
        assert!(operation.contrast_target().is_none());
        // A shift is not a hard level.
        let mut soft = joint_query(&[y]);
        soft.interventions = Arc::from([
            Intervention::shift(x1, Value::f64(1.0)),
            Intervention::set(x2, Value::f64(1.0)),
        ]);
        assert!(CheckedCounterfactualOperation::compile(graph.clone(), soft).is_err());
        // A nested step needs the query's `allow_nested`.
        let sequence = |level: f64| {
            Intervention::sequence(InterventionSequence::new(vec![
                SequencedIntervention::new(
                    Intervention::set(x1, Value::f64(level)),
                    TemporalPolicy::pulse(0),
                ),
                SequencedIntervention::new(
                    Intervention::set(x2, Value::f64(level)),
                    TemporalPolicy::pulse(0),
                ),
            ]))
        };
        let nested = CounterfactualQuery::new(y, vec![sequence(1.0)]).with_control(sequence(0.0));
        assert!(CheckedCounterfactualOperation::compile(graph.clone(), nested.clone()).is_err());
        let operation =
            CheckedCounterfactualOperation::compile(graph, nested.with_nested(true)).unwrap();
        let target = operation.contrast_target().unwrap();
        assert_eq!(target.treatments(), vec![x1, x2]);
    }

    #[test]
    fn a_contrast_set_executes_both_outcomes_and_both_interventions_against_linear_truth() {
        let (data, graph) = two_cause_data(60);
        let (x2, y) = (VariableId::from_raw(1), VariableId::from_raw(2));
        let operation =
            CheckedCounterfactualOperation::compile(graph, joint_query(&[y, x2])).unwrap();
        let ctx = ExecutionContext::for_tests(3);
        let (_, set) = operation.execute_contrasts(&data, &ctx).unwrap();
        // Both interventions move `y` by 2 + 3; `x2` moves by exactly its own contrast.
        let effect = set.contrast(y).unwrap();
        assert!((effect.mean_ite - 5.0).abs() < 0.05, "mean_ite={}", effect.mean_ite);
        assert_eq!(effect.unit_effects.len(), data.row_count());
        let own = set.contrast(x2).unwrap();
        assert!(own.unit_effects.iter().all(|e| (e - 1.0).abs() < 1e-12));
        // The single-outcome executor refuses a contrast set rather than answer part of it.
        assert!(operation.execute(&data, &ctx).is_err());
    }

    #[test]
    fn nested_operation_refuses_world_level_mismatch() {
        let mut graph = Dag::with_variables(3);
        for (source, target) in [(0, 1), (0, 2), (1, 2)] {
            graph
                .insert_directed(DenseNodeId::from_raw(source), DenseNodeId::from_raw(target))
                .unwrap();
        }
        let query = NestedCounterfactualQuery::new(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        )
        .unwrap();
        let changed_worlds = NestedCounterfactualQuery::with_levels(
            query.treatment,
            query.mediator,
            query.outcome,
            -1.0,
            1.0,
        )
        .unwrap();
        let plan = NestedCounterfactualOperation::compile(graph.clone(), query).unwrap();
        assert!(plan.matches(&graph, &query));
        assert!(!plan.matches(&graph, &changed_worlds));
    }

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

    /// Deterministic conditional mean of a fitted slot at explicit parent
    /// columns (noise held at zero). This forward-evaluates the fitted SCM
    /// independently of the counterfactual engine.
    fn conditional_mean(slot: &MechanismSlot, columns: &[&[f64]]) -> Vec<f64> {
        use antecedent_model::{ParentBatch, evaluate_column};
        let rows = columns[0].len();
        let mut flat = Vec::with_capacity(columns.len() * rows);
        for column in columns {
            assert_eq!(column.len(), rows);
            flat.extend_from_slice(column);
        }
        let batch = ParentBatch { n_rows: rows, n_parents: columns.len(), values: &flat };
        let zeros = vec![0.0_f64; rows];
        let mut out = vec![0.0_f64; rows];
        let mut ws = MechanismWorkspace::default();
        evaluate_column(slot, batch, &zeros, &mut out, &mut ws).unwrap();
        out
    }

    /// The natural-direct-effect route shares one abduced exogenous draw across
    /// both worlds, and under a non-separable outcome that sharing is
    /// observable: the direct effect reads the frozen mediator's disturbance, so
    /// the route's value matches the shared-abduction truth and is decisively
    /// separated from the disturbance-free (independent) value.
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one test walks the whole shared-vs-independent proof"
    )]
    fn nested_route_shares_abduced_exogenous_draw_under_nonseparable_outcome() {
        // Non-separable structural model with an analytic natural direct effect:
        //   M = 0.8*X + U_M
        //   Y = 1.7*X + 0.5*M + 0.9*X*M^2      (+ an additive U_Y that cancels
        //                                        from the direct effect)
        // The X*M^2 term makes Y convex in the mediator, so the direct effect
        //   NDE_u = (active-control)*(1.7 + 0.9*M0_u^2)
        // depends on the frozen mediator M0_u = 0.8*control + U_M,u through its
        // square. Averaging keeps E[M0^2] = mean(M0)^2 + Var(U_M): the abduced
        // disturbance's variance survives. Both levels sit inside the observed X
        // range so the fitted spline interpolates.
        let n = 600usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
        let m: Vec<f64> =
            x.iter().enumerate().map(|(i, x)| 0.8 * x + (i as f64 * 0.91).cos()).collect();
        let y: Vec<f64> =
            x.iter().zip(&m).map(|(x, m)| 1.7 * x + 0.5 * m + 0.9 * x * m * m).collect();
        let data = TabularData::from_f64_columns([
            ("x", x.as_slice()),
            ("m", m.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(3);
        for (s, t) in [(0, 1), (0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(t)).unwrap();
        }
        let treatment = VariableId::from_raw(0);
        let mediator = VariableId::from_raw(1);
        let outcome = VariableId::from_raw(2);
        let (control, active) = (-1.0, 2.0);
        let query =
            NestedCounterfactualQuery::with_levels(treatment, mediator, outcome, control, active)
                .unwrap();
        let ctx = ExecutionContext::for_tests(41);

        // The route, retargeted to fit the non-separable outcome mechanism.
        let operation = NestedCounterfactualOperation::compile(graph.clone(), query)
            .unwrap()
            .with_non_separable_outcome();
        assert_eq!(operation.outcome_mechanism(), NestedOutcomeMechanism::NonSeparableBasis);
        let route = operation.execute(&data, &ctx).unwrap();

        // Reconstruct the identical fitted mechanism to (a) confirm the outcome
        // is a genuine non-additive basis and (b) simulate the SCM forward,
        // independently of the counterfactual engine the route drives.
        let compiled = CompiledCausalModel::compile(graph.clone()).unwrap();
        let store = fit_non_separable_nested_outcome(&compiled, &data, outcome).unwrap();
        let outcome_slot = store.get(DenseNodeId::from_raw(2));
        let MechanismSlot::LinearBasis { basis, .. } = outcome_slot else {
            panic!("expected a non-separable LinearBasis outcome, got {outcome_slot:?}");
        };
        assert!(basis.has_cross_parent_product(), "outcome basis is separable: {basis:?}");

        // Parent gather order the fitted basis expects for its two columns.
        let outcome_parents: Vec<usize> = compiled
            .parent_gathers
            .iter()
            .find(|gather| gather.child == DenseNodeId::from_raw(2))
            .unwrap()
            .parents
            .iter()
            .map(|p| p.as_usize())
            .collect();
        assert_eq!(outcome_parents, vec![0, 1], "expected parent order [X, M]");

        // Abduct the mediator disturbance by hand: U_M,u = m_u - ĝ_M(x_u); the
        // control-world mediator realisation is M0_u = ĝ_M(control) + U_M,u.
        let mediator_slot = store.get(DenseNodeId::from_raw(1));
        let mhat_obs = conditional_mean(mediator_slot, &[&x]);
        let mhat_control = conditional_mean(mediator_slot, &[&vec![control; n]]);
        let m0: Vec<f64> = (0..n).map(|u| mhat_control[u] + (m[u] - mhat_obs[u])).collect();
        let mbar0 = m0.iter().sum::<f64>() / n as f64;

        // SHARED truth: each world evaluates the outcome at the unit's own
        // abduced mediator realisation M0_u.
        let y_active_shared = conditional_mean(outcome_slot, &[&vec![active; n], &m0]);
        let y_control_shared = conditional_mean(outcome_slot, &[&vec![control; n], &m0]);
        let shared_truth =
            (0..n).map(|u| y_active_shared[u] - y_control_shared[u]).sum::<f64>() / n as f64;

        // INDEPENDENT value: the mediator is frozen at its disturbance-free
        // counterfactual mean m̄0 instead of the unit's abduced draw, so the two
        // worlds no longer share the abduced exogenous term.
        let y_active_free = conditional_mean(outcome_slot, &[&vec![active; n], &vec![mbar0; n]]);
        let y_control_free = conditional_mean(outcome_slot, &[&vec![control; n], &vec![mbar0; n]]);
        let independent_value =
            (0..n).map(|u| y_active_free[u] - y_control_free[u]).sum::<f64>() / n as f64;

        // The route reproduces the shared-abduction forward simulation exactly.
        assert!(
            (route - shared_truth).abs() < 1e-9,
            "route {route} != shared truth {shared_truth}"
        );
        // Shared and independent draws give genuinely different answers, and the
        // route matches the shared one, not the independent one.
        assert!(
            (shared_truth - independent_value).abs() > 1.0,
            "shared {shared_truth} and independent {independent_value} do not differ"
        );
        assert!(
            (route - independent_value).abs() > 1.0,
            "route {route} coincides with the independent value {independent_value}"
        );
        assert!(
            (route - shared_truth).abs() < (route - independent_value).abs(),
            "route is closer to the independent value than to the shared truth"
        );

        // Fidelity to the known analytic SCM: shared ≈ 3*(1.7 + 0.9*E[M0^2]),
        // independent ≈ 3*(1.7 + 0.9*m̄0^2); their gap is 3*0.9*Var(U_M) > 0.
        let em2 = m0.iter().map(|v| v * v).sum::<f64>() / n as f64;
        let analytic_shared = (active - control) * (1.7 + 0.9 * em2);
        let analytic_independent = (active - control) * (1.7 + 0.9 * mbar0 * mbar0);
        assert!(
            (route - analytic_shared).abs() < 0.25,
            "route {route} vs analytic {analytic_shared}"
        );
        assert!(
            (analytic_shared - analytic_independent).abs() > 1.0,
            "analytic shared and independent must differ"
        );
    }

    /// Negative control: under the supported *separable* linear-Gaussian outcome
    /// the direct effect is `a·(active − control)` and never reads the frozen
    /// mediator, so the shared and disturbance-free values coincide even though
    /// the mediator carries a real abduced disturbance. This is why the linear
    /// cell alone cannot exhibit the shared-exogenous property, and why the
    /// non-separable outcome above is needed to make it observable.
    #[test]
    fn nested_shared_equals_independent_under_separable_outcome() {
        let n = 600usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
        let m: Vec<f64> =
            x.iter().enumerate().map(|(i, x)| 0.8 * x + (i as f64 * 0.91).cos()).collect();
        // Separable additive outcome: no treatment×mediator interaction.
        let y: Vec<f64> = x.iter().zip(&m).map(|(x, m)| 1.7 * x + 4.0 * m).collect();
        let data = TabularData::from_f64_columns([
            ("x", x.as_slice()),
            ("m", m.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(3);
        for (s, t) in [(0, 1), (0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(t)).unwrap();
        }
        let compiled = CompiledCausalModel::compile(graph.clone()).unwrap();
        let (store, _) = MechanismRegistry::standard()
            .assign_and_fit(
                &compiled,
                &data,
                SelectionPolicy::RequireFamily(MechanismFamily::LinearGaussian),
            )
            .unwrap();
        let outcome_slot = store.get(DenseNodeId::from_raw(2));
        let mediator_slot = store.get(DenseNodeId::from_raw(1));
        let (control, active) = (-1.0, 2.0);
        let mhat_obs = conditional_mean(mediator_slot, &[&x]);
        let mhat_control = conditional_mean(mediator_slot, &[&vec![control; n]]);
        let m0: Vec<f64> = (0..n).map(|u| mhat_control[u] + (m[u] - mhat_obs[u])).collect();
        let mbar0 = m0.iter().sum::<f64>() / n as f64;
        // Variance is genuinely present in the abduced mediator draw.
        let var_m0 = m0.iter().map(|v| (v - mbar0).powi(2)).sum::<f64>() / n as f64;
        assert!(var_m0 > 0.1, "mediator draw should carry real variance: {var_m0}");
        let y_active_shared = conditional_mean(outcome_slot, &[&vec![active; n], &m0]);
        let y_control_shared = conditional_mean(outcome_slot, &[&vec![control; n], &m0]);
        let shared =
            (0..n).map(|u| y_active_shared[u] - y_control_shared[u]).sum::<f64>() / n as f64;
        let independent = conditional_mean(outcome_slot, &[&vec![active; n], &vec![mbar0; n]])[0]
            - conditional_mean(outcome_slot, &[&vec![control; n], &vec![mbar0; n]])[0];
        assert!(
            (shared - independent).abs() < 1e-9,
            "separable outcome, yet shared {shared} != independent {independent}"
        );
    }

    /// Adversarial mechanism-underdetermination: the non-separable outcome basis
    /// cannot be fit from too few rows, and the route refuses with a typed error
    /// rather than fabricating a direct effect. This is distinct from the
    /// graph-shape scope refusal (the graph here is the supported nested DAG).
    #[test]
    fn nested_non_separable_outcome_refuses_when_basis_underdetermined() {
        let n = 8usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 - 2.0).collect();
        let m: Vec<f64> = x.iter().map(|x| 0.8 * x).collect();
        let y: Vec<f64> =
            x.iter().zip(&m).map(|(x, m)| 1.7 * x + 0.5 * m + 0.9 * x * m * m).collect();
        let data = TabularData::from_f64_columns([
            ("x", x.as_slice()),
            ("m", m.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(3);
        for (s, t) in [(0, 1), (0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(t)).unwrap();
        }
        let query = NestedCounterfactualQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            VariableId::from_raw(2),
            -1.0,
            2.0,
        )
        .unwrap();
        let operation = NestedCounterfactualOperation::compile(graph, query)
            .unwrap()
            .with_non_separable_outcome();
        let err = operation.execute(&data, &ExecutionContext::for_tests(7)).unwrap_err();
        // The linear-Gaussian supported path fits the same tiny sample fine, so
        // the refusal is specific to the non-separable outcome basis.
        assert!(
            matches!(err, CausalError::Unsupported { .. } | CausalError::Model(_)),
            "expected a typed mechanism refusal, got {err:?}"
        );
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
        let err = antecedent_attribution::anomaly::score_anomalies(&model, &data, &q).unwrap_err();
        assert_eq!(err, AttributionError::PopulationOutOfRange { kind: "row", index: n, limit: n });
    }
}
