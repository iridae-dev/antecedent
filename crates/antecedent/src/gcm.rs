//! GCM workflow helpers (fit → sample → CF → anomaly).
//!
//! Thin facade over `antecedent-model` / `antecedent-counterfactual` / `antecedent-attribution`
//! so planners and Python bind once at the library boundary.
//!
//! Attribution and simulation on a fitted model are compiled routes: a study with a
//! [`antecedent_core::CausalQuery::Attribute`] or
//! [`antecedent_core::CausalQuery::Simulate`] query on a supplied or accepted DAG
//! fits its mechanisms through a [`crate::MechanismProgram`] (held-out
//! family selection, with the selection record on the result) and returns the
//! engine's answer with a [`crate::CausalContract`] and a claim. The `sample_do`,
//! `policy_value`, `counterfactual_ite`, `anomaly_attribution*`, `attribute_*` and
//! `rank_root_causes` functions are the engines those routes run; they are hidden
//! from the public surface.
//!
//! # Example
//!
//! ```rust,ignore
//! use antecedent::gcm::fit_gcm;
//!
//! let fitted = fit_gcm(dag, &data)?;
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AnomalyAttributionQuery, CausalRng, ChangeAttributionQuery, CounterfactualQuery,
    ExecutionContext, Intervention, InterventionalDistributionQuery, MechanismChangeQuery,
    NestedCounterfactualQuery, PathSpecificEffectQuery, TargetPopulation, UnitChangeQuery, Value,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::Dag;
use antecedent_model::{
    InterventionOverlay, ModelView, NoiseFix, NoiseReplayer, NoiseSource, ValueBatch,
};

use crate::error::CausalError;

pub use antecedent_attribution::{
    AnomalyScores, ArrowStrength, AttributionError, ChangeAttribution, ChangeAttributionResult,
    ComponentStatus, ComponentUncertainty, DifferenceMeasure, DistributionChangeOptions,
    FeatureRelevance, FitResampling, INTRINSIC_MAX_EXACT_PLAYERS, IntrinsicAllocation,
    IntrinsicBudget, IntrinsicInfluenceQuery, IntrinsicInfluenceResult, IntrinsicMethod,
    MechanismChangeDetection, MechanismChangeMethod, NoiseIndependence, PartialStderr,
    PopulationDoContrast, RobustChangeOptions, RootCauseRank, ShapleyMethod, SourceInfluence,
    StructureChangeOptions, UncertaintyComponent, UnitChangeResult, arrow_strengths,
    detect_mechanism_changes, distribution_change, distribution_change_robust,
    distribution_change_with_row_weights, feature_relevance, intrinsic_influence, path_decompose,
    population_do_contrast, root_cause_rank, score_anomalies, score_anomalies_with,
    structure_change, unit_change,
};
pub use antecedent_counterfactual::{
    AbductionMissingPolicy, CompiledCounterfactualPlan, CounterfactualEngine, CounterfactualError,
    CounterfactualResult, CounterfactualWorld, ExogenousPosterior, NoiseInferenceKind,
    RANK_PRESERVING_ASSUMPTION, nested_counterfactual, nested_counterfactual_with_exo,
    nested_hard_counterfactual, simultaneous_hard_counterfactual,
};
pub use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, DoSampleResult, DynamicMechanism,
    GraphFalsificationConfig, GraphFalsificationOutcome, GraphFalsificationReport,
    GraphUninformativeReason, InvertibleStructuralCausalModel, KdeDoSampler, MarkovImplicationTest,
    McmcDoSampler, MechanismAssignment, MechanismFamily, MechanismRegistry, MechanismSlot,
    MechanismTyping, MechanismWorkspace, ModelCollection, ModelError, ModelEvaluator,
    NodeFamilyConstraint, NoiseTable, PolicyRoute, PolicySupport, ProbabilisticCausalModel,
    RouteDecision, SelectionOptions, SelectionPolicy, SelectionProvenance, SelectionRule,
    StructuralCausalModel, TemporalNoise, TemporalPath, TemporalPlan, ValidationSplits,
    WeightingDoSampler, interventional_mean, policy_support, sample_interventional,
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
    fit_with_registry(&MechanismRegistry::standard(), graph, data, CausalError::from)
}

/// Compile `graph` and fit `registry` to `data`; `fit_error` maps a failure of the
/// fit itself (compile failures always convert plainly).
fn fit_with_registry(
    registry: &MechanismRegistry,
    graph: Dag,
    data: &TabularData,
    fit_error: fn(ModelError) -> CausalError,
) -> Result<FittedGcm, CausalError> {
    let compiled = CompiledCausalModel::compile(graph)?;
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
pub fn fit_gcm_counterfactual(graph: Dag, data: &TabularData) -> Result<FittedGcm, CausalError> {
    fit_with_registry(
        &MechanismRegistry::with_heterogeneity_families(),
        graph,
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

/// Frozen single-intervention ITE target and mechanism-fitting procedure.
/// The licensed high-level counterfactual route uses heterogeneity-capable
/// mechanism selection and shared abducted disturbances for both worlds.
#[derive(Clone, Debug)]
pub struct CheckedCounterfactualOperation {
    graph: Dag,
    query: CounterfactualQuery,
    treatment: VariableId,
    outcome: VariableId,
    active: f64,
    control: f64,
}

impl CheckedCounterfactualOperation {
    /// Check and freeze the currently licensed one-outcome, one-hard-set ITE.
    pub(crate) fn compile(graph: Dag, query: CounterfactualQuery) -> Result<Self, CausalError> {
        query.validate()?;
        if query.allow_nested || query.outcomes.len() != 1 || query.interventions.len() != 1 {
            return Err(CausalError::Unsupported {
                message: "Study counterfactual path currently supports a single hard intervention for ITE (use gcm helpers for multi-world predict)",
            });
        }
        let Intervention::Set { variable: treatment, value } = &query.interventions[0] else {
            return Err(CausalError::Unsupported {
                message: "Study counterfactual path requires a hard Set intervention",
            });
        };
        let Intervention::Set { variable: control_variable, value: control_value } = &query.control
        else {
            return Err(CausalError::Unsupported {
                message: "Study counterfactual path requires a hard Set control intervention",
            });
        };
        if treatment != control_variable {
            return Err(CausalError::Compile {
                message: "counterfactual control targets a different treatment".into(),
            });
        }
        let active = value.as_f64().ok_or_else(|| CausalError::Compile {
            message: "counterfactual intervention value must be f64".into(),
        })?;
        let control = control_value.as_f64().ok_or_else(|| CausalError::Compile {
            message: "counterfactual control value must be f64".into(),
        })?;
        if !active.is_finite() || !control.is_finite() {
            return Err(CausalError::Unsupported {
                message: "counterfactual treatment levels must be finite",
            });
        }
        let treatment = *treatment;
        let outcome = query.outcomes[0];
        if usize::try_from(treatment.raw()).map_or(true, |id| id >= graph.node_count())
            || usize::try_from(outcome.raw()).map_or(true, |id| id >= graph.node_count())
        {
            return Err(CausalError::Compile {
                message: "counterfactual target variable is absent from the graph".into(),
            });
        }
        Ok(Self { graph, query, treatment, outcome, active, control })
    }

    /// The frozen semantic target.
    #[must_use]
    pub const fn query(&self) -> &CounterfactualQuery {
        &self.query
    }

    /// The frozen graph and mechanism assignment scope.
    #[must_use]
    pub const fn graph(&self) -> &Dag {
        &self.graph
    }

    pub(crate) fn matches(&self, graph: &Dag, query: &CounterfactualQuery) -> bool {
        self.query == *query && crate::graph_identity::same_dag_structure(graph, &self.graph)
    }

    /// Fit mechanisms under the frozen graph, then abduct once for both worlds.
    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<(FittedGcm, IteResult), CausalError> {
        let fitted = fit_gcm_counterfactual(self.graph.clone(), data)?;
        let ite = counterfactual_ite(
            fitted.model.clone(),
            data,
            self.treatment,
            self.outcome,
            self.active,
            self.control,
            ctx,
        )?;
        Ok((fitted, ite))
    }
}

/// Outcome mechanism the nested natural-direct-effect route fits before
/// abducting and replaying both worlds.
///
/// The choice is what makes the shared abduced exogenous draw either invisible
/// or observable. Under a separable outcome the frozen mediator's disturbance
/// cancels out of the direct effect; under a non-separable outcome it does not.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NestedOutcomeMechanism {
    /// Separable linear-Gaussian additive-noise outcome (the licensed
    /// point-only cell). The natural direct effect is `a·(active − control)`,
    /// which does not read the abducted disturbance, so a shared and an
    /// independent exogenous draw yield the identical value.
    LinearGaussian,
    /// Non-separable outcome basis (cross-parent products and per-parent smooth
    /// terms, [`MechanismFamily::LinearSpline`]). The outcome is a non-additive
    /// function of its parents, so a treatment-interacted mediator term carries
    /// the frozen mediator's abducted disturbance into the direct effect. The
    /// disturbance is additive, so abduction still inverts it exactly and the
    /// counterfactual replay round-trips.
    NonSeparableBasis,
}

impl NestedOutcomeMechanism {
    /// Tag a persisted artifact stores for this mechanism family.
    pub(crate) const fn wire_tag(self) -> &'static str {
        match self {
            Self::LinearGaussian => "linear_gaussian",
            Self::NonSeparableBasis => "non_separable_basis",
        }
    }

    /// Inverse of [`Self::wire_tag`]; `None` for a tag no family carries.
    pub(crate) fn from_wire_tag(tag: &str) -> Option<Self> {
        match tag {
            "linear_gaussian" => Some(Self::LinearGaussian),
            "non_separable_basis" => Some(Self::NonSeparableBasis),
            _ => None,
        }
    }
}

/// Complete checked operation for the licensed natural direct effect route.
///
/// The operation fixes the graph, cross-world target roles, treatment levels,
/// and shared-exogenous execution procedure at preparation time.
#[derive(Clone, Debug)]
pub struct NestedCounterfactualOperation {
    graph: Dag,
    query: NestedCounterfactualQuery,
    mediation_query: antecedent_core::MediationQuery,
    outer_world: Intervention,
    active_inner_world: Intervention,
    control_inner_world: Intervention,
    frozen: Arc<[VariableId]>,
    outcome_mechanism: NestedOutcomeMechanism,
}

impl NestedCounterfactualOperation {
    /// The frozen cross-world query and treatment levels.
    #[must_use]
    pub const fn query(&self) -> &NestedCounterfactualQuery {
        &self.query
    }

    /// The three-node Markovian mediation graph required by this operation.
    #[must_use]
    pub const fn graph(&self) -> &Dag {
        &self.graph
    }

    pub(crate) fn compile(
        graph: Dag,
        query: NestedCounterfactualQuery,
    ) -> Result<Self, CausalError> {
        query.validate()?;
        let observed = graph.directed_edge_pairs_sorted();
        let mut expected = vec![
            (query.treatment.raw(), query.mediator.raw()),
            (query.treatment.raw(), query.outcome.raw()),
            (query.mediator.raw(), query.outcome.raw()),
        ];
        expected.sort_unstable();
        if graph.node_count() != 3 || observed != expected {
            return Err(crate::unsupported_reason!(
                "cross_world_not_identified",
                "query requires exactly X -> M, X -> Y, M -> Y with no other nodes or edges"
            ));
        }
        Ok(Self {
            graph,
            query,
            mediation_query: query.as_mediation_query(),
            outer_world: Intervention::set(query.treatment, Value::f64(query.control_value())),
            active_inner_world: Intervention::set(
                query.treatment,
                Value::f64(query.active_value()),
            ),
            control_inner_world: Intervention::set(
                query.treatment,
                Value::f64(query.control_value()),
            ),
            frozen: Arc::from([query.mediator]),
            outcome_mechanism: NestedOutcomeMechanism::LinearGaussian,
        })
    }

    /// Re-target this operation to fit a non-separable outcome basis instead of
    /// the separable linear-Gaussian outcome.
    ///
    /// The graph, cross-world roles, and treatment levels are unchanged; only
    /// the fitted outcome mechanism differs. This is the variant under which the
    /// shared abduced exogenous draw is observable: the direct effect reads the
    /// frozen mediator's disturbance because the outcome is non-additive in its
    /// parents. The licensed linear-Gaussian path (used by the staged Study
    /// route) is untouched.
    #[must_use]
    pub fn with_non_separable_outcome(mut self) -> Self {
        self.outcome_mechanism = NestedOutcomeMechanism::NonSeparableBasis;
        self
    }

    /// The outcome mechanism this operation fits.
    #[must_use]
    pub const fn outcome_mechanism(&self) -> NestedOutcomeMechanism {
        self.outcome_mechanism
    }

    pub(crate) fn matches(&self, graph: &Dag, query: &NestedCounterfactualQuery) -> bool {
        self.query == *query && crate::graph_identity::same_dag_structure(graph, &self.graph)
    }

    pub(crate) fn mediation_query(&self) -> &antecedent_core::MediationQuery {
        &self.mediation_query
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<f64, CausalError> {
        let compiled = CompiledCausalModel::compile(self.graph.clone())?;
        let store =
            fit_nested_mechanisms(&compiled, data, self.outcome_mechanism, self.query.outcome)?;
        let engine = CounterfactualEngine::new(compiled.with_mechanisms(store));
        let exo = engine
            .abduct(data, AbductionMissingPolicy::Error, ctx)
            .map_err(CausalError::rejected)?;
        let control_outer = [self.outer_world.clone()];
        let active_inner = [self.active_inner_world.clone()];
        let control_inner = [self.control_inner_world.clone()];
        let mut workspace = MechanismWorkspace::default();
        let active = nested_counterfactual_with_exo(
            &engine,
            &exo,
            &antecedent_counterfactual::NestedSpec {
                outer: &control_outer,
                inner: &active_inner,
                frozen: &self.frozen,
                outcome: self.query.outcome,
            },
            &mut workspace,
            ctx,
        )
        .map_err(CausalError::rejected)?;
        let control = nested_counterfactual_with_exo(
            &engine,
            &exo,
            &antecedent_counterfactual::NestedSpec {
                outer: &control_outer,
                inner: &control_inner,
                frozen: &self.frozen,
                outcome: self.query.outcome,
            },
            &mut workspace,
            ctx,
        )
        .map_err(CausalError::rejected)?;
        Ok(active - control)
    }
}

/// Fit every mechanism of the nested route's model under `mechanism`.
///
/// Shared by the natural-direct-effect operation and the cross-world edge
/// contrast, so both read the same fitted structural model. The natural direct
/// effect route calls this unpolled form, so its behavior is unchanged.
pub(crate) fn fit_nested_mechanisms(
    compiled: &CompiledCausalModel,
    data: &TabularData,
    mechanism: NestedOutcomeMechanism,
    outcome: VariableId,
) -> Result<CompiledMechanismStore, CausalError> {
    fit_nested_mechanisms_polled(compiled, data, mechanism, outcome, "", &|| false)
}

/// [`fit_nested_mechanisms`] that asks `cancelled` before every node, every
/// candidate family and every cross-validation fold fit of the model crate's
/// fitting loops, and stops with [`CausalError::Cancelled`] (`stage`, no partial
/// store) when it answers `true`. The fitted mechanisms are identical to the
/// unpolled form's: polling only reads the token.
pub(crate) fn fit_nested_mechanisms_polled(
    compiled: &CompiledCausalModel,
    data: &TabularData,
    mechanism: NestedOutcomeMechanism,
    outcome: VariableId,
    stage: &'static str,
    cancelled: &dyn Fn() -> bool,
) -> Result<CompiledMechanismStore, CausalError> {
    let map = |error: ModelError| match error {
        ModelError::Cancelled => CausalError::Cancelled { stage },
        other => CausalError::from(other),
    };
    match mechanism {
        NestedOutcomeMechanism::LinearGaussian => Ok(MechanismRegistry::standard()
            .assign_and_fit_polled(
                compiled,
                data,
                SelectionPolicy::RequireFamily(MechanismFamily::LinearGaussian),
                cancelled,
            )
            .map_err(map)?
            .0),
        NestedOutcomeMechanism::NonSeparableBasis => {
            fit_non_separable_nested_outcome_polled(compiled, data, outcome, stage, cancelled)
        }
    }
}

/// Fit the nested route's mechanisms with a non-separable outcome basis.
///
/// Every parent-free and single-parent node keeps the linear-Gaussian family
/// (a cross-parent product needs two parents), so the mediator equation stays
/// separable and its abducted disturbance is exact. The two-parent outcome is
/// refit as a [`MechanismFamily::LinearSpline`] basis, whose smooth per-parent
/// expansion and cross-parent products make the outcome a non-additive function
/// of treatment and mediator. The disturbance stays additive, so abduction
/// inverts it and the counterfactual replay round-trips.
#[cfg(test)]
fn fit_non_separable_nested_outcome(
    compiled: &CompiledCausalModel,
    data: &TabularData,
    outcome: VariableId,
) -> Result<CompiledMechanismStore, CausalError> {
    fit_non_separable_nested_outcome_polled(compiled, data, outcome, "", &|| false)
}

/// [`fit_non_separable_nested_outcome`] that polls `cancelled` (see
/// [`fit_nested_mechanisms_polled`]).
fn fit_non_separable_nested_outcome_polled(
    compiled: &CompiledCausalModel,
    data: &TabularData,
    outcome: VariableId,
    stage: &'static str,
    cancelled: &dyn Fn() -> bool,
) -> Result<CompiledMechanismStore, CausalError> {
    let registry = MechanismRegistry::standard();
    let (_, mut assignments) = registry
        .assign_and_fit_polled(
            compiled,
            data,
            SelectionPolicy::RequireFamily(MechanismFamily::LinearGaussian),
            cancelled,
        )
        .map_err(|e| match e {
            ModelError::Cancelled => CausalError::Cancelled { stage },
            other => CausalError::from(other),
        })?;
    let mut retargeted = false;
    for assignment in &mut assignments {
        if assignment.variable == outcome {
            assignment.selected = MechanismFamily::LinearSpline;
            retargeted = true;
        }
    }
    if !retargeted {
        return Err(CausalError::Compile {
            message: "nested outcome variable is absent from the fitted mechanism assignments"
                .into(),
        });
    }
    let weights = vec![1.0_f64; data.row_count()];
    registry.refit_weighted_polled(compiled, data, &assignments, &weights, cancelled).map_err(|e| {
        match e {
            ModelError::Cancelled => CausalError::Cancelled { stage },
            other => map_mechanism_fit(other),
        }
    })
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
    sample_interventional(model, interventions, n, rng, &mut ws, ctx).map_err(CausalError::from)
}

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

/// Model-based value of a policy on one outcome, with the unintervened outcome on the same
/// noise, and the per-route licence the policy was run under.
///
/// The means are properties of the fitted model, not of the population: they are as good as the
/// structure and the mechanisms are. Only the simulation error is quantified here
/// ([`Self::policy_simulation_stderr`], [`Self::difference_simulation_stderr`]); the model-fit
/// uncertainty is not, and no interval is claimed.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyValue {
    /// The outcome variable.
    pub outcome: VariableId,
    /// Draws of the policy run and of the baseline replayed on its noise.
    pub n_draws: usize,
    /// Mean outcome under the policy.
    pub policy_mean: f64,
    /// Mean outcome with no intervention, on the same noise.
    pub baseline_mean: f64,
    /// `policy_mean - baseline_mean`, from the paired draws.
    pub mean_difference: f64,
    /// Monte Carlo standard error of `policy_mean`.
    pub policy_simulation_stderr: f64,
    /// Monte Carlo standard error of `mean_difference` (paired, so common noise cancels).
    pub difference_simulation_stderr: f64,
    /// The licence of each route for this policy.
    pub support: PolicySupport,
}

/// Value of `interventions` on `outcome` under the fitted model: draw `n` rows with their noise
/// under the policy, replay the same noise with no intervention, and report both means and their
/// paired difference with the simulation error.
///
/// This is [`sample_with_noise`] and [`replay_noise`], the executor every policy route shares;
/// a policy the simulation route refuses is refused here with the same registered reason code.
///
/// # Errors
///
/// A refused policy, an outcome absent from the model, fewer than two draws, a table over the
/// context's memory limit, or cancellation.
#[doc(hidden)]
pub fn policy_value(
    model: &CompiledCausalModel,
    interventions: &[Intervention],
    outcome: VariableId,
    n: usize,
    ctx: &ExecutionContext,
) -> Result<PolicyValue, CausalError> {
    let node = policy_value_request(model, outcome, n)?;
    let support = policy_support(model, interventions);
    let overlay = InterventionOverlay::from_interventions(model, interventions)?;
    policy_value_on(model, &overlay, support, (outcome, node), n, ctx)
}

/// The outcome's dense node, after checking the request can report a simulation error.
fn policy_value_request(
    model: &CompiledCausalModel,
    outcome: VariableId,
    n: usize,
) -> Result<antecedent_graph::DenseNodeId, CausalError> {
    if n < 2 {
        return Err(CausalError::Compile {
            message: "a policy value needs at least two draws to report its simulation error"
                .into(),
        });
    }
    model.dense_of(outcome).ok_or_else(|| CausalError::Compile {
        message: format!("outcome {outcome} is absent from the fitted model"),
    })
}

/// The executor of [`policy_value`] on an already compiled overlay and resolved licence, shared
/// by the one-shot call and [`PreparedPolicy`].
fn policy_value_on(
    model: &CompiledCausalModel,
    overlay: &InterventionOverlay,
    support: PolicySupport,
    (outcome, node): (VariableId, antecedent_graph::DenseNodeId),
    n: usize,
    ctx: &ExecutionContext,
) -> Result<PolicyValue, CausalError> {
    let (policy, noise) =
        NoiseReplayer::new().sample(&ModelView::with_overlay(model, overlay.clone()), n, ctx)?;
    let baseline = replay_noise(model, &[], &noise, 0, ctx)?;
    let treated = policy.column(node.as_usize()).map_err(CausalError::from)?;
    let untreated = baseline.column(node.as_usize()).map_err(CausalError::from)?;
    let count = n as f64;
    let mean = |column: &[f64]| column.iter().sum::<f64>() / count;
    let stderr = |values: &mut dyn Iterator<Item = f64>, centre: f64| {
        let ss: f64 = values.map(|v| (v - centre) * (v - centre)).sum();
        (ss / (count - 1.0) / count).sqrt()
    };
    let (policy_mean, baseline_mean) = (mean(treated), mean(untreated));
    let mean_difference = policy_mean - baseline_mean;
    Ok(PolicyValue {
        outcome,
        n_draws: n,
        policy_mean,
        baseline_mean,
        mean_difference,
        policy_simulation_stderr: stderr(&mut treated.iter().copied(), policy_mean),
        difference_simulation_stderr: stderr(
            &mut treated.iter().zip(untreated).map(|(t, u)| t - u),
            mean_difference,
        ),
        support,
    })
}

/// A policy program compiled and licensed once against one fitted model, rerun without being
/// compiled again: the retained form of [`policy_value`].
///
/// [`Self::prepare`] validates the program against the model (a refused program is refused
/// there, with the registered reason code), compiles its overlay and resolves the licence of
/// every route. [`Self::value`] then runs the executor [`policy_value`] runs, on the retained
/// overlay, so it equals the one-shot value bit for bit at the same context.
/// [`Self::rebind`] carries the program to a refitted model: the overlay and licence are
/// resolved against that model, and the program itself is not compiled again.
#[derive(Clone, Debug)]
pub struct PreparedPolicy {
    interventions: Arc<[Intervention]>,
    overlay: InterventionOverlay,
    support: PolicySupport,
    program_compilations: u64,
    model_bindings: u64,
}

impl PreparedPolicy {
    /// Compile `interventions` against `model` and resolve the licence of each route.
    ///
    /// # Errors
    ///
    /// A program the model refuses (an unknown variable, a mistyped or circular expression, a
    /// node both set and shifted), with its registered reason code.
    pub fn prepare(
        model: &CompiledCausalModel,
        interventions: Vec<Intervention>,
    ) -> Result<Self, CausalError> {
        let overlay = InterventionOverlay::from_interventions(model, &interventions)?;
        let support = policy_support(model, &interventions);
        Ok(Self {
            interventions: Arc::from(interventions),
            overlay,
            support,
            program_compilations: 1,
            model_bindings: 1,
        })
    }

    /// The same program bound to `model` (for example the model refitted on refreshed data):
    /// the overlay and licence are resolved against it; [`Self::program_compilations`] is
    /// unchanged and [`Self::model_bindings`] grows by one.
    ///
    /// # Errors
    ///
    /// A model that refuses the program.
    pub fn rebind(&self, model: &CompiledCausalModel) -> Result<Self, CausalError> {
        let overlay = InterventionOverlay::from_interventions(model, &self.interventions)?;
        Ok(Self {
            interventions: Arc::clone(&self.interventions),
            overlay,
            support: policy_support(model, &self.interventions),
            program_compilations: self.program_compilations,
            model_bindings: self.model_bindings.saturating_add(1),
        })
    }

    /// The program.
    #[must_use]
    pub fn interventions(&self) -> &[Intervention] {
        &self.interventions
    }

    /// The licence of each route, fixed at preparation.
    #[must_use]
    pub fn support(&self) -> &PolicySupport {
        &self.support
    }

    /// How many times the program was compiled: one, however often it is rerun or rebound.
    #[must_use]
    pub fn program_compilations(&self) -> u64 {
        self.program_compilations
    }

    /// How many models the program's overlay and licence were resolved against.
    #[must_use]
    pub fn model_bindings(&self) -> u64 {
        self.model_bindings
    }

    /// Value of the retained program on `outcome`; equals [`policy_value`] on the same model,
    /// program and context. `model` must be the model the policy was prepared or rebound on.
    ///
    /// # Errors
    ///
    /// An outcome absent from the model, fewer than two draws, a table over the context's
    /// memory limit, or cancellation.
    #[doc(hidden)]
    pub fn value(
        &self,
        model: &CompiledCausalModel,
        outcome: VariableId,
        n: usize,
        ctx: &ExecutionContext,
    ) -> Result<PolicyValue, CausalError> {
        let node = policy_value_request(model, outcome, n)?;
        policy_value_on(model, &self.overlay, self.support.clone(), (outcome, node), n, ctx)
    }
}

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

/// Abduction once, then unit-level ITE for binary hard interventions on `treatment`.
///
/// # Errors
///
/// Abduction / prediction failures.
#[doc(hidden)]
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
    let exo = engine.abduct(data, AbductionMissingPolicy::Error, ctx)?;
    ite_from_abduction(&engine, exo, treatment, outcome, (active, control), ctx)
}

/// [`counterfactual_ite`] with the noise of many-to-one (categorical) nodes drawn from `rng`.
///
/// One posterior draw of the mechanisms owns one abduction: a categorical node's disturbance is
/// uniform within the factual category's CDF bin given that draw's parameters, so a Bayesian
/// refit loop passes one stream through every draw and the noise posterior varies with the
/// parameters instead of repeating one seeded draw. Invertible mechanisms consume no randomness
/// and give the same unit effects as [`counterfactual_ite`].
///
/// # Errors
///
/// Abduction / prediction failures.
pub(crate) fn counterfactual_ite_with_noise(
    model: CompiledCausalModel,
    data: &TabularData,
    contrast: (VariableId, VariableId, f64, f64),
    ctx: &ExecutionContext,
    rng: &mut CausalRng,
) -> Result<IteResult, CausalError> {
    let (treatment, outcome, active, control) = contrast;
    let engine = CounterfactualEngine::new(model);
    let exo = engine.abduct_with_rng(data, AbductionMissingPolicy::Error, rng)?;
    ite_from_abduction(&engine, exo, treatment, outcome, (active, control), ctx)
}

/// Replay one abduced noise state under both interventions and summarize the unit effects.
fn ite_from_abduction(
    engine: &CounterfactualEngine,
    exo: ExogenousPosterior,
    treatment: VariableId,
    outcome: VariableId,
    levels: (f64, f64),
    ctx: &ExecutionContext,
) -> Result<IteResult, CausalError> {
    let (active, control) = levels;
    let mut ws = MechanismWorkspace::default();
    let ite = engine.individual_treatment_effect(
        &exo,
        outcome,
        Intervention::set(treatment, Value::f64(active)),
        Intervention::set(treatment, Value::f64(control)),
        &mut ws,
        ctx,
    )?;
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
    /// draws (Bayesian: `unit_posterior_quantile`, a credible interval for the
    /// observed unit's contrast under the fitted mechanism, mechanism-refit
    /// uncertainty only, not a predictive interval for a new unit).
    ///
    /// `None` for Frequentist results, deliberately. The unit effect
    /// `Y_i(a) - Y_i(a0)` is not identified from data; it is computed under the
    /// fitted SCM. Its value depends jointly on every path mechanism's
    /// parameters and on disturbances abducted with those same parameters (or,
    /// downstream of a discrete mechanism, on one seeded noise draw under a
    /// rank-preserving coupling, which no sampling interval describes), and
    /// each mechanism family is selected on the same data, so a delta-method
    /// interval conditional on the selected family would ignore selection. The
    /// refit construction that carries this uncertainty is the Bayesian
    /// Dirichlet mechanism-refit posterior, already published and calibrated
    /// under `InferenceMode::Bayesian`; a frequentist copy of it would target
    /// the same object. A pointwise interval for the conditional average effect
    /// `tau(x)` (not a unit ITE) is `DrLearner::fit_pointwise_profiles`
    /// (uncalibrated). The diagnostic `gcm.counterfactual.uncertainty_unavailable`
    /// carries this reason on the result.
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
#[doc(hidden)]
pub fn anomaly_attribution(
    model: &CompiledCausalModel,
    data: &TabularData,
    outcomes: impl IntoIterator<Item = VariableId>,
    max_units: usize,
) -> Result<Vec<AnomalyScores>, CausalError> {
    let targets: Arc<[VariableId]> = outcomes.into_iter().collect::<Vec<_>>().into();
    let q = AnomalyAttributionQuery::new(targets, max_units);
    score_anomalies(model, data, &q).map_err(CausalError::from)
}

/// [`anomaly_attribution`] under an execution context: rows are scored in parallel up to its
/// thread budget and a cancelled context stops the run.
///
/// # Errors
///
/// Attribution failures.
#[doc(hidden)]
pub fn anomaly_attribution_with(
    model: &CompiledCausalModel,
    data: &TabularData,
    outcomes: impl IntoIterator<Item = VariableId>,
    max_units: usize,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Vec<AnomalyScores>, CausalError> {
    let targets: Arc<[VariableId]> = outcomes.into_iter().collect::<Vec<_>>().into();
    let q = AnomalyAttributionQuery::new(targets, max_units);
    score_anomalies_with(model, data, &q, ctx).map_err(CausalError::from)
}

/// [`anomaly_attribution_with`] against a full [`AnomalyAttributionQuery`], so the
/// query's reference (`Empirical` / `Fixed`) is honored rather than reconstructed.
///
/// # Errors
///
/// Attribution failures.
#[doc(hidden)]
pub fn anomaly_attribution_query_with(
    model: &CompiledCausalModel,
    data: &TabularData,
    query: &AnomalyAttributionQuery,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Vec<AnomalyScores>, CausalError> {
    score_anomalies_with(model, data, query, ctx).map_err(CausalError::from)
}

/// Intrinsic causal influence: the Shapley split of `query.target`'s variance among the
/// exogenous noise terms of its ancestors and itself, with permutation-sampling, simulation and
/// mechanism-fit uncertainty reported as separate components.
///
/// `noise` must declare the noise terms independent (or pass a structure without bidirected
/// edges); `data`, when supplied with a requested bootstrap, is resampled to quantify
/// mechanism-fitting uncertainty, which is otherwise marked unquantified.
///
/// # Errors
///
/// Coded refusals for undeclared or correlated noise, unsupported mechanisms, oversized plans
/// and invalid settings; model or replay failures; cancellation.
#[doc(hidden)]
pub fn attribute_intrinsic_influence(
    model: &CompiledCausalModel,
    noise: NoiseIndependence<'_>,
    data: Option<&TabularData>,
    query: &IntrinsicInfluenceQuery,
    ctx: &ExecutionContext,
) -> Result<IntrinsicInfluenceResult, CausalError> {
    intrinsic_influence(model, noise, data, query, ctx).map_err(CausalError::from)
}

/// Distribution-change attribution (pinned baseline-GCM parity).
///
/// # Errors
///
/// Attribution failures.
#[doc(hidden)]
pub fn attribute_distribution_change(
    model: &CompiledCausalModel,
    data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &DistributionChangeOptions,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, CausalError> {
    distribution_change(model, data, query, options, ctx).map_err(CausalError::from)
}

/// Robust distribution-change attribution.
///
/// # Errors
///
/// Attribution failures.
#[doc(hidden)]
pub fn attribute_distribution_change_robust(
    model: &CompiledCausalModel,
    data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &RobustChangeOptions,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, CausalError> {
    distribution_change_robust(model, data, query, options, ctx).map_err(CausalError::from)
}

/// Structure-change attribution between two DAGs (parent-set Shapley).
///
/// # Errors
///
/// Attribution failures.
#[doc(hidden)]
pub fn attribute_structure_change(
    baseline_model: &CompiledCausalModel,
    comparison_model: &CompiledCausalModel,
    data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &StructureChangeOptions,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, CausalError> {
    structure_change(baseline_model, comparison_model, data, query, options, ctx)
        .map_err(CausalError::from)
}

/// Mechanism-change detection (not attribution).
///
/// # Errors
///
/// Detection failures.
#[doc(hidden)]
pub fn mechanism_change_detection(
    model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    method: MechanismChangeMethod,
    ctx: &ExecutionContext,
) -> Result<Vec<antecedent_attribution::MechanismChangeDetection>, CausalError> {
    detect_mechanism_changes(model, data, query, method, ctx).map_err(CausalError::from)
}

/// Unit-change attribution.
///
/// # Errors
///
/// Attribution failures.
#[doc(hidden)]
pub fn attribute_unit_change(
    model: &CompiledCausalModel,
    data: &TabularData,
    query: &UnitChangeQuery,
    ctx: &ExecutionContext,
) -> Result<UnitChangeResult, CausalError> {
    unit_change(model, data, query, ctx).map_err(CausalError::from)
}

/// Sample an interventional distribution under [`InterventionalDistributionQuery`].
///
/// Thin wrapper over [`sample_do`]. Only [`TargetPopulation::AllObserved`] is supported.
///
/// # Errors
///
/// Query validation, unsupported target population, or sampling failures.
#[doc(hidden)]
pub fn sample_interventional_distribution(
    model: &CompiledCausalModel,
    query: &InterventionalDistributionQuery,
    n: usize,
    rng: &mut CausalRng,
    ctx: &ExecutionContext,
) -> Result<ValueBatch, CausalError> {
    query.validate()?;
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
#[doc(hidden)]
pub fn attribute_path_specific(
    model: &CompiledCausalModel,
    query: &PathSpecificEffectQuery,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, CausalError> {
    query.validate()?;
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
    )?;
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
#[doc(hidden)]
pub fn attribute_paths(
    model: &CompiledCausalModel,
    sources: &[VariableId],
    outcome: VariableId,
    max_paths: usize,
    max_len: usize,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, CausalError> {
    path_decompose(model, sources, outcome, max_paths, max_len, ctx).map_err(CausalError::from)
}

/// Feature relevance under interventions, with the sampling options as one value.
///
/// # Errors
///
/// Sampling failures.
#[doc(hidden)]
pub fn attribute_feature_relevance_with(
    model: &CompiledCausalModel,
    data: &TabularData,
    outcome: VariableId,
    features: &[VariableId],
    options: &antecedent_attribution::FeatureRelevanceOptions,
    ctx: &ExecutionContext,
) -> Result<Vec<FeatureRelevance>, CausalError> {
    feature_relevance(model, data, outcome, features, options, ctx).map_err(CausalError::from)
}

/// Feature relevance under interventions.
///
/// Positional form of [`attribute_feature_relevance_with`], kept for the Python binding's
/// keyword arguments.
///
/// # Errors
///
/// Sampling failures.
#[allow(
    clippy::too_many_arguments,
    reason = "positional form of attribute_feature_relevance_with, which the Python binding calls with its keyword arguments"
)]
#[doc(hidden)]
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
    let options =
        antecedent_attribution::FeatureRelevanceOptions { delta, n_samples, max_features };
    attribute_feature_relevance_with(model, data, outcome, features, &options, ctx)
}

/// Rank root causes from an attribution result.
///
/// # Errors
///
/// Ranking failures.
#[doc(hidden)]
pub fn rank_root_causes(
    attribution: &ChangeAttributionResult,
    ctx: &ExecutionContext,
) -> Result<Vec<RootCauseRank>, CausalError> {
    root_cause_rank(attribution, None, None, ctx).map_err(CausalError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        AllocationMethod, CausalSchemaBuilder, MeasurementSpec, RoleHint, ShapleyConfig,
        SmallRoleSet, ValueType,
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

    /// Negative control: under the licensed *separable* linear-Gaussian outcome
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
    /// graph-shape scope refusal (the graph here is the licensed nested DAG).
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
        // The linear-Gaussian licensed path fits the same tiny sample fine, so
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
        let err = score_anomalies(&model, &data, &q).unwrap_err();
        assert_eq!(err, AttributionError::PopulationOutOfRange { kind: "row", index: n, limit: n });
    }
}
