//! Ordered phases of `StudyBuilder::finish`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::CausalQuery;
use antecedent_data::TableView;
use antecedent_discovery::GraphPosterior;

use crate::accepted::{AcceptedGraph, GraphClass, GraphRef};
use crate::error::CausalError;
use crate::estimator_spec::EstimatorSpec;
use crate::inference::InferenceMode;
use crate::strategy_table::{EstimatorId, IdentifierId};
use crate::support::{CellStatus, StructureSource};

use super::super::execute::{Study, StudyConfig, StudyDeclarations, StudyInference};
use super::super::latency::{ResolvedLatencyBudget, refuse_non_report_hmc};
use super::super::prepared::{CachedStaticIdentification, IdentificationCache};
use super::super::query_license::is_graphless_design;
use super::settings::{Declarations, ExecutionSettings, ExposureInputs, InferenceSettings};
use super::{
    DataInput, InterferenceSpec, RefuteSuite, TransportTrialSpec, custom_validators_apply,
    data_schema, estimator_setting_conflict, omitted_bootstrap_resamples, population_estimable,
    refuse_estimator_inference_mismatch, refuse_unlicensed_interference,
    refuse_unlicensed_panel_route, refuse_unlicensed_transport, refuse_unsupported_draw_controls,
    refuse_unsupported_likelihood, stub_accepted_graph_for, tiered_graph_conflict,
    validate_schema_binding,
};

/// Builder fields moved into the pipeline, plus the `inspect_only` flag.
pub(super) struct FinishInput {
    pub(super) inspect_only: bool,
    pub(super) data: DataInput,
    pub(super) graph: Option<AcceptedGraph>,
    pub(super) graph_posterior: Option<GraphPosterior>,
    pub(super) query: Option<CausalQuery>,
    pub(super) caller_graph: bool,
    pub(super) declared: Declarations,
    pub(super) exposure: ExposureInputs,
    pub(super) settings: InferenceSettings,
    pub(super) execution: ExecutionSettings,
}

/// State that passes through the phases. A phase writes the fields it decides.
struct Carry {
    inspect_only: bool,
    data: DataInput,
    declared: Declarations,
    exposure: ExposureInputs,
    settings: InferenceSettings,
    execution: ExecutionSettings,
    /// The suite requested before a default downgrade to `RefuteSuite::None`.
    refute_default_downgrade: Option<RefuteSuite>,
    /// `(tier_replicates, configured_replicates)` when the configured estimator's own
    /// replicate count won over the latency tier's.
    latency_bootstrap_not_applied: Option<(u32, u32)>,
    /// The requested `AverageEffect` when it was compiled as its lag-0 pulse contrast.
    lag_zero_average_effect: Option<antecedent_core::AverageEffectQuery>,
}

/// Graph and query slot shared by every phase after the schema is bound.
struct Spine<Q> {
    carry: Carry,
    graph: AcceptedGraph,
    graph_posterior: Option<GraphPosterior>,
    query: Q,
}

/// Schema phase: exclusive inputs, then one graph bound to the data schema.
struct BoundSchema(Spine<Option<CausalQuery>>);

/// Inference phase: latency tier and compute budget applied to draws, bootstrap, and refute.
struct BudgetedInference(Spine<Option<CausalQuery>>);

/// Query phase: a query is present, and a design-based query has its fixed contract.
struct ContractedQuery(Spine<CausalQuery>);

/// Estimand phase: a sharp RD design retargets the default population to its cutoff.
struct RetargetedEstimand(Spine<CausalQuery>);

/// Route phase: likelihood, population, bootstrap ownership, and custom validators.
struct ExecutableRoute(Spine<CausalQuery>);

/// Graph phase: structure source, effective class, matrix class, and outcome functional.
struct ClassifiedGraph {
    spine: Spine<CausalQuery>,
    structure: StructureSource,
    graph_class: GraphClass,
    matrix_class: &'static str,
    selected: Option<EstimatorId>,
}

/// Estimator phase: identifier and estimator compatibility, plus a cell-AIPW identification.
struct ChosenEstimator {
    classified: ClassifiedGraph,
    identification_cache: Option<Arc<CachedStaticIdentification>>,
}

/// Refute phase: default-suite downgrade, then unknown-tier, panel, and class-prior licenses.
struct SettledRefute(ChosenEstimator);

/// Support phase: matrix cell classification for this query.
struct ClassifiedSupport {
    settled: SettledRefute,
    support_status: Option<CellStatus>,
}

/// Exposure phase: transport selection diagram or interference network bound to the unit table.
struct BoundExposure {
    support: ClassifiedSupport,
    selection_diagram: Option<antecedent_graph::SelectionDiagram>,
    transport_trial: Option<TransportTrialSpec>,
    interference: Option<InterferenceSpec>,
}

pub(super) fn assemble(input: FinishInput) -> Result<Study, CausalError> {
    let schema = bind_schema(input)?;
    let inference = budget_inference(schema)?;
    let query = contract_query(inference)?;
    let estimand = retarget_cutoff_estimand(query)?;
    let route = admit_executable_route(estimand)?;
    let graph = classify_graph(route)?;
    let estimator = choose_estimator(graph)?;
    let refute = settle_refute(estimator)?;
    let support = classify_support(refute)?;
    let exposure = bind_exposure(support)?;
    Ok(into_study(exposure))
}

fn bind_schema(input: FinishInput) -> Result<BoundSchema, CausalError> {
    refuse_conflicting_inputs(&input)?;

    let FinishInput {
        inspect_only,
        data,
        graph,
        graph_posterior,
        query,
        caller_graph: _,
        mut declared,
        exposure,
        settings,
        execution,
    } = input;
    if let Some(design) = declared.sampling_design {
        if let Some(spec) = declared.estimator_spec.as_mut() {
            spec.bind_sampling_design(design)?;
        }
    }
    let design_based = query.as_ref().is_some_and(is_graphless_design);
    let discovered = match &query {
        Some(CausalQuery::Discover(search)) => Some(super::discovery_stub_class(&search.method)),
        _ => None,
    };
    let (graph, graph_posterior) = match discovered {
        Some(class) => {
            if graph.is_some() || graph_posterior.is_some() {
                return Err(CausalError::Conflict {
                    what: "graph",
                    detail: "a structure-discovery study takes no graph or graph posterior; the \
                             discovered structure is its answer",
                });
            }
            (super::discovery_stub_graph(&data, class)?, None)
        }
        None => bind_graph(graph, graph_posterior, &data, design_based)?,
    };
    Ok(BoundSchema(Spine {
        carry: Carry {
            inspect_only,
            data,
            declared,
            exposure,
            settings,
            execution,
            refute_default_downgrade: None,
            latency_bootstrap_not_applied: None,
            lag_zero_average_effect: None,
        },
        graph,
        graph_posterior,
        query,
    }))
}

/// Refuses builder inputs that are mutually exclusive before any field is moved.
fn refuse_conflicting_inputs(input: &FinishInput) -> Result<(), CausalError> {
    crate::analysis::missing_data::validate_declaration(
        &crate::analysis::missing_data::DeclarationContext {
            query: input.query.as_ref(),
            data: &input.data,
            inference: &input.settings.inference,
            estimator: input.declared.estimator,
            graph_posterior: input.graph_posterior.is_some(),
            sampling_design: input.declared.sampling_design,
        },
        input.declared.missing.as_ref(),
    )?;
    if let Some(spec) = &input.declared.estimator_spec {
        if spec.is_configured() {
            if input.settings.bootstrap_explicit {
                return Err(estimator_setting_conflict("bootstrap_replicates"));
            }
            if input.declared.overlap_policy.is_some() {
                return Err(estimator_setting_conflict("overlap_policy"));
            }
        }
    }
    if let (Some(design), Some(_)) = (input.declared.sampling_design, input.declared.split.as_ref())
    {
        // A discovery/estimation split cuts the series into contiguous blocks.
        design.apply_splitting(antecedent_core::SplittingRule::ContiguousBlock).map_err(
            |error| CausalError::Data(antecedent_data::DataError::SamplingDesign(error)),
        )?;
    }
    if input.caller_graph && input.declared.tiered.is_some() {
        return Err(tiered_graph_conflict());
    }
    if input.declared.class_prior.is_some() && input.graph_posterior.is_some() {
        return Err(CausalError::Conflict {
            what: "class_prior",
            detail: "class_prior and graph_posterior are distinct structural-mass contracts; \
                     supply exactly one",
        });
    }
    Ok(())
}

/// Binds the supplied graph or graph posterior to the data schema, or the trial graph of a design-based query.
fn bind_graph(
    graph: Option<AcceptedGraph>,
    graph_posterior: Option<GraphPosterior>,
    data: &DataInput,
    design_based: bool,
) -> Result<(AcceptedGraph, Option<GraphPosterior>), CausalError> {
    Ok(match (graph, graph_posterior) {
        (Some(_), Some(_)) => {
            return Err(CausalError::Conflict {
                what: "graph",
                detail: "both .graph(..) and .graph_posterior(..) were set; supply exactly \
                         one causal-structure input",
            });
        }
        (Some(g), None) => {
            validate_schema_binding(&g, data_schema(data))?;
            (g, None)
        }
        (None, Some(gp)) => {
            let stub = stub_accepted_graph_for(data, gp.n_vars(), gp.atom_kind())?;
            (stub, Some(gp))
        }
        (None, None) if design_based => (AcceptedGraph::randomized_trial(data_schema(data)), None),
        (None, None) => return Err(CausalError::Missing { field: "graph" }),
    })
}

fn budget_inference(bound: BoundSchema) -> Result<BudgetedInference, CausalError> {
    let BoundSchema(Spine { mut carry, graph, graph_posterior, query }) = bound;
    let mut refute = carry.settings.refute;
    let mut bootstrap_replicates = carry.settings.bootstrap_replicates;
    // Moved, not cloned, so a boxed external prior keeps its allocation.
    let mut inference =
        std::mem::replace(&mut carry.settings.inference, InferenceMode::Frequentist);
    let latency_mode = carry.execution.latency_mode;
    let compute_budget = carry.execution.compute_budget;

    if let Some(mode) = latency_mode {
        refuse_non_report_hmc(mode, &inference)?;
        let resolved = ResolvedLatencyBudget::from_mode(mode).with_overrides(compute_budget);
        if !carry.settings.bootstrap_explicit {
            bootstrap_replicates = resolved.bootstrap;
        } else if let Some(b) = compute_budget.bootstrap {
            bootstrap_replicates = b;
        }
        if !carry.settings.refute_explicit {
            refute = resolved.refute;
        } else if let Some(v) = compute_budget.validators {
            refute = v;
        }
        inference = match inference {
            InferenceMode::Bayesian(cfg) => {
                // A budget field wins, then a count the caller set on the config; only
                // a backend-default count is resized by the tier.
                let draws = if carry.settings.n_draws_explicit {
                    compute_budget.n_draws.unwrap_or(cfg.n_draws)
                } else if cfg.n_draws_explicit {
                    cfg.n_draws
                } else {
                    resolved.n_draws
                };
                let mut cfg = cfg;
                cfg.n_draws = draws;
                InferenceMode::Bayesian(cfg)
            }
            InferenceMode::Frequentist => InferenceMode::Frequentist,
        };
    } else if compute_budget.bootstrap.is_some()
        || compute_budget.validators.is_some()
        || compute_budget.n_draws.is_some()
    {
        if let Some(b) = compute_budget.bootstrap {
            bootstrap_replicates = b;
        }
        if let Some(v) = compute_budget.validators {
            refute = v;
        }
        if let Some(n) = compute_budget.n_draws {
            inference = match inference {
                InferenceMode::Bayesian(cfg) => InferenceMode::Bayesian(cfg.n_draws(n)),
                InferenceMode::Frequentist => InferenceMode::Frequentist,
            };
        }
    }

    carry.settings.refute = refute;
    carry.settings.bootstrap_replicates = bootstrap_replicates;
    carry.settings.inference = inference;
    Ok(BudgetedInference(Spine { carry, graph, graph_posterior, query }))
}

fn contract_query(budget: BudgetedInference) -> Result<ContractedQuery, CausalError> {
    let BudgetedInference(Spine { mut carry, graph, graph_posterior, query }) = budget;
    let query = query.ok_or(CausalError::Missing { field: "query" })?;
    let query = lag_zero_average_effect(query, &mut carry, &graph)?;
    if matches!(
        query,
        CausalQuery::Discover(_) | CausalQuery::Attribute(_) | CausalQuery::Simulate(_)
    ) {
        refuse_structure_route_options(&query, &carry)?;
        // These routes run no refutation suite and do not resample: the search
        // or the fitted model is the procedure, and its uncertainty is what the
        // engine reports.
        carry.settings.refute = RefuteSuite::None;
        // The mechanism-refit row bootstrap of a fitted-model attribution runs only when a
        // replicate count was asked for.
        if !(carry.settings.bootstrap_explicit
            && crate::analysis::execute::attribute_bootstrap_supported(&query))
        {
            carry.settings.bootstrap_replicates = 0;
        }
    }
    if carry.settings.retain_joint_horizons {
        refuse_joint_horizons(&query, &carry, &graph)?;
    }
    if is_graphless_design(&query) {
        apply_design_contract(&query, &graph, &carry)?;
        // Design-based queries have no refutation suite. Bootstrap stays only for an
        // explicit survival or temporal-design count (the temporal design refuses it where
        // its interval does not resample); every other design-based query reports zero
        // replicates.
        carry.settings.refute = RefuteSuite::None;
        if !matches!(
            query,
            CausalQuery::Survival(_)
                | CausalQuery::TemporalDesign(_)
                | CausalQuery::IvBounds(_)
                | CausalQuery::Association(_)
        ) || !carry.settings.bootstrap_explicit
        {
            carry.settings.bootstrap_replicates = 0;
        }
        if carry.declared.split.is_some()
            || carry.declared.tiered.is_some()
            || carry.exposure.selection_targets.is_some()
            || carry.exposure.transport_trial.is_some()
            || carry.exposure.interference.is_some()
            || carry.declared.rd.is_some()
            || carry.declared.population_registry.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "design-based queries do not accept discovery splits, graph, transport, interference, RD, or population options",
            });
        }
    }
    Ok(ContractedQuery(Spine { carry, graph, graph_posterior, query }))
}

/// An `AverageEffect` on one series (or event series) under a `TemporalDag`, `TemporalCpdag` or
/// `TemporalPag` denotes the lag-0 contrast `E[Y_t | do(X_t = active)] - E[Y_t | do(X_t =
/// control)]`: the pulse at offset 0 read at horizon 1. On a class the envelope identifies it on
/// every completion (temporal backdoor adjustment for the parents of `X_t` on a `TemporalCpdag`
/// completion, visibility-aware MAG adjustment on a `TemporalPag` completion) and its Bayesian
/// reading is the per-completion conjugate posterior with draw-wise identified-set bounds; a
/// `TemporalDag` (or a class whose matrix reading collapses to one) is the single completion. The query is
/// compiled as that pulse; the requested query is kept so the support cell stays the
/// `AverageEffect` cell (executing, unlicensed) and every result discloses the reading.
///
/// Over a graph posterior the same pulse is compiled and the per-atom proofs mix as for any
/// other temporal pulse. Effect modifiers, a non-mean outcome functional and a tier background
/// are refused with the next attempt named. Every other query passes through unchanged.
fn lag_zero_average_effect(
    query: CausalQuery,
    carry: &mut Carry,
    graph: &AcceptedGraph,
) -> Result<CausalQuery, CausalError> {
    let CausalQuery::AverageEffect(average) = &query else {
        return Ok(query);
    };
    if !matches!(carry.data, DataInput::Temporal(_) | DataInput::Event(_))
        || !matches!(
            graph.class(),
            GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
        )
    {
        return Ok(query);
    }
    if carry.declared.tiered.is_some() {
        return Err(crate::unsupported_reason!(
            "route_not_supported",
            "a lag-0 AverageEffect on a temporal graph reads a supplied TemporalDag, \
             TemporalCpdag or TemporalPag or a posterior over them, not a tier background"
        ));
    }
    if !average.effect_modifiers.is_empty()
        || average.outcome_functional != antecedent_core::OutcomeFunctional::Mean
        || average.target_population != antecedent_core::TargetPopulation::AllObserved
    {
        return Err(crate::unsupported_reason!(
            "route_not_supported",
            "a lag-0 AverageEffect on a temporal graph is the mean contrast over all observed \
             periods, without modifiers. Next attempt: a TemporalConditionalPulse for a \
             modifier, or a temporal ResponseCurve for another functional"
        ));
    }
    let lag_zero = antecedent_core::TemporalEffectQuery {
        treatment: average.treatment,
        outcome: average.outcome,
        policy: antecedent_core::TemporalPolicy::pulse(0),
        control: average.control.clone(),
        active: average.active.clone(),
        horizon_steps: 1,
        max_history_lag: None,
        target_population: antecedent_core::TargetPopulation::AllObserved,
        effect_modifiers: Arc::from([]),
    };
    lag_zero.validate().map_err(|error| CausalError::Compile {
        message: format!("lag-0 AverageEffect: {error}"),
    })?;
    // `StudyBuilder::inference` binds the static Bayesian g-computation to every
    // AverageEffect; the temporal class envelope selects its own estimator.
    if carry.declared.estimator_spec.is_none()
        && carry.declared.estimator == Some(EstimatorId::BayesianGcomp)
    {
        carry.declared.estimator = None;
    }
    carry.lag_zero_average_effect = Some(average.clone());
    Ok(CausalQuery::TemporalEffect(lag_zero))
}

/// Refuses `retain_joint_horizons` on a study whose response has no joint horizon replicates to
/// retain: only a temporal response surface on a supplied `TemporalDag` resamples every horizon
/// together, and each inference mode needs what produces them.
fn refuse_joint_horizons(
    query: &CausalQuery,
    carry: &Carry,
    graph: &AcceptedGraph,
) -> Result<(), CausalError> {
    use antecedent_core::ResponseFunctional as F;
    if matches!(
        query,
        CausalQuery::Response(q)
            if matches!(
                q.functional,
                F::AverageDerivative { .. }
                    | F::PointDerivative { .. }
                    | F::DirectionalDerivative { .. }
                    | F::Jacobian { .. }
            )
    ) {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "a derivative response is the local slope at one horizon, so it has no horizons to \
             compose; omit retain_joint_horizons"
        ));
    }
    let temporal = matches!(query, CausalQuery::Response(q) if q.is_temporal());
    if !temporal || graph.as_temporal_dag().is_none() {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "joint horizon replicates are retained by a temporal response on a supplied \
             TemporalDag; omit retain_joint_horizons"
        ));
    }
    match &carry.settings.inference {
        InferenceMode::Frequentist
            if !(carry.settings.bootstrap_explicit && carry.settings.bootstrap_replicates > 0) =>
        {
            Err(crate::unsupported_reason!(
                "option_not_applicable",
                "the joint horizon replicates of a Frequentist response are its circular-block \
                 bootstrap replicates; request bootstrap_replicates"
            ))
        }
        InferenceMode::Bayesian(cfg)
            if cfg.prior.is_some()
                || cfg.prior_artifact.is_some()
                || cfg.external_compose.is_some()
                || cfg.white_innovation.is_some() =>
        {
            Err(crate::unsupported_reason!(
                "option_not_applicable",
                "the joint horizon draws are the block-weight posterior over the surface, which \
                 takes no coefficient prior, transferred prior or innovation declaration; leave \
                 them unset"
            ))
        }
        _ => Ok(()),
    }
}

/// Refuses the options a structure-discovery, attribution or simulation study
/// does not read, so none is accepted and then ignored.
fn refuse_structure_route_options(query: &CausalQuery, carry: &Carry) -> Result<(), CausalError> {
    if (carry.settings.refute_explicit && carry.settings.refute != RefuteSuite::None)
        || !carry.declared.custom_validators.is_empty()
    {
        return Err(crate::unsupported_reason!(
            "validators_not_applicable",
            "structure discovery, attribution and simulation run no refutation suite or custom \
             validator; omit refute and validators"
        ));
    }
    if carry.settings.bootstrap_explicit
        && carry.settings.bootstrap_replicates != 0
        && !crate::analysis::execute::attribute_bootstrap_supported(query)
    {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "structure discovery, attribution and simulation do not resample; omit \
             bootstrap_replicates"
        ));
    }
    if carry.declared.split.is_some()
        || carry.declared.tiered.is_some()
        || carry.exposure.selection_targets.is_some()
        || carry.exposure.transport_trial.is_some()
        || carry.exposure.interference.is_some()
        || carry.declared.rd.is_some()
        || carry.declared.population_registry.is_some()
        || carry.declared.identifier.is_some()
        || carry.declared.estimator.is_some()
        || carry.declared.estimator_spec.is_some()
    {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "structure discovery, attribution and simulation fix their own procedure; they take \
             no identifier, estimator, split, tier background, transport, interference, RD or \
             population option"
        ));
    }
    if matches!(carry.settings.inference, InferenceMode::Bayesian(_))
        && !crate::analysis::execute::attribute_posterior_supported(query)
        && !crate::analysis::execute::mechanism_change_posterior_declared(query)
    {
        return Err(if let CausalQuery::Discover(_) = query {
            crate::unsupported_reason!(
                "option_not_applicable",
                "a discovery method declares its own inference mode: a structure posterior \
                 (exact, order, structure, CI-screened or DBN) is Bayesian by its method; leave \
                 the study's inference at its default"
            )
        } else {
            crate::unsupported_reason!(
                "option_not_applicable",
                "an Attribute mechanism-change detection is a two-sample test whose p-value is a \
                 tail probability of a null law of its statistic, not a smooth functional of the \
                 refitted mechanisms; under Bayesian inference it reads as the posterior \
                 probability of a change, which needs a prior over the change indicator. Next \
                 attempt: declare it with MechanismChangeQuery::with_change_prior"
            )
        });
    }
    Ok(())
}

/// One refusal of an estimator selected for a design-based query it does not answer.
enum DesignCheck {
    /// A query of this kind that selected an estimator takes one of `allowed`.
    QueryTakes {
        query: fn(&CausalQuery) -> bool,
        allowed: &'static [EstimatorId],
        message: &'static str,
    },
    /// These estimators answer a query of this kind only.
    EstimatorsAnswer {
        query: fn(&CausalQuery) -> bool,
        estimators: &'static [EstimatorId],
        message: &'static str,
    },
    /// A rule that reads the query's own fields.
    Custom(fn(&CausalQuery, Option<EstimatorId>) -> Result<(), CausalError>),
}

impl DesignCheck {
    fn apply(
        &self,
        query: &CausalQuery,
        estimator: Option<EstimatorId>,
    ) -> Result<(), CausalError> {
        let refusal = match self {
            Self::QueryTakes { query: is_kind, allowed, message } => (is_kind(query)
                && estimator.is_some_and(|id| !allowed.contains(&id)))
            .then_some(*message),
            Self::EstimatorsAnswer { query: is_kind, estimators, message } => (!is_kind(query)
                && estimator.is_some_and(|id| estimators.contains(&id)))
            .then_some(*message),
            Self::Custom(check) => return check(query, estimator),
        };
        refusal.map_or(Ok(()), |message| Err(CausalError::Unsupported { message }))
    }
}

/// Defines the predicate "the query is one of these kinds" a [`DesignCheck`] reads.
macro_rules! query_kind {
    ($name:ident, $($pattern:pat_param)|+) => {
        fn $name(query: &CausalQuery) -> bool {
            matches!(query, $($pattern)|+)
        }
    };
}

query_kind!(is_panel_did, CausalQuery::PanelDid(_));
query_kind!(is_synthetic_control, CausalQuery::SyntheticControl(_));
query_kind!(is_local_ratio, CausalQuery::LocalPolynomialRatio(_));
query_kind!(
    is_bayesian_design,
    CausalQuery::PanelDid(_)
        | CausalQuery::SyntheticControl(_)
        | CausalQuery::LocalPolynomialRatio(_)
);
query_kind!(is_policy_value, CausalQuery::PolicyValue(_));
query_kind!(is_survival, CausalQuery::Survival(_));
query_kind!(is_dose_response, CausalQuery::ContinuousDoseResponse(_));
query_kind!(is_history_match, CausalQuery::HistoryMatch(_));
query_kind!(is_sequential_contrast, CausalQuery::SequentialContrast(_));
query_kind!(is_temporal_design, CausalQuery::TemporalDesign(_));
query_kind!(is_iv_bounds, CausalQuery::IvBounds(_));
query_kind!(is_longitudinal_regime, CausalQuery::LongitudinalRegime(_));

const DESIGN_ESTIMATOR_MISMATCH: &str = "the selected estimator does not match this design query; select its Bayesian estimator or none";
const HISTORY_ESTIMATORS: &[EstimatorId] =
    &[EstimatorId::PeriodHistoryMatching, EstimatorId::BayesianPeriodHistoryOutcomeModel];
const CONTRAST_ESTIMATORS: &[EstimatorId] =
    &[EstimatorId::SequentialCrossfitContrastWald, EstimatorId::BayesianSequentialCrossfitContrast];
const TEMPORAL_DESIGN_ESTIMATORS: &[EstimatorId] = &[
    EstimatorId::TemporalDesignExcursion,
    EstimatorId::BayesianTemporalDesignExcursion,
    EstimatorId::TemporalRegimeSurvival,
    EstimatorId::BayesianTemporalRegimeSurvival,
];
const IV_BOUNDS_ESTIMATORS: &[EstimatorId] =
    &[EstimatorId::IvBoundsSharp, EstimatorId::BayesianIvBounds];

/// The refusals of an estimator selected for a design-based query, in the order they apply.
const DESIGN_CHECKS: &[DesignCheck] = &[
    DesignCheck::Custom(refuse_randomized_effect_mismatch),
    DesignCheck::Custom(refuse_policy_value_mismatch),
    DesignCheck::QueryTakes {
        query: is_panel_did,
        allowed: &[EstimatorId::BayesianPanelDid],
        message: DESIGN_ESTIMATOR_MISMATCH,
    },
    DesignCheck::QueryTakes {
        query: is_synthetic_control,
        allowed: &[EstimatorId::BayesianSyntheticControl],
        message: DESIGN_ESTIMATOR_MISMATCH,
    },
    DesignCheck::QueryTakes {
        query: is_local_ratio,
        allowed: &[EstimatorId::BayesianLocalRatio],
        message: DESIGN_ESTIMATOR_MISMATCH,
    },
    DesignCheck::EstimatorsAnswer {
        query: is_bayesian_design,
        estimators: &[
            EstimatorId::BayesianPanelDid,
            EstimatorId::BayesianSyntheticControl,
            EstimatorId::BayesianLocalRatio,
        ],
        message: "the selected Bayesian design estimator does not match this query",
    },
    DesignCheck::EstimatorsAnswer {
        query: is_policy_value,
        estimators: &[EstimatorId::BayesianBinaryPolicyValue],
        message: "bayesian.binary_policy_value answers a PolicyValue query only",
    },
    DesignCheck::QueryTakes {
        query: is_survival,
        allowed: &[
            EstimatorId::RandomizedSurvivalProductLimit,
            EstimatorId::BayesianPiecewiseSurvival,
        ],
        message: "randomized survival requires the product-limit or the piecewise-exponential Bayesian estimator",
    },
    DesignCheck::QueryTakes {
        query: is_dose_response,
        allowed: &[EstimatorId::ContinuousDoseKernel, EstimatorId::BayesianDosePolicy],
        message: "continuous-dose response requires the kernel or the Bayesian dose-policy estimator",
    },
    DesignCheck::EstimatorsAnswer {
        query: is_survival,
        estimators: &[EstimatorId::BayesianPiecewiseSurvival],
        message: "bayesian.piecewise_survival answers a randomized Survival query only",
    },
    DesignCheck::EstimatorsAnswer {
        query: is_dose_response,
        estimators: &[EstimatorId::BayesianDosePolicy],
        message: "bayesian.continuous_dose_policy answers a ContinuousDoseResponse query only",
    },
    DesignCheck::QueryTakes {
        query: is_history_match,
        allowed: HISTORY_ESTIMATORS,
        message: "a history match query takes the period history matching estimator, and that estimator answers that query only",
    },
    DesignCheck::EstimatorsAnswer {
        query: is_history_match,
        estimators: HISTORY_ESTIMATORS,
        message: "a history match query takes the period history matching estimator, and that estimator answers that query only",
    },
    DesignCheck::QueryTakes {
        query: is_sequential_contrast,
        allowed: CONTRAST_ESTIMATORS,
        message: "a sequential contrast query takes the sequential cross-fit contrast estimators, and they answer that query only",
    },
    DesignCheck::EstimatorsAnswer {
        query: is_sequential_contrast,
        estimators: CONTRAST_ESTIMATORS,
        message: "a sequential contrast query takes the sequential cross-fit contrast estimators, and they answer that query only",
    },
    DesignCheck::EstimatorsAnswer {
        query: is_temporal_design,
        estimators: TEMPORAL_DESIGN_ESTIMATORS,
        message: "the temporal-design estimators answer a TemporalDesign query only",
    },
    DesignCheck::QueryTakes {
        query: is_temporal_design,
        allowed: TEMPORAL_DESIGN_ESTIMATORS,
        message: "a temporal-design query takes the excursion estimators for an arm or dose design and the regime-survival estimators for a survival design",
    },
    DesignCheck::Custom(refuse_temporal_design_assignment_mismatch),
    DesignCheck::QueryTakes {
        query: is_iv_bounds,
        allowed: IV_BOUNDS_ESTIMATORS,
        message: "an IV bounds query takes the IV bounds estimators, and they answer that query only",
    },
    DesignCheck::EstimatorsAnswer {
        query: is_iv_bounds,
        estimators: IV_BOUNDS_ESTIMATORS,
        message: "an IV bounds query takes the IV bounds estimators, and they answer that query only",
    },
    DesignCheck::EstimatorsAnswer {
        query: is_longitudinal_regime,
        estimators: &[EstimatorId::BayesianSequentialRegimeValue],
        message: "bayesian.sequential_regime_value answers a LongitudinalRegime query only",
    },
    DesignCheck::Custom(refuse_longitudinal_regime_method_mismatch),
];

/// Refuses an explicitly selected estimator that does not match the design-based query's frozen inputs.
fn refuse_estimator_design_mismatch(
    query: &CausalQuery,
    estimator: Option<EstimatorId>,
) -> Result<(), CausalError> {
    DESIGN_CHECKS.iter().try_for_each(|check| check.apply(query, estimator))
}

/// Refuses the survival estimators on an arm design, or the arm estimators on a survival
/// design, for a temporal-design query that selected one of the temporal-design estimators.
fn refuse_temporal_design_assignment_mismatch(
    query: &CausalQuery,
    estimator: Option<EstimatorId>,
) -> Result<(), CausalError> {
    let (CausalQuery::TemporalDesign(design), Some(estimator)) = (query, estimator) else {
        return Ok(());
    };
    let survival = matches!(
        estimator,
        EstimatorId::TemporalRegimeSurvival | EstimatorId::BayesianTemporalRegimeSurvival
    );
    let survival_design =
        matches!(design.assignment, antecedent_core::TemporalAssignment::RegimeSurvival { .. });
    if TEMPORAL_DESIGN_ESTIMATORS.contains(&estimator) && survival != survival_design {
        return Err(CausalError::Unsupported {
            message: "a temporal-design query takes the excursion estimators for an arm or dose \
                      design and the regime-survival estimators for a survival design",
        });
    }
    Ok(())
}

/// Refuses a longitudinal-regime estimator that does not match the query's method.
fn refuse_longitudinal_regime_method_mismatch(
    query: &CausalQuery,
    estimator: Option<EstimatorId>,
) -> Result<(), CausalError> {
    if let CausalQuery::LongitudinalRegime(regime) = query {
        let expected = match regime.method {
            antecedent_core::LongitudinalRegimeMethod::Ipw => EstimatorId::LongitudinalIpwRegime,
            antecedent_core::LongitudinalRegimeMethod::GFormula => {
                EstimatorId::LongitudinalGFormulaRegime
            }
            antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust => {
                EstimatorId::LongitudinalSequentialDrRegime
            }
            antecedent_core::LongitudinalRegimeMethod::MarginalStructuralModel => {
                EstimatorId::LongitudinalMarginalStructuralModel
            }
        };
        let bayesian_twin = estimator == Some(EstimatorId::BayesianSequentialRegimeValue)
            && regime.method == antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust;
        if estimator.is_some_and(|id| id != expected) && !bayesian_twin {
            return Err(CausalError::Unsupported {
                message: "longitudinal regime estimator must match the selected method",
            });
        }
    }
    Ok(())
}

/// Refuses a randomized-effect estimator that does not match the assignment design.
fn refuse_randomized_effect_mismatch(
    query: &CausalQuery,
    estimator: Option<EstimatorId>,
) -> Result<(), CausalError> {
    if let CausalQuery::RandomizedEffect(randomized) = query {
        let expected = match &randomized.design {
            antecedent_core::RandomizationDesign::Bernoulli => {
                if randomized.received_treatment.is_some() {
                    EstimatorId::RandomizedWaldCace
                } else if randomized.fixed_cuped.is_some() {
                    EstimatorId::RandomizedFixedCupedHt
                } else if !randomized.ancova_covariates.is_empty() {
                    EstimatorId::RandomizedAncova
                } else {
                    EstimatorId::RandomizedHt
                }
            }
            antecedent_core::RandomizationDesign::Complete { .. }
            | antecedent_core::RandomizationDesign::Stratified { .. }
            | antecedent_core::RandomizationDesign::Factorial2x2 { .. }
            | antecedent_core::RandomizationDesign::Cluster { .. } => EstimatorId::RandomizedNeyman,
            antecedent_core::RandomizationDesign::MultiArm { .. } => EstimatorId::RandomizedHt,
            antecedent_core::RandomizationDesign::Switchback { .. } => {
                EstimatorId::RandomizedSwitchbackHt
            }
        };
        let bayesian_arms = matches!(
            randomized.design,
            antecedent_core::RandomizationDesign::Bernoulli
                | antecedent_core::RandomizationDesign::Complete { .. }
                | antecedent_core::RandomizationDesign::Factorial2x2 { .. }
                | antecedent_core::RandomizationDesign::MultiArm { .. }
        );
        let selected_bayesian = matches!(
            estimator,
            Some(
                EstimatorId::BayesianRandomizedFinitePopulation
                    | EstimatorId::BayesianRandomizedSuperpopulation
                    | EstimatorId::BayesianComplierEffect
            )
        );
        let selected_switchback = estimator == Some(EstimatorId::BayesianSwitchback);
        if selected_switchback
            && !matches!(randomized.design, antecedent_core::RandomizationDesign::Switchback { .. })
        {
            return Err(CausalError::Unsupported {
                message: "the Bayesian switchback posterior needs a switchback assignment design",
            });
        }
        if selected_bayesian && !bayesian_arms {
            return Err(CausalError::Unsupported {
                message: "the Bayesian arm-contrast posterior models Bernoulli, complete, \
                          multi-arm and 2x2 factorial assignment; cluster, stratified and \
                          switchback designs have no arm-exchangeable likelihood here",
            });
        }
        if estimator.is_some_and(|id| id != expected) && !selected_bayesian && !selected_switchback
        {
            return Err(CausalError::Unsupported {
                message: "the selected randomized estimator does not match the assignment design",
            });
        }
    }
    Ok(())
}

/// Refuses a policy estimator that does not match the policy query's frozen inputs.
fn refuse_policy_value_mismatch(
    query: &CausalQuery,
    estimator: Option<EstimatorId>,
) -> Result<(), CausalError> {
    if let CausalQuery::PolicyValue(policy) = query {
        let expected = if policy.multi_action.is_some() {
            EstimatorId::RandomizedMultiActionIpwPolicy
        } else if policy.mu0.is_empty() {
            EstimatorId::RandomizedIpwPolicy
        } else {
            EstimatorId::RandomizedDrPolicy
        };
        if estimator
            .is_some_and(|id| id != expected && id != EstimatorId::BayesianBinaryPolicyValue)
        {
            return Err(CausalError::Unsupported {
                message: "the selected policy estimator does not match its frozen inputs",
            });
        }
    }
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "one linear sequence of design-contract refusals checked in a fixed order"
)]
fn apply_design_contract(
    query: &CausalQuery,
    graph: &AcceptedGraph,
    carry: &Carry,
) -> Result<(), CausalError> {
    refuse_estimator_design_mismatch(query, carry.declared.estimator)?;
    if let CausalQuery::Association(association) = query {
        association.validate()?;
    }
    if graph.class() != GraphClass::RandomizedTrial {
        return Err(CausalError::Unsupported {
            message: "design-based queries carry their own identification contract and must not be combined with graph= or discovery=",
        });
    }
    if matches!(query, CausalQuery::HistoryMatch(_))
        && matches!(carry.settings.inference, InferenceMode::Bayesian(_))
        && carry.declared.estimator != Some(EstimatorId::BayesianPeriodHistoryOutcomeModel)
    {
        // A matching rule has no likelihood: the typed boundary names the premise and the
        // next attempt instead of a mislabelled interval.
        return Err(super::super::matching_posterior_refusal::matching_posterior_refusal(
            super::super::matching_posterior_refusal::MatchingSetting::WithinPeriodHistory,
        ));
    }
    let bayesian_design = matches!(
        query,
        CausalQuery::PanelDid(_)
            | CausalQuery::SyntheticControl(_)
            | CausalQuery::LocalPolynomialRatio(_)
            | CausalQuery::PolicyValue(_)
            | CausalQuery::LongitudinalRegime(_)
            | CausalQuery::SequentialContrast(_)
            | CausalQuery::RandomizedEffect(_)
            | CausalQuery::Survival(_)
            | CausalQuery::ContinuousDoseResponse(_)
            | CausalQuery::TemporalDesign(_)
            | CausalQuery::HistoryMatch(_)
            | CausalQuery::IvBounds(_)
            | CausalQuery::Association(_)
    );
    if !matches!(carry.settings.inference, InferenceMode::Frequentist) && !bayesian_design {
        return Err(CausalError::Unsupported {
            message: "design-based queries currently support Frequentist inference only",
        });
    }
    // Survival and the dose policy read the declared design model and refuse an unusable
    // backend themselves, naming their own next attempt.
    let reads_own_model = matches!(
        query,
        CausalQuery::Survival(_)
            | CausalQuery::ContinuousDoseResponse(_)
            | CausalQuery::TemporalDesign(_)
            | CausalQuery::IvBounds(_)
            | CausalQuery::Association(_)
    );
    if let (InferenceMode::Bayesian(config), false) = (&carry.settings.inference, reads_own_model) {
        if config.backend != antecedent_estimate::BayesianBackendKind::ConjugateGaussian
            || config.has_transfer()
        {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "the Bayesian panel DiD, synthetic control and local ratio fit their own \
                 built-in Gaussian models; use the conjugate backend with prior_scale and \
                 n_draws only, and no transferred or custom prior"
            ));
        }
    }
    if carry.declared.identifier.is_some_and(|id| id != IdentifierId::RandomizedDesign)
        || carry.declared.estimator.is_some_and(|id| {
            !matches!(
                id,
                EstimatorId::RandomizedHt
                    | EstimatorId::RandomizedFixedCupedHt
                    | EstimatorId::RandomizedAncova
                    | EstimatorId::RandomizedWaldCace
                    | EstimatorId::RandomizedSwitchbackHt
                    | EstimatorId::RandomizedNeyman
                    | EstimatorId::RandomizedSurvivalProductLimit
                    | EstimatorId::LongitudinalIpwRegime
                    | EstimatorId::LongitudinalGFormulaRegime
                    | EstimatorId::BayesianPiecewiseSurvival
                    | EstimatorId::BayesianDosePolicy
                    | EstimatorId::ContinuousDoseKernel
                    | EstimatorId::BayesianPanelDid
                    | EstimatorId::BayesianSyntheticControl
                    | EstimatorId::BayesianLocalRatio
                    | EstimatorId::BayesianBinaryPolicyValue
                    | EstimatorId::BayesianSequentialRegimeValue
                    | EstimatorId::SequentialCrossfitContrastWald
                    | EstimatorId::BayesianSequentialCrossfitContrast
                    | EstimatorId::PeriodHistoryMatching
                    | EstimatorId::BayesianPeriodHistoryOutcomeModel
                    | EstimatorId::BayesianRandomizedFinitePopulation
                    | EstimatorId::BayesianRandomizedSuperpopulation
                    | EstimatorId::BayesianComplierEffect
                    | EstimatorId::BayesianSwitchback
                    | EstimatorId::TemporalDesignExcursion
                    | EstimatorId::BayesianTemporalDesignExcursion
                    | EstimatorId::TemporalRegimeSurvival
                    | EstimatorId::BayesianTemporalRegimeSurvival
                    | EstimatorId::IvBoundsSharp
                    | EstimatorId::BayesianIvBounds
            )
        })
        || carry
            .declared
            .estimator_spec
            .as_ref()
            .is_some_and(|spec| !matches!(spec, EstimatorSpec::Default(_)))
    {
        return Err(CausalError::Unsupported {
            message: "design-based query estimators are fixed; custom estimators are not accepted",
        });
    }
    if (carry.settings.refute_explicit && carry.settings.refute != RefuteSuite::None)
        || !carry.declared.custom_validators.is_empty()
    {
        return Err(CausalError::Unsupported {
            message: "design-based queries have no refutation suite or custom validator route",
        });
    }
    if carry.settings.bootstrap_explicit
        && carry.settings.bootstrap_replicates != 0
        && !matches!(
            query,
            CausalQuery::Survival(_)
                | CausalQuery::TemporalDesign(_)
                | CausalQuery::IvBounds(_)
                | CausalQuery::Association(_)
        )
    {
        return Err(CausalError::Unsupported {
            message: "this design-based query does not support bootstrap intervals",
        });
    }
    Ok(())
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "one stage of the finish chain, every stage of which returns Result so the chain composes with `?`"
)]
fn retarget_cutoff_estimand(
    contracted: ContractedQuery,
) -> Result<RetargetedEstimand, CausalError> {
    let ContractedQuery(Spine { carry, graph, graph_posterior, mut query }) = contracted;
    // A sharp RD design identifies the effect for units at its cutoff and nothing
    // else. A study that selects it while leaving the population at the default is
    // retargeted to that population here, so the contract, the calibration key and
    // the result all name the estimand that is actually identified; the
    // identification diagnostics say that the population-wide effect was not.
    if matches!(
        carry.declared.estimator,
        Some(EstimatorId::RdSharp | EstimatorId::BayesianRdLocalLinear)
    ) || carry.declared.identifier == Some(IdentifierId::RdSharp)
    {
        if let (Some(rd), CausalQuery::AverageEffect(average)) =
            (carry.declared.rd.as_ref(), &mut query)
        {
            if average.target_population == antecedent_core::TargetPopulation::AllObserved {
                average.target_population = antecedent_core::TargetPopulation::local_at_cutoff(
                    rd.running_variable,
                    rd.cutoff,
                );
            }
        }
    }
    Ok(RetargetedEstimand(Spine { carry, graph, graph_posterior, query }))
}

fn admit_executable_route(estimand: RetargetedEstimand) -> Result<ExecutableRoute, CausalError> {
    let RetargetedEstimand(Spine { mut carry, graph, graph_posterior, query }) = estimand;
    // An estimator that does not implement this inference mode is refused for that
    // mismatch before a likelihood or route check the estimator never reads.
    if let Some(estimator) = carry.declared.estimator {
        refuse_estimator_inference_mismatch(&query, estimator, &carry.settings.inference)?;
    }
    refuse_unsupported_likelihood(
        &query,
        &carry.data,
        graph.class(),
        graph_posterior.is_none() && carry.declared.tiered.is_none(),
        carry.declared.estimator,
        &carry.settings.inference,
    )?;
    super::refuse_unsupported_temporal_link(
        &query,
        &carry.data,
        graph.class(),
        graph_posterior.is_none() && carry.declared.tiered.is_none(),
        carry.declared.estimator,
    )?;
    super::refuse_unsupported_temporal_instrument(
        &query,
        &carry.data,
        graph.class(),
        graph_posterior.is_none() && carry.declared.tiered.is_none(),
        carry.declared.identifier,
        carry.declared.estimator,
    )?;
    super::refuse_unsupported_white_innovation(
        &query,
        &carry.data,
        graph.class(),
        graph_posterior.is_none() && carry.declared.tiered.is_none(),
        &carry.settings.inference,
    )?;
    refuse_unsupported_draw_controls(
        &query,
        &carry.data,
        graph.class(),
        graph_posterior.is_none() && carry.declared.tiered.is_none(),
        carry.declared.estimator,
        carry.declared.draw_controls,
        &carry.settings.inference,
    )?;
    super::refuse_unsupported_predictive_replicates(
        &query,
        &carry.data,
        graph.class(),
        graph_posterior.is_none() && carry.declared.tiered.is_none(),
        carry.declared.estimator,
        &carry.settings.inference,
    )?;
    if !population_estimable(&query) {
        return Err(crate::unsupported_reason!(
            "population_not_estimable",
            "only an AverageEffect estimates a target population other than AllObserved; \
             this query kind has no weighting or subpopulation estimator for another target. \
             Declare the AllObserved population, or prepare an AllObserved AIPW or cell-AIPW \
             study and retarget its frozen scores"
        ));
    }
    if let Some(configured) =
        carry.declared.estimator_spec.as_ref().and_then(EstimatorSpec::bootstrap_replicates)
    {
        // A configured estimator owns its replicate count (an explicit
        // builder count beside it is refused above). The study reports and
        // executes that same count instead of its own omitted default.
        if carry.execution.latency_mode.is_some()
            && carry.settings.bootstrap_replicates != configured
        {
            carry.latency_bootstrap_not_applied =
                Some((carry.settings.bootstrap_replicates, configured));
        }
        carry.settings.bootstrap_replicates = configured;
    }
    if !carry.settings.bootstrap_explicit
        && !omitted_bootstrap_resamples(&query, &carry.settings.inference)
    {
        carry.settings.bootstrap_replicates = 0;
    }
    if !carry.declared.custom_validators.is_empty()
        && !custom_validators_apply(
            &query,
            &carry.data,
            graph.class(),
            graph_posterior.is_some(),
            &carry.settings.inference,
        )
    {
        return Err(crate::unsupported_reason!(
            "validators_not_applicable",
            "custom validators refute one scalar average effect; this query and data route \
             has no scalar refutation problem for them to run on"
        ));
    }
    require_declared_dependence_read(&carry, &query)?;
    Ok(ExecutableRoute(Spine { carry, graph, graph_posterior, query }))
}

/// Refuse a declared clustered or serial design whose analytic error ignores it.
///
/// With the bootstrap off, the only uncertainty an average effect, conditional effect or
/// response publishes is the analytic error, and the study-level design cannot supply the
/// cluster labels or lag it needs. An error that reads rows as independent under a declared
/// clustered panel or dependent time series is too small, so the study is held to the
/// declaration: the configured estimator must carry the matching standard-error kind (cluster,
/// multiway or panel HAC for a panel, Newey-West or panel HAC for a series), which only the
/// average effect through a configured regression, weighting or AIPW estimator reads.
fn require_declared_dependence_read(
    carry: &Carry,
    query: &CausalQuery,
) -> Result<(), CausalError> {
    use antecedent_core::DeclaredSamplingDesign as Design;
    use antecedent_estimate::AnalyticSeKind as Kind;
    let Some(design @ (Design::ClusteredPanel | Design::DependentTimeSeries)) =
        carry.declared.sampling_design
    else {
        return Ok(());
    };
    let reads_estimator_dependence = match query {
        CausalQuery::AverageEffect(_) => true,
        CausalQuery::ConditionalEffect(_) | CausalQuery::Response(_) => false,
        _ => return Ok(()),
    };
    if !matches!(carry.data, DataInput::Tabular(_)) || carry.settings.inference.is_bayesian() {
        return Ok(());
    }
    let inference = carry.declared.estimator_spec.as_ref().and_then(EstimatorSpec::inference_spec);
    let replicates = inference.map_or(carry.settings.bootstrap_replicates, |i| i.bootstrap_replicates);
    if replicates > 0 {
        return Ok(());
    }
    let reads = reads_estimator_dependence
        && inference.is_some_and(|inference| match (design, inference.se_kind) {
            (Design::ClusteredPanel, Kind::Cluster | Kind::Multiway | Kind::PanelClusterHac { .. }) => {
                true
            }
            (Design::DependentTimeSeries, Kind::NeweyWest { .. } | Kind::PanelClusterHac { .. }) => {
                true
            }
            _ => false,
        });
    if reads {
        return Ok(());
    }
    Err(match design {
        Design::ClusteredPanel => crate::unsupported_reason!(
            "option_not_applicable",
            "a clustered-panel design is declared with the bootstrap off, but the analytic error \
             this route publishes treats rows as independent and would be too small. Next \
             attempt: configure an average-effect estimator whose standard error is the cluster \
             sandwich (for example LinearAdjustmentAte::new().with_se_kind(AnalyticSeKind::Cluster) \
             with its cluster ids), or declare independent subject trajectories if the rows are \
             independent"
        ),
        _ => crate::unsupported_reason!(
            "option_not_applicable",
            "a dependent-time-series design is declared with the bootstrap off, but the analytic \
             error this route publishes treats rows as independent and would be too small. Next \
             attempt: configure an average-effect estimator whose standard error is Newey-West \
             (for example LinearAdjustmentAte::new().with_se_kind(AnalyticSeKind::NeweyWest { lag })), \
             or declare independent subject trajectories if the rows are independent"
        ),
    })
}

fn classify_graph(route: ExecutableRoute) -> Result<ClassifiedGraph, CausalError> {
    let ExecutableRoute(Spine { mut carry, graph, graph_posterior, query }) = route;
    let structure = if graph_posterior.is_some() {
        StructureSource::GraphPosterior
    } else if matches!(query, CausalQuery::Discover(_)) {
        StructureSource::Discovered
    } else if graph.class() == GraphClass::RandomizedTrial {
        StructureSource::RandomizedTrial
    } else {
        carry.declared.structure_source.unwrap_or(StructureSource::Explicit)
    };
    // Graph-posterior stubs are empty bookkeeping graphs. Collapse based on
    // stub edges (an empty Admg looks like a DAG) would relabel licensed
    // Admg posterior cells. Trust the atom-kind stub class instead.
    let graph_class = if graph_posterior.is_some() {
        graph.class()
    } else {
        crate::support::effective_graph_class(&graph, &query)
    };
    let matrix_class = if graph_posterior.is_some() {
        graph.class().as_str()
    } else {
        crate::support::matrix_graph_class(&graph, &query, carry.declared.tiered.as_ref())
    };
    let selected = carry.declared.estimator_spec.as_ref().map(EstimatorSpec::id);
    refuse_outcome_functional(
        &query,
        &mut carry,
        graph_posterior.is_some(),
        graph_class,
        selected,
    )?;
    Ok(ClassifiedGraph {
        spine: Spine { carry, graph, graph_posterior, query },
        structure,
        graph_class,
        matrix_class,
        selected,
    })
}

fn refuse_outcome_functional(
    query: &CausalQuery,
    carry: &mut Carry,
    graph_posterior: bool,
    graph_class: GraphClass,
    selected: Option<EstimatorId>,
) -> Result<(), CausalError> {
    let Carry { declared, settings, refute_default_downgrade, .. } = carry;
    let tiered = declared.tiered.as_ref();
    let InferenceSettings { inference, refute_explicit, refute, .. } = settings;
    let inference = &*inference;
    let refute_explicit = *refute_explicit;
    let functional = match query {
        CausalQuery::AverageEffect(q) => Some(&q.outcome_functional),
        CausalQuery::ConditionalEffect(q) => Some(&q.inner.outcome_functional),
        CausalQuery::Response(q) => Some(&q.outcome_functional),
        _ => None,
    };
    // The randomized quantile posterior reads the outcome functional itself: it is the
    // Bayesian counterpart of the quantile coordinate, selected explicitly on a Dag.
    let bayesian_quantile = matches!(inference, InferenceMode::Bayesian(_))
        && selected == Some(EstimatorId::BayesianRandomizedQuantile)
        && !graph_posterior
        && graph_class == GraphClass::Dag
        && matches!(query, CausalQuery::AverageEffect(q)
            if q.outcome_functional.quantile_level().is_some());
    if functional.is_some_and(|f| !matches!(f, antecedent_core::OutcomeFunctional::Mean)) {
        let frequentist_transform = matches!(inference, InferenceMode::Frequentist)
            && !graph_posterior
            && match query {
                CausalQuery::AverageEffect(_) => graph_class == GraphClass::Dag || tiered.is_some(),
                CausalQuery::ConditionalEffect(_) => {
                    matches!(graph_class, GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag)
                }
                CausalQuery::Response(q) => q.temporal.is_none(),
                _ => false,
            };
        let transforms_outcome = bayesian_quantile || frequentist_transform;
        if !transforms_outcome {
            return Err(CausalError::Unsupported {
                message: "this inference/graph/query path does not implement the requested outcome functional",
            });
        }
    }
    if functional.and_then(antecedent_core::OutcomeFunctional::quantile_level).is_some() {
        let frequentist_supported = matches!(inference, InferenceMode::Frequentist)
            && !graph_posterior
            && match query {
                CausalQuery::AverageEffect(q) => {
                    selected == Some(EstimatorId::Aipw)
                        && q.target_population == antecedent_core::TargetPopulation::AllObserved
                        && (graph_class == GraphClass::Dag
                            || tiered.is_some_and(|b| {
                                b.within_tier == antecedent_graph::WithinTier::CoDetermined
                            }))
                }
                CausalQuery::ConditionalEffect(q) => {
                    q.inner.target_population == antecedent_core::TargetPopulation::AllObserved
                        && matches!(
                            graph_class,
                            GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag
                        )
                }
                CausalQuery::Response(q) => {
                    selected == Some(EstimatorId::CellAipw)
                        && q.target_population == antecedent_core::TargetPopulation::AllObserved
                        && q.temporal.is_none()
                }
                _ => false,
            };
        let supported = bayesian_quantile || frequentist_supported;
        if !supported {
            return Err(CausalError::Unsupported {
                message: "quantiles require Frequentist AllObserved AIPW AverageEffect, binary ConditionalEffect, or cell-AIPW joint response; use prepare + retarget for score-table target weights",
            });
        }
        if *refute != RefuteSuite::None {
            // An omitted suite is a default the study may drop; a suite the caller
            // asked for (directly or through a validator budget) is refused.
            if refute_explicit {
                return Err(CausalError::Unsupported {
                    message: "quantile functionals currently require refute=none; mean-effect refuters do not validate a quantile",
                });
            }
            *refute_default_downgrade = Some(*refute);
            *refute = RefuteSuite::None;
        }
    }
    Ok(())
}

fn choose_estimator(classified: ClassifiedGraph) -> Result<ChosenEstimator, CausalError> {
    if classified.spine.carry.declared.tiered.is_some()
        && classified.spine.carry.declared.identifier.is_some()
    {
        return Err(CausalError::Unsupported {
            message: "TieredBackground selects its own identifier; omit identifier",
        });
    }
    if classified.spine.carry.declared.continuous_cell.is_some() {
        return Err(CausalError::Unsupported {
            message: antecedent_estimate::POINT_CDE_UNLICENSED,
        });
    }
    let identification_cache = cache_cell_aipw(&classified)?;
    refuse_exceedance_grid(&classified)?;
    if let Some(estimator) = classified.spine.carry.declared.estimator {
        refuse_estimator_inference_mismatch(
            &classified.spine.query,
            estimator,
            &classified.spine.carry.settings.inference,
        )?;
    }
    refuse_incompatible_estimator_spec(&classified)?;
    Ok(ChosenEstimator { classified, identification_cache })
}

fn cache_cell_aipw(
    classified: &ClassifiedGraph,
) -> Result<Option<Arc<CachedStaticIdentification>>, CausalError> {
    let carry = &classified.spine.carry;
    let query = &classified.spine.query;
    let graph = &classified.spine.graph;
    let codetermined_joint = carry
        .declared
        .tiered
        .as_ref()
        .is_some_and(|b| b.within_tier == antecedent_graph::WithinTier::CoDetermined)
        && matches!(
            query,
            CausalQuery::Response(q)
                if q.temporal.is_none()
                    && matches!(
                        &q.functional,
                        antecedent_core::ResponseFunctional::InterventionResponse {
                            interventions,
                            ..
                        } if interventions.len() >= 2
                    )
        );
    // A Bayesian joint level on the closure runs response.bayesian over the same
    // generalized adjustment set, so it caches the same tier identification.
    let bayesian_joint = codetermined_joint
        && matches!(carry.settings.inference, InferenceMode::Bayesian(_))
        && classified.selected.is_none_or(|id| id == EstimatorId::ResponseBayesian);
    let cell_aipw = classified.selected == Some(EstimatorId::CellAipw)
        || (classified.selected.is_none() && codetermined_joint);
    if !cell_aipw && !bayesian_joint {
        return Ok(None);
    }
    if let Some(background) = &carry.declared.tiered {
        let CausalQuery::Response(response) = query else {
            return Err(CausalError::Unsupported {
                message: "cell-AIPW on a tiered background requires joint InterventionResponse",
            });
        };
        if matches!(carry.settings.inference, InferenceMode::Bayesian(_)) {
            if !bayesian_joint {
                return Err(CausalError::Unsupported {
                    message: "a Bayesian joint InterventionResponse on a CoDetermined tier \
                              closure runs response.bayesian over the closure's generalized \
                              adjustment set; select no estimator or response.bayesian",
                });
            }
        } else if classified.selected.is_some_and(|id| id != EstimatorId::CellAipw) {
            return Err(CausalError::Unsupported {
                message: "CoDetermined joint cells require estimator cell.aipw",
            });
        }
        let identification = match graph.graph_ref() {
            GraphRef::Admg(admg) => {
                antecedent_identify::identify_tiered_joint_on(background, admg, response)?
            }
            _ => antecedent_identify::identify_tiered_joint(
                background,
                data_schema(&carry.data),
                response,
            )?,
        };
        if !matches!(
            identification.status,
            antecedent_core::IdentificationStatus::NonparametricallyIdentified
                | antecedent_core::IdentificationStatus::PartiallyIdentified
        ) || identification.estimands.is_empty()
        {
            return Err(CausalError::Unsupported {
                message: antecedent_identify::TIERED_JOINT_ADJUSTMENT_REFUSE,
            });
        }
        let estimand = identification.estimands[0].clone();
        return Ok(Some(Arc::new(CachedStaticIdentification { identification, estimand })));
    } else if classified.graph_class != GraphClass::Dag {
        return Err(CausalError::Unsupported {
            message: "cell-AIPW requires a static DAG with one certified common adjustment set",
        });
    }
    Ok(None)
}

fn refuse_exceedance_grid(classified: &ClassifiedGraph) -> Result<(), CausalError> {
    let carry = &classified.spine.carry;
    let query = &classified.spine.query;
    let functional = match query {
        CausalQuery::AverageEffect(q) => Some(&q.outcome_functional),
        CausalQuery::ConditionalEffect(q) => Some(&q.inner.outcome_functional),
        CausalQuery::Response(q) => Some(&q.outcome_functional),
        _ => None,
    };
    if !matches!(functional, Some(antecedent_core::OutcomeFunctional::ExceedanceGrid(_))) {
        return Ok(());
    }
    let score_grid = matches!(carry.settings.inference, InferenceMode::Frequentist)
        && classified.spine.graph_posterior.is_none()
        && match query {
            CausalQuery::AverageEffect(q) => {
                classified.selected == Some(EstimatorId::Aipw)
                    && matches!(q.target_population, antecedent_core::TargetPopulation::AllObserved)
                    && (classified.graph_class == GraphClass::Dag
                        || carry.declared.tiered.as_ref().is_some_and(|b| {
                            b.within_tier == antecedent_graph::WithinTier::CoDetermined
                        }))
            }
            CausalQuery::Response(q) => {
                classified.selected == Some(EstimatorId::CellAipw)
                    && q.temporal.is_none()
                    && (classified.graph_class == GraphClass::Dag
                        || carry.declared.tiered.as_ref().is_some_and(|b| {
                            b.within_tier == antecedent_graph::WithinTier::CoDetermined
                        }))
            }
            CausalQuery::ConditionalEffect(_) => {
                matches!(
                    classified.graph_class,
                    GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag
                )
            }
            _ => false,
        };
    if !score_grid {
        return Err(CausalError::Unsupported {
            message: "full exceedance grids require explicit iid AIPW, cell-AIPW joint response, or Frequentist ConditionalEffect on Dag/Cpdag/Pag",
        });
    }
    Ok(())
}

fn refuse_incompatible_estimator_spec(classified: &ClassifiedGraph) -> Result<(), CausalError> {
    let Some(spec) = &classified.spine.carry.declared.estimator_spec else {
        return Ok(());
    };
    let query = &classified.spine.query;
    let inference = &classified.spine.carry.settings.inference;
    let bayesian = matches!(inference, InferenceMode::Bayesian(_));
    let expected = match query {
        CausalQuery::ConditionalEffect(_) => Some(if bayesian {
            EstimatorId::BayesianConditional
        } else {
            EstimatorId::ConditionalLinearAdjustment
        }),
        CausalQuery::Response(q) => Some(if q.temporal.is_some() {
            if bayesian {
                EstimatorId::TemporalResponseBayesian
            } else {
                EstimatorId::TemporalResponseGcomp
            }
        } else if bayesian
            && !matches!(
                &q.functional,
                antecedent_core::ResponseFunctional::AverageDerivative { .. }
                    | antecedent_core::ResponseFunctional::PointDerivative { .. }
                    | antecedent_core::ResponseFunctional::DirectionalDerivative { .. }
                    | antecedent_core::ResponseFunctional::Jacobian { .. }
            )
        {
            EstimatorId::ResponseBayesian
        } else {
            EstimatorId::default_for_response(&q.functional)
        }),
        CausalQuery::Mediation(_) if classified.graph_class == GraphClass::TemporalDag => {
            Some(if bayesian {
                EstimatorId::BayesianTemporalMediation
            } else {
                EstimatorId::TemporalMediation
            })
        }
        _ => None,
    };
    if let Some(expected) = expected {
        let cell_aipw_ok = spec.id() == EstimatorId::CellAipw
            && matches!(
                query,
                CausalQuery::Response(q)
                    if q.temporal.is_none()
                        && matches!(
                            q.functional,
                            antecedent_core::ResponseFunctional::InterventionResponse { .. }
                        )
            );
        let admg_functional_ok = spec.id() == EstimatorId::FunctionalEffect
            && matches!(classified.graph_class, GraphClass::Admg | GraphClass::TemporalAdmg)
            && matches!(
                query,
                CausalQuery::Response(q)
                    if q.temporal.is_none()
                        && matches!(
                            q.functional,
                            antecedent_core::ResponseFunctional::InterventionResponse { .. }
                                | antecedent_core::ResponseFunctional::MeanCurve { .. }
                        )
            );
        let admg_conditional_ok = spec.id() == EstimatorId::FunctionalEffect
            && classified.graph_class == GraphClass::Admg
            && matches!(query, CausalQuery::ConditionalEffect(_));
        let basis_gcomp_ok = spec.id() == EstimatorId::BayesianBasisGcomp
            && bayesian
            && classified.graph_class == GraphClass::Dag
            && matches!(query, CausalQuery::AverageEffect(_) | CausalQuery::ConditionalEffect(_));
        if spec.id() != expected
            && !cell_aipw_ok
            && !admg_functional_ok
            && !admg_conditional_ok
            && !basis_gcomp_ok
        {
            return Err(crate::compile_reason!(
                "strategy_incompatible",
                "query and inference require estimator {}; got {}",
                expected.as_str(),
                spec.id().as_str()
            ));
        }
    }
    Ok(())
}

fn settle_refute(mut chosen: ChosenEstimator) -> Result<SettledRefute, CausalError> {
    let structure = chosen.classified.structure;
    let matrix_class = chosen.classified.matrix_class;
    {
        let Spine { carry, graph, query, graph_posterior: _ } = &mut chosen.classified.spine;
        if !carry.settings.refute_explicit && carry.refute_default_downgrade.is_none() {
            let support_query = carry
                .lag_zero_average_effect
                .clone()
                .map_or_else(|| query.clone(), CausalQuery::AverageEffect);
            let query = &support_query;
            let requested = crate::support::support_cell_named(
                query,
                matrix_class,
                structure,
                &carry.settings.inference,
                carry.settings.refute,
            );
            let without_validation = crate::support::support_cell_named(
                query,
                matrix_class,
                structure,
                &carry.settings.inference,
                RefuteSuite::None,
            );
            if requested.is_some_and(|cell| {
                matches!(
                    crate::support::classify(cell),
                    CellStatus::NotApplicable { .. } | CellStatus::Refused
                )
            }) && without_validation.is_some_and(|cell| {
                matches!(
                    crate::support::classify(cell),
                    CellStatus::Licensed | CellStatus::Allowlisted { .. }
                ) || crate::support::executes_unlicensed(cell)
            }) {
                carry.refute_default_downgrade = Some(carry.settings.refute);
                carry.settings.refute = RefuteSuite::None;
            }
        }
        if carry
            .declared
            .tiered
            .as_ref()
            .is_some_and(|b| b.within_tier == antecedent_graph::WithinTier::Unknown)
        {
            match query {
                CausalQuery::ConditionalEffect(_) => {
                    return Err(CausalError::Unsupported {
                        message: "Unknown-tier ConditionalEffect is not licensed; the \
                              two-scenario envelope is AverageEffect only",
                    });
                }
                CausalQuery::Response(q) if q.functional.treatment_ids().len() <= 1 => {
                    return Err(CausalError::Unsupported {
                        message: "Unknown-tier single-treatment Response is not licensed; the \
                              two-scenario envelope is AverageEffect only",
                    });
                }
                _ => {}
            }
        }
        if let DataInput::Panel(panel) = &carry.data {
            refuse_unlicensed_panel_route(
                query,
                graph.class(),
                &carry.settings.inference,
                panel,
                carry.declared.split.as_ref(),
            )?;
        }
        if carry.declared.class_prior.is_some()
            && matches!(carry.settings.inference, InferenceMode::Frequentist)
        {
            return Err(CausalError::Unsupported {
                message: "class_prior is a structural probability over class members and \
                      requires Bayesian inference; Frequentist incomplete-class Pulse \
                      keeps enumeration-weighted ATE without treating those weights as \
                      probabilities",
            });
        }
        if carry.declared.class_prior.is_some() && !graph.class().is_incomplete_temporal() {
            return Err(CausalError::Unsupported {
                message: "class_prior requires an incomplete temporal graph class",
            });
        }
    }
    Ok(SettledRefute(chosen))
}

fn classify_support(settled: SettledRefute) -> Result<ClassifiedSupport, CausalError> {
    let matrix_class = settled.0.classified.matrix_class;
    let structure = settled.0.classified.structure;
    let support_status = {
        let spine = &settled.0.classified.spine;
        let carry = &spine.carry;
        let query = &spine.query;
        if !carry.inspect_only {
            crate::support::refuse_undeclared_off_axis(query)?;
        }
        let requested = carry.lag_zero_average_effect.clone().map(CausalQuery::AverageEffect);
        let mut support_status = if let Some(cell) = crate::support::support_cell_named(
            requested.as_ref().unwrap_or(query),
            matrix_class,
            structure,
            &carry.settings.inference,
            carry.settings.refute,
        ) {
            // Geometric n/a / refused still refuse the build (except inspect).
            // Geometric licensed is reclassified once an estimator is known so
            // matching / IV / RD / forest / two-stage front-door cannot inherit
            // a license whose evidence never ran them.
            match crate::support::classify(cell) {
                CellStatus::NotApplicable { .. }
                | CellStatus::Refused
                | CellStatus::Allowlisted { .. } => {
                    if carry.inspect_only {
                        Some(crate::support::classify(cell))
                    } else if crate::support::executes_unlicensed(cell) {
                        // A refused cell its closed rule marks as executing: it runs and
                        // the result reports `refused`, so no license is implied.
                        Some(CellStatus::Refused)
                    } else {
                        Some(crate::support::refuse_if_not_applicable(cell)?)
                    }
                }
                CellStatus::Licensed => {
                    // `inference()` auto-binds `BayesianGcomp` for every static
                    // AverageEffect, including classes whose compiler route is
                    // identifier-native (`general.id` + `functional.effect` on
                    // Admg). Only a caller-selected estimator (`estimator_spec`)
                    // is classified as bound; otherwise the cell's
                    // licensed_routes default is what will actually run.
                    let estimator = if carry.declared.estimator_spec.is_some() {
                        carry.declared.estimator
                    } else {
                        crate::support::licensed_route_estimator(cell).or(carry.declared.estimator)
                    };
                    match estimator {
                        Some(est) => Some(crate::support::classify_estimator(cell, est)),
                        None => Some(CellStatus::Refused),
                    }
                }
            }
        } else {
            None
        };
        // The matrix's InterferenceQuery/Frequentist cell licenses only Bernoulli +
        // NeighborCount and its Young-bound calibration. A cluster total effect has
        // separate execution evidence but no matrix cell or interval calibration yet. A cell
        // the closed rules already mark as executing unlicensed keeps its refused status.
        if support_status != Some(CellStatus::Refused)
            && matches!(
                query,
                CausalQuery::Interference(q)
                    if matches!(
                        q.assignment,
                        antecedent_core::AssignmentDesign::ClusterRandomization { .. }
                            | antecedent_core::AssignmentDesign::TwoStageSaturation { .. }
                            | antecedent_core::AssignmentDesign::ObservedExposure { .. }
                    )
            )
        {
            support_status = None;
        }
        // The untempered posterior of a declared white-innovation premise is a different
        // construction from the licensed tempered posterior of the same cell: it is unlicensed
        // and its calibration is unmeasured, so the cell's license does not carry to it.
        if carry.settings.retain_joint_horizons
            && matches!(carry.settings.inference, InferenceMode::Bayesian(_))
            && support_status == Some(CellStatus::Licensed)
        {
            // The block-weight posterior over the surface replaces the licensed conjugate fits.
            support_status = Some(CellStatus::Refused);
        }
        if matches!(&carry.settings.inference, InferenceMode::Bayesian(cfg) if cfg.white_innovation.is_some())
            && support_status == Some(CellStatus::Licensed)
        {
            support_status = Some(CellStatus::Refused);
        }
        // The Frequentist counterfactual and attribution cells are licensed as point
        // values. A requested mechanism-refit bootstrap interval is a construction no
        // coverage record measures: it executes, and the result is unlicensed.
        if matches!(carry.settings.inference, InferenceMode::Frequentist)
            && carry.settings.bootstrap_replicates > 0
            && matrix_class == "Dag"
            && matches!(
                query,
                CausalQuery::Counterfactual(_)
                    | CausalQuery::AnomalyAttribution(_)
                    | CausalQuery::ChangeAttribution(_)
            )
            && support_status == Some(CellStatus::Licensed)
        {
            support_status = Some(CellStatus::Refused);
        }
        require_licensed_bayesian_estimator(carry, support_status)?;
        support_status
    };
    Ok(ClassifiedSupport { settled, support_status })
}

fn require_licensed_bayesian_estimator(
    carry: &Carry,
    support_status: Option<CellStatus>,
) -> Result<(), CausalError> {
    if !carry.inspect_only
        && matches!(
            carry.declared.estimator,
            Some(
                EstimatorId::BayesianIvJointLinear
                    | EstimatorId::BayesianRdLocalLinear
                    | EstimatorId::BayesianBasisGcomp
                    | EstimatorId::BayesianRobustAte
            )
        )
        && support_status != Some(CellStatus::Licensed)
    {
        return Err(CausalError::Unsupported {
            message: "selected Bayesian estimator is not licensed for this support cell",
        });
    }
    Ok(())
}

fn bind_exposure(settled: ClassifiedSupport) -> Result<BoundExposure, CausalError> {
    let ClassifiedSupport { settled, support_status } = settled;
    let spine = &settled.0.classified.spine;
    let carry = &spine.carry;
    let (selection_diagram, transport_trial, interference) = match &spine.query {
        CausalQuery::Transport(transport) => {
            if !carry.inspect_only {
                refuse_unlicensed_transport(transport)?;
            }
            if carry.exposure.interference.is_some() {
                return Err(CausalError::Unsupported {
                    message: "interference network is not used by TransportQuery",
                });
            }
            // A DAG is the ADMG with no latent confounding: the selection diagram of a DAG
            // is its ADMG embedding with the declared selection nodes.
            let admg = if spine.graph_posterior.is_some()
                || matches!(spine.graph.class(), GraphClass::Cpdag | GraphClass::Pag)
            {
                // A graph posterior, a Cpdag and a Pag name no single graph: every atom or
                // class member is run through sID on its own selection diagram when it is
                // identified. The study keeps the declared selection targets on an edgeless
                // diagram over the data's variables.
                let nodes = u32::try_from(data_schema(&carry.data).len()).map_err(|_| {
                    CausalError::Compile {
                        message: "too many variables for a graph-posterior selection diagram"
                            .into(),
                    }
                })?;
                antecedent_graph::Admg::with_variables(nodes)
            } else {
                match spine.graph.graph_ref() {
                    GraphRef::Dag(dag) => {
                        Some(antecedent_graph::Admg::from_dag(dag).map_err(CausalError::rejected)?)
                    }
                    other => other.projected_admg()?,
                }
                .ok_or(CausalError::Unsupported {
                    message: "TransportQuery requires a supplied Dag, Admg or TemporalAdmg",
                })?
            };
            let targets = carry.exposure.selection_targets.clone().unwrap_or_default();
            let diagram = antecedent_graph::SelectionDiagram::try_new(admg.clone(), targets)
                .map_err(CausalError::rejected)?;
            let trial = carry.exposure.transport_trial.clone().ok_or(CausalError::Unsupported {
                message: "TransportQuery requires StudyBuilder::transport_trial",
            })?;
            (Some(diagram), Some(trial), None)
        }
        CausalQuery::Interference(design) => {
            if !carry.inspect_only {
                refuse_unlicensed_interference(design)?;
            }
            if carry.exposure.transport_trial.is_some()
                || carry.exposure.selection_targets.is_some()
            {
                return Err(CausalError::Unsupported {
                    message: "transport trial columns are not used by InterferenceQuery",
                });
            }
            let spec = carry.exposure.interference.clone().ok_or(CausalError::Unsupported {
                message: "InterferenceQuery requires StudyBuilder::interference",
            })?;
            let DataInput::Tabular(units) = &carry.data else {
                return Err(CausalError::Unsupported {
                    message: "InterferenceQuery requires tabular unit data",
                });
            };
            if spec.network.units().row_count() != units.row_count()
                || spec.assignment.len() != units.row_count()
            {
                return Err(CausalError::Compile {
                    message: "interference network, assignment, and unit table row counts \
                              must match"
                        .into(),
                });
            }
            (None, None, Some(spec.bound_to(units)?))
        }
        _ if carry.exposure.transport_trial.is_some()
            || carry.exposure.selection_targets.is_some()
            || carry.exposure.interference.is_some() =>
        {
            return Err(CausalError::Unsupported {
                message: "selection_targets / transport_trial / interference are only valid \
                          for TransportQuery or InterferenceQuery",
            });
        }
        _ => (None, None, None),
    };
    Ok(BoundExposure {
        support: ClassifiedSupport { settled, support_status },
        selection_diagram,
        transport_trial,
        interference,
    })
}

fn into_study(exposure: BoundExposure) -> Study {
    let BoundExposure { support, selection_diagram, transport_trial, interference } = exposure;
    let ClassifiedSupport { settled, support_status } = support;
    let SettledRefute(ChosenEstimator { classified, identification_cache }) = settled;
    let ClassifiedGraph { spine, structure, graph_class: _, matrix_class: _, selected: _ } =
        classified;
    let Spine { carry, graph, graph_posterior, query } = spine;
    let Carry {
        inspect_only: _,
        data,
        declared,
        exposure: _,
        settings,
        execution,
        refute_default_downgrade,
        latency_bootstrap_not_applied,
        lag_zero_average_effect,
    } = carry;
    Study {
        config: Arc::new(StudyConfig {
            refute_default_downgrade,
            latency_bootstrap_not_applied,
            estimator_spec_identity: declared
                .estimator_spec
                .as_ref()
                .map(super::super::contract_identity::estimator_spec_identity),
            response_options: declared.response_options,
            observation_options: declared.observation_options,
            draw_controls: declared.draw_controls,
            observation_delayed_entry: declared.observation_delayed_entry,
            rd: declared.rd,
            stage_sink: execution.stage_sink,
            continuous_cell: declared.continuous_cell,
            selection_diagram,
            transport_trial,
            discovery: None,
            mechanisms: None,
            posterior_provenance: None,
            lag_zero_average_effect,
        }),
        data,
        graph: Arc::new(graph),
        graph_posterior,
        structure_source: structure,
        support_status,
        query,
        declared: StudyDeclarations {
            class_prior: declared.class_prior,
            max_completions: declared.max_completions,
            split: declared.split,
            identifier: declared.identifier,
            estimator: declared.estimator,
            estimator_spec: declared.estimator_spec.map(Arc::new),
            overlap_policy: declared.overlap_policy,
            population_registry: declared.population_registry,
            participants: declared.participants,
            missing: declared.missing,
            sampling_design: declared.sampling_design,
            custom_validators: declared.custom_validators,
            tiered: declared.tiered,
        },
        settings: StudyInference {
            inference: settings.inference,
            refute: settings.refute,
            bootstrap_replicates: settings.bootstrap_replicates,
            retain_joint_horizons: settings.retain_joint_horizons,
        },
        latency_mode: execution.latency_mode,
        identification: identification_cache.map(IdentificationCache::Static),
        shared_batch_design: None,
        interference,
    }
}

#[cfg(test)]
mod design_check_tests {
    use antecedent_core::{AverageEffectQuery, CausalQuery, VariableId};

    use super::refuse_estimator_design_mismatch;
    use crate::strategy_table::EstimatorId;

    fn average_effect() -> CausalQuery {
        CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ))
    }

    #[test]
    fn a_query_with_no_estimator_selected_is_never_refused() {
        // Acceptance companion to the refusal test below: `Ok(())` is the whole result.
        assert!(refuse_estimator_design_mismatch(&average_effect(), None).is_ok());
    }

    #[test]
    fn an_estimator_reserved_to_a_design_query_is_refused_on_another_query() {
        for estimator in [
            EstimatorId::IvBoundsSharp,
            EstimatorId::BayesianPanelDid,
            EstimatorId::BayesianBinaryPolicyValue,
            EstimatorId::TemporalDesignExcursion,
            EstimatorId::BayesianPiecewiseSurvival,
            EstimatorId::PeriodHistoryMatching,
            EstimatorId::SequentialCrossfitContrastWald,
            EstimatorId::BayesianSequentialRegimeValue,
        ] {
            assert!(
                refuse_estimator_design_mismatch(&average_effect(), Some(estimator)).is_err(),
                "{estimator:?} answers a design query only"
            );
        }
    }

    #[test]
    fn an_estimator_without_a_design_query_is_not_refused() {
        // Acceptance companion to the refusal test above: a general estimator on an average
        // effect is admitted; `Ok(())` is the whole result.
        assert!(
            refuse_estimator_design_mismatch(&average_effect(), Some(EstimatorId::Aipw)).is_ok()
        );
    }
}
