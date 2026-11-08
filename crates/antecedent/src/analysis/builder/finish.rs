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
use crate::strategy_table::{
    EstimatorId, IdentifierId, Site, estimator_targets, estimator_validation,
};
use crate::support::{CellStatus, StructureSource};

use super::super::execute::{Study, StudyConfig, StudyDeclarations, StudyInference};
use super::super::latency::{ResolvedLatencyBudget, refuse_non_report_hmc};
use super::super::prepared::{CachedStaticIdentification, IdentificationCache};
use super::super::query_support::is_graphless_design;
use super::settings::{Declarations, ExecutionSettings, ExposureInputs, InferenceSettings};
use super::{
    DataInput, InterferenceSpec, RefuteSuite, TransportTrialSpec, custom_validators_apply,
    data_schema, estimator_setting_conflict, omitted_bootstrap_resamples, population_estimable,
    refuse_estimator_inference_mismatch, refuse_unsupported_draw_controls,
    refuse_unsupported_interference, refuse_unsupported_likelihood, refuse_unsupported_panel_route,
    refuse_unsupported_transport, stub_accepted_graph_for, tiered_graph_conflict,
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

/// Refute phase: default-suite downgrade, then unknown-tier, panel, and class-prior supports.
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
    // Entries a failed earlier build on this thread left behind belong to no study.
    drop(antecedent_core::support_gate::drain_unscoped());
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
        // The shared admission holds the selected estimator to the declared dependence; an
        // estimator selected by id alone (no spec, as a batch does) is resolved the same way.
        match (declared.estimator_spec.as_mut(), declared.estimator) {
            (Some(spec), _) => spec.bind_sampling_design(design)?,
            (None, Some(id)) => crate::estimator_spec::ensure_dependence_honoured(
                &EstimatorSpec::Default(id),
                design,
            )?,
            (None, None) => {}
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
    if let Some(spec) = &input.declared.estimator_spec
        && spec.is_configured()
    {
        if input.settings.bootstrap_explicit {
            return Err(estimator_setting_conflict("bootstrap_replicates"));
        }
        if input.declared.overlap_policy.is_some() {
            return Err(estimator_setting_conflict("overlap_policy"));
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
    let requested = query.ok_or(CausalError::Missing { field: "query" })?;
    let query = lag_zero_average_effect(requested.clone(), &mut carry, &graph)?;
    refuse_estimator_target_mismatch(
        &requested,
        &query,
        &carry,
        &graph,
        graph_posterior.is_some(),
    )?;
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
/// `AverageEffect` cell (executing, unsupported) and every result discloses the reading.
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
    // The static search engines test and score rows as independent draws; a bootstrap-stability
    // search reads the declared design itself when it resamples, and a series search reads a
    // dependent series by construction.
    if matches!(
        query,
        CausalQuery::Discover(search)
            if !matches!(search.method, antecedent_core::DiscoveryMethod::BootstrapStability { .. })
    ) && matches!(carry.data, DataInput::Tabular(_))
        && matches!(
            carry.declared.sampling_design,
            Some(
                antecedent_core::DeclaredSamplingDesign::ClusteredPanel
                    | antecedent_core::DeclaredSamplingDesign::DependentTimeSeries
            )
        )
    {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "a static structure search tests and scores rows as independent draws and reads no \
             cluster label or lag, so a declared clustered-panel or dependent-time-series design \
             would be silently ignored and its edges over-selected. Next attempt: declare \
             independent subject trajectories if the rows are independent, or search the series \
             with a lagged method such as PCMCI"
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

/// Refuses an explicitly selected estimator that does not estimate the query's quantity.
///
/// What each estimator estimates is its declared target
/// ([`crate::strategy_table::EstimatorTarget`]); a mismatch is a scientific refusal in every
/// build and runs before the support gate. The legacy inference refusal runs first so a
/// matching rule's typed posterior boundary keeps its premise and next attempt. The query a
/// lag-0 average effect was compiled from is read as the temporal pulse it became, and the
/// average effect it was asked as: the estimator may answer either.
fn refuse_estimator_target_mismatch(
    requested: &CausalQuery,
    query: &CausalQuery,
    carry: &Carry,
    graph: &AcceptedGraph,
    graph_posterior: bool,
) -> Result<(), CausalError> {
    let Some(estimator) = carry.declared.estimator else {
        return Ok(());
    };
    refuse_estimator_inference_mismatch(query, estimator, &carry.settings.inference)?;
    // The estimators the builder binds itself (`query`, `inference`) are placeholders for the
    // route's own default, not a selection: the route replaces them, so they are not checked.
    if carry.declared.estimator_spec.is_none()
        && matches!(
            estimator,
            EstimatorId::BayesianGcomp
                | EstimatorId::BayesianBasisGcomp
                | EstimatorId::TemporalLinearAdjustment
        )
    {
        return Ok(());
    }
    // The site is a premise of the question only for an estimator the caller selected: a
    // placeholder the builder bound is not the caller's selection. A tier background reads as
    // a DAG identified by generalized adjustment; a graph posterior keeps its atom class.
    let site = |query: &CausalQuery| {
        carry.declared.estimator_spec.is_some().then(|| {
            let tier = carry.declared.tiered.is_some();
            let class = if tier {
                GraphClass::Dag
            } else if graph_posterior {
                graph.class()
            } else {
                crate::support::effective_graph_class(graph, query)
            };
            Site { class, tier, posterior: graph_posterior }
        })
    };
    let verdict = estimator_targets(query, &carry.settings.inference, site(query), estimator);
    match verdict {
        Err(_) if requested != query => {
            estimator_targets(requested, &carry.settings.inference, site(requested), estimator)?;
        }
        other => other?,
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
    if let (InferenceMode::Bayesian(config), false) = (&carry.settings.inference, reads_own_model)
        && (config.backend != antecedent_estimate::BayesianBackendKind::ConjugateGaussian
            || config.has_transfer())
    {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "the Bayesian panel DiD, synthetic control and local ratio fit their own \
                 built-in Gaussian models; use the conjugate backend with prior_scale and \
                 n_draws only, and no transferred or custom prior"
        ));
    }
    if carry.declared.identifier.is_some_and(|id| id != IdentifierId::for_design_query(query))
        || carry.declared.estimator.is_some_and(|id| {
            !matches!(
                id,
                EstimatorId::RandomizedHt
                    | EstimatorId::RandomizedFixedCupedHt
                    | EstimatorId::RandomizedClusterFixedCuped
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
    if (matches!(
        carry.declared.estimator,
        Some(EstimatorId::RdSharp | EstimatorId::BayesianRdLocalLinear)
    ) || carry.declared.identifier == Some(IdentifierId::RdSharp))
        && let (Some(rd), CausalQuery::AverageEffect(average)) =
            (carry.declared.rd.as_ref(), &mut query)
        && average.target_population == antecedent_core::TargetPopulation::AllObserved
    {
        average.target_population =
            antecedent_core::TargetPopulation::local_at_cutoff(rd.running_variable, rd.cutoff);
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
    require_declared_dependence_read(&carry, &query, graph.class())?;
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
#[allow(
    clippy::too_many_lines,
    reason = "the complete route keeps related contract checks together"
)]
fn require_declared_dependence_read(
    carry: &Carry,
    query: &CausalQuery,
    graph_class: GraphClass,
) -> Result<(), CausalError> {
    use antecedent_core::DeclaredSamplingDesign as Design;
    use antecedent_estimate::AnalyticSeKind as Kind;
    let joint_cell = carry.declared.estimator == Some(EstimatorId::CellAipw)
        && matches!(carry.data, DataInput::Tabular(_))
        && matches!(
            query,
            CausalQuery::Response(response)
                if matches!(
                    response.functional,
                    antecedent_core::ResponseFunctional::InterventionResponse { .. }
                )
        );
    // The trial-to-target contrast reads the labels or the lag in its own error and bootstrap.
    let transport =
        matches!(query, CausalQuery::Transport(_)) && matches!(carry.data, DataInput::Tabular(_));
    if let Some(dependence) = carry.declared.score_dependence.as_ref() {
        // Cluster labels or a serial kernel are read by two routes only: the joint-cell
        // response builds its variance from the influence values of its score table, and the
        // trial-to-target transport contrast from its per-row influence.
        if !joint_cell && !transport {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "score_dependence is read only by the joint-cell response (a tabular \
                 InterventionResponse with the CellAipw estimator) and by a trial-to-target \
                 TransportQuery, which build a cluster-robust or Newey-West variance from \
                 their influence values; this route \
                 would silently ignore the declared labels or kernel. Next attempt: configure \
                 an average-effect estimator whose standard error is the cluster sandwich or \
                 Newey-West with its own labels, or drop the declaration"
            ));
        }
        match (dependence, carry.declared.sampling_design) {
            (antecedent_estimate::ScoreDependence::Clusters(_), Some(Design::ClusteredPanel))
            | (
                antecedent_estimate::ScoreDependence::NeweyWest { .. },
                Some(Design::DependentTimeSeries),
            ) => {}
            (antecedent_estimate::ScoreDependence::Clusters(_), _) => {
                return Err(crate::unsupported_reason!(
                    "option_not_applicable",
                    "the declared score dependence and the declared sampling design disagree: \
                     cluster labels go with a clustered-panel design. Next attempt: declare \
                     the clustered-panel sampling_design beside the labels, or drop them"
                ));
            }
            (antecedent_estimate::ScoreDependence::NeweyWest { .. }, _) => {
                return Err(crate::unsupported_reason!(
                    "option_not_applicable",
                    "the declared score dependence and the declared sampling design disagree: \
                     a Newey-West lag goes with a dependent-time-series design. Next attempt: \
                     declare the dependent-time-series sampling_design beside the lag, or drop it"
                ));
            }
        }
        if joint_cell && carry.settings.bootstrap_replicates > 0 {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "the joint cell publishes an influence-function interval and does not \
                 bootstrap, so a bootstrap requested beside the declared dependence would be \
                 silently ignored. Next attempt: set bootstrap_replicates(0)"
            ));
        }
        return Ok(());
    }
    let Some(design @ (Design::ClusteredPanel | Design::DependentTimeSeries)) =
        carry.declared.sampling_design
    else {
        return Ok(());
    };
    // The policy value and the continuous-dose response publish an influence-function or
    // independent-row variance and interval over independent evaluation subjects, whatever the
    // bootstrap setting, and nothing declares clusters or a lag to them: a declared dependence
    // they cannot read is refused rather than answered with that interval.
    if matches!(query, CausalQuery::PolicyValue(_) | CausalQuery::ContinuousDoseResponse(_))
        && matches!(carry.data, DataInput::Tabular(_))
    {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "this route publishes an influence-function or independent-row interval over \
             independent evaluation subjects and reads no cluster or serial dependence, so the \
             declared clustered-panel or dependent-time-series design would be silently \
             ignored. Next attempt: declare independent subject trajectories if the evaluation \
             subjects are independent, or estimate an average effect with a cluster or \
             Newey-West standard error"
        ));
    }
    // A Bayesian static route conditions on a likelihood whose rows are independent given the
    // model, so its posterior would be too narrow for a declared clustered panel or dependent
    // series; no static Bayesian likelihood reads a cluster label or a lag.
    if matches!(carry.data, DataInput::Tabular(_))
        && carry.settings.inference.is_bayesian()
        && matches!(
            query,
            CausalQuery::AverageEffect(_)
                | CausalQuery::ConditionalEffect(_)
                | CausalQuery::Response(_)
                | CausalQuery::Mediation(_)
                | CausalQuery::Counterfactual(_)
        )
    {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "a Bayesian static route conditions on a likelihood whose rows are independent given \
             the model and reads no cluster label or lag, so a declared clustered-panel or \
             dependent-time-series design would be silently ignored and the posterior too \
             narrow. Next attempt: declare independent subject trajectories if the rows are \
             independent, or use the frequentist average effect with a cluster or Newey-West \
             standard error"
        ));
    }
    // The sharp-discontinuity estimator resamples and sandwiches rows one by one: neither its
    // bootstrap nor its heteroskedasticity-robust error reads a cluster label or a lag.
    if carry.declared.rd.is_some()
        && matches!(query, CausalQuery::AverageEffect(_))
        && matches!(carry.data, DataInput::Tabular(_))
    {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "the sharp regression-discontinuity estimator resamples and sandwiches rows one by \
             one and reads no cluster label or lag, so a declared clustered-panel or \
             dependent-time-series design would be silently ignored and its error too small. \
             Next attempt: declare independent subject trajectories if the rows are independent, \
             or use the fuzzy local-polynomial ratio with a calendar-time running variable for a \
             dependent series"
        ));
    }
    // The interference contrast is design-based over one outcome per unit of a fixed network:
    // it reads the cluster labels only through a clustered assignment design and no lag at all.
    if let CausalQuery::Interference(interference) = query {
        let clustered_assignment = matches!(
            interference.assignment,
            antecedent_core::AssignmentDesign::ClusterRandomization { .. }
                | antecedent_core::AssignmentDesign::TwoStageSaturation { .. }
        );
        if design == Design::DependentTimeSeries || !clustered_assignment {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "the interference contrast conditions on the realized assignment of a fixed \
                 network and reads one outcome per unit, with cluster labels only from a \
                 clustered assignment design and no lag, so the declared clustered-panel or \
                 dependent-time-series design would be silently ignored. Next attempt: declare \
                 independent subject trajectories, use a cluster-randomized or two-stage \
                 assignment design for clusters, or use the unit-period interference design \
                 whose cluster-robust contrasts read serial dependence"
            ));
        }
        return Ok(());
    }
    let reads_estimator_dependence = match query {
        CausalQuery::AverageEffect(_) => true,
        CausalQuery::ConditionalEffect(_) | CausalQuery::Response(_) => false,
        _ => return Ok(()),
    };
    if !matches!(carry.data, DataInput::Tabular(_)) || carry.settings.inference.is_bayesian() {
        return Ok(());
    }
    // A cross-fitted estimator with declared independence units reads the declared clustered
    // panel in its own whole-cluster cross-fit and cluster sandwich (the shared admission has
    // already refused the designs and configurations it does not read), so neither a
    // standard-error kind nor a bootstrap setting has to carry it.
    if carry
        .declared
        .estimator_spec
        .as_ref()
        .is_some_and(EstimatorSpec::reads_dependence_through_units)
    {
        return Ok(());
    }
    let inference = carry.declared.estimator_spec.as_ref().and_then(EstimatorSpec::inference_spec);
    let replicates =
        inference.map_or(carry.settings.bootstrap_replicates, |i| i.bootstrap_replicates);
    // The default kernel-smoothed curve and plug-in intervention response of a DAG publish an
    // influence-function interval over independent rows and do not bootstrap: the replicates
    // setting changes nothing in them, so it is not a way to read a declared clustered panel or
    // dependent series.
    let cell_aipw = carry.declared.estimator == Some(EstimatorId::CellAipw);
    if let CausalQuery::Response(response) = query
        && replicates > 0
        && response.temporal.is_none()
        && (cell_aipw
            || (graph_class == GraphClass::Dag
                && carry.declared.estimator_spec.is_none()
                && matches!(
                    response.functional,
                    antecedent_core::ResponseFunctional::MeanCurve { .. }
                        | antecedent_core::ResponseFunctional::InterventionResponse { .. }
                )))
    {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "the default response curve, the intervention response and the joint cell AIPW publish an \
                 influence-function interval over independent rows and do not bootstrap, so a \
                 declared clustered-panel or dependent-time-series design would be silently \
                 ignored whatever the bootstrap setting. Next attempt: declare independent \
                 subject trajectories if the rows are independent, declare the labels or lag \
                 with score_dependence on the joint cell (bootstrap off), or estimate an \
                 average effect with a cluster or Newey-West standard error"
        ));
    }
    if replicates > 0 {
        return Ok(());
    }
    let reads = reads_estimator_dependence
        && inference.is_some_and(|inference| {
            matches!(
                (design, inference.se_kind),
                (
                    Design::ClusteredPanel,
                    Kind::Cluster | Kind::Multiway | Kind::PanelClusterHac { .. }
                ) | (
                    Design::DependentTimeSeries,
                    Kind::NeweyWest { .. } | Kind::PanelClusterHac { .. }
                )
            )
        });
    if reads {
        return Ok(());
    }
    Err(if design == Design::ClusteredPanel {
        crate::unsupported_reason!(
            "option_not_applicable",
            "a clustered-panel design is declared with the bootstrap off, but the analytic error \
             this route publishes treats rows as independent and would be too small. Next \
             attempt: configure an average-effect estimator whose standard error is the cluster \
             sandwich (for example LinearAdjustmentAte::new().with_se_kind(AnalyticSeKind::Cluster) \
             with its cluster ids), or declare independent subject trajectories if the rows are \
             independent"
        )
    } else {
        crate::unsupported_reason!(
            "option_not_applicable",
            "a dependent-time-series design is declared with the bootstrap off, but the analytic \
             error this route publishes treats rows as independent and would be too small. Next \
             attempt: configure an average-effect estimator whose standard error is Newey-West \
             (for example LinearAdjustmentAte::new().with_se_kind(AnalyticSeKind::NeweyWest { lag })), \
             or declare independent subject trajectories if the rows are independent"
        )
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
    // stub edges (an empty Admg looks like a DAG) would relabel supported
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
    refuse_unimplemented_validation(&mut carry)?;
    Ok(ClassifiedGraph {
        spine: Spine { carry, graph, graph_posterior, query },
        structure,
        graph_class,
        matrix_class,
        selected,
    })
}

/// Refuses a validation suite the selected estimator's route never reads.
///
/// An omitted suite is a default the study may drop (and discloses as a downgrade); a suite
/// the caller asked for is refused rather than run as no validation.
fn refuse_unimplemented_validation(carry: &mut Carry) -> Result<(), CausalError> {
    let Some(estimator) = carry.declared.estimator else {
        return Ok(());
    };
    if carry.declared.estimator_spec.is_none()
        && matches!(
            estimator,
            EstimatorId::BayesianGcomp
                | EstimatorId::BayesianBasisGcomp
                | EstimatorId::TemporalLinearAdjustment
        )
    {
        return Ok(());
    }
    let suite = crate::support::validation_axis(carry.settings.refute);
    let verdict = estimator_validation(estimator, suite);
    if verdict.is_ok() {
        return Ok(());
    }
    if carry.settings.refute_explicit {
        verdict?;
    }
    carry.refute_default_downgrade = Some(carry.settings.refute);
    carry.settings.refute = RefuteSuite::None;
    Ok(())
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
                    q.target_population == antecedent_core::TargetPopulation::AllObserved
                        && ((selected == Some(EstimatorId::Aipw)
                            && (graph_class == GraphClass::Dag
                                || tiered.is_some_and(|b| {
                                    b.within_tier == antecedent_graph::WithinTier::CoDetermined
                                })))
                            || (selected == Some(EstimatorId::RandomizedQuantile)
                                && graph_class == GraphClass::Dag))
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
                message: "quantiles require Frequentist AllObserved AIPW or randomized-quantile AverageEffect, binary ConditionalEffect, or cell-AIPW joint response; use prepare + retarget for score-table target weights",
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
            message: antecedent_estimate::POINT_CDE_UNSUPPORTED,
        });
    }
    let identification_cache = cache_cell_aipw(&classified)?;
    refuse_exceedance_grid(&classified)?;
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
            let series = matches!(carry.data, DataInput::Temporal(_));
            let requested = crate::support::support_cell_named_for_data(
                query,
                matrix_class,
                structure,
                &carry.settings.inference,
                carry.settings.refute,
                series,
            );
            let without_validation = crate::support::support_cell_named_for_data(
                query,
                matrix_class,
                structure,
                &carry.settings.inference,
                RefuteSuite::None,
                series,
            );
            if requested.is_some_and(|cell| {
                matches!(
                    crate::support::classify(cell),
                    CellStatus::NotApplicable { .. } | CellStatus::NotSupported
                )
            }) && without_validation
                .is_some_and(|cell| matches!(crate::support::classify(cell), CellStatus::Supported))
            {
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
                        message: "Unknown-tier ConditionalEffect is not supported; the \
                              two-scenario envelope is AverageEffect only",
                    });
                }
                CausalQuery::Response(q) if q.functional.treatment_ids().len() <= 1 => {
                    return Err(CausalError::Unsupported {
                        message: "Unknown-tier single-treatment Response is not supported; the \
                              two-scenario envelope is AverageEffect only",
                    });
                }
                _ => {}
            }
        }
        if let DataInput::Panel(panel) = &carry.data {
            refuse_unsupported_panel_route(
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

#[allow(
    clippy::too_many_lines,
    reason = "one linear classification of the settled refute; splitting it scatters the support rules"
)]
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
        // Evidence the selected route lacks although its cell is supported. Any entry refuses
        // the build as `not_supported`; a development build runs it and records the entries.
        let mut missing_evidence: Vec<String> = Vec::new();
        let requested = carry.lag_zero_average_effect.clone().map(CausalQuery::AverageEffect);
        let mut support_status = if let Some(cell) = crate::support::support_cell_named_for_data(
            requested.as_ref().unwrap_or(query),
            matrix_class,
            structure,
            &carry.settings.inference,
            carry.settings.refute,
            matches!(carry.data, DataInput::Temporal(_)),
        ) {
            // Geometric n/a / refused still refuse the build (except inspect).
            // Geometric supported is reclassified once an estimator is known so
            // matching / IV / RD / forest / two-stage front-door cannot inherit
            // support whose evidence never ran them.
            match crate::support::classify(cell) {
                CellStatus::NotApplicable { .. } | CellStatus::NotSupported => {
                    if carry.inspect_only {
                        Some(crate::support::classify(cell))
                    } else {
                        Some(crate::support::refuse_if_not_applicable(cell)?)
                    }
                }
                CellStatus::Supported => {
                    // `inference()` auto-binds `BayesianGcomp` for every static
                    // AverageEffect, including classes whose compiler route is
                    // identifier-native (`general.id` + `functional.effect` on
                    // Admg). Only a caller-selected estimator (`estimator_spec`)
                    // is classified as bound; otherwise the cell's
                    // supported_routes default is what will actually run.
                    //
                    // A declared identifier that carries its own estimator (general.id
                    // evaluates a functional) replaces the cell's default the same way.
                    let identifier_estimator = carry
                        .declared
                        .identifier
                        .and_then(IdentifierId::static_native_estimator)
                        .filter(|_| {
                            matches!(query, CausalQuery::AverageEffect(_)) && matrix_class == "Dag"
                        });
                    let estimator = if carry.declared.estimator_spec.is_some() {
                        carry.declared.estimator
                    } else if let Some(native) = identifier_estimator {
                        Some(native)
                    } else {
                        crate::support::supported_route_estimator(cell).or(carry.declared.estimator)
                    };
                    if let Some(est) = estimator {
                        let status = crate::support::classify_estimator(cell, est);
                        if status == CellStatus::NotSupported {
                            missing_evidence.push(format!("estimator_evidence:{}", est.as_str()));
                        }
                        Some(status)
                    } else {
                        missing_evidence.push("estimator_evidence:none_selected".to_owned());
                        Some(CellStatus::NotSupported)
                    }
                }
            }
        } else {
            None
        };
        // The matrix's InterferenceQuery/Frequentist cell supports only Bernoulli +
        // NeighborCount and its Young-bound calibration. A cluster total effect has
        // separate execution evidence but no matrix cell or interval calibration yet. A cell
        // the closed rules already mark as executing unsupported keeps its refused status.
        if support_status != Some(CellStatus::NotSupported)
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
        // construction from the supported tempered posterior of the same cell: it is unsupported
        // and its calibration is not_measured, so the cell's support does not carry to it.
        if carry.settings.retain_joint_horizons
            && matches!(carry.settings.inference, InferenceMode::Bayesian(_))
            && support_status == Some(CellStatus::Supported)
            && !construction_checked("joint_horizon_block_weight_posterior")
        {
            // The block-weight posterior over the surface replaces the supported conjugate fits.
            support_status = Some(CellStatus::NotSupported);
            missing_evidence
                .push("construction_evidence:joint_horizon_block_weight_posterior".to_owned());
        }
        if matches!(&carry.settings.inference, InferenceMode::Bayesian(cfg) if cfg.white_innovation.is_some())
            && support_status == Some(CellStatus::Supported)
            && !construction_checked("untempered_white_innovation_posterior")
        {
            support_status = Some(CellStatus::NotSupported);
            missing_evidence
                .push("construction_evidence:untempered_white_innovation_posterior".to_owned());
        }
        // The Frequentist counterfactual and attribution cells are supported as point
        // values. A requested mechanism-refit bootstrap interval is a construction no
        // coverage record measures: it executes, and the result is unsupported.
        if matches!(carry.settings.inference, InferenceMode::Frequentist)
            && carry.settings.bootstrap_replicates > 0
            && matrix_class == "Dag"
            && matches!(
                query,
                CausalQuery::Counterfactual(_)
                    | CausalQuery::AnomalyAttribution(_)
                    | CausalQuery::ChangeAttribution(_)
            )
            && support_status == Some(CellStatus::Supported)
            && !construction_checked("mechanism_refit_bootstrap_interval")
        {
            support_status = Some(CellStatus::NotSupported);
            missing_evidence
                .push("construction_evidence:mechanism_refit_bootstrap_interval".to_owned());
        }
        // The joint cell is supported on independent, unweighted rows. Sampling weights and a
        // declared cluster or serial dependence are read by it (weighted fits, cluster-robust or
        // Newey-West influence variance, against an exact reference) but no coverage record
        // measures those intervals, so the cell's support does not carry to them.
        if support_status == Some(CellStatus::Supported)
            && carry.declared.estimator == Some(EstimatorId::CellAipw)
            && matches!(query, CausalQuery::Response(_))
            && (carry.declared.score_dependence.is_some()
                || matches!(&carry.data, DataInput::Tabular(t) if t.storage().weights().is_some()))
        {
            if construction_checked("weighted_or_dependent_cell_aipw_interval") {
                // The construction has an executed independent check: the cell's interval is
                // supported and calibration only annotates it.
            } else {
                support_status = Some(CellStatus::NotSupported);
                missing_evidence.push(
                    "construction_evidence:weighted_or_dependent_cell_aipw_interval".to_owned(),
                );
            }
        }
        if !carry.inspect_only && !missing_evidence.is_empty() {
            let missing: Vec<&str> = missing_evidence.iter().map(String::as_str).collect();
            antecedent_core::support_gate::require_supported_unscoped(
                "the selected route has no supported evidence record (parity/support_cells.toml)",
                &missing,
            )?;
        }
        require_supported_bayesian_estimator(carry, support_status)?;
        support_status
    };
    Ok(ClassifiedSupport { settled, support_status })
}

/// Constructions whose interval has an executed independent construction check, keyed like the
/// `construction_evidence:*` entries the support gate reports for the ones that do not. Each entry
/// names the test that compares the interval with values computed independently (closed form,
/// exact rational arithmetic, hand-derived formula); a construction absent from this list stays
/// unsupported. The same test is registered as a `construction` fixture on the owning capability
/// record (`estimate.factorized_joint_cells`).
const CONSTRUCTION_CHECKS: &[(&str, &str)] = &[
    (
        "weighted_or_dependent_cell_aipw_interval",
        "crates/antecedent/tests/cells_04_joint_intervention.rs::the_joint_cell_equals_the_exact_weighted_cross_fit_aipw_and_its_sandwich_variances",
    ),
    (
        "mechanism_refit_bootstrap_interval",
        "crates/antecedent/tests/mechanism_refit_bootstrap_construction.rs::the_refit_bootstrap_interval_equals_the_hand_refit_of_the_same_seeded_resamples",
    ),
    (
        "untempered_white_innovation_posterior",
        "crates/antecedent/tests/white_innovation_posterior_construction.rs::the_untempered_posterior_is_the_exact_normal_inverse_gamma_posterior_of_the_lag_window",
    ),
    (
        "joint_horizon_block_weight_posterior",
        "crates/antecedent/tests/joint_horizon_block_weight_construction.rs::the_block_weight_draws_are_the_exp1_block_weighted_refits_up_to_one_dispersion_factor_per_cell",
    ),
];

fn construction_checked(key: &str) -> bool {
    CONSTRUCTION_CHECKS.iter().any(|(checked, _)| *checked == key)
}

fn require_supported_bayesian_estimator(
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
        && support_status != Some(CellStatus::Supported)
    {
        let estimator = carry.declared.estimator.map_or("none", |id| id.as_str());
        let evidence = format!("estimator_evidence:{estimator}");
        antecedent_core::support_gate::require_supported_unscoped(
            "selected Bayesian estimator is not supported for this support cell",
            &[evidence.as_str()],
        )?;
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
                refuse_unsupported_transport(transport)?;
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
                refuse_unsupported_interference(design)?;
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
            data_diagnostics: Some(crate::DataDiagnostics::default()),
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
            would_refuse: antecedent_core::support_gate::drain_unscoped(),
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
            score_dependence: declared.score_dependence,
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
        retain_response_witnesses: false,
        temporal_validation_budget: None,
        interference,
    }
}

#[cfg(test)]
mod design_check_tests {
    use antecedent_core::{AverageEffectQuery, CausalQuery, VariableId};

    use crate::inference::InferenceMode;
    use crate::strategy_table::{EstimatorId, estimator_targets};

    fn average_effect() -> CausalQuery {
        CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ))
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
                estimator_targets(&average_effect(), &InferenceMode::Frequentist, None, estimator)
                    .is_err(),
                "{estimator:?} answers a design query only"
            );
        }
    }

    #[test]
    fn an_estimator_without_a_design_query_is_not_refused() {
        // Acceptance companion to the refusal test above: a general estimator on an average
        // effect is admitted; `Ok(())` is the whole result.
        assert!(
            estimator_targets(
                &average_effect(),
                &InferenceMode::Frequentist,
                None,
                EstimatorId::Aipw,
            )
            .is_ok()
        );
    }

    #[test]
    fn a_bayesian_only_estimator_is_refused_under_frequentist_inference() {
        let refusal = estimator_targets(
            &average_effect(),
            &InferenceMode::Frequentist,
            None,
            EstimatorId::BayesianGcomp,
        )
        .expect_err("a Bayesian estimator does not run under Frequentist inference");
        assert_eq!(refusal.code, "estimator_inference_mismatch");
    }
}
