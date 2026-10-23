//! Ordered phases of `StudyBuilder::finish`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalQuery, PopulationRegistry, VariableId};
use antecedent_data::{DiscoveryEstimationSplit, TableView};
use antecedent_discovery::GraphPosterior;
use antecedent_estimate::{ContinuousResponseOptions, OverlapPolicy};
use antecedent_graph::TieredBackground;
use antecedent_validate::CustomEffectValidator;

use crate::accepted::{AcceptedGraph, GraphClass, GraphRef};
use crate::error::CausalError;
use crate::estimator_spec::EstimatorSpec;
use crate::inference::InferenceMode;
use crate::strategy_table::{EstimatorId, IdentifierId};
use crate::support::{CellStatus, StructureSource};

use super::super::execute::{Study, StudyConfig};
use super::super::latency::{
    ComputeBudget, LatencyMode, ResolvedLatencyBudget, refuse_non_report_hmc,
};
use super::super::prepared::CachedStaticIdentification;
use super::super::query_license::is_graphless_design;
use super::super::stage::StageResultSink;
use super::{
    DataInput, InterferenceSpec, RdConfig, RefuteSuite, TransportTrialSpec,
    custom_validators_apply, data_schema, estimator_setting_conflict, omitted_bootstrap_resamples,
    population_estimable, refuse_estimator_inference_mismatch, refuse_unlicensed_interference,
    refuse_unlicensed_panel_route, refuse_unlicensed_transport, refuse_unsupported_draw_controls,
    refuse_unsupported_likelihood, stub_accepted_graph_for, tiered_graph_conflict,
    validate_schema_binding,
};

/// Builder fields moved into the pipeline, plus the `inspect_only` flag.
#[allow(
    clippy::struct_excessive_bools,
    reason = "inspect_only and caller_graph are independent facts about the call, and refute_explicit, bootstrap_explicit and n_draws_explicit mirror the builder's explicit-versus-defaulted markers one for one"
)]
pub(super) struct FinishInput {
    pub(super) inspect_only: bool,
    pub(super) data: DataInput,
    pub(super) graph: Option<AcceptedGraph>,
    pub(super) graph_posterior: Option<GraphPosterior>,
    pub(super) class_prior: Option<crate::ClassPrior>,
    pub(super) max_completions: Option<usize>,
    pub(super) structure_source: Option<StructureSource>,
    pub(super) query: Option<CausalQuery>,
    pub(super) refute: RefuteSuite,
    pub(super) refute_explicit: bool,
    pub(super) bootstrap_replicates: u32,
    pub(super) bootstrap_explicit: bool,
    pub(super) split: Option<DiscoveryEstimationSplit>,
    pub(super) identifier: Option<IdentifierId>,
    pub(super) estimator: Option<EstimatorId>,
    pub(super) estimator_spec: Option<EstimatorSpec>,
    pub(super) response_options: Option<ContinuousResponseOptions>,
    pub(super) observation_options: antecedent_estimate::ObservationEstimatorOptions,
    pub(super) draw_controls: Option<crate::PosteriorDrawControls>,
    pub(super) observation_delayed_entry: Option<VariableId>,
    pub(super) rd: Option<RdConfig>,
    pub(super) inference: InferenceMode,
    pub(super) n_draws_explicit: bool,
    pub(super) overlap_policy: Option<OverlapPolicy>,
    pub(super) population_registry: Option<PopulationRegistry>,
    pub(super) participants: Option<antecedent_io::ParticipantBindingWire>,
    pub(super) missing: Option<antecedent_data::MissingDeclaration>,
    pub(super) sampling_design: Option<antecedent_core::DeclaredSamplingDesign>,
    pub(super) custom_validators: Vec<Arc<dyn CustomEffectValidator>>,
    pub(super) tiered: Option<TieredBackground>,
    pub(super) caller_graph: bool,
    pub(super) continuous_cell: Option<(VariableId, Arc<[f64]>)>,
    pub(super) latency_mode: Option<LatencyMode>,
    pub(super) compute_budget: ComputeBudget,
    pub(super) stage_sink: Option<Arc<dyn StageResultSink>>,
    pub(super) selection_targets: Option<Arc<[VariableId]>>,
    pub(super) transport_trial: Option<TransportTrialSpec>,
    pub(super) interference: Option<InterferenceSpec>,
}

/// Fields that pass through the phases. A phase writes the fields it decides.
#[allow(
    clippy::struct_excessive_bools,
    reason = "refute_explicit, bootstrap_explicit and n_draws_explicit mirror the builder's explicit-versus-defaulted markers that decide whether a latency tier may override a setting, and inspect_only is the independent inspect-versus-estimate flag"
)]
struct Carry {
    inspect_only: bool,
    data: DataInput,
    class_prior: Option<crate::ClassPrior>,
    max_completions: Option<usize>,
    structure_source: Option<StructureSource>,
    refute: RefuteSuite,
    refute_explicit: bool,
    refute_default_downgrade: Option<RefuteSuite>,
    latency_bootstrap_not_applied: Option<(u32, u32)>,
    bootstrap_replicates: u32,
    bootstrap_explicit: bool,
    split: Option<DiscoveryEstimationSplit>,
    identifier: Option<IdentifierId>,
    estimator: Option<EstimatorId>,
    estimator_spec: Option<EstimatorSpec>,
    response_options: Option<ContinuousResponseOptions>,
    observation_options: antecedent_estimate::ObservationEstimatorOptions,
    draw_controls: Option<crate::PosteriorDrawControls>,
    observation_delayed_entry: Option<VariableId>,
    rd: Option<RdConfig>,
    inference: InferenceMode,
    n_draws_explicit: bool,
    overlap_policy: Option<OverlapPolicy>,
    population_registry: Option<PopulationRegistry>,
    participants: Option<antecedent_io::ParticipantBindingWire>,
    missing: Option<antecedent_data::MissingDeclaration>,
    sampling_design: Option<antecedent_core::DeclaredSamplingDesign>,
    custom_validators: Vec<Arc<dyn CustomEffectValidator>>,
    tiered: Option<TieredBackground>,
    continuous_cell: Option<(VariableId, Arc<[f64]>)>,
    latency_mode: Option<LatencyMode>,
    compute_budget: ComputeBudget,
    stage_sink: Option<Arc<dyn StageResultSink>>,
    selection_targets: Option<Arc<[VariableId]>>,
    transport_trial: Option<TransportTrialSpec>,
    interference: Option<InterferenceSpec>,
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
        class_prior,
        max_completions,
        structure_source,
        query,
        refute,
        refute_explicit,
        bootstrap_replicates,
        bootstrap_explicit,
        split,
        identifier,
        estimator,
        estimator_spec,
        response_options,
        observation_options,
        draw_controls,
        observation_delayed_entry,
        rd,
        inference,
        n_draws_explicit,
        overlap_policy,
        population_registry,
        participants,
        missing,
        sampling_design,
        custom_validators,
        tiered,
        caller_graph: _,
        continuous_cell,
        latency_mode,
        compute_budget,
        stage_sink,
        selection_targets,
        transport_trial,
        interference,
    } = input;
    let mut estimator_spec = estimator_spec;
    if let Some(design) = sampling_design {
        if let Some(spec) = estimator_spec.as_mut() {
            spec.bind_sampling_design(design)?;
        }
    }
    let design_based = query.as_ref().is_some_and(is_graphless_design);
    let discovered = match &query {
        Some(CausalQuery::Discover(search)) => Some(search.output()),
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
            class_prior,
            max_completions,
            structure_source,
            refute,
            refute_explicit,
            refute_default_downgrade: None,
            latency_bootstrap_not_applied: None,
            bootstrap_replicates,
            bootstrap_explicit,
            split,
            identifier,
            estimator,
            estimator_spec,
            response_options,
            observation_options,
            draw_controls,
            observation_delayed_entry,
            rd,
            inference,
            n_draws_explicit,
            overlap_policy,
            population_registry,
            participants,
            missing,
            sampling_design,
            custom_validators,
            tiered,
            continuous_cell,
            latency_mode,
            compute_budget,
            stage_sink,
            selection_targets,
            transport_trial,
            interference,
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
            inference: &input.inference,
            estimator: input.estimator,
            graph_posterior: input.graph_posterior.is_some(),
            sampling_design: input.sampling_design,
        },
        input.missing.as_ref(),
    )?;
    if let Some(spec) = &input.estimator_spec {
        if spec.is_configured() {
            if input.bootstrap_explicit {
                return Err(estimator_setting_conflict("bootstrap_replicates"));
            }
            if input.overlap_policy.is_some() {
                return Err(estimator_setting_conflict("overlap_policy"));
            }
        }
    }
    if let (Some(design), Some(_)) = (input.sampling_design, input.split.as_ref()) {
        // A discovery/estimation split cuts the series into contiguous blocks.
        design.apply_splitting(antecedent_core::SplittingRule::ContiguousBlock).map_err(
            |error| CausalError::Data(antecedent_data::DataError::SamplingDesign(error)),
        )?;
    }
    if input.caller_graph && input.tiered.is_some() {
        return Err(tiered_graph_conflict());
    }
    if input.class_prior.is_some() && input.graph_posterior.is_some() {
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
            let stub = stub_accepted_graph_for(data, gp.n_vars, gp.atom_kind)?;
            (stub, Some(gp))
        }
        (None, None) if design_based => (AcceptedGraph::randomized_trial(data_schema(data)), None),
        (None, None) => return Err(CausalError::Missing { field: "graph" }),
    })
}

fn budget_inference(bound: BoundSchema) -> Result<BudgetedInference, CausalError> {
    let BoundSchema(Spine { mut carry, graph, graph_posterior, query }) = bound;
    let mut refute = carry.refute;
    let mut bootstrap_replicates = carry.bootstrap_replicates;
    // Moved, not cloned, so a boxed external prior keeps its allocation.
    let mut inference = std::mem::replace(&mut carry.inference, InferenceMode::Frequentist);
    let latency_mode = carry.latency_mode;
    let compute_budget = carry.compute_budget;

    if let Some(mode) = latency_mode {
        refuse_non_report_hmc(mode, &inference)?;
        let resolved = ResolvedLatencyBudget::from_mode(mode).with_overrides(compute_budget);
        if !carry.bootstrap_explicit {
            bootstrap_replicates = resolved.bootstrap;
        } else if let Some(b) = compute_budget.bootstrap {
            bootstrap_replicates = b;
        }
        if !carry.refute_explicit {
            refute = resolved.refute;
        } else if let Some(v) = compute_budget.validators {
            refute = v;
        }
        inference = match inference {
            InferenceMode::Bayesian(cfg) => {
                // A budget field wins, then a count the caller set on the config; only
                // a backend-default count is resized by the tier.
                let draws = if carry.n_draws_explicit {
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

    carry.refute = refute;
    carry.bootstrap_replicates = bootstrap_replicates;
    carry.inference = inference;
    Ok(BudgetedInference(Spine { carry, graph, graph_posterior, query }))
}

fn contract_query(budget: BudgetedInference) -> Result<ContractedQuery, CausalError> {
    let BudgetedInference(Spine { mut carry, graph, graph_posterior, query }) = budget;
    let query = query.ok_or(CausalError::Missing { field: "query" })?;
    if matches!(
        query,
        CausalQuery::Discover(_) | CausalQuery::Attribute(_) | CausalQuery::Simulate(_)
    ) {
        refuse_structure_route_options(&query, &carry)?;
        // These routes run no refutation suite and do not resample: the search
        // or the fitted model is the procedure, and its uncertainty is what the
        // engine reports.
        carry.refute = RefuteSuite::None;
        carry.bootstrap_replicates = 0;
    }
    if is_graphless_design(&query) {
        apply_design_contract(&query, &graph, &carry)?;
        // Design-based queries have no refutation suite. Bootstrap stays only for an
        // explicit survival or temporal-design count (the temporal design refuses it where
        // its interval does not resample); every other design-based query reports zero
        // replicates.
        carry.refute = RefuteSuite::None;
        if !matches!(query, CausalQuery::Survival(_) | CausalQuery::TemporalDesign(_))
            || !carry.bootstrap_explicit
        {
            carry.bootstrap_replicates = 0;
        }
        if carry.split.is_some()
            || carry.tiered.is_some()
            || carry.selection_targets.is_some()
            || carry.transport_trial.is_some()
            || carry.interference.is_some()
            || carry.rd.is_some()
            || carry.population_registry.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "design-based queries do not accept discovery splits, graph, transport, interference, RD, or population options",
            });
        }
    }
    Ok(ContractedQuery(Spine { carry, graph, graph_posterior, query }))
}

/// Refuses the options a structure-discovery, attribution or simulation study
/// does not read, so none is accepted and then ignored.
fn refuse_structure_route_options(query: &CausalQuery, carry: &Carry) -> Result<(), CausalError> {
    if (carry.refute_explicit && carry.refute != RefuteSuite::None)
        || !carry.custom_validators.is_empty()
    {
        return Err(crate::unsupported_reason!(
            "validators_not_applicable",
            "structure discovery, attribution and simulation run no refutation suite or custom \
             validator; omit refute and validators"
        ));
    }
    if carry.bootstrap_explicit && carry.bootstrap_replicates != 0 {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "structure discovery, attribution and simulation do not resample; omit \
             bootstrap_replicates"
        ));
    }
    if carry.split.is_some()
        || carry.tiered.is_some()
        || carry.selection_targets.is_some()
        || carry.transport_trial.is_some()
        || carry.interference.is_some()
        || carry.rd.is_some()
        || carry.population_registry.is_some()
        || carry.identifier.is_some()
        || carry.estimator.is_some()
        || carry.estimator_spec.is_some()
    {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "structure discovery, attribution and simulation fix their own procedure; they take \
             no identifier, estimator, split, tier background, transport, interference, RD or \
             population option"
        ));
    }
    if matches!(carry.inference, InferenceMode::Bayesian(_)) {
        return Err(match query {
            CausalQuery::Discover(_) => crate::unsupported_reason!(
                "option_not_applicable",
                "a discovery method declares its own inference mode: a structure posterior \
                 (exact, order, structure, CI-screened or DBN) is Bayesian by its method; leave \
                 the study's inference at its default"
            ),
            _ => crate::unsupported_reason!(
                "option_not_applicable",
                "attribution and simulation on a fitted model report the fitted model's answer; \
                 for the Dirichlet mechanism-refit posterior use AnomalyAttribution, \
                 ChangeAttribution or Counterfactual under Bayesian inference"
            ),
        });
    }
    Ok(())
}

/// Refuses an explicitly selected estimator that does not match the design-based query's frozen inputs.
fn refuse_estimator_design_mismatch(
    query: &CausalQuery,
    estimator: Option<EstimatorId>,
) -> Result<(), CausalError> {
    refuse_randomized_effect_mismatch(query, estimator)?;
    refuse_policy_value_mismatch(query, estimator)?;
    refuse_bayesian_design_mismatch(query, estimator)?;
    refuse_survival_dose_mismatch(query, estimator)?;
    refuse_history_contrast_mismatch(query, estimator)?;
    refuse_temporal_design_mismatch(query, estimator)?;
    refuse_longitudinal_regime_mismatch(query, estimator)
}

/// Refuses a temporal-design estimator selected for another query, and another estimator, or
/// the survival estimators on an arm design (or the reverse), for a temporal-design query.
fn refuse_temporal_design_mismatch(
    query: &CausalQuery,
    estimator: Option<EstimatorId>,
) -> Result<(), CausalError> {
    let excursion = matches!(
        estimator,
        Some(EstimatorId::TemporalDesignExcursion | EstimatorId::BayesianTemporalDesignExcursion)
    );
    let survival = matches!(
        estimator,
        Some(EstimatorId::TemporalRegimeSurvival | EstimatorId::BayesianTemporalRegimeSurvival)
    );
    if estimator.is_none() {
        return Ok(());
    }
    let CausalQuery::TemporalDesign(design) = query else {
        if excursion || survival {
            return Err(CausalError::Unsupported {
                message: "the temporal-design estimators answer a TemporalDesign query only",
            });
        }
        return Ok(());
    };
    let survival_design =
        matches!(design.assignment, antecedent_core::TemporalAssignment::RegimeSurvival { .. });
    if !(excursion || survival) || survival != survival_design {
        return Err(CausalError::Unsupported {
            message: "a temporal-design query takes the excursion estimators for an arm or dose \
                      design and the regime-survival estimators for a survival design",
        });
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

/// Refuses a Bayesian design estimator selected for a query of another design family.
fn refuse_bayesian_design_mismatch(
    query: &CausalQuery,
    estimator: Option<EstimatorId>,
) -> Result<(), CausalError> {
    for (matches_query, id) in [
        (matches!(query, CausalQuery::PanelDid(_)), EstimatorId::BayesianPanelDid),
        (matches!(query, CausalQuery::SyntheticControl(_)), EstimatorId::BayesianSyntheticControl),
        (matches!(query, CausalQuery::LocalPolynomialRatio(_)), EstimatorId::BayesianLocalRatio),
    ] {
        if matches_query && estimator.is_some_and(|selected| selected != id) {
            return Err(CausalError::Unsupported {
                message: "the selected estimator does not match this design query; select its Bayesian estimator or none",
            });
        }
    }
    if !matches!(
        query,
        CausalQuery::PanelDid(_)
            | CausalQuery::SyntheticControl(_)
            | CausalQuery::LocalPolynomialRatio(_)
    ) && matches!(
        estimator,
        Some(
            EstimatorId::BayesianPanelDid
                | EstimatorId::BayesianSyntheticControl
                | EstimatorId::BayesianLocalRatio
        )
    ) {
        return Err(CausalError::Unsupported {
            message: "the selected Bayesian design estimator does not match this query",
        });
    }
    if !matches!(query, CausalQuery::PolicyValue(_))
        && estimator == Some(EstimatorId::BayesianBinaryPolicyValue)
    {
        return Err(CausalError::Unsupported {
            message: "bayesian.binary_policy_value answers a PolicyValue query only",
        });
    }
    Ok(())
}

/// Refuses survival and dose-response estimators selected for the wrong query.
fn refuse_survival_dose_mismatch(
    query: &CausalQuery,
    estimator: Option<EstimatorId>,
) -> Result<(), CausalError> {
    if matches!(query, CausalQuery::Survival(_))
        && estimator.is_some_and(|id| {
            id != EstimatorId::RandomizedSurvivalProductLimit
                && id != EstimatorId::BayesianPiecewiseSurvival
        })
    {
        return Err(CausalError::Unsupported {
            message: "randomized survival requires the product-limit or the piecewise-exponential Bayesian estimator",
        });
    }
    if matches!(query, CausalQuery::ContinuousDoseResponse(_))
        && estimator.is_some_and(|id| {
            id != EstimatorId::ContinuousDoseKernel && id != EstimatorId::BayesianDosePolicy
        })
    {
        return Err(CausalError::Unsupported {
            message: "continuous-dose response requires the kernel or the Bayesian dose-policy estimator",
        });
    }
    if !matches!(query, CausalQuery::Survival(_))
        && estimator == Some(EstimatorId::BayesianPiecewiseSurvival)
    {
        return Err(CausalError::Unsupported {
            message: "bayesian.piecewise_survival answers a randomized Survival query only",
        });
    }
    if !matches!(query, CausalQuery::ContinuousDoseResponse(_))
        && estimator == Some(EstimatorId::BayesianDosePolicy)
    {
        return Err(CausalError::Unsupported {
            message: "bayesian.continuous_dose_policy answers a ContinuousDoseResponse query only",
        });
    }
    Ok(())
}

/// Refuses history-matching and sequential-contrast estimators selected for the wrong query.
fn refuse_history_contrast_mismatch(
    query: &CausalQuery,
    estimator: Option<EstimatorId>,
) -> Result<(), CausalError> {
    let history_estimator = estimator == Some(EstimatorId::PeriodHistoryMatching);
    if matches!(query, CausalQuery::HistoryMatch(_)) != history_estimator
        && (history_estimator || estimator.is_some())
    {
        return Err(CausalError::Unsupported {
            message: "a history match query takes the period history matching estimator, and that estimator answers that query only",
        });
    }
    let contrast_estimator = matches!(
        estimator,
        Some(
            EstimatorId::SequentialCrossfitContrastWald
                | EstimatorId::BayesianSequentialCrossfitContrast
        )
    );
    if matches!(query, CausalQuery::SequentialContrast(_)) != contrast_estimator
        && (contrast_estimator || estimator.is_some())
    {
        return Err(CausalError::Unsupported {
            message: "a sequential contrast query takes the sequential cross-fit contrast estimators, and they answer that query only",
        });
    }
    Ok(())
}

/// Refuses a longitudinal-regime estimator that does not match the query's method.
fn refuse_longitudinal_regime_mismatch(
    query: &CausalQuery,
    estimator: Option<EstimatorId>,
) -> Result<(), CausalError> {
    if !matches!(query, CausalQuery::LongitudinalRegime(_))
        && estimator == Some(EstimatorId::BayesianSequentialRegimeValue)
    {
        return Err(CausalError::Unsupported {
            message: "bayesian.sequential_regime_value answers a LongitudinalRegime query only",
        });
    }
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

fn apply_design_contract(
    query: &CausalQuery,
    graph: &AcceptedGraph,
    carry: &Carry,
) -> Result<(), CausalError> {
    refuse_estimator_design_mismatch(query, carry.estimator)?;
    if graph.class() != GraphClass::RandomizedTrial {
        return Err(CausalError::Unsupported {
            message: "design-based queries carry their own identification contract and must not be combined with graph= or discovery=",
        });
    }
    if matches!(query, CausalQuery::HistoryMatch(_))
        && matches!(carry.inference, InferenceMode::Bayesian(_))
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
    );
    if !matches!(carry.inference, InferenceMode::Frequentist) && !bayesian_design {
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
    );
    if let (InferenceMode::Bayesian(config), false) = (&carry.inference, reads_own_model) {
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
    if carry.identifier.is_some_and(|id| id != IdentifierId::RandomizedDesign)
        || carry.estimator.is_some_and(|id| {
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
                    | EstimatorId::BayesianRandomizedFinitePopulation
                    | EstimatorId::BayesianRandomizedSuperpopulation
                    | EstimatorId::BayesianSwitchback
                    | EstimatorId::TemporalDesignExcursion
                    | EstimatorId::BayesianTemporalDesignExcursion
                    | EstimatorId::TemporalRegimeSurvival
                    | EstimatorId::BayesianTemporalRegimeSurvival
            )
        })
        || carry
            .estimator_spec
            .as_ref()
            .is_some_and(|spec| !matches!(spec, EstimatorSpec::Default(_)))
    {
        return Err(CausalError::Unsupported {
            message: "design-based query estimators are fixed; custom estimators are not accepted",
        });
    }
    if (carry.refute_explicit && carry.refute != RefuteSuite::None)
        || !carry.custom_validators.is_empty()
    {
        return Err(CausalError::Unsupported {
            message: "design-based queries have no refutation suite or custom validator route",
        });
    }
    if carry.bootstrap_explicit
        && carry.bootstrap_replicates != 0
        && !matches!(query, CausalQuery::Survival(_) | CausalQuery::TemporalDesign(_))
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
    if matches!(carry.estimator, Some(EstimatorId::RdSharp | EstimatorId::BayesianRdLocalLinear))
        || carry.identifier == Some(IdentifierId::RdSharp)
    {
        if let (Some(rd), CausalQuery::AverageEffect(average)) = (carry.rd.as_ref(), &mut query) {
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
    if let Some(estimator) = carry.estimator {
        refuse_estimator_inference_mismatch(&query, estimator, &carry.inference)?;
    }
    refuse_unsupported_likelihood(
        &query,
        &carry.data,
        graph.class(),
        graph_posterior.is_none() && carry.tiered.is_none(),
        carry.estimator,
        &carry.inference,
    )?;
    super::refuse_unsupported_temporal_link(
        &query,
        &carry.data,
        graph.class(),
        graph_posterior.is_none() && carry.tiered.is_none(),
        carry.estimator,
    )?;
    super::refuse_unsupported_temporal_instrument(
        &query,
        &carry.data,
        graph.class(),
        graph_posterior.is_none() && carry.tiered.is_none(),
        carry.identifier,
        carry.estimator,
    )?;
    super::refuse_unsupported_white_innovation(
        &query,
        &carry.data,
        graph.class(),
        graph_posterior.is_none() && carry.tiered.is_none(),
        &carry.inference,
    )?;
    refuse_unsupported_draw_controls(
        &query,
        &carry.data,
        graph.class(),
        graph_posterior.is_none() && carry.tiered.is_none(),
        carry.estimator,
        carry.draw_controls,
        &carry.inference,
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
        carry.estimator_spec.as_ref().and_then(EstimatorSpec::bootstrap_replicates)
    {
        // A configured estimator owns its replicate count (an explicit
        // builder count beside it is refused above). The study reports and
        // executes that same count instead of its own omitted default.
        if carry.latency_mode.is_some() && carry.bootstrap_replicates != configured {
            carry.latency_bootstrap_not_applied = Some((carry.bootstrap_replicates, configured));
        }
        carry.bootstrap_replicates = configured;
    }
    if !carry.bootstrap_explicit && !omitted_bootstrap_resamples(&query, &carry.inference) {
        carry.bootstrap_replicates = 0;
    }
    if !carry.custom_validators.is_empty()
        && !custom_validators_apply(
            &query,
            &carry.data,
            graph.class(),
            graph_posterior.is_some(),
            &carry.inference,
        )
    {
        return Err(crate::unsupported_reason!(
            "validators_not_applicable",
            "custom validators refute one scalar average effect; this query and data route \
             has no scalar refutation problem for them to run on"
        ));
    }
    Ok(ExecutableRoute(Spine { carry, graph, graph_posterior, query }))
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
        carry.structure_source.unwrap_or(StructureSource::Explicit)
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
        crate::support::matrix_graph_class(&graph, &query, carry.tiered.as_ref())
    };
    let selected = carry.estimator_spec.as_ref().map(EstimatorSpec::id);
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
    let Carry { inference, tiered, refute_explicit, refute, refute_default_downgrade, .. } = carry;
    let inference = &*inference;
    let tiered = tiered.as_ref();
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
                        && q.inner.effect_modifiers.len() == 1
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
                message: "quantiles require Frequentist AllObserved AIPW AverageEffect, binary ConditionalEffect with one modifier, or cell-AIPW joint response; use prepare + retarget for score-table target weights",
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
    if classified.spine.carry.tiered.is_some() && classified.spine.carry.identifier.is_some() {
        return Err(CausalError::Unsupported {
            message: "TieredBackground selects its own identifier; omit identifier",
        });
    }
    if classified.spine.carry.continuous_cell.is_some() {
        return Err(CausalError::Unsupported {
            message: antecedent_estimate::POINT_CDE_UNLICENSED,
        });
    }
    let identification_cache = cache_cell_aipw(&classified)?;
    refuse_exceedance_grid(&classified)?;
    if let Some(estimator) = classified.spine.carry.estimator {
        refuse_estimator_inference_mismatch(
            &classified.spine.query,
            estimator,
            &classified.spine.carry.inference,
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
        && matches!(carry.inference, InferenceMode::Bayesian(_))
        && classified.selected.is_none_or(|id| id == EstimatorId::ResponseBayesian);
    let cell_aipw = classified.selected == Some(EstimatorId::CellAipw)
        || (classified.selected.is_none() && codetermined_joint);
    if !cell_aipw && !bayesian_joint {
        return Ok(None);
    }
    if let Some(background) = &carry.tiered {
        let CausalQuery::Response(response) = query else {
            return Err(CausalError::Unsupported {
                message: "cell-AIPW on a tiered background requires joint InterventionResponse",
            });
        };
        if matches!(carry.inference, InferenceMode::Bayesian(_)) {
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
    let score_grid = matches!(carry.inference, InferenceMode::Frequentist)
        && classified.spine.graph_posterior.is_none()
        && match query {
            CausalQuery::AverageEffect(q) => {
                classified.selected == Some(EstimatorId::Aipw)
                    && matches!(q.target_population, antecedent_core::TargetPopulation::AllObserved)
                    && (classified.graph_class == GraphClass::Dag
                        || carry.tiered.as_ref().is_some_and(|b| {
                            b.within_tier == antecedent_graph::WithinTier::CoDetermined
                        }))
            }
            CausalQuery::Response(q) => {
                classified.selected == Some(EstimatorId::CellAipw)
                    && q.temporal.is_none()
                    && (classified.graph_class == GraphClass::Dag
                        || carry.tiered.as_ref().is_some_and(|b| {
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
    let Some(spec) = &classified.spine.carry.estimator_spec else {
        return Ok(());
    };
    let query = &classified.spine.query;
    let inference = &classified.spine.carry.inference;
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
        if !carry.refute_explicit && carry.refute_default_downgrade.is_none() {
            let requested = crate::support::support_cell_named(
                query,
                matrix_class,
                structure,
                &carry.inference,
                carry.refute,
            );
            let without_validation = crate::support::support_cell_named(
                query,
                matrix_class,
                structure,
                &carry.inference,
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
                carry.refute_default_downgrade = Some(carry.refute);
                carry.refute = RefuteSuite::None;
            }
        }
        if carry
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
                &carry.inference,
                panel,
                carry.split.as_ref(),
            )?;
        }
        if carry.class_prior.is_some() && matches!(carry.inference, InferenceMode::Frequentist) {
            return Err(CausalError::Unsupported {
                message: "class_prior is a structural probability over class members and \
                      requires Bayesian inference; Frequentist incomplete-class Pulse \
                      keeps enumeration-weighted ATE without treating those weights as \
                      probabilities",
            });
        }
        if carry.class_prior.is_some() && !graph.class().is_incomplete_temporal() {
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
        let mut support_status = if let Some(cell) = crate::support::support_cell_named(
            query,
            matrix_class,
            structure,
            &carry.inference,
            carry.refute,
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
                    let estimator = if carry.estimator_spec.is_some() {
                        carry.estimator
                    } else {
                        crate::support::licensed_route_estimator(cell).or(carry.estimator)
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
        // separate execution evidence but no matrix cell or interval calibration yet.
        if matches!(
            query,
            CausalQuery::Interference(q)
                if matches!(
                    q.assignment,
                    antecedent_core::AssignmentDesign::ClusterRandomization { .. }
                        | antecedent_core::AssignmentDesign::TwoStageSaturation { .. }
                        | antecedent_core::AssignmentDesign::ObservedExposure { .. }
                )
        ) {
            support_status = None;
        }
        // The untempered posterior of a declared white-innovation premise is a different
        // construction from the licensed tempered posterior of the same cell: it is unlicensed
        // and its calibration is unmeasured, so the cell's license does not carry to it.
        if matches!(&carry.inference, InferenceMode::Bayesian(cfg) if cfg.white_innovation.is_some())
            && support_status == Some(CellStatus::Licensed)
        {
            support_status = Some(CellStatus::Refused);
        }
        // The Frequentist counterfactual and attribution cells are licensed as point
        // values. A requested mechanism-refit bootstrap interval is a construction no
        // coverage record measures: it executes, and the result is unlicensed.
        if matches!(carry.inference, InferenceMode::Frequentist)
            && carry.bootstrap_replicates > 0
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
        if !carry.inspect_only
            && matches!(
                carry.estimator,
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
        support_status
    };
    Ok(ClassifiedSupport { settled, support_status })
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
            if carry.interference.is_some() {
                return Err(CausalError::Unsupported {
                    message: "interference network is not used by TransportQuery",
                });
            }
            let admg =
                spine.graph.graph_ref().projected_admg()?.ok_or(CausalError::Unsupported {
                    message: "TransportQuery requires a supplied Admg or TemporalAdmg",
                })?;
            let targets = carry.selection_targets.clone().unwrap_or_default();
            let diagram = antecedent_graph::SelectionDiagram::try_new(admg.clone(), targets)
                .map_err(CausalError::rejected)?;
            let trial = carry.transport_trial.clone().ok_or(CausalError::Unsupported {
                message: "TransportQuery requires StudyBuilder::transport_trial",
            })?;
            (Some(diagram), Some(trial), None)
        }
        CausalQuery::Interference(design) => {
            if !carry.inspect_only {
                refuse_unlicensed_interference(design)?;
            }
            if carry.transport_trial.is_some() || carry.selection_targets.is_some() {
                return Err(CausalError::Unsupported {
                    message: "transport trial columns are not used by InterferenceQuery",
                });
            }
            let spec = carry.interference.clone().ok_or(CausalError::Unsupported {
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
        _ if carry.transport_trial.is_some()
            || carry.selection_targets.is_some()
            || carry.interference.is_some() =>
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
    Study {
        config: Arc::new(StudyConfig {
            refute_default_downgrade: carry.refute_default_downgrade,
            latency_bootstrap_not_applied: carry.latency_bootstrap_not_applied,
            estimator_spec_identity: carry
                .estimator_spec
                .as_ref()
                .map(super::super::contract_identity::estimator_spec_identity),
            response_options: carry.response_options,
            observation_options: carry.observation_options,
            draw_controls: carry.draw_controls,
            observation_delayed_entry: carry.observation_delayed_entry,
            rd: carry.rd,
            stage_sink: carry.stage_sink,
            continuous_cell: carry.continuous_cell,
            selection_diagram,
            transport_trial,
            transport_identification_cache: None,
            discovery: None,
            mechanisms: None,
        }),
        data: carry.data,
        graph: Arc::new(graph),
        graph_posterior,
        class_prior: carry.class_prior,
        max_completions: carry.max_completions,
        structure_source: structure,
        support_status,
        query,
        refute: carry.refute,
        bootstrap_replicates: carry.bootstrap_replicates,
        split: carry.split,
        identifier: carry.identifier,
        estimator: carry.estimator,
        estimator_spec: carry.estimator_spec.map(Arc::new),
        inference: carry.inference,
        overlap_policy: carry.overlap_policy,
        population_registry: carry.population_registry,
        participants: carry.participants,
        missing: carry.missing,
        sampling_design: carry.sampling_design,
        custom_validators: carry.custom_validators,
        latency_mode: carry.latency_mode,
        identification_cache,
        pag_identification_cache: None,
        cpdag_identification_cache: None,
        temporal_identification_cache: None,
        temporal_class_identification_cache: None,
        graph_posterior_identification_cache: None,
        dbn_posterior_identification_cache: None,
        temporal_class_posterior_identification_cache: None,
        tiered: carry.tiered,
        shared_batch_design: None,
        interference,
    }
}
