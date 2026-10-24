//! Executors of the structure-discovery, attribution and simulation routes.
//!
//! Each runs the existing search, attribution or simulation engine (through the
//! runners in [`crate::analysis::structure_route`]) inside the lowered
//! program's dispatch scope: it checks cancellation, leases its buffers from
//! the admission pool before the engine allocates them, reuses the
//! identification the program retained, and returns a result the caller stamps
//! with the lowered identity and the executed contract.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus, DiscoverQuery,
    DiscoveryInference, DiscoveryMethod, InterventionSequence, SimulateQuery, SimulationTarget,
};

use super::{
    Arc, AssembleArgs, CausalError, CausalExprArena, CausalQuery, Dag, DataInput, DerivationMeta,
    DerivationTrace, Diagnostic, DiagnosticKind, DiagnosticSeverity, DomainRef, EffectEstimate,
    EstimatorId, ExecutionContext, ExprNode, GcmSlot, IdentificationPerformanceRecord,
    IdentificationResult, IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, IdentifierId, Instant, OutcomeExprId, OverlapPolicy,
    PhysicalExecutionPlan, ProvenanceGraph, Strategy, StudyResult, TableView, TabularData,
    VariableId, assemble_result, elapsed_ns, gcm_query_vars, identification_from_cache_or,
    identify_cached_diagnostic, nan_effect, parametric_scm_identification,
};
use crate::analysis::structure_route::{
    AttributionFit, AttributionOutcome, AttributionPayload, ChangeReading, DiscoverData,
    DiscoveredStructure, DiscoveryPayload, MechanismProgram, SimulationOutcome, SimulationPayload,
    admit, discovery_quote, model_quote, run_attribute_with_fit, run_discovery, run_simulate,
};

/// Identifier a structure search records: it identifies no causal contrast;
/// the search states its premises and the structure is reviewed before use.
pub(crate) const STRUCTURE_SEARCH_IDENTIFIER: &str = "discovery.structure_search";

/// Stage name of a structure search, for cancellation.
const DISCOVER_STAGE: &str = "discover";
/// Stage name of a fitted-model route, for cancellation.
const MODEL_STAGE: &str = "estimate.model";

fn premise(assumption: antecedent_core::Assumption, algorithm: &str) -> AssumptionRecord {
    AssumptionRecord {
        assumption,
        source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from(algorithm) },
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    }
}

fn restriction(id: &str, description: &str, algorithm: &str) -> AssumptionRecord {
    premise(
        antecedent_core::Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
            id: Arc::from(id),
            description: Arc::from(description),
        }),
        algorithm,
    )
}

/// The premises a search method's answer is correct under.
fn search_premises(method: &DiscoveryMethod) -> antecedent_core::AssumptionSet {
    use antecedent_core::Assumption;
    match method {
        // A wrapper is correct under what the search it wraps is correct under, and its own
        // restriction.
        DiscoveryMethod::ClassPosterior { base } => {
            let mut set = search_premises(base);
            set.push(restriction(
                "discovery.equivalence_class_collapse",
                "Markov equivalence is the identifiability limit of the Gaussian BIC score: \
                 DAGs of one class score equally, so the class posterior is the DAG posterior \
                 summed over each class and adds no assumption beyond the DAG posterior's.",
                method.id(),
            ));
            return set;
        }
        DiscoveryMethod::BootstrapStability { base, .. } => {
            let mut set = search_premises(base);
            set.push(restriction(
                "discovery.resampling_design",
                "The rows were sampled under the declared sampling design, so resamples drawn \
                 under it reproduce the search's sampling law: independent subjects for a \
                 table, a stationary dependent series for circular blocks, independent units \
                 for a panel. The frequencies are a bootstrap distribution of the search, not \
                 a posterior, and their coverage is not measured.",
                method.id(),
            ));
            return set;
        }
        _ => {}
    }
    let id = method.id();
    let mut set = antecedent_core::AssumptionSet::default();
    set.push(premise(Assumption::CausalMarkov, id));
    let latent_tolerant = matches!(
        method,
        DiscoveryMethod::Fci
            | DiscoveryMethod::Rfci
            | DiscoveryMethod::Lpcmci
            | DiscoveryMethod::ExactPagPosterior
            | DiscoveryMethod::PagMcmc
    );
    if !latent_tolerant {
        set.push(premise(Assumption::CausalSufficiency, id));
    }
    if method.uses_independence_tests() {
        set.push(premise(Assumption::Faithfulness, id));
        set.push(restriction(
            "discovery.independence_test_validity",
            "The declared conditional-independence test is valid for the data's dependence: its \
             error rates are the ones the significance level states.",
            id,
        ));
    }
    if method.is_temporal() {
        set.push(premise(Assumption::Stationarity, id));
    }
    match method {
        DiscoveryMethod::DirectLingam { .. } => set.push(restriction(
            "discovery.linear_non_gaussian",
            "Every mechanism is linear with independent non-Gaussian noise.",
            id,
        )),
        DiscoveryMethod::Notears { .. } => set.push(restriction(
            "discovery.linear_sem",
            "The data follow an acyclic linear structural equation model.",
            id,
        )),
        DiscoveryMethod::Ges { .. } => set.push(restriction(
            "discovery.gaussian_bic_score",
            "The Gaussian BIC score is consistent for the data's mechanisms.",
            id,
        )),
        _ => {}
    }
    if method.inference() == DiscoveryInference::Bayesian {
        set.push(restriction(
            "discovery.structural_prior_and_score",
            "The posterior is over structures under the declared structural prior and the \
             Gaussian BIC marginal score; it does not address structures outside the prior's \
             support.",
            id,
        ));
    }
    set
}

/// Identification product of a structure search: the searched variables'
/// observational law, under the premises the method declares.
///
/// A search identifies no causal contrast. The product states what the answer
/// is a function of (the observational joint law of the searched variables) and
/// the premises it is correct under, so a contract and a claim can carry them.
/// Its status is [`IdentificationStatus::IdentifiedUnderParametricRestrictions`]:
/// the class is identified from that law only under those premises.
#[must_use]
pub(crate) fn structure_search_identification(
    query: CausalQuery,
    search: &DiscoverQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let mut arena = CausalExprArena::new();
    let first = search.variables.first().copied().unwrap_or(VariableId::from_raw(0));
    let joint = arena.intern_var_set(search.variables.iter().copied());
    let none = arena.empty_intervention_set();
    let empty = arena.empty_var_set();
    let distribution = arena.intern_distribution(joint, empty, none, DomainRef::Observational);
    let functional = arena
        .intern(ExprNode::Expectation { function: OutcomeExprId::identity(first), distribution });
    arena.set_derivation(
        functional,
        DerivationMeta::rule(
            STRUCTURE_SEARCH_IDENTIFIER,
            Some(Arc::from(format!(
                "structure search {} over {} variables; the answer is a {} read from the \
                 observational law, reviewed before any downstream use",
                search.method.id(),
                search.variables.len(),
                search.output().as_str()
            ))),
        ),
    );
    let estimand =
        IdentifiedEstimand::backdoor(STRUCTURE_SEARCH_IDENTIFIER, Arc::from([]), functional);
    let identification =
        IdentificationResult::from_parts(antecedent_identify::IdentificationResultParts {
            status: IdentificationStatus::IdentifiedUnderParametricRestrictions,
            query,
            estimands: vec![estimand.clone()],
            arena,
            derivation: DerivationTrace::default(),
            required_assumptions: search_premises(&search.method),
            diagnostics: Vec::new(),
            performance: IdentificationPerformanceRecord::default(),
            hedge: None,
        });
    (identification, estimand)
}

fn modality_refusal(method: &DiscoveryMethod) -> CausalError {
    crate::compile_reason!(
        "data_modality_not_licensed",
        "the {} search does not run on this data: static searches read tabular data, temporal \
         searches read series or panel data, J-PCMCI+ reads multi-environment data, and regime \
         PCMCI reads one series",
        method.id()
    )
}

/// The note on how a discovered structure is to be read.
fn discovery_note(structure: &DiscoveredStructure, query: &DiscoverQuery) -> Diagnostic {
    match structure {
        DiscoveredStructure::Stability(table) => Diagnostic::new(
            "discovery.stability_table",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "the stability search tabulated {} of {} resamples ({} failed and are kept \
                 as failed mass); the frequencies are a bootstrap distribution of the \
                 search under the declared sampling design, not a posterior, and carry no \
                 review gate. Their coverage is not measured.",
                table.succeeded, table.requested, table.failed
            ),
        ),
        DiscoveredStructure::RegimeStability(tables) => Diagnostic::new(
            "discovery.stability_table",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "the stability search tabulated one table for each of {} labelled regime(s), \
                 re-running regime PCMCI on circular-block resamples that carry their regime \
                 labels; the frequencies are a bootstrap distribution of the search, not a \
                 posterior, and carry no review gate. Their coverage is not measured.",
                tables.len()
            ),
        ),
        _ => Diagnostic::new(
            "discovery.review_gate",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "the {} search returned a {}; a downstream study reads it only after its \
                 review gate (accept_discovered), which records this search's provenance on \
                 the accepted graph. CPDAG and PAG marks stay class information.",
                query.method.id(),
                query.output().as_str()
            ),
        ),
    }
}

impl super::Study {
    /// Admission of a structure search: the query is valid and the method runs
    /// on the study's data modality.
    ///
    /// # Errors
    ///
    /// An invalid query, or a method on data it does not run on
    /// (`data_modality_not_licensed`).
    pub(super) fn ensure_discovery_admitted(
        &self,
        query: &DiscoverQuery,
    ) -> Result<(), CausalError> {
        query.validate()?;
        // A wrapper runs on the data the search it wraps runs on.
        let method = match &query.method {
            DiscoveryMethod::ClassPosterior { base }
            | DiscoveryMethod::BootstrapStability { base, .. } => &**base,
            other => other,
        };
        let admitted = match &self.data {
            DataInput::Tabular(_) => !method.is_temporal(),
            DataInput::Temporal(_) | DataInput::Event(_) => {
                method.is_temporal() && !matches!(method, DiscoveryMethod::JpcmciPlus)
            }
            DataInput::Panel(panel) => {
                crate::analysis::builder::panel_shared_regularity(panel)?;
                matches!(
                    method,
                    DiscoveryMethod::Pcmci
                        | DiscoveryMethod::PcmciPlus
                        | DiscoveryMethod::Lpcmci
                        | DiscoveryMethod::DbnPosterior { .. }
                )
            }
            DataInput::MultiEnv(_) => matches!(method, DiscoveryMethod::JpcmciPlus),
        };
        if admitted { Ok(()) } else { Err(modality_refusal(method)) }
    }

    /// Execute a structure search on the study's tabular rows.
    pub(crate) fn execute_discover_tabular(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.run_discover(&DiscoverData::Tabular(data), physical, ctx)
    }

    /// Execute a structure search on `data`, whatever its modality.
    pub(crate) fn execute_discover(
        &self,
        data: &DataInput,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let data = match data {
            DataInput::Tabular(data) => DiscoverData::Tabular(data),
            DataInput::Temporal(series) | DataInput::Event(series) => DiscoverData::Series(series),
            DataInput::Panel(panel) => DiscoverData::Panel(panel),
            DataInput::MultiEnv(multi) => DiscoverData::MultiEnv(multi),
        };
        self.run_discover(&data, physical, ctx)
    }

    fn run_discover(
        &self,
        data: &DiscoverData<'_>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        // The search runs only from the program compile lowered for this study's cell, whether
        // it is entered through the dispatcher or directly.
        physical.require_lowered(crate::analysis::lowering::expected_cell(self))?;
        let started = Instant::now();
        let CausalQuery::Discover(query) = &self.query else {
            return Err(CausalError::Compile {
                message: "structure-search execution was handed a study that searches nothing"
                    .into(),
            });
        };
        self.ensure_discovery_admitted(query)?;
        let spec = self.discovery.as_deref().ok_or_else(|| {
            crate::compile_reason!(
                "required_option_missing",
                "the study carries no executable search; build it with StudyBuilder::discovery"
            )
        })?;
        spec.check_query(query)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: DISCOVER_STAGE });
        }
        let lease = admit(ctx, discovery_quote(query, data.rows()), "structure search")?;
        let structure = run_discovery(spec, query, data, self.sampling_design, ctx)?;
        drop(lease);
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: DISCOVER_STAGE });
        }
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.reuse_static_identification(physical), || {
                Ok(structure_search_identification(self.query.clone(), query))
            })?;
        let mut diagnostics = vec![discovery_note(&structure, query)];
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let first = query.variables.first().copied().unwrap_or(VariableId::from_raw(0));
        let mut result = assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical.record,
            identification,
            estimand,
            estimate: nan_effect(),
            posterior: None,
            payload: crate::result::ResultPayload::Discovery(Box::new(DiscoveryPayload {
                structure,
                query: query.clone(),
            })),
            refutations: Vec::new(),
            diagnostics,
            provenance: ProvenanceGraph::new(),
            treatment: first,
            outcome: first,
            wall_time_ns: elapsed_ns(started),
            latency_mode: self.latency_mode.map(|mode| Arc::from(mode.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: None,
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
            bayesian: query.inference() == DiscoveryInference::Bayesian,
        });
        crate::analysis::helpers::stamp_result_context(
            &mut result,
            self.support_status,
            self.structure_source,
            self.population_registry.as_ref(),
            &[],
        );
        Ok(result)
    }

    /// Execute an attribution or simulation on mechanisms fitted on `graph`.
    pub(super) fn execute_model_route(
        &self,
        data: &TabularData,
        graph: &Dag,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (treatment, outcome) = gcm_query_vars(&self.query)?;
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.reuse_static_identification(physical), || {
                Ok(parametric_scm_identification(self.query.clone(), treatment, outcome))
            })?;
        let standard = MechanismProgram::standard();
        let program = self.mechanisms.as_deref().unwrap_or(&standard);
        let answer = run_model_query(program, graph, data, &self.query, ctx)?;
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                answer.estimate,
                Strategy::new(IdentifierId::GcmParametric, EstimatorId::GcmFit),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                diagnostics: Some(answer.diagnostics),
                gcm: Some(answer.slot),
                empty_provenance: true,
                ..Default::default()
            }),
        ))
    }
}

/// The answer of a fitted-model route, before it is assembled into a result.
pub(super) struct ModelAnswer {
    pub(super) slot: GcmSlot,
    pub(super) estimate: EffectEstimate,
    pub(super) diagnostics: Vec<Diagnostic>,
}

fn point(value: f64) -> EffectEstimate {
    EffectEstimate::new(
        value,
        f64::NAN,
        antecedent_core::AssumptionSet::default(),
        OverlapPolicy::ExplicitOverride,
    )
}

fn model_diagnostic(code: &'static str, message: &str) -> Diagnostic {
    Diagnostic::new(code, DiagnosticKind::Scientific, DiagnosticSeverity::Info, message.to_owned())
}

/// The point value and the note a simulation outcome is reported with.
fn simulation_reading(outcome: &SimulationOutcome) -> (f64, &'static str) {
    match outcome {
        SimulationOutcome::PolicyValue(value) => (
            value.mean_difference,
            "the policy value is the fitted model's; only the Monte Carlo error of the \
             draws is quantified",
        ),
        SimulationOutcome::UnitEffects(ite) => (
            ite.mean_ite,
            "unit effects are computed under the fitted mechanisms after abduction; they \
             are not identified from data",
        ),
        SimulationOutcome::Counterfactual(_) => (
            f64::NAN,
            "counterfactual values are computed under the fitted mechanisms after \
             abduction; they are not identified from data",
        ),
        SimulationOutcome::Temporal(_) => (
            f64::NAN,
            "a rollout of the fitted model under the declared schedule; only the draws' \
             own variation is reported",
        ),
        _ => (f64::NAN, "draws of the fitted model under the declared interventions"),
    }
}

/// Fit the program's mechanisms and run `query` on them: an attribution, a
/// simulation, or the mechanism-change and unit-change studies.
///
/// # Errors
///
/// A fit or engine refusal, a refused lease, or cancellation.
pub(super) fn run_model_query(
    program: &MechanismProgram,
    graph: &Dag,
    data: &TabularData,
    query: &CausalQuery,
    ctx: &ExecutionContext,
) -> Result<ModelAnswer, CausalError> {
    run_model_query_with_fit(program, graph, data, query, None, ModelRun::default(), ctx)
}

/// How a fitted-model query reads its answer beyond the point: a mechanism-change reading and,
/// on the lag-aligned unfolding of a `TemporalDag`, the unfolding a rollout steps.
#[derive(Clone, Copy, Default)]
pub(super) struct ModelRun<'a> {
    pub(super) reading: ChangeReading,
    pub(super) unfolding: Option<&'a super::temporal_attribution::LagAlignedScm>,
}

/// Roll the dynamics of a fitted lag-aligned model forward.
///
/// The unfolded model is one slice: the contemporaneous nodes keep their fitted mechanisms and
/// every lagged copy `v@L` is assigned the value of `v` read `L` steps back from the rollout's
/// history, which starts at the last rows of the series. The caller's schedule is applied on top.
/// Sampling the unfolded graph once would instead treat the lagged copies as exogenous columns.
///
/// # Errors
///
/// A non-rollout target, a caller-supplied history (the history is the data's last rows), a
/// series shorter than the history, or a rollout failure.
pub(super) fn rollout_on_unfolding(
    fitted: &crate::gcm::FittedGcm,
    data: &TabularData,
    aligned: &super::temporal_attribution::LagAlignedScm,
    simulate: &SimulateQuery,
    ctx: &ExecutionContext,
) -> Result<SimulationOutcome, CausalError> {
    let SimulationTarget::TemporalRollout { schedule, n_rows, n_steps, window, history, seed } =
        &simulate.target
    else {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "a simulation on a time series is a temporal rollout of the fitted dynamics"
        ));
    };
    if !history.is_empty() {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "the rollout of a fitted TemporalDag starts from the last rows of the series; omit \
             the history"
        ));
    }
    let mut steps = aligned.lag_assignments(*n_steps);
    steps.extend(schedule.steps.iter().cloned());
    let sequence = InterventionSequence::new(steps);
    let initial = aligned.initial_history(data, *n_rows)?;
    let plan = crate::gcm::TemporalPlan {
        sequence: &sequence,
        n_steps: *n_steps,
        n_rows: *n_rows,
        window: (*window).max(aligned.history()),
        initial_history: &initial,
    };
    let mut rng = antecedent_core::CausalRng::from_seed(*seed);
    Ok(SimulationOutcome::Temporal(crate::gcm::sample_temporal_policy(
        &fitted.model,
        &plan,
        &mut rng,
        ctx,
    )?))
}

/// [`run_model_query`] with the mechanism-fit refits of an intrinsic-influence attribution
/// set by the study's inference mode (`fit`) instead of by the query.
///
/// # Errors
///
/// As [`run_model_query`].
pub(super) fn run_model_query_with_fit(
    program: &MechanismProgram,
    graph: &Dag,
    data: &TabularData,
    query: &CausalQuery,
    fit: Option<AttributionFit>,
    run: ModelRun<'_>,
    ctx: &ExecutionContext,
) -> Result<ModelAnswer, CausalError> {
    let reading = run.reading;
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: MODEL_STAGE });
    }
    let fitted = program.fit(graph, data)?;
    let nodes = graph.node_count() as u64;
    let rows = data.row_count() as u64;
    let mechanisms: Arc<[crate::gcm::MechanismAssignment]> = Arc::from(fitted.assignments.clone());
    let fitted_note = model_diagnostic(
        "gcm.model.fitted_answer",
        "computed under the mechanisms fitted on the supplied structure; mechanism-fit \
         uncertainty is not part of the reported value",
    );
    match query {
        CausalQuery::Attribute(attribute) => {
            let _lease = admit(ctx, model_quote(rows, nodes, 0), "attribution")?;
            let outcome =
                run_attribute_with_fit(&fitted, graph, data, attribute, fit, reading, ctx)?;
            let value = match &outcome {
                AttributionOutcome::RootCauses { attribution, .. } => attribution.total_change,
                AttributionOutcome::Change(change) => change.total_change,
                AttributionOutcome::IntrinsicInfluence(influence) => influence.total_variance,
                _ => f64::NAN,
            };
            Ok(ModelAnswer {
                slot: GcmSlot::Attribution(Box::new(AttributionPayload { outcome, mechanisms })),
                estimate: point(value),
                diagnostics: vec![fitted_note],
            })
        }
        CausalQuery::Simulate(simulate) => {
            let draws = crate::analysis::structure_route::simulated_rows(&simulate.target, rows);
            let _lease = admit(ctx, model_quote(rows, nodes, draws), "simulation")?;
            if let Some(aligned) = run.unfolding {
                let outcome = rollout_on_unfolding(&fitted, data, aligned, simulate, ctx)?;
                return Ok(ModelAnswer {
                    slot: GcmSlot::Simulation(Box::new(SimulationPayload { outcome, mechanisms })),
                    estimate: point(f64::NAN),
                    diagnostics: vec![
                        fitted_note,
                        model_diagnostic(
                            "gcm.simulation.unfolded_rollout",
                            "a rollout of the fitted dynamics: the contemporaneous mechanisms of \
                             the lag-aligned model are stepped forward and every lagged node reads \
                             the history of the rollout, which starts at the last rows of the \
                             series",
                        ),
                    ],
                });
            }
            let outcome = run_simulate(&fitted, data, simulate, ctx)?;
            let (value, note) = simulation_reading(&outcome);
            Ok(ModelAnswer {
                slot: GcmSlot::Simulation(Box::new(SimulationPayload { outcome, mechanisms })),
                estimate: point(value),
                diagnostics: vec![fitted_note, model_diagnostic("gcm.simulation", note)],
            })
        }
        CausalQuery::MechanismChange(detection) => {
            detection.validate()?;
            let detections = crate::analysis::structure_route::run_mechanism_change(
                &fitted,
                data,
                detection,
                antecedent_attribution::MechanismChangeMethod::LikelihoodRatio,
                reading,
                ctx,
            )?;
            Ok(ModelAnswer {
                slot: GcmSlot::Mechanism(detections),
                estimate: nan_effect(),
                diagnostics: Vec::new(),
            })
        }
        CausalQuery::UnitChange(unit) => {
            unit.validate()?;
            let result = crate::gcm::attribute_unit_change(&fitted.model, data, unit, ctx)?;
            Ok(ModelAnswer {
                slot: GcmSlot::Unit(result),
                estimate: nan_effect(),
                diagnostics: Vec::new(),
            })
        }
        _ => Err(CausalError::Compile {
            message: "a fitted-model route was handed a query it does not run".into(),
        }),
    }
}
