//! Assembly of executed study results and provenance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, AssumptionSet, BufferMaterialization, Diagnostic, EffectEstimate,
    ExecutionPerformanceRecord, IdentifiedEstimand, LogicalAnalysisPlanRecord,
    PhysicalExecutionPlanRecord, ProvenanceGraph, ProvenanceNode, RefutationReport, StudyResult,
    VERSION, VariableId, bootstrap_limit_diagnostic, sampler_limit_diagnostic,
};

pub(crate) struct AssembleArgs<'a> {
    pub(crate) logical: &'a LogicalAnalysisPlanRecord,
    pub(crate) physical: &'a PhysicalExecutionPlanRecord,
    pub(crate) identification: antecedent_identify::IdentificationResult,
    pub(crate) estimand: IdentifiedEstimand,
    pub(crate) estimate: EffectEstimate,
    pub(crate) posterior: Option<antecedent_estimate::CausalPosterior>,
    pub(crate) payload: crate::result::ResultPayload,
    pub(crate) refutations: Vec<RefutationReport>,
    pub(crate) diagnostics: Vec<Diagnostic>,
    pub(crate) provenance: ProvenanceGraph,
    pub(crate) treatment: VariableId,
    pub(crate) outcome: VariableId,
    /// Wall-clock nanoseconds for identify→estimate→refute.
    pub(crate) wall_time_ns: u64,
    /// Latency mode label when a tier was requested.
    pub(crate) latency_mode: Option<Arc<str>>,
    /// Per-stage timings.
    pub(crate) stage_timings_ns: Vec<(Arc<str>, u64)>,
    /// Bootstrap replicates requested.
    pub(crate) bootstrap_replicates_requested: Option<u32>,
    /// Bootstrap replicates that succeeded.
    pub(crate) bootstrap_replicates_ok: Option<u32>,
    /// Posterior draws (Bayesian).
    pub(crate) n_draws: Option<u32>,
    /// Cancellation observed during execute.
    pub(crate) cancelled: bool,
    /// Adaptive early-stop (bootstrap SE and/or Bayesian draws).
    pub(crate) early_stopped: bool,
    /// Whether the program's inference family is Bayesian.
    pub(crate) bayesian: bool,
}

pub(crate) fn assemble_result(mut args: AssembleArgs<'_>) -> StudyResult {
    let bootstrap = args.estimate.uncertainty.bootstrap;
    args.diagnostics.extend(bootstrap_limit_diagnostic(
        args.bootstrap_replicates_requested,
        args.bootstrap_replicates_ok.or(bootstrap.replicates_ok),
        bootstrap.replicates_failed,
        args.early_stopped,
    ));
    args.diagnostics.extend(
        args.posterior
            .as_ref()
            .and_then(|posterior| sampler_limit_diagnostic(&posterior.diagnostics)),
    );
    let copy_count = args
        .physical
        .materializations
        .iter()
        .filter(|(_, m)| !matches!(m, BufferMaterialization::Borrowed))
        .count() as u64;
    let mut result = StudyResult {
        retained_resources: Vec::new(),
        executed_input: None,
        temporal_response_witnesses: None,
        temporal_posterior_witnesses: None,
        static_class_witnesses: None,
        static_posterior_source: None,
        full_target_diagnostics: None,
        authority: crate::result::RuntimeRefutationAuthority::Fresh,
        view: crate::result::StudyResultView {
            performance: ExecutionPerformanceRecord {
                native_execution: None,
                wall_time_ns: Some(args.wall_time_ns),
                peak_rss_bytes: None,
                copy_count,
                scalar_fallback_count: 0,
                latency_mode: args.latency_mode,
                stage_timings_ns: args.stage_timings_ns,
                bootstrap_replicates_requested: args.bootstrap_replicates_requested,
                bootstrap_replicates_ok: args.bootstrap_replicates_ok,
                n_draws: args.n_draws,
                cancelled: args.cancelled,
                early_stopped: args.early_stopped,
                bytes_borrowed: None,
            },
            logical_plan: args.logical.clone(),
            physical_plan: args.physical.clone(),
            identification: args.identification,
            certificate: None,
            estimand: args.estimand,
            structural_response: None,
            posterior: args.posterior,
            refutations: args.refutations,
            predictive_checks: Vec::new(),
            diagnostics: args.diagnostics,
            provenance: args.provenance,
            structure_source: crate::support::StructureSource::Explicit,
            treatment: Some(args.treatment),
            outcome: args.outcome,
            population_registry: None,
        },
        estimate: crate::PrimaryEstimate::Effect(args.estimate),
        payload: args.payload,
        predictive_replicates: None,
        support_status: None,

        extras: crate::result::ResultExtras {
            candidate_selection: None,
            interval: None,
            row_weights: None,
            missing_data: None,
            custom_validator_names: Vec::new(),
            would_refuse: Vec::new(),
        },
        executed_contract: None,
        lowered_identity: None,
        lowered_quadrant: None,
    };
    result.rebind_interval(args.bayesian);
    result
}

/// Copy the support, structure, population, and caller-validator fields a
/// checked execution stamps after [`assemble_result`].
///
/// The assembled result starts with an empty registry, so assigning the cloned
/// registry matches `clone_from` on that empty slot.
pub(crate) fn stamp_result_context(
    result: &mut StudyResult,
    support_status: Option<crate::support::CellStatus>,
    structure_source: crate::support::StructureSource,
    population_registry: Option<&antecedent_core::PopulationRegistry>,
    custom_validator_names: &[Arc<str>],
) {
    result.support_status = support_status;
    result.structure_source = structure_source;
    result.population_registry = population_registry.cloned();
    result.extras.custom_validator_names = custom_validator_names.to_vec();
}

pub(crate) type ProvStep<'a> = (&'a str, &'a str, &'a [&'a str], &'a AssumptionSet);

pub(crate) fn provenance_pair(first: ProvStep<'_>, second: ProvStep<'_>) -> ProvenanceGraph {
    let mut provenance = ProvenanceGraph::new();
    for (artifact_id, operation, parents, assumptions) in [first, second] {
        let parent_arcs: Arc<[Arc<str>]> =
            parents.iter().map(|p| Arc::<str>::from(*p)).collect::<Vec<_>>().into();
        provenance.push(ProvenanceNode {
            artifact_id: Arc::from(artifact_id),
            operation: Arc::from(operation),
            parents: parent_arcs,
            assumptions: assumptions.clone(),
            library_version: Arc::from(VERSION),
            config_digest: Some(Arc::from("temporal")),
        });
    }
    provenance
}
