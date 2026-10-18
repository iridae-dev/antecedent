//! The earlier public surface still resolves by its earlier paths and signatures.
//!
//! Nothing here runs a scientific route: each item is named the way a downstream crate that
//! was written against the earlier release names it, so a removed item, a changed signature or
//! a lost auto trait stops this file compiling.
#![allow(deprecated, reason = "the retained earlier names are what this file checks")]
#![allow(clippy::float_cmp, reason = "exact distances between integer-valued points")]

use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention, InterventionSequence,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::{
    DiscoveryError, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary,
    adjacency_mask_from_admg, adjacency_mask_from_cpdag, adjacency_masks_from_pag,
    admg_from_adjacency_mask, cpdag_from_adjacency_mask, dag_from_adjacency_mask, has_edge,
    mask_is_dag, pag_from_adjacency_mask, parents_of, set_edge,
};
use antecedent_graph::{Admg, Cpdag, Dag, Pag};
use antecedent_stats::{MatchingDistance, MatchingIndex};

fn assert_unwind_safe<T: UnwindSafe + RefUnwindSafe>() {}

#[test]
fn query_and_intervention_types_keep_their_auto_traits() {
    assert_unwind_safe::<Intervention>();
    assert_unwind_safe::<InterventionSequence>();
    assert_unwind_safe::<CausalQuery>();
    assert_unwind_safe::<AverageEffectQuery>();
    assert_unwind_safe::<antecedent_core::CausalResponse>();
    assert_unwind_safe::<antecedent_core::TransportQuery>();
    assert_unwind_safe::<antecedent_core::TemporalEffectQuery>();
    assert_unwind_safe::<antecedent_core::ResponseQuery>();
    assert_unwind_safe::<antecedent_core::SequencedIntervention>();
    assert_unwind_safe::<antecedent_counterfactual::CounterfactualWorld>();
    assert_unwind_safe::<antecedent_model::InterventionOverlay>();
    assert_unwind_safe::<antecedent_graph::CpdagCompletionSampler>();
    assert_unwind_safe::<antecedent_graph::TemporalCpdagCompletionSampler>();
    assert_unwind_safe::<antecedent_identify::IdentificationResult>();
    assert_unwind_safe::<antecedent_identify::TemporalClassEnvelope>();
    assert_unwind_safe::<antecedent_state::CausalState>();
    assert_unwind_safe::<antecedent_state::QueryStore>();
    assert_unwind_safe::<antecedent_state::QueryRecord>();
    assert_unwind_safe::<antecedent_state::StateEvent>();
    assert_unwind_safe::<antecedent_estimate::CheckedAipwPreparation>();
    assert_unwind_safe::<antecedent_estimate::CheckedLinearAdjustmentAte>();
    assert_unwind_safe::<antecedent::StudyResult>();
    assert_unwind_safe::<antecedent::BatchQuery>();
    assert_unwind_safe::<antecedent::Identification>();
    assert_unwind_safe::<antecedent::ResultPayload>();
    assert_unwind_safe::<antecedent::StageEvent>();
    assert_unwind_safe::<antecedent::planner::LogicalAnalysisPlan>();
    assert_unwind_safe::<antecedent::planner::PhysicalExecutionPlan>();
}

#[test]
fn single_word_mask_functions_keep_their_signatures() {
    let _: fn(u64, usize, usize, usize) -> bool = has_edge;
    let _: fn(u64, usize, usize, usize, bool) -> u64 = set_edge;
    let _: fn(u64, usize) -> bool = mask_is_dag;
    let _: fn(u64, usize, usize) -> Vec<u32> = parents_of;
    let _: fn(u64, usize) -> Result<Dag, DiscoveryError> = dag_from_adjacency_mask;
    let _: fn(u64, usize) -> Result<Cpdag, DiscoveryError> = cpdag_from_adjacency_mask;
    let _: fn(u64, usize) -> Result<Admg, DiscoveryError> = admg_from_adjacency_mask;
    let _: fn(u64, u64, usize) -> Result<Pag, DiscoveryError> = pag_from_adjacency_mask;
    let _: fn(&Cpdag) -> Result<u64, DiscoveryError> = adjacency_mask_from_cpdag;
    let _: fn(&Admg) -> Result<u64, DiscoveryError> = adjacency_mask_from_admg;
    let _: fn(&Pag) -> Result<(u64, u64), DiscoveryError> = adjacency_masks_from_pag;
    let _: fn(u64, usize, usize, usize, bool) -> u64 =
        antecedent_discovery::graph_posterior::set_edge;
    let _: fn(u64, usize) -> bool = antecedent_discovery::graph_posterior::mask_is_dag;
    let _: fn(u64, u64, usize) -> Result<Pag, DiscoveryError> =
        antecedent_discovery::graph_posterior::pag_from_adjacency_mask;
}

#[test]
fn single_word_masks_round_trip_on_eight_nodes() {
    let mask = set_edge(set_edge(0, 8, 0, 7, true), 8, 7, 3, true);
    assert!(has_edge(mask, 8, 0, 7) && has_edge(mask, 8, 7, 3) && !has_edge(mask, 8, 3, 7));
    assert!(mask_is_dag(mask, 8));
    assert_eq!(parents_of(mask, 8, 3), vec![7]);
    dag_from_adjacency_mask(mask, 8).expect("a DAG on eight nodes");
    // A graph on nine nodes does not fit one word and is refused, not truncated.
    assert!(dag_from_adjacency_mask(0, 9).is_err());
}

#[test]
fn graph_posterior_keeps_its_single_word_fields_and_constructor() {
    let mut diagnostics = antecedent_prob::InferenceDiagnostics::analytic("released_surface");
    diagnostics.converged = true;
    let adjacency: Arc<[u64]> = Arc::from(vec![0_u64, set_edge(0, 2, 0, 1, true)]);
    let posterior = GraphPosterior::new(
        2,
        vec![0.5, 0.5],
        Arc::clone(&adjacency),
        vec![0.0, 0.5, 0.0, 0.0],
        vec![0.0, 0.5, 0.0, 0.0],
        SamplingSummary { ess: 2.0, diagnostics, rejected_invalid: 0 },
    )
    .expect("a valid posterior");
    let words: &Arc<[u64]> = &posterior.adjacency;
    assert_eq!(words.as_ref(), adjacency.as_ref());
    assert_eq!(posterior.atoms.len(), 2);
    let posterior = posterior
        .with_atom_kind(GraphPosteriorAtomKind::Pag)
        .with_mark_masks(vec![0_u64, 0])
        .expect("matching length");
    let marks: &Option<Arc<[u64]>> = &posterior.mark_masks;
    assert_eq!(marks.as_deref(), Some([0_u64, 0].as_slice()));
}

#[test]
fn temporal_mask_builders_keep_their_signatures() {
    use antecedent_discovery::dbn_posterior::{
        temporal_cpdag_from_dbn_masks, temporal_dag_from_dbn_masks, temporal_pag_from_dbn_masks,
    };
    use antecedent_graph::{TemporalCpdag, TemporalDag, TemporalPag};
    let _: fn(u64, u64, usize, u32, &[VariableId]) -> Result<TemporalDag, DiscoveryError> =
        temporal_dag_from_dbn_masks;
    let _: fn(u64, u64, usize, u32, &[VariableId]) -> Result<TemporalCpdag, DiscoveryError> =
        temporal_cpdag_from_dbn_masks;
    let _: fn(u64, u64, u64, usize, u32, &[VariableId]) -> Result<TemporalPag, DiscoveryError> =
        temporal_pag_from_dbn_masks;
    let _ = antecedent_discovery::temporal_dag_from_dbn_masks;
}

#[test]
fn matching_index_keeps_its_context_free_entry_points() {
    let _ = antecedent_stats::EXACT_MATCHING_ROW_LIMIT;
    let _ = antecedent_stats::matching::EXACT_MATCHING_ROW_LIMIT;
    let donors = [0.0, 1.0, 5.0];
    let index = MatchingIndex::exact(&donors, 1, &[10, 11, 12], MatchingDistance::Absolute)
        .expect("an index");
    let mut rows = [0usize; 2];
    let mut distances = [0.0; 2];
    let matched = index
        .match_all(&[0.9, 4.0], 2, None, &mut rows, &mut distances)
        .expect("both queries match");
    assert_eq!(matched, 2);
    assert_eq!(rows, [11, 12]);
    let mut kth = [0.0; 3];
    index.kth_self_distances_of_donors(1, &mut kth).expect("k-th self distances");
    assert_eq!(kth, [1.0, 1.0, 4.0]);
    let mut out = [0.0; 3];
    let mut scratch = Vec::new();
    MatchingIndex::kth_self_distances(
        &donors,
        3,
        1,
        MatchingDistance::Absolute,
        1,
        &mut out,
        &mut scratch,
    )
    .expect("k-th self distances");
    assert_eq!(out, kth);
}

#[test]
fn retired_ceilings_keep_their_released_values() {
    assert_eq!(antecedent_identify::sid::temporal_sequence::TEMPORAL_MAX_OBSERVED, 12);
    assert_eq!(antecedent_identify::sid::scenarios::SCENARIO_MAX_OBSERVED, 12);
    assert_eq!(antecedent_identify::MZ_TRANSPORT_MAX_OBSERVED, 12);
    assert_eq!(antecedent_identify::sid::MZ_TRANSPORT_MAX_OBSERVED, 12);
    assert_eq!(antecedent_identify::sid::cpdag_completion::CPDAG_MAX_NODES, 6);
    assert_eq!(
        antecedent_identify::sid::cpdag_completion::CPDAG_BOUNDS_EXCEEDED_DETAIL,
        "cpdag_scenarios.bounds_exceeded"
    );
    assert_eq!(antecedent_io::latent_class_artifact::LATENT_CLASS_ARTIFACT_MAX_ROWS, 10_000);
    assert_eq!(
        antecedent_io::nonlinear_mediation_artifact::NONLINEAR_MEDIATION_ARTIFACT_MAX_ROWS,
        20_000
    );
    assert_eq!(antecedent::analysis::latent_class_effects::LATENT_CLASS_ARTIFACT_MAX_ROWS, 10_000);
    assert_eq!(
        antecedent::analysis::nonlinear_mediation::NONLINEAR_MEDIATION_ARTIFACT_MAX_ROWS,
        20_000
    );
}

#[test]
fn refusal_constructor_and_matrix_rows_keep_their_shape() {
    let refusal = antecedent_identify::discovered_transport::refuse_pag_transport(Vec::new(), 0);
    let antecedent_identify::discovered_transport::DiscoveredTransportError::Refused(refusal) =
        refusal
    else {
        panic!("the constructor builds a refusal");
    };
    assert_eq!(refusal.detail, "discovered_transport.pag_members_not_causal_graphs");
    let rows: &[antecedent_counterfactual::selective::MatrixRow; 11] =
        &antecedent_counterfactual::selective::matrix::ROWS;
    assert_eq!(rows.len(), 11);
    assert!(antecedent_counterfactual::selective::matrix::rows().len() >= rows.len());
}

#[test]
fn design_record_functional_keeps_the_exact_count_label() {
    let _: fn(&antecedent_design::decision_reliability::CalibrationCoordinate) -> String =
        antecedent_design::decision_reliability::calibration::record_functional;
}

#[test]
fn notears_keeps_its_context_only_signature() {
    let _: fn(
        &TabularData,
        &[VariableId],
        &antecedent::discovery::StaticDiscoverParams,
        f64,
        f64,
        bool,
        &ExecutionContext,
    ) -> Result<
        antecedent_discovery::notears::NotearsDiscoveryResult,
        antecedent::CausalError,
    > = antecedent::discovery::discover_notears;
}

#[test]
fn adaptive_weighting_body_keeps_its_variant_shape() {
    use antecedent::analysis::ProcedureReportBody;
    fn is_weighting(body: &ProcedureReportBody) -> bool {
        matches!(
            body,
            ProcedureReportBody::AdaptiveWeighting {
                summary: _,
                truth_value: _,
                mean_enrolled: _,
                rule_stop_fraction: _,
                replications: _
            }
        )
    }
    let _ = is_weighting;
}
