//! The parameter bridge read against every declared structure form, against readings,
//! mixtures and laws computed by hand in each test.

use std::sync::Arc;

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, QuantityClass, ValueKind,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::execution::RngFactory;
use antecedent_core::{CausalSchemaBuilder, ExecutionContext};
use antecedent_discovery::adjacency_mask::set_edge;
use antecedent_discovery::{
    AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary,
};
use antecedent_graph::{Admg, Cpdag, CyclicMixedGraph, Dag, DenseNodeId};
use antecedent_identify::cyclic_id::CyclicIdPremises;
use antecedent_prob::InferenceDiagnostics;

use super::challenge::{DiagnosticStatus, ExcitationReading};
use super::hawkes::{self, HawkesParameters, INTERVENE};
use super::question::{before_release, release_window};
use super::{
    AcceptedExcitation, Arrival, Continuation, EventHistory, EventWindow, ExcitationSubject,
    ReleaseAction, ReleaseDraw,
};
use crate::AcceptedGraph;
use crate::Study;
use crate::accepted::StructureProvenance;
use crate::analysis::structure_members::{DeclaredStructure, MemberStatus, StructureForm};

const NAMES: [&str; 2] = ["release", "activity"];

fn node(index: u32) -> DenseNodeId {
    DenseNodeId::from_raw(index)
}

/// The horizon direct mass alpha (1 - e^{-beta H}) of the release -> activity edge, beta = 1,
/// H = 1.
fn direct(alpha: f64) -> f64 {
    alpha * (1.0 - (-1.0_f64).exp())
}

fn quantity(regime: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "edge.release_to_activity".into(),
        variable_name: "offspring".into(),
        role: QuantityRole::Outcome,
        units: "count".into(),
        population_id: "observed_stream".into(),
        regime_id: regime.into(),
        horizon: 1,
        functional_id: functional.into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn bridge(functional: &str, mapping: BridgeMapping) -> BridgeContract {
    let source = quantity("observational", functional);
    BridgeContract {
        kind: BridgeKind::ParameterInterventionResponse,
        source: BridgeEndpoint { quantity: source.clone(), class: QuantityClass::ModelParameter },
        target: BridgeEndpoint {
            quantity: quantity("do_one_release", functional),
            class: QuantityClass::InterventionResponse,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope {
            population_id: source.population_id.clone(),
            regime_id: source.regime_id.clone(),
        },
        mapping,
        assumptions: vec![BridgeAssumption {
            id: "structural_kernel".into(),
            kind: AssumptionKind::ParameterInterpretation,
            statement: "one added event yields this offspring under the declared kernel".into(),
            evidence: None,
        }],
        transport: None,
        mapping_uncertainty: None,
        source_dependence: None,
        estimand_bridge: None,
    }
}

fn agreed(alpha: f64) -> ExcitationReading {
    ExcitationReading {
        status: DiagnosticStatus::Agreed,
        reason: None,
        source: 0,
        target: 1,
        horizon: 1.0,
        conditional_direct: direct(alpha),
        alternative_direct: direct(alpha),
        lifetime_direct: alpha,
        lifetime_descendants: alpha,
        horizon_descendants: 0.0,
    }
}

fn history() -> EventHistory {
    EventHistory {
        catalog: NAMES.map(String::from).to_vec(),
        time_unit: "hour".into(),
        origin: "origin".into(),
        source: "log".into(),
        empty_origin: 0.0,
        window: EventWindow { start: 0.0, end: 1.0 },
        left: vec![],
        events: (1..9)
            .map(|index| Arrival {
                time: 0.1 * f64::from(index),
                kind: usize::try_from(index).unwrap_or(0) % 2,
            })
            .collect(),
        gaps: vec![],
    }
}

/// Release excites activity with mass `forward`; activity excites release with `backward`.
fn parameters(forward: f64, backward: f64) -> HawkesParameters {
    HawkesParameters {
        baseline: vec![0.2, 0.2],
        branching: vec![0.0, backward, forward, 0.0],
        decay: vec![0.4, 1.7, 1.0, 0.3],
    }
}

fn subject(parameters: HawkesParameters) -> ExcitationSubject {
    ExcitationSubject::Catalog { parameters, history: history(), column_cap: 0.9 }
}

fn schema(names: &[&str]) -> antecedent_core::CausalSchema {
    let mut builder =
        CausalSchemaBuilder::new().continuous(names[0]).treatment().continuous(names[1]).outcome();
    for &name in &names[2..] {
        builder = builder.continuous(name).context();
    }
    builder.build().unwrap()
}

fn dag(edges: &[(u32, u32)]) -> AcceptedGraph {
    let mut graph = Dag::with_variables(2);
    for &(from, to) in edges {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    AcceptedGraph::dag(graph).with_schema(&schema(&NAMES))
}

fn provenance() -> Arc<StructureProvenance> {
    Arc::new(StructureProvenance {
        method: "pc".into(),
        inference: "frequentist".into(),
        discovered_class: "cpdag".into(),
        query: [1; 32],
        search_bounds: "alpha=0.05;max_cond_size=1;max_lag=1;schedule=none".into(),
        data_snapshot: [2; 32],
        program: [3; 32],
        claim: [4; 32],
        review: "accepted".into(),
        resolved_marks: 0,
        structure_prior: None,
        pooling: None,
    })
}

fn accept(
    structure: DeclaredStructure,
    parameters: HawkesParameters,
    alpha: f64,
    bridge: BridgeContract,
) -> AcceptedExcitation {
    let ctx = ExecutionContext::for_tests(0);
    Study::prepare_excitation_question(agreed(alpha), subject(parameters), bridge, structure, &ctx)
        .unwrap()
        .execute(&ctx)
        .unwrap()
}

#[test]
fn an_unoriented_pair_reads_an_envelope_from_zero_to_the_mass() {
    // release - activity holds release -> activity (reads the mass) and activity -> release
    // (no directed path from release: an identified zero). Identity map: envelope [0, mass].
    let mut cpdag = Cpdag::with_variables(2);
    cpdag.insert_undirected(node(0), node(1)).unwrap();
    let graph = AcceptedGraph::cpdag(cpdag)
        .unwrap()
        .with_schema(&schema(&NAMES))
        .with_provenance(provenance());
    let accepted = accept(
        DeclaredStructure::graph(graph),
        parameters(0.4, 0.0),
        0.4,
        bridge("direct_offspring", BridgeMapping::Identity),
    );
    let mass = direct(0.4);
    let reading = &accepted.structure;
    assert_eq!(reading.form, StructureForm::Graph);
    assert_eq!(reading.members.len(), 2);
    assert!(reading.members.iter().all(|member| member.status == MemberStatus::Identified));
    assert!((reading.envelope[0] - 0.0).abs() < 1e-15);
    assert!((reading.envelope[1] - mass).abs() < 1e-12);
    assert_eq!(reading.mixture, None);
    assert!((reading.identified_weight - 1.0).abs() < 1e-12);
    assert_eq!(reading.evidence, crate::accepted::digest_hex(&provenance().digest()));
    assert!(reading.cyclic_reading.is_none());
}

#[test]
fn a_weighted_candidate_set_mixes_its_identified_readings() {
    // Weights 0.5, 0.3, 0.2 on release -> activity (mass), activity -> release (zero) and the
    // bow release -> activity, release <-> activity (unidentified). Identified weight 0.8;
    // conditional mixture (0.5 mass + 0.3 * 0) / 0.8 = 0.625 mass; envelope [0, mass].
    let mut bow = Admg::with_variables(2);
    bow.insert_directed(node(0), node(1)).unwrap();
    bow.insert_bidirected(node(0), node(1)).unwrap();
    let structure = DeclaredStructure::candidates(
        vec![dag(&[(0, 1)]), dag(&[(1, 0)]), AcceptedGraph::admg(bow).with_schema(&schema(&NAMES))],
        Some(vec![0.5, 0.3, 0.2]),
    );
    let identity = structure.identity().unwrap().to_hex();
    let accepted = accept(
        structure,
        parameters(0.4, 0.0),
        0.4,
        bridge("direct_offspring", BridgeMapping::Identity),
    );
    let mass = direct(0.4);
    let reading = &accepted.structure;
    assert_eq!(reading.form, StructureForm::Candidates);
    let bridged: Vec<Option<f64>> = reading.members.iter().map(|member| member.bridged).collect();
    assert!((bridged[0].unwrap() - mass).abs() < 1e-12);
    assert_eq!(bridged[1], Some(0.0));
    assert_eq!(bridged[2], None);
    assert_eq!(reading.members[2].status, MemberStatus::NotIdentified);
    assert!((reading.identified_weight - 0.8).abs() < 1e-12);
    assert!((reading.mixture.unwrap() - 0.625 * mass).abs() < 1e-12);
    assert!((reading.envelope[1] - mass).abs() < 1e-12);
    // A declared structure with no search behind it cites its canonical identity.
    assert_eq!(reading.evidence, identity);
}

#[test]
fn a_posterior_mixes_conditional_on_its_identified_atoms() {
    // ADMG atoms over release, activity, w, z with masses 0.6 (release -> activity), 0.3
    // (activity -> release) and 0.1 (release -> z -> activity, w -> activity, release <-> w
    // <-> z: unidentified by Tian's criterion). The edge probability of release -> activity is
    // 0.6; identified weight 0.9; the mixture conditional on identified atoms is 0.6 / 0.9 of
    // the mass.
    const N: usize = 4;
    let edges = |list: &[(usize, usize)]| {
        list.iter()
            .fold(AdjacencyMask::EMPTY, |mask, &(from, to)| set_edge(mask, N, from, to, true))
    };
    let mut marginals = vec![0.0; N * N];
    marginals[1] = 0.6;
    marginals[N] = 0.3;
    let posterior = GraphPosterior::from_atoms(
        N,
        vec![0.6, 0.3, 0.1],
        vec![
            edges(&[(0, 1)]),
            edges(&[(1, 0)]),
            edges(&[(0, 3), (3, 1), (2, 1), (0, 2), (2, 0), (2, 3), (3, 2)]),
        ],
        marginals.clone(),
        marginals,
        SamplingSummary {
            ess: 3.0,
            diagnostics: InferenceDiagnostics::analytic("structure_members"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Admg);
    assert!((posterior.edge_marginals()[1] - 0.6).abs() < 1e-15);
    let names = ["release", "activity", "w", "z"].map(String::from).to_vec();
    let accepted = accept(
        DeclaredStructure::posterior(posterior, names),
        parameters(0.4, 0.0),
        0.4,
        bridge("direct_offspring", BridgeMapping::Identity),
    );
    let mass = direct(0.4);
    let reading = &accepted.structure;
    assert_eq!(reading.form, StructureForm::Posterior);
    assert!((reading.identified_weight - 0.9).abs() < 1e-12);
    assert!((reading.mixture.unwrap() - 0.6 / 0.9 * mass).abs() < 1e-12);
    assert_eq!(reading.members[2].bridged, None);
}

#[test]
fn a_feedback_cycle_carries_the_total_offspring_of_the_resolvent() {
    // release -> activity with mass a = 0.4 and activity -> release with b = 0.3. The
    // resolvent column of release is [(I - A)^{-1} - I]_{., release} = [ab, a] / (1 - ab), so
    // the lifetime total offspring of activity is a / (1 - ab): the direct mass a plus the
    // feedback descendants a^2 b / (1 - ab).
    let (a, b) = (0.4_f64, 0.3_f64);
    let total = a / (1.0 - a * b);
    let mut graph = CyclicMixedGraph::with_variables(2);
    graph.insert_directed(node(0), node(1)).unwrap();
    graph.insert_directed(node(1), node(0)).unwrap();
    let structure = DeclaredStructure::cyclic(
        graph,
        NAMES.map(String::from).to_vec(),
        CyclicIdPremises { io_scm: true, scc_product_density: true },
    );
    let identity = accept(
        structure.clone(),
        parameters(a, b),
        a,
        bridge("total_offspring", BridgeMapping::Identity),
    );
    assert!((identity.bridged_direct - total).abs() < 1e-12);
    assert!((identity.bridged_direct - (a + a * a * b / (1.0 - a * b))).abs() < 1e-12);
    assert_eq!(identity.structure.functional, "total_offspring");
    assert!(identity.structure.members[0].relation.feedback());
    assert!(identity.structure.cyclic_reading.is_none());
    // Under an affine map the direct mass crosses and the resolvent reading sits beside it.
    let affine = accept(
        structure,
        parameters(a, b),
        a,
        bridge("direct_offspring", BridgeMapping::Affine { intercept: 0.0, slope: 2.0 }),
    );
    assert!((affine.bridged_direct - 2.0 * direct(a)).abs() < 1e-12);
    assert!((affine.structure.cyclic_reading.unwrap() - total).abs() < 1e-12);
    // A total offspring offered to an affine map refuses.
    let ctx = ExecutionContext::for_tests(0);
    let refused = Study::prepare_excitation_question(
        agreed(a),
        subject(parameters(a, b)),
        bridge("total_offspring", BridgeMapping::Affine { intercept: 0.0, slope: 2.0 }),
        dag(&[(0, 1)]),
        &ctx,
    )
    .unwrap()
    .execute(&ctx);
    let Err(error) = refused else { panic!("an affine total offspring crossed the bridge") };
    assert_eq!(
        error.detail(),
        Some(antecedent_core::reason_details::events::DESCENDANT_NOT_CARRIED)
    );
}

#[test]
fn a_two_way_kernel_on_an_oriented_dag_reports_the_resolvent_beside_the_direct_mass() {
    let (a, b) = (0.4_f64, 0.3_f64);
    let accepted = accept(
        DeclaredStructure::graph(dag(&[(0, 1)]).with_provenance(provenance())),
        parameters(a, b),
        a,
        bridge("direct_offspring", BridgeMapping::Affine { intercept: 0.0, slope: 2.0 }),
    );
    assert!((accepted.bridged_direct - 2.0 * direct(a)).abs() < 1e-12);
    assert!((accepted.structure.cyclic_reading.unwrap() - a / (1.0 - a * b)).abs() < 1e-12);
}

#[test]
fn a_mixture_law_draws_null_members_on_the_same_streams() {
    // Candidates release -> activity (0.25) and activity -> release (0.75). The law draws
    // each path index twice on one stream: with the added release exciting, then without it.
    let ctx = ExecutionContext::for_tests(0);
    let observed = history();
    let kernel = parameters(0.4, 0.0);
    let accepted = accept(
        DeclaredStructure::candidates(vec![dag(&[(0, 1)]), dag(&[(1, 0)])], Some(vec![1.0, 3.0])),
        kernel.clone(),
        0.4,
        bridge("direct_offspring", BridgeMapping::Identity),
    );
    let paths = 200;
    let draw = ReleaseDraw { release: 0.0, window_end: 2.0, paths, max_events: 64, seed: 11 };
    let law = accepted
        .release_law(Continuation::Catalog(&observed), ReleaseAction::Now, draw, &ctx)
        .unwrap();
    assert_eq!(law.paths().len(), 2 * paths);
    let weights = law.weights().unwrap();
    assert!(weights[..paths].iter().all(|w| (w - 0.25 / paths as f64).abs() < 1e-15));
    assert!(weights[paths..].iter().all(|w| (w - 0.75 / paths as f64).abs() < 1e-15));
    // The null group is exactly the continuation without the added event on stream i; the
    // oriented group is the continuation with it, on the same stream.
    let window = release_window(ReleaseDraw { release: 1.0, ..draw }).unwrap();
    let rng = RngFactory::from_seed(11);
    let mut work = hawkes::Work::new(&ctx);
    let added = Arrival { time: 1.0, kind: 0 };
    for index in 0..paths {
        let prefix = before_release(&observed, 1.0);
        let without = hawkes::sample(
            &observed,
            &prefix,
            &kernel,
            window,
            (&rng, INTERVENE, index as u64),
            64,
            &mut work,
        )
        .unwrap();
        assert_eq!(law.paths()[paths + index], without);
        let mut excited = prefix.clone();
        excited.push(added);
        let with = hawkes::sample(
            &observed,
            &excited,
            &kernel,
            window,
            (&rng, INTERVENE, index as u64),
            64,
            &mut work,
        )
        .unwrap();
        assert_eq!(law.paths()[index][0], added);
        assert_eq!(law.paths()[index][1..], with[..]);
    }
    // The weighted mean activity count of the mixture equals the weighted member means.
    let activity = |path: &Vec<Arrival>| path.iter().filter(|event| event.kind == 1).count() as f64;
    let mean = |range: std::ops::Range<usize>| {
        law.paths()[range.clone()].iter().map(activity).sum::<f64>() / range.len() as f64
    };
    let mixture: f64 =
        law.paths().iter().zip(weights).map(|(path, weight)| weight * activity(path)).sum();
    let members = 0.25 * mean(0..paths) + 0.75 * mean(paths..2 * paths);
    assert!((mixture - members).abs() < 1e-12, "{mixture} against {members}");
}

#[test]
fn a_structure_no_member_identifies_refuses() {
    let mut bow = Admg::with_variables(2);
    bow.insert_directed(node(0), node(1)).unwrap();
    bow.insert_bidirected(node(0), node(1)).unwrap();
    let ctx = ExecutionContext::for_tests(0);
    let refused = Study::prepare_excitation_question(
        agreed(0.4),
        subject(parameters(0.4, 0.0)),
        bridge("direct_offspring", BridgeMapping::Identity),
        AcceptedGraph::admg(bow).with_schema(&schema(&NAMES)),
        &ctx,
    );
    let Err(error) = refused else { panic!("an unidentified structure was read") };
    assert_eq!(
        error.detail(),
        Some(antecedent_core::reason_details::events::READING_NOT_IDENTIFIED)
    );
}

#[test]
fn a_single_dag_keeps_its_program_identity() {
    // The identity of a question read against one accepted DAG is the digest a discovered DAG
    // always had: the reading, the subject, the bridge and the provenance digest.
    let ctx = ExecutionContext::for_tests(0);
    let graph = dag(&[(0, 1)]).with_provenance(provenance());
    let reading = agreed(0.4);
    let bridge = bridge("direct_offspring", BridgeMapping::Identity);
    let prepared = Study::prepare_excitation_question(
        reading.clone(),
        subject(parameters(0.4, 0.0)),
        bridge.clone(),
        &graph,
        &ctx,
    )
    .unwrap();
    let subject_hex = super::digest(
        "antecedent.events.excitation_subject.catalog.v1",
        &(&parameters(0.4, 0.0), &history(), &0.9_f64),
    )
    .unwrap()
    .to_hex();
    let source = super::digest(
        "antecedent.events.excitation_question.v2",
        &(
            &reading,
            subject_hex,
            bridge.identity().to_hex(),
            Some(crate::accepted::digest_hex(&provenance().digest())),
        ),
    )
    .unwrap();
    let expected = super::PreparedKernel::on(
        crate::planner::lowered::EventProcessRoute::ParameterBridge,
        source,
        "excitation_question",
    )
    .unwrap();
    assert_eq!(prepared.plan_view().identity(), expected.plan_view().identity());
}
