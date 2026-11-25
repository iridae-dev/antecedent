//! Member enumeration and per-member identification of each declared structure form, against
//! members and identification statuses written out by hand.

use std::sync::Arc;

use antecedent_core::{CausalSchemaBuilder, ExecutionContext};
use antecedent_discovery::adjacency_mask::set_edge;
use antecedent_discovery::{
    AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary,
};
use antecedent_graph::{Admg, Cpdag, CyclicMixedGraph, Dag, DenseNodeId};
use antecedent_identify::cyclic_id::CyclicIdPremises;
use antecedent_prob::InferenceDiagnostics;

use super::{CyclicReading, DeclaredStructure, MemberStatus, StructureForm};
use crate::AcceptedGraph;

fn node(index: u32) -> DenseNodeId {
    DenseNodeId::from_raw(index)
}

fn schema(names: &[&str]) -> antecedent_core::CausalSchema {
    let mut builder =
        CausalSchemaBuilder::new().continuous(names[0]).treatment().continuous(names[1]).outcome();
    for &name in &names[2..] {
        builder = builder.continuous(name).context();
    }
    builder.build().unwrap()
}

fn dag(names: &[&str], edges: &[(u32, u32)]) -> AcceptedGraph {
    let mut graph = Dag::with_variables(u32::try_from(names.len()).unwrap());
    for &(from, to) in edges {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    AcceptedGraph::dag(graph).with_schema(&schema(names))
}

fn premises() -> CyclicIdPremises {
    CyclicIdPremises { io_scm: true, scc_product_density: true }
}

#[test]
fn an_unoriented_pair_has_both_orientations_as_identified_members() {
    // release - activity: the class holds release -> activity and activity -> release. The
    // complete ID algorithm identifies both: P(activity | do(release)) is P(activity | release)
    // in the first and P(activity) in the second, where no directed path leaves release.
    let ctx = ExecutionContext::for_tests(0);
    let mut cpdag = Cpdag::with_variables(2);
    cpdag.insert_undirected(node(0), node(1)).unwrap();
    let names = ["release", "activity"];
    let graph = AcceptedGraph::cpdag(cpdag).unwrap().with_schema(&schema(&names));
    let members = DeclaredStructure::graph(graph)
        .members("release", "activity", CyclicReading::Lagged, &ctx)
        .unwrap();
    assert_eq!(members.form(), StructureForm::Graph);
    assert!(!members.weighted());
    assert!(!members.truncated());
    assert_eq!(members.members().len(), 2);
    let forward = members.members().iter().filter(|m| m.relation().forward()).count();
    let reverse = members.members().iter().filter(|m| m.relation().reverse()).count();
    assert_eq!((forward, reverse), (1, 1));
    assert!(members.members().iter().all(|m| m.status() == MemberStatus::Identified));
    // Uniform enumeration weight: the reading 1 on the forward member and 0 on the reverse one.
    let summary = members.summarize(|m| if m.relation().forward() { 1.0 } else { 0.0 });
    assert_eq!(summary.envelope, Some([0.0, 1.0]));
    assert_eq!(summary.mixture, None);
    assert!((summary.identified_weight - 1.0).abs() < 1e-12);
}

#[test]
fn a_weighted_candidate_set_keeps_an_unidentified_bow_out_of_the_mixture() {
    // Candidates: release -> activity (0.5), activity -> release (0.3), and an ADMG with
    // release -> activity and release <-> activity (0.2), a bow the ID algorithm cannot
    // identify. Identified weight 0.8; the forward reading 1 mixes to 0.5 / 0.8.
    let ctx = ExecutionContext::for_tests(0);
    let names = ["release", "activity"];
    let mut bow = Admg::with_variables(2);
    bow.insert_directed(node(0), node(1)).unwrap();
    bow.insert_bidirected(node(0), node(1)).unwrap();
    let structure = DeclaredStructure::candidates(
        vec![
            dag(&names, &[(0, 1)]),
            dag(&names, &[(1, 0)]),
            AcceptedGraph::admg(bow).with_schema(&schema(&names)),
        ],
        Some(vec![5.0, 3.0, 2.0]),
    );
    let members = structure.members("release", "activity", CyclicReading::Lagged, &ctx).unwrap();
    assert_eq!(members.form(), StructureForm::Candidates);
    assert!(members.weighted());
    let statuses: Vec<_> = members.members().iter().map(super::StructureMember::status).collect();
    assert_eq!(
        statuses,
        [MemberStatus::Identified, MemberStatus::Identified, MemberStatus::NotIdentified]
    );
    assert!(members.members()[2].relation().bidirected());
    let summary = members.summarize(|m| if m.relation().forward() { 1.0 } else { 0.0 });
    assert_eq!(summary.envelope, Some([0.0, 1.0]));
    assert!((summary.identified_weight - 0.8).abs() < 1e-12);
    assert!((summary.mixture.unwrap() - 0.625).abs() < 1e-12);
}

#[test]
fn a_posterior_reads_its_atoms_with_their_mass() {
    // ADMG atoms over release, activity, w, z. Atom A (0.6): release -> activity. Atom B
    // (0.3): activity -> release. Atom C (0.1): release -> z -> activity, w -> activity,
    // release <-> w <-> z: a bidirected path from release to its child z inside the
    // ancestors of activity, so Tian's criterion leaves C unidentified. The edge marginal of
    // release -> activity is 0.6; identified weight 0.9; conditional mixture 0.6 / 0.9.
    const N: usize = 4;
    let ctx = ExecutionContext::for_tests(0);
    let edges = |list: &[(usize, usize)]| {
        list.iter()
            .fold(AdjacencyMask::EMPTY, |mask, &(from, to)| set_edge(mask, N, from, to, true))
    };
    let atoms = vec![
        edges(&[(0, 1)]),
        edges(&[(1, 0)]),
        edges(&[(0, 3), (3, 1), (2, 1), (0, 2), (2, 0), (2, 3), (3, 2)]),
    ];
    let mut marginals = vec![0.0; N * N];
    marginals[1] = 0.6;
    marginals[N] = 0.3;
    let posterior = GraphPosterior::from_atoms(
        N,
        vec![0.6, 0.3, 0.1],
        atoms,
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
    let names = ["release", "activity", "w", "z"].map(String::from).to_vec();
    let members = DeclaredStructure::posterior(posterior, names)
        .members("release", "activity", CyclicReading::Lagged, &ctx)
        .unwrap();
    assert_eq!(members.form(), StructureForm::Posterior);
    let statuses: Vec<_> = members.members().iter().map(super::StructureMember::status).collect();
    assert_eq!(
        statuses,
        [MemberStatus::Identified, MemberStatus::Identified, MemberStatus::NotIdentified]
    );
    let summary = members.summarize(|m| if m.relation().forward() { 1.0 } else { 0.0 });
    assert!((summary.identified_weight - 0.9).abs() < 1e-12);
    assert!((summary.mixture.unwrap() - 0.6 / 0.9).abs() < 1e-12);
}

#[test]
fn a_feedback_pair_is_identified_when_its_edges_are_lags() {
    // release <-> activity as a cycle. Read as lags, release@t-1 -> activity@t and
    // activity@t-1 -> release@t: a pulse of release at t reaches activity at t+1 only along
    // its own edge, so the unrolled member identifies.
    let ctx = ExecutionContext::for_tests(0);
    let mut graph = CyclicMixedGraph::with_variables(2);
    graph.insert_directed(node(0), node(1)).unwrap();
    graph.insert_directed(node(1), node(0)).unwrap();
    let names = vec!["release".to_owned(), "activity".to_owned()];
    let structure = DeclaredStructure::cyclic(graph, names, premises());
    let members = structure.members("release", "activity", CyclicReading::Lagged, &ctx).unwrap();
    assert_eq!(members.form(), StructureForm::Cyclic);
    assert_eq!(members.members().len(), 1);
    assert!(members.members()[0].relation().feedback());
    assert_eq!(members.members()[0].status(), MemberStatus::Identified);
}

#[test]
fn identity_covers_the_form_the_graph_and_its_labels() {
    let names = ["release", "activity"];
    let forward = DeclaredStructure::graph(dag(&names, &[(0, 1)])).identity().unwrap();
    let reverse = DeclaredStructure::graph(dag(&names, &[(1, 0)])).identity().unwrap();
    let renamed =
        DeclaredStructure::graph(dag(&["launch", "activity"], &[(0, 1)])).identity().unwrap();
    let candidates =
        DeclaredStructure::candidates(vec![dag(&names, &[(0, 1)])], None).identity().unwrap();
    assert_ne!(forward, reverse);
    assert_ne!(forward, renamed);
    assert_ne!(forward, candidates);
    assert_eq!(forward, DeclaredStructure::graph(dag(&names, &[(0, 1)])).identity().unwrap());
    let provenance = Arc::new(crate::accepted::StructureProvenance {
        method: "pc".into(),
        inference: "frequentist".into(),
        discovered_class: "dag".into(),
        query: [1; 32],
        search_bounds: "alpha=0.05".into(),
        data_snapshot: [2; 32],
        program: [3; 32],
        claim: [4; 32],
        review: "accepted".into(),
        resolved_marks: 0,
        structure_prior: None,
        pooling: None,
    });
    let cited = DeclaredStructure::graph(dag(&names, &[(0, 1)]))
        .with_provenance(provenance)
        .identity()
        .unwrap();
    assert_ne!(forward, cited);
}

#[test]
fn names_outside_the_structure_refuse() {
    let ctx = ExecutionContext::for_tests(0);
    let structure = DeclaredStructure::graph(dag(&["release", "activity"], &[(0, 1)]));
    assert!(structure.members("release", "launch", CyclicReading::Lagged, &ctx).is_err());
    assert!(structure.members("release", "release", CyclicReading::Lagged, &ctx).is_err());
    let unbound = DeclaredStructure::graph(AcceptedGraph::dag(Dag::with_variables(2)));
    assert!(unbound.members("release", "activity", CyclicReading::Lagged, &ctx).is_err());
}

#[test]
fn site_a_member_is_identified_only_under_the_shared_point_rule() {
    use antecedent_core::IdentificationStatus;
    for status in IdentificationStatus::ALL {
        let expected = if antecedent_io_core::contract_section::status_identifies_a_point(status) {
            MemberStatus::Identified
        } else {
            MemberStatus::NotIdentified
        };
        assert_eq!(super::member_status(status), expected, "{status:?}");
    }
    // A partially identified member gives a set, never the member's effect.
    assert_eq!(
        super::member_status(IdentificationStatus::PartiallyIdentified),
        MemberStatus::NotIdentified
    );
}

fn cyclic_graph(edges: &[(u32, u32)]) -> Option<CyclicMixedGraph> {
    let mut graph = CyclicMixedGraph::with_variables(3);
    for &(from, to) in edges {
        graph.insert_directed(node(from), node(to)).ok()?;
    }
    Some(graph)
}

/// `DeclaredStructure::identity`: candidate graphs and their weights are ordered (members are
/// enumerated and weighted in candidate order); a cyclic graph's edges are a set
/// (`CyclicIdentity` writes them sorted); weights enter by their bits.
#[test]
fn site_structure_members_declared_identity_orders_candidates_and_sets_cyclic_edges() {
    let names = ["release", "activity", "w"];
    let graphs = [dag(&names, &[(0, 1)]), dag(&names, &[(1, 0)]), dag(&names, &[(0, 1), (2, 1)])];
    let candidates = |order: &[usize], weights: Vec<f64>| {
        let chosen = order.iter().map(|&i| graphs[i].clone()).collect();
        DeclaredStructure::candidates(chosen, Some(weights)).identity().ok()
    };
    crate::identity_laws::ordered_laws(&[0_usize, 1, 2], |order| {
        candidates(order, vec![5.0, 3.0, 2.0])
    });
    crate::identity_laws::ordered_laws(&[5.0, 3.0, 2.0], |weights| {
        candidates(&[0, 1, 2], weights.to_vec())
    });
    crate::identity_laws::bit_exact(|x| candidates(&[0, 1, 2], vec![x, 3.0, 2.0]));

    let cyclic_names: Vec<String> = names.iter().map(|name| (*name).to_owned()).collect();
    crate::identity_laws::set_laws(&[(0_u32, 1_u32), (1, 0), (1, 2)], |edges| {
        DeclaredStructure::cyclic(cyclic_graph(edges)?, cyclic_names.clone(), premises())
            .identity()
            .ok()
    });
}

/// `member_key`: a member's edges are a set (the graph's canonical identity sorts them, and
/// a cyclic member's `CyclicIdentity` writes them sorted); each member kind is its own tag.
#[test]
fn site_structure_members_member_key_sets_edges_and_tags_kinds() {
    crate::identity_laws::set_laws(&[(0_u32, 1_u32), (0, 2), (1, 2)], |edges| {
        let mut graph = Dag::with_variables(3);
        for &(from, to) in edges {
            graph.insert_directed(node(from), node(to)).ok()?;
        }
        super::member_key(&super::MemberGraph::Dag(graph)).ok()
    });
    crate::identity_laws::set_laws(&[(0_u32, 1_u32), (1, 0), (1, 2)], |edges| {
        super::member_key(&super::MemberGraph::Cyclic(cyclic_graph(edges)?)).ok()
    });
    crate::identity_laws::all_distinct(&[
        super::member_key(&super::MemberGraph::Dag(Dag::with_variables(2))).expect("a key"),
        super::member_key(&super::MemberGraph::Admg(Admg::with_variables(2))).expect("a key"),
        super::member_key(&super::MemberGraph::Cyclic(CyclicMixedGraph::with_variables(2)))
            .expect("a key"),
    ]);
}
