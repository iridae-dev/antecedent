//! 2.3 B2 / X10 second row: decision, witness and refusals of the ordered-response
//! (`R -> R`) recovery row. Numerical truth from an enumerated SCM lives in
//! `antecedent-estimate/tests/b2_recovery_chain_truth.rs`.
//!
//! Role ids: `X1 = 0, X2 = 1, R1 = 2, R2 = 3, X*1 = 4, X*2 = 5` unless relabelled.

use std::collections::BTreeMap;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{
    ChainPartial, ChainRecoveryDecision, ChainRecoveryDetail, ChainRecoveryError,
    ChainRecoveryQuery, decide_chain_recovery, verify_chain_witness,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

/// Graph over the raw ids in `order` (node insertion order) with directed `edges`
/// and optional bidirected pairs.
fn build(order: &[u32], edges: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::empty();
    let mut dense: BTreeMap<u32, DenseNodeId> = BTreeMap::new();
    for raw in order {
        let id = graph.add_node(NodeRef::Static(VariableId::from_raw(*raw))).unwrap();
        dense.insert(*raw, id);
    }
    for (a, b) in edges {
        graph.insert_directed(dense[a], dense[b]).unwrap();
    }
    for (a, b) in bidirected {
        graph.insert_bidirected(dense[a], dense[b]).unwrap();
    }
    graph
}

/// `ids = [X1, X2, R1, R2, X*1, X*2]`.
fn query(ids: [u32; 6]) -> ChainRecoveryQuery {
    let v = VariableId::from_raw;
    ChainRecoveryQuery {
        first: ChainPartial { variable: v(ids[0]), response: v(ids[2]), proxy: v(ids[4]) },
        second: ChainPartial { variable: v(ids[1]), response: v(ids[3]), proxy: v(ids[5]) },
    }
}

const IDS: [u32; 6] = [0, 1, 2, 3, 4, 5];

/// Edges of the row with the chain head on axis `head`; `dependent` adds
/// `X_head -> R_tail`; `extra` appends further edges.
fn edges(ids: [u32; 6], head: usize, dependent: bool, extra: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let [x1, x2, r1, r2, p1, p2] = ids;
    let mut out = vec![(x1, x2), (x1, p1), (r1, p1), (x2, p2), (r2, p2)];
    if head == 0 {
        out.push((r1, r2));
        if dependent {
            out.push((x1, r2));
        }
    } else {
        out.push((r2, r1));
        if dependent {
            out.push((x2, r1));
        }
    }
    out.extend_from_slice(extra);
    out
}

fn decide(
    ids: [u32; 6],
    order: &[u32],
    edge_list: &[(u32, u32)],
    bidirected: &[(u32, u32)],
) -> Result<ChainRecoveryDecision, ChainRecoveryError> {
    decide_chain_recovery(&build(order, edge_list, bidirected), &query(ids), &ctx())
}

fn recovered(decision: ChainRecoveryDecision) -> antecedent_identify::ChainRecoveryPlan {
    match decision {
        ChainRecoveryDecision::Recovered(plan) => *plan,
        ChainRecoveryDecision::NonRecoverable(_) => panic!("expected a recovery plan"),
    }
}

#[test]
fn b2_chain_graph_is_recovered_with_the_head_response_first() {
    let plan = recovered(decide(IDS, &IDS, &edges(IDS, 0, true, &[]), &[]).unwrap());
    assert_eq!(plan.head, 0);
    assert!(plan.tail_depends_on_head_variable);
    assert_eq!(plan.rule_version, "b2.recovery_chain.v1");
    assert!(plan.premises.contains(&"response_chain:2->3".to_owned()));
    assert!(plan.operations_consumed > 0);

    let independent = recovered(decide(IDS, &IDS, &edges(IDS, 0, false, &[]), &[]).unwrap());
    assert_eq!(independent.head, 0);
    assert!(!independent.tail_depends_on_head_variable);

    // The mirrored chain R2 -> R1 (with X2 -> R1) has its head on the second axis.
    let mirrored = recovered(decide(IDS, &IDS, &edges(IDS, 1, true, &[]), &[]).unwrap());
    assert_eq!(mirrored.head, 1);
    assert!(mirrored.tail_depends_on_head_variable);
}

#[test]
fn b2_chain_decision_is_invariant_to_node_order_and_labels() {
    let reference = recovered(decide(IDS, &IDS, &edges(IDS, 0, true, &[]), &[]).unwrap());
    // Relabelled variables, nodes inserted in reverse order.
    let relabelled: [u32; 6] = [41, 17, 9, 30, 2, 25];
    let mut reversed = relabelled;
    reversed.reverse();
    let permuted =
        recovered(decide(relabelled, &reversed, &edges(relabelled, 0, true, &[]), &[]).unwrap());
    assert_eq!(permuted.head, reference.head);
    assert_eq!(permuted.tail_depends_on_head_variable, reference.tail_depends_on_head_variable);
    assert_eq!(permuted.formula, reference.formula);
    assert_eq!(permuted.rule_version, reference.rule_version);
}

#[test]
fn b2_self_censoring_chain_is_nonrecoverable_with_an_exact_witness() {
    // X2 -> R2 on top of the chain: P(X2) is not recoverable from the pattern law.
    let extra = [(1, 3)];
    let graph = build(&IDS, &edges(IDS, 0, true, &extra), &[]);
    let q = query(IDS);
    let ChainRecoveryDecision::NonRecoverable(witness) =
        decide_chain_recovery(&graph, &q, &ctx()).unwrap()
    else {
        panic!("a self-censoring edge is nonrecoverable");
    };
    assert_eq!(witness.edge, (1, 3));
    let check = verify_chain_witness(&graph, &q, &witness, &ctx()).unwrap();
    assert_eq!(check.observed_cells, 9);
    assert_eq!(check.denominator, 60u128.pow(4));
    assert_eq!(check.differing_cell, (0, 0));
    // P(X1 = 0) P(X2 = 0) over 60^4: (30 / 60)(30 / 60) against (30 / 60)(36 / 60).
    assert_eq!(check.masses, (30 * 30 * 3600, 30 * 36 * 3600));
    assert_eq!(
        ChainRecoveryDetail::NonrecoverableWitness.reason_code(),
        "transport_proven_non_transportable"
    );
    assert_eq!(
        ChainRecoveryDetail::NonrecoverableWitness.detail(),
        "recovery_chain.nonrecoverable_witness"
    );
}

#[test]
fn b2_head_self_censoring_and_independent_tail_are_also_witnessed() {
    // X1 -> R1 (head self-censoring) with an independent tail.
    let extra = [(0, 2)];
    let graph = build(&IDS, &edges(IDS, 0, false, &extra), &[]);
    let q = query(IDS);
    let ChainRecoveryDecision::NonRecoverable(witness) =
        decide_chain_recovery(&graph, &q, &ctx()).unwrap()
    else {
        panic!("a self-censoring edge is nonrecoverable");
    };
    assert_eq!(witness.edge, (0, 2));
    let check = verify_chain_witness(&graph, &q, &witness, &ctx()).unwrap();
    assert_eq!(check.observed_cells, 9);
    assert_ne!(check.masses.0, check.masses.1);
}

#[test]
fn b2_a_tampered_witness_fails_the_exact_verifier() {
    let extra = [(1, 3)];
    let graph = build(&IDS, &edges(IDS, 0, true, &extra), &[]);
    let q = query(IDS);
    let ChainRecoveryDecision::NonRecoverable(witness) =
        decide_chain_recovery(&graph, &q, &ctx()).unwrap()
    else {
        panic!("nonrecoverable");
    };
    // Perturbing a response mechanism of the second model breaks observed equality.
    let mut observed_broken = (*witness).clone();
    observed_broken.second[3].numerators[0] += 1;
    let error = verify_chain_witness(&graph, &q, &observed_broken, &ctx()).unwrap_err();
    assert_eq!(error.detail, ChainRecoveryDetail::InvalidDerivation);
    // Making the models identical leaves no target difference.
    let mut same = (*witness).clone();
    same.second = same.first.clone();
    let error = verify_chain_witness(&graph, &q, &same, &ctx()).unwrap_err();
    assert_eq!(error.detail, ChainRecoveryDetail::InvalidDerivation);
    // A mechanism that disagrees with the graph parents is refused.
    let mut wrong_parents = (*witness).clone();
    wrong_parents.first[0].parents = vec![9];
    assert_eq!(
        verify_chain_witness(&graph, &q, &wrong_parents, &ctx()).unwrap_err().detail,
        ChainRecoveryDetail::InvalidDerivation
    );
}

fn assert_not_supported(result: Result<ChainRecoveryDecision, ChainRecoveryError>) {
    let error = result.unwrap_err();
    assert_eq!(error.detail, ChainRecoveryDetail::UnsupportedMechanism, "{error}");
    assert_eq!(error.reason_code(), "route_not_supported");
    assert_eq!(error.detail.detail(), "recovery_chain.unsupported_mechanism");
}

#[test]
fn b2_adjacent_graphs_are_not_supported_never_nonrecoverable() {
    // The tail variable drives the head response: X2 -> R1 with chain R1 -> R2.
    assert_not_supported(decide(IDS, &IDS, &edges(IDS, 0, true, &[(1, 2)]), &[]));
    // No response chain: the 2.2 route's graph, not this row's.
    let no_chain = vec![(0, 1), (0, 4), (2, 4), (1, 5), (3, 5), (0, 3)];
    assert_not_supported(decide(IDS, &IDS, &no_chain, &[]));
    // Bidirected edge (unmeasured confounding).
    assert_not_supported(decide(IDS, &IDS, &edges(IDS, 0, true, &[]), &[(0, 1)]));
    // A response drives the other variable's proxy.
    assert_not_supported(decide(IDS, &IDS, &edges(IDS, 0, true, &[(2, 5)]), &[]));
    // A response causes a substantive variable.
    assert_not_supported(decide(IDS, &IDS, &edges(IDS, 0, false, &[(3, 1)]), &[]));
    // An unsupported edge outranks a self-censoring edge: no witness is claimed.
    assert_not_supported(decide(IDS, &IDS, &edges(IDS, 0, true, &[(1, 3), (1, 2)]), &[]));
}

#[test]
fn b2_malformed_roles_are_invalid_queries() {
    // A proxy not wired to its response.
    let mut broken = edges(IDS, 0, true, &[]);
    broken.retain(|e| *e != (3, 5));
    let error = decide(IDS, &IDS, &broken, &[]).unwrap_err();
    assert_eq!(error.detail, ChainRecoveryDetail::InvalidQuery);
    assert_eq!(error.reason_code(), "invalid_argument");
    // A role declared twice.
    let duplicate = ChainRecoveryQuery {
        first: query(IDS).first,
        second: ChainPartial {
            variable: VariableId::from_raw(0),
            response: VariableId::from_raw(3),
            proxy: VariableId::from_raw(5),
        },
    };
    let graph = build(&IDS, &edges(IDS, 0, true, &[]), &[]);
    let error = decide_chain_recovery(&graph, &duplicate, &ctx()).unwrap_err();
    assert_eq!(error.detail, ChainRecoveryDetail::InvalidQuery);
    // A graph node without a role.
    let extra_node = [0, 1, 2, 3, 4, 5, 6];
    let graph = build(&extra_node, &edges(IDS, 0, true, &[]), &[]);
    let error = decide_chain_recovery(&graph, &query(IDS), &ctx()).unwrap_err();
    assert_eq!(error.detail, ChainRecoveryDetail::InvalidQuery);
}

#[test]
fn b2_reason_codes_and_details_are_fixed_per_kind() {
    use ChainRecoveryDetail::{
        Budget, InvalidDerivation, InvalidObservedLaw, InvalidQuery, NonrecoverableWitness,
        Positivity, UnsupportedMechanism,
    };
    let table = [
        (NonrecoverableWitness, "recovery_chain.nonrecoverable_witness"),
        (UnsupportedMechanism, "recovery_chain.unsupported_mechanism"),
        (InvalidQuery, "recovery_chain.invalid_query"),
        (Positivity, "recovery_chain.positivity"),
        (InvalidObservedLaw, "recovery_chain.invalid_observed_law"),
        (InvalidDerivation, "recovery_chain.invalid_derivation"),
        (Budget, "recovery_chain.budget"),
    ];
    for (kind, detail) in table {
        assert_eq!(kind.detail(), detail);
    }
    assert_eq!(Positivity.reason_code(), "transport_support_failure");
    assert_eq!(Budget.reason_code(), "transport_budget_cancel");
    assert_eq!(InvalidDerivation.reason_code(), "transport_not_certified");
}
