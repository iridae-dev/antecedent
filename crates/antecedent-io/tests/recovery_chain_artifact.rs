//! 2.3 B2 (X10 second row): the ordered-response recovery artifact through its re-deciding
//! consumer.
//!
//! The truth is the enumerated binary SCM of `antecedent-estimate/tests/
//! b2_recovery_chain_truth.rs`: `P(X1 = 1) = 0.37`, `P(X2 = 1 | X1) = (0.2, 0.7)`, head
//! response `P(R_h = 1) = 0.65`, tail response `P(R_t = 1 | R_h, X_h)` =
//! `[[0.4, 0.55], [0.3, 0.8]]` (dependent) or `P(R_t = 1 | R_h) = (0.3, 0.8)` (independent).
//! The nonrecoverable fixture adds the self-censoring edge `X2 -> R2`. A fresh consumer must
//! re-decide, re-evaluate and re-verify from the artifact alone and refuse every change,
//! including one resealed with recomputed digests.
//!
//! Role ids: `X1 = 0, X2 = 1, R1 = 2, R2 = 3, X*1 = 4, X*2 = 5` (also the dense node order).
#![allow(clippy::float_cmp, reason = "bit-identical replay is the property under test")]

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_estimate::{ChainPatternLaw, evaluate_chain_recovery};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{
    ChainPartial, ChainRecoveryDecision, ChainRecoveryDetail, ChainRecoveryQuery,
    decide_chain_recovery, verify_chain_witness,
};
use antecedent_io::IoError;
use antecedent_io::recovery_chain_artifact::{
    ChainGraphWire, RECOVERY_CHAIN_ARTIFACT_VERSION, RECOVERY_CHAIN_NONRECOVERABLE,
    RECOVERY_CHAIN_RECOVERED, RecoveryChainArtifactError, RecoveryChainArtifactWire,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn names() -> Vec<String> {
    ["x1", "x2", "r1", "r2", "p1", "p2"].map(str::to_owned).to_vec()
}

fn build(edges: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::empty();
    let mut dense = Vec::new();
    for raw in 0..6u32 {
        dense.push(graph.add_node(NodeRef::Static(VariableId::from_raw(raw))).unwrap());
    }
    for (a, b) in edges {
        graph.insert_directed(dense[*a as usize], dense[*b as usize]).unwrap();
    }
    graph
}

fn edges(head: usize, dependent: bool, extra: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out = vec![(0, 1), (0, 4), (2, 4), (1, 5), (3, 5)];
    if head == 0 {
        out.push((2, 3));
        if dependent {
            out.push((0, 3));
        }
    } else {
        out.push((3, 2));
        if dependent {
            out.push((1, 2));
        }
    }
    out.extend_from_slice(extra);
    out
}

fn query() -> ChainRecoveryQuery {
    let v = VariableId::from_raw;
    ChainRecoveryQuery {
        first: ChainPartial { variable: v(0), response: v(2), proxy: v(4) },
        second: ChainPartial { variable: v(1), response: v(3), proxy: v(5) },
    }
}

/// Enumerated SCM: `(pattern law, truth)` with axes in declaration order.
fn scm(head: usize, dependent: bool) -> (ChainPatternLaw, [[f64; 2]; 2]) {
    let tail = 1 - head;
    let mut both = [[0.0; 2]; 2];
    let mut only = [[0.0; 2]; 2];
    let mut neither = 0.0;
    let mut truth = [[0.0; 2]; 2];
    for cell in 0..16usize {
        let x = [(cell >> 3) & 1, (cell >> 2) & 1];
        let r = [(cell >> 1) & 1, cell & 1];
        let p_x1 = if x[0] == 1 { 0.37 } else { 0.63 };
        let p_x2_one = if x[0] == 1 { 0.7 } else { 0.2 };
        let p_x2 = if x[1] == 1 { p_x2_one } else { 1.0 - p_x2_one };
        let p_head = if r[head] == 1 { 0.65 } else { 0.35 };
        let tail_one = if dependent {
            [[0.4, 0.55], [0.3, 0.8]][r[head]][x[head]]
        } else {
            [0.3, 0.8][r[head]]
        };
        let p_tail = if r[tail] == 1 { tail_one } else { 1.0 - tail_one };
        let mass = p_x1 * p_x2 * p_head * p_tail;
        truth[x[0]][x[1]] += mass;
        match (r[0], r[1]) {
            (1, 1) => both[x[0]][x[1]] += mass,
            (1, 0) => only[0][x[0]] += mass,
            (0, 1) => only[1][x[1]] += mass,
            _ => neither += mass,
        }
    }
    (ChainPatternLaw::new(both, only[0], only[1], neither).unwrap(), truth)
}

fn recovered_wire(head: usize, dependent: bool) -> (RecoveryChainArtifactWire, [[f64; 2]; 2]) {
    let graph = build(&edges(head, dependent, &[]));
    let ChainRecoveryDecision::Recovered(plan) =
        decide_chain_recovery(&graph, &query(), &ctx()).unwrap()
    else {
        panic!("expected a recovery plan");
    };
    let (law, truth) = scm(head, dependent);
    let recovered = evaluate_chain_recovery(&plan, &law).unwrap();
    let wire = RecoveryChainArtifactWire::from_recovered(
        &graph,
        &query(),
        &names(),
        &plan,
        &law,
        &recovered,
    )
    .unwrap();
    (wire, truth)
}

fn nonrecoverable_wire() -> RecoveryChainArtifactWire {
    let graph = build(&edges(0, true, &[(1, 3)]));
    let ChainRecoveryDecision::NonRecoverable(witness) =
        decide_chain_recovery(&graph, &query(), &ctx()).unwrap()
    else {
        panic!("a self-censoring edge is nonrecoverable");
    };
    let check = verify_chain_witness(&graph, &query(), &witness, &ctx()).unwrap();
    RecoveryChainArtifactWire::from_nonrecoverable(&graph, &query(), &names(), &witness, &check)
        .unwrap()
}

fn reseal(mut wire: RecoveryChainArtifactWire) -> Vec<u8> {
    wire.premises_digest = wire.expected_premises_digest().unwrap();
    wire.data_digest = wire.expected_data_digest().unwrap();
    wire.export().unwrap()
}

#[test]
fn b2_artifact_recovered_round_trip_equals_enumerated_truth_for_every_chain_shape() {
    for head in 0..2 {
        for dependent in [true, false] {
            let (wire, truth) = recovered_wire(head, dependent);
            assert_eq!(wire.outcome, RECOVERY_CHAIN_RECOVERED);
            let consumed =
                RecoveryChainArtifactWire::consume_typed(&wire.export().unwrap(), &ctx()).unwrap();
            let recovered = consumed.recovered.as_ref().unwrap();
            for (x1, row) in truth.iter().enumerate() {
                for (x2, expected) in row.iter().enumerate() {
                    let got = recovered.cells()[x1][x2];
                    assert!((got - expected).abs() < 1e-12, "({x1},{x2}): {got} vs {expected}");
                }
            }
            assert_eq!(recovered.rule_version(), "b2.recovery_chain.v1");
            let plan = wire.plan.as_ref().unwrap();
            assert_eq!(plan.head, head);
            assert_eq!(plan.tail_depends_on_head_variable, dependent);
            assert!(consumed.check.is_none());
            assert_eq!(consumed.wire.premises_digest, wire.premises_digest);
            assert_eq!(consumed.wire.data_digest, wire.data_digest);
        }
    }
}

#[test]
fn b2_artifact_producer_refuses_a_mutated_checked_plan() {
    let graph = build(&edges(0, true, &[]));
    let ChainRecoveryDecision::Recovered(plan) =
        decide_chain_recovery(&graph, &query(), &ctx()).unwrap()
    else {
        panic!("expected a recovery plan");
    };
    let (law, _) = scm(0, true);
    let recovered = evaluate_chain_recovery(&plan, &law).unwrap();
    let mut altered = *plan;
    altered.tail_depends_on_head_variable = false;
    assert!(
        RecoveryChainArtifactWire::from_recovered(
            &graph,
            &query(),
            &names(),
            &altered,
            &law,
            &recovered,
        )
        .is_err()
    );
}

#[test]
fn b2_artifact_nonrecoverable_carries_a_reverified_witness() {
    let wire = nonrecoverable_wire();
    assert_eq!(wire.outcome, RECOVERY_CHAIN_NONRECOVERABLE);
    let consumed =
        RecoveryChainArtifactWire::consume_typed(&wire.export().unwrap(), &ctx()).unwrap();
    assert!(matches!(consumed.decision, ChainRecoveryDecision::NonRecoverable(_)));
    let check = consumed.check.as_ref().unwrap();
    assert_eq!(check.observed_cells, 9);
    assert_eq!(check.denominator, 60u128.pow(4));
    assert_eq!(check.differing_cell, (0, 0));
    // P(X1 = 0) P(X2 = 0) over 60^4: (30 / 60)(30 / 60) against (30 / 60)(36 / 60).
    assert_eq!(check.masses, (30 * 30 * 3600, 30 * 36 * 3600));
    let stored = wire.witness_check.as_ref().unwrap();
    assert_eq!(stored.masses, ((30 * 30 * 3600).to_string(), (30 * 36 * 3600).to_string()));
    assert_eq!(wire.witness.as_ref().unwrap().edge, (1, 3));
    assert!(consumed.recovered.is_none() && wire.observed.is_none() && wire.plan.is_none());
}

#[test]
fn b2_resealed_graph_changes_are_refused() {
    // Removing X1 -> R2 turns the dependent chain into an independent one: the stored plan
    // no longer equals the re-decided plan.
    let (mut dropped, _) = recovered_wire(0, true);
    dropped.graph.directed.retain(|e| *e != (0, 3));
    assert_eq!(
        RecoveryChainArtifactWire::consume_typed(&reseal(dropped), &ctx()).unwrap_err(),
        RecoveryChainArtifactError::DecisionMismatch("plan")
    );
    // Adding a self-censoring edge turns the re-decision into a nonrecoverable one.
    let (mut censored, _) = recovered_wire(0, true);
    censored.graph.directed.push((1, 3));
    censored.graph.directed.sort_unstable();
    assert_eq!(
        RecoveryChainArtifactWire::consume_typed(&reseal(censored), &ctx()).unwrap_err(),
        RecoveryChainArtifactError::DecisionMismatch("outcome")
    );
    // Adding an edge outside the row is the route's typed unsupported-mechanism refusal.
    let (mut outside, _) = recovered_wire(0, true);
    outside.graph.directed.push((2, 5));
    outside.graph.directed.sort_unstable();
    let error = RecoveryChainArtifactWire::consume_typed(&reseal(outside), &ctx()).unwrap_err();
    assert_eq!(error.refusal(), ("route_not_supported", "recovery_chain.unsupported_mechanism"));
}

#[test]
fn b2_resealed_query_change_is_refused() {
    // Declaring the axes in the opposite order re-decides a different head axis.
    let (mut swapped, _) = recovered_wire(0, true);
    swapped.query.first = (1, 3, 5);
    swapped.query.second = (0, 2, 4);
    assert_eq!(
        RecoveryChainArtifactWire::consume_typed(&reseal(swapped), &ctx()).unwrap_err(),
        RecoveryChainArtifactError::DecisionMismatch("plan")
    );
}

#[test]
fn b2_resealed_observed_and_recovered_cell_changes_are_refused() {
    // A shifted observed cell (the cells still sum to one) no longer reproduces the stored law.
    let (mut observed, _) = recovered_wire(0, true);
    {
        let law = observed.observed.as_mut().unwrap();
        law.both[0][0] += 0.01;
        law.neither -= 0.01;
    }
    assert_eq!(
        RecoveryChainArtifactWire::consume_typed(&reseal(observed), &ctx()).unwrap_err(),
        RecoveryChainArtifactError::RecoveredMismatch
    );
    // A tampered recovered cell.
    let (mut recovered, _) = recovered_wire(1, false);
    recovered.recovered.as_mut().unwrap().cells[1][1] += 1e-12;
    assert_eq!(
        RecoveryChainArtifactWire::consume_typed(&reseal(recovered), &ctx()).unwrap_err(),
        RecoveryChainArtifactError::RecoveredMismatch
    );
    // A tampered rule version.
    let (mut rule, _) = recovered_wire(1, true);
    rule.recovered.as_mut().unwrap().rule_version = "b2.recovery_chain.v0".into();
    assert_eq!(
        RecoveryChainArtifactWire::consume_typed(&reseal(rule), &ctx()).unwrap_err(),
        RecoveryChainArtifactError::RecoveredMismatch
    );
}

#[test]
fn b2_resealed_witness_changes_are_refused() {
    let mut model = nonrecoverable_wire();
    model.witness.as_mut().unwrap().second[3].numerators[0] += 1;
    assert_eq!(
        RecoveryChainArtifactWire::consume_typed(&reseal(model), &ctx()).unwrap_err(),
        RecoveryChainArtifactError::DecisionMismatch("witness")
    );
    let mut edge = nonrecoverable_wire();
    edge.witness.as_mut().unwrap().edge = (0, 2);
    assert_eq!(
        RecoveryChainArtifactWire::consume_typed(&reseal(edge), &ctx()).unwrap_err(),
        RecoveryChainArtifactError::DecisionMismatch("witness")
    );
    let mut check = nonrecoverable_wire();
    check.witness_check.as_mut().unwrap().masses.1 = "1".into();
    assert_eq!(
        RecoveryChainArtifactWire::consume_typed(&reseal(check), &ctx()).unwrap_err(),
        RecoveryChainArtifactError::DecisionMismatch("witness check")
    );
}

#[test]
fn b2_unresealed_changes_hit_the_digests() {
    let (mut graph, _) = recovered_wire(0, true);
    graph.graph.directed.retain(|e| *e != (0, 3));
    assert_eq!(
        RecoveryChainArtifactWire::consume_typed(&graph.export().unwrap(), &ctx()).unwrap_err(),
        RecoveryChainArtifactError::PremisesMismatch
    );
    let (mut law, _) = recovered_wire(0, true);
    law.observed.as_mut().unwrap().both[0][0] += 0.01;
    assert_eq!(
        RecoveryChainArtifactWire::consume_typed(&law.export().unwrap(), &ctx()).unwrap_err(),
        RecoveryChainArtifactError::DataMismatch
    );
}

#[test]
fn b2_artifact_without_mass_in_a_required_cell_is_a_positivity_refusal() {
    let (mut wire, _) = recovered_wire(0, true);
    {
        let law = wire.observed.as_mut().unwrap();
        let moved = law.both[1][1];
        law.both[1][1] = 0.0;
        law.neither += moved;
    }
    let error = RecoveryChainArtifactWire::consume_typed(&reseal(wire), &ctx()).unwrap_err();
    assert!(matches!(&error, RecoveryChainArtifactError::Route(inner)
        if inner.detail == ChainRecoveryDetail::Positivity));
    assert_eq!(error.refusal(), ("transport_support_failure", "recovery_chain.positivity"));
}

#[test]
fn b2_artifact_invalid_observed_law_is_refused() {
    let (mut wire, _) = recovered_wire(0, true);
    wire.observed.as_mut().unwrap().neither += 0.5;
    let error = RecoveryChainArtifactWire::consume_typed(&reseal(wire), &ctx()).unwrap_err();
    assert_eq!(error.refusal(), ("invalid_argument", "recovery_chain.invalid_observed_law"));
}

#[test]
fn b2_artifact_version_features_outcome_and_names_are_checked() {
    let (mut future, _) = recovered_wire(0, true);
    future.version = RECOVERY_CHAIN_ARTIFACT_VERSION + 1;
    let bytes = future.export().unwrap();
    assert!(matches!(
        RecoveryChainArtifactWire::decode(&bytes),
        Err(IoError::UnsupportedVersion { version }) if version == RECOVERY_CHAIN_ARTIFACT_VERSION + 1
    ));
    assert!(matches!(
        RecoveryChainArtifactWire::consume_typed(&bytes, &ctx()),
        Err(RecoveryChainArtifactError::Undecodable(_))
    ));

    let (mut feature, _) = recovered_wire(0, true);
    feature.required_features.clear();
    assert!(matches!(
        RecoveryChainArtifactWire::consume_typed(&reseal(feature), &ctx()),
        Err(RecoveryChainArtifactError::UnsupportedSemantics(_))
    ));

    // A recovered outcome that carries a witness, and an unknown outcome, are shape errors.
    let (mut mixed, _) = recovered_wire(0, true);
    mixed.witness = nonrecoverable_wire().witness;
    assert!(matches!(
        RecoveryChainArtifactWire::consume_typed(&reseal(mixed), &ctx()),
        Err(RecoveryChainArtifactError::UnsupportedSemantics(_))
    ));
    let (mut unknown, _) = recovered_wire(0, true);
    unknown.outcome = "maybe".into();
    assert!(matches!(
        RecoveryChainArtifactWire::consume_typed(&reseal(unknown), &ctx()),
        Err(RecoveryChainArtifactError::UnsupportedSemantics(_))
    ));

    let (wire, _) = recovered_wire(0, true);
    assert!(wire.check_variable_names(&names()).is_ok());
    let mut renamed = names();
    renamed[0] = "other".into();
    assert_eq!(
        wire.check_variable_names(&renamed).unwrap_err(),
        RecoveryChainArtifactError::NamesMismatch
    );
}

#[test]
fn b2_artifact_graph_wire_round_trips_and_refuses_a_bidirected_graph() {
    let graph = build(&edges(0, true, &[]));
    let wire = ChainGraphWire::from_graph(&graph).unwrap();
    assert_eq!(wire.nodes, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(wire.to_graph().unwrap().node_count(), 6);

    let mut confounded = build(&edges(0, true, &[]));
    confounded.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    assert!(ChainGraphWire::from_graph(&confounded).is_err());
}

#[test]
fn b2_artifact_refusals_map_to_io_errors_with_their_reason_codes() {
    let (mut wire, _) = recovered_wire(0, true);
    wire.graph.directed.push((2, 5));
    wire.graph.directed.sort_unstable();
    let refused = RecoveryChainArtifactWire::consume(&reseal(wire), &ctx()).unwrap_err();
    assert_eq!(refused.reason_code(), Some("route_not_supported"));
}
