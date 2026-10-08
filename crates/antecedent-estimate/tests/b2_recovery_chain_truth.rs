//! 2.3 B2 / X10 second row: the ordered-response recovery formula against the
//! truth of an enumerated binary SCM, written out independently of the library.
//!
//! SCM (axis 0 = X1, axis 1 = X2): `P(X1 = 1) = 0.37`, `P(X2 = 1 | X1) = (0.2,
//! 0.7)`; the head response is `P(R_h = 1) = 0.65`; the tail response is
//! `P(R_t = 1 | R_h, X_h) = [[0.4, 0.55], [0.3, 0.8]][R_h][X_h]` (dependent) or
//! `P(R_t = 1 | R_h) = (0.3, 0.8)` (independent). The 16-cell joint is
//! enumerated, the proxies are deterministic, and the pattern law and the target
//! law are marginals of that joint.

use std::collections::BTreeMap;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_estimate::{ChainPatternLaw, ChainRecoveredLaw, evaluate_chain_recovery};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{
    ChainPartial, ChainRecoveryDecision, ChainRecoveryDetail, ChainRecoveryPlan,
    ChainRecoveryQuery, decide_chain_recovery,
};

const IDS: [u32; 6] = [0, 1, 2, 3, 4, 5];

/// Enumerated SCM: returns `(pattern law, truth)` with axes in declaration order.
fn scm(head: usize, dependent: bool) -> (ChainPatternLaw, [[f64; 2]; 2]) {
    let tail = 1 - head;
    let mut both = [[0.0; 2]; 2];
    let mut only = [[0.0; 2]; 2]; // only[axis][x]: that axis observed, the other missing
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
    let law = ChainPatternLaw::new(both, only[0], only[1], neither).unwrap();
    (law, truth)
}

fn build(order: &[u32], edges: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::empty();
    let mut dense: BTreeMap<u32, DenseNodeId> = BTreeMap::new();
    for raw in order {
        let id = graph.add_node(NodeRef::Static(VariableId::from_raw(*raw))).unwrap();
        dense.insert(*raw, id);
    }
    for (a, b) in edges {
        graph.insert_directed(dense[a], dense[b]).unwrap();
    }
    graph
}

fn edges(ids: [u32; 6], head: usize, dependent: bool) -> Vec<(u32, u32)> {
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
    out
}

fn query(ids: [u32; 6]) -> ChainRecoveryQuery {
    let v = VariableId::from_raw;
    ChainRecoveryQuery {
        first: ChainPartial { variable: v(ids[0]), response: v(ids[2]), proxy: v(ids[4]) },
        second: ChainPartial { variable: v(ids[1]), response: v(ids[3]), proxy: v(ids[5]) },
    }
}

fn plan(ids: [u32; 6], order: &[u32], head: usize, dependent: bool) -> ChainRecoveryPlan {
    let ctx = ExecutionContext::for_tests(1);
    match decide_chain_recovery(&build(order, &edges(ids, head, dependent)), &query(ids), &ctx)
        .unwrap()
    {
        ChainRecoveryDecision::Recovered(plan) => *plan,
        ChainRecoveryDecision::NonRecoverable(_) => panic!("expected a recovery plan"),
    }
}

fn assert_close(law: &ChainRecoveredLaw, truth: &[[f64; 2]; 2]) {
    for (x1, row) in truth.iter().enumerate() {
        for (x2, expected) in row.iter().enumerate() {
            let got = law.cells()[x1][x2];
            assert!((got - expected).abs() < 1e-12, "cell ({x1},{x2}): {got} vs {expected}");
        }
    }
}

#[test]
fn b2_chain_formula_equals_the_enumerated_truth_for_every_chain_shape() {
    for head in 0..2 {
        for dependent in [true, false] {
            let (observed, truth) = scm(head, dependent);
            let plan = plan(IDS, &IDS, head, dependent);
            assert_eq!(plan.head, head);
            assert_eq!(plan.tail_depends_on_head_variable, dependent);
            let recovered = evaluate_chain_recovery(&plan, &observed).unwrap();
            assert_close(&recovered, &truth);
            let total: f64 = recovered.cells().iter().flatten().sum();
            assert!((total - 1.0).abs() < 1e-12);
            assert_eq!(recovered.rule_version(), "b2.recovery_chain.v1");
        }
    }
}

#[test]
fn b2_chain_recovery_is_not_complete_case_analysis() {
    // In the dependent SCM the complete-case law is biased: the recovery is not
    // MAR or an unweighted complete-case fallback.
    let (observed, truth) = scm(0, true);
    let cc_total: f64 = observed.both().iter().flatten().sum();
    let complete_case_gap = (observed.both()[0][0] / cc_total - truth[0][0]).abs();
    assert!(complete_case_gap > 1e-3, "{complete_case_gap}");
    let recovered = evaluate_chain_recovery(&plan(IDS, &IDS, 0, true), &observed).unwrap();
    assert_close(&recovered, &truth);
}

#[test]
fn b2_chain_answer_is_invariant_to_variable_order_and_labels() {
    let (observed, truth) = scm(0, true);
    let reference = evaluate_chain_recovery(&plan(IDS, &IDS, 0, true), &observed).unwrap();
    let relabelled: [u32; 6] = [41, 17, 9, 30, 2, 25];
    let mut reversed = relabelled;
    reversed.reverse();
    let permuted = plan(relabelled, &reversed, 0, true);
    let answer = evaluate_chain_recovery(&permuted, &observed).unwrap();
    for x1 in 0..2 {
        for x2 in 0..2 {
            assert!((answer.cells()[x1][x2] - reference.cells()[x1][x2]).abs() < 1e-15);
        }
    }
    assert_close(&answer, &truth);
    // Declaring the variables in the opposite axis order transposes the answer.
    let swapped_ids: [u32; 6] = [1, 0, 3, 2, 5, 4];
    let swapped_plan = plan(swapped_ids, &IDS, 1, true);
    assert_eq!(swapped_plan.head, 1);
    let both = *observed.both();
    let swapped_observed = ChainPatternLaw::new(
        [[both[0][0], both[1][0]], [both[0][1], both[1][1]]],
        *observed.only_second(),
        *observed.only_first(),
        observed.neither(),
    )
    .unwrap();
    let swapped = evaluate_chain_recovery(&swapped_plan, &swapped_observed).unwrap();
    for x1 in 0..2 {
        for x2 in 0..2 {
            assert!((swapped.cells()[x2][x1] - reference.cells()[x1][x2]).abs() < 1e-15);
        }
    }
}

#[test]
fn b2_chain_zero_cell_is_a_positivity_refusal() {
    let (observed, _) = scm(0, true);
    let mut both = *observed.both();
    let moved = both[1][1];
    both[1][1] = 0.0;
    let broken = ChainPatternLaw::new(
        both,
        [0.0, 0.0],
        [0.0, 0.0],
        1.0 - both.iter().flatten().sum::<f64>(),
    )
    .unwrap();
    assert!(moved > 0.0);
    let error = evaluate_chain_recovery(&plan(IDS, &IDS, 0, true), &broken).unwrap_err();
    assert_eq!(error.detail, ChainRecoveryDetail::Positivity);
    assert_eq!(error.reason_code(), "transport_support_failure");
    assert_eq!(error.detail.detail(), "recovery_chain.positivity");
}

#[test]
fn b2_chain_head_response_without_mass_is_a_positivity_refusal() {
    // Head R1 never observed: P(R1 = 1) = 0 and every complete-case cell is empty.
    let law = ChainPatternLaw::new([[0.0; 2]; 2], [0.0; 2], [0.5, 0.25], 0.25).unwrap();
    let error = evaluate_chain_recovery(&plan(IDS, &IDS, 0, true), &law).unwrap_err();
    assert_eq!(error.detail, ChainRecoveryDetail::Positivity);
}

#[test]
fn b2_chain_invalid_observed_laws_are_refused() {
    let negative = ChainPatternLaw::new([[0.5, 0.5], [0.5, -0.5]], [0.0; 2], [0.0; 2], 0.0);
    assert_eq!(negative.unwrap_err().detail, ChainRecoveryDetail::InvalidObservedLaw);
    let not_normalized = ChainPatternLaw::new([[0.1; 2]; 2], [0.1; 2], [0.1; 2], 0.1);
    let error = not_normalized.unwrap_err();
    assert_eq!(error.detail, ChainRecoveryDetail::InvalidObservedLaw);
    assert_eq!(error.reason_code(), "invalid_argument");
    let not_finite = ChainPatternLaw::new([[f64::NAN; 2]; 2], [0.0; 2], [0.0; 2], 0.0);
    assert_eq!(not_finite.unwrap_err().detail, ChainRecoveryDetail::InvalidObservedLaw);
}

#[test]
fn b2_chain_malformed_plan_is_refused() {
    let (observed, _) = scm(0, true);
    let original = plan(IDS, &IDS, 0, true);
    assert!(original.is_intact());
    let mut wrong_head = original.clone();
    wrong_head.head = 1; // still in range, but not the checked response order
    let mut wrong_formula = original.clone();
    wrong_formula.tail_depends_on_head_variable = false;
    let mut wrong_version = original.clone();
    wrong_version.rule_version = "b2.recovery_chain.v0";
    let mut wrong_premise = original.clone();
    wrong_premise.premises.clear();
    for bad in [wrong_head, wrong_formula, wrong_version, wrong_premise] {
        assert!(!bad.is_intact());
        let error = evaluate_chain_recovery(&bad, &observed).unwrap_err();
        assert_eq!(error.detail, ChainRecoveryDetail::InvalidDerivation);
    }
}
