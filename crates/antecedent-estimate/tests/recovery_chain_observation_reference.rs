//! The ordered-response recovery of a binary m-graph against exact-rational enumeration.
//!
//! `oracle_data/recovery_chain_observation_reference.py` (frozen in the JSON beside it)
//! enumerates 24 structural models over `X1, X2, R1, R2` with exact fractions, both chain
//! heads and both tail shapes (the tail response with and without a dependence on the head
//! variable); eight use moderate probabilities and sixteen put probabilities at 1/1000 or
//! 999/1000, which makes a complete-case cell as small as 4e-10. The observed nine-cell pattern
//! law is the marginal of the enumerated joint, and the oracle answer is the enumerated
//! `P(X1, X2)`, which the recovery must return. The same script re-enumerates the self-censoring
//! witness over fractions; this test re-enumerates the witness the decision returns with its own
//! integer code and compares it with that frozen result.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_estimate::{ChainPatternLaw, evaluate_chain_recovery};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{
    ChainPartial, ChainRecoveryDecision, ChainRecoveryDetail, ChainRecoveryPlan,
    ChainRecoveryQuery, ChainRecoveryWitness, decide_chain_recovery, verify_chain_witness,
};
use serde_json::Value;

const IDS: [u32; 6] = [0, 1, 2, 3, 4, 5];

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/recovery_chain_observation_reference.json"))
        .expect("frozen reference parses")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("number")).collect()
}

fn build(edges: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::empty();
    let mut dense: BTreeMap<u32, DenseNodeId> = BTreeMap::new();
    for raw in IDS {
        dense.insert(raw, graph.add_node(NodeRef::Static(VariableId::from_raw(raw))).unwrap());
    }
    for (a, b) in edges {
        graph.insert_directed(dense[a], dense[b]).unwrap();
    }
    graph
}

fn edges(head: usize, dependent: bool, extra: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let [x1, x2, r1, r2, p1, p2] = IDS;
    let mut out = vec![(x1, x2), (x1, p1), (r1, p1), (x2, p2), (r2, p2)];
    let (rh, rt, xh) = if head == 0 { (r1, r2, x1) } else { (r2, r1, x2) };
    out.push((rh, rt));
    if dependent {
        out.push((xh, rt));
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

fn decide(edge_list: &[(u32, u32)]) -> ChainRecoveryDecision {
    decide_chain_recovery(&build(edge_list), &query(), &ExecutionContext::for_tests(1)).unwrap()
}

fn plan(head: usize, dependent: bool) -> ChainRecoveryPlan {
    match decide(&edges(head, dependent, &[])) {
        ChainRecoveryDecision::Recovered(plan) => *plan,
        ChainRecoveryDecision::NonRecoverable(_) => panic!("expected a recovery plan"),
    }
}

fn law(case: &Value) -> ChainPatternLaw {
    let both = case["both"].as_array().unwrap();
    let (b0, b1) = (floats(&both[0]), floats(&both[1]));
    let (f, s) = (floats(&case["only_first"]), floats(&case["only_second"]));
    ChainPatternLaw::new(
        [[b0[0], b0[1]], [b1[0], b1[1]]],
        [f[0], f[1]],
        [s[0], s[1]],
        case["neither"].as_f64().unwrap(),
    )
    .unwrap()
}

#[test]
fn recovery_equals_the_exactly_enumerated_law_including_extreme_probabilities() {
    let data = reference();
    let cases = data["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 24);
    let mut smallest = f64::INFINITY;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let head = usize::try_from(case["head"].as_u64().unwrap()).unwrap();
        let dependent = case["dependent"].as_bool().unwrap();
        let plan = plan(head, dependent);
        assert_eq!(plan.head, head, "{name}");
        assert_eq!(plan.tail_depends_on_head_variable, dependent, "{name}");
        smallest = smallest.min(case["smallest_complete_cell"].as_f64().unwrap());
        let recovered =
            evaluate_chain_recovery(&plan, &law(case)).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let truth = case["truth"].as_array().unwrap();
        for x1 in 0..2 {
            let row = floats(&truth[x1]);
            for x2 in 0..2 {
                let got = recovered.cells()[x1][x2];
                assert!(
                    (got - row[x2]).abs() <= 1e-12 * row[x2].max(1e-300),
                    "{name} ({x1},{x2}): {got} vs {}",
                    row[x2]
                );
            }
        }
    }
    assert!(smallest < 1e-8, "the cases must reach a tiny complete-case cell: {smallest}");
}

#[test]
fn the_wrong_response_order_or_tail_shape_does_not_recover_the_truth() {
    // The recovery is a function of the declared chain, not of the data alone: the plan for the
    // opposite head, or the independent-tail plan for a tail that depends on the head variable,
    // returns a different law on the same observed cells.
    let data = reference();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| {
            c["head"] == 0
                && c["dependent"] == true
                && c["smallest_complete_cell"].as_f64().unwrap() > 1e-3
        })
        .expect("a moderate dependent case");
    let observed = law(case);
    let truth = case["truth"].as_array().unwrap();
    let distance = |plan: &ChainRecoveryPlan| {
        let recovered = evaluate_chain_recovery(plan, &observed).unwrap();
        (0..2)
            .flat_map(|a| (0..2).map(move |b| (a, b)))
            .map(|(a, b)| (recovered.cells()[a][b] - floats(&truth[a])[b]).abs())
            .fold(0.0_f64, f64::max)
    };
    assert!(distance(&plan(0, true)) < 1e-12);
    assert!(distance(&plan(0, false)) > 1e-4, "an independent-tail plan must not recover it");
}

/// Exact integer mass over `60^4` of the cells of one witness model, by own enumeration.
fn witness_cells(
    model: &[antecedent_identify::ChainWitnessMechanism],
    axis: usize,
) -> (Vec<u128>, u128) {
    let mut cells = vec![0_u128; 9];
    let mut p_axis_one = 0_u128;
    let value = |assignment: &[u8; 4], node: u32| assignment[usize::try_from(node).unwrap()];
    for code in 0..16_u32 {
        let assignment =
            [(code >> 3 & 1) as u8, (code >> 2 & 1) as u8, (code >> 1 & 1) as u8, (code & 1) as u8];
        let mut mass = 1_u128;
        for mechanism in model {
            let mut config = 0_usize;
            for &parent in &mechanism.parents {
                config = config * 2 + usize::from(value(&assignment, parent));
            }
            let one = u128::from(mechanism.numerators[config]);
            mass *= if value(&assignment, mechanism.node) == 1 { one } else { 60 - one };
        }
        let [x1, x2, r1, r2] = assignment;
        let slot = match (r1, r2) {
            (1, 1) => usize::from(x1) * 2 + usize::from(x2),
            (1, 0) => 4 + usize::from(x1),
            (0, 1) => 6 + usize::from(x2),
            _ => 8,
        };
        cells[slot] += mass;
        if assignment[axis] == 1 {
            p_axis_one += mass;
        }
    }
    (cells, p_axis_one)
}

fn fraction(text: &str) -> (u128, u128) {
    let (n, d) = text.split_once('/').map_or((text, "1"), |(n, d)| (n, d));
    (n.parse().unwrap(), d.parse().unwrap())
}

#[test]
fn self_censoring_witness_models_agree_on_every_observed_cell_and_differ_on_the_target() {
    let data = reference();
    for (axis, witness_json) in data["witnesses"].as_array().unwrap().iter().enumerate() {
        assert_eq!(witness_json["cells_equal"], true);
        let [x1, x2, r1, r2, _, _] = IDS;
        let censoring = if axis == 0 { (x1, r1) } else { (x2, r2) };
        let extra = [censoring];
        let ChainRecoveryDecision::NonRecoverable(witness) = decide(&edges(0, false, &extra))
        else {
            panic!("a self-censoring edge must be a nonrecoverability witness");
        };
        let witness: ChainRecoveryWitness = *witness;
        assert_eq!(witness.edge, censoring);
        let check = verify_chain_witness(
            &build(&edges(0, false, &extra)),
            &query(),
            &witness,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(check.denominator, 60_u128.pow(4));
        let (first, first_p) = witness_cells(&witness.first, axis);
        let (second, second_p) = witness_cells(&witness.second, axis);
        assert_eq!(first, second, "axis {axis}: the observed cells must coincide");
        // Compare with the frozen exact fractions: cell / 60^4 == numerator / denominator.
        for (k, cell) in first.iter().enumerate() {
            for (key, got) in [("first_cells", *cell), ("second_cells", second[k])] {
                let (n, d) = fraction(witness_json[key][k].as_str().unwrap());
                assert_eq!(got * d, n * 60_u128.pow(4), "axis {axis} {key} cell {k}");
            }
        }
        // The censored variable's marginal P(X_axis = 1) differs between the models: 1/2 against 2/5.
        assert_ne!(first_p, second_p);
        let (n1, d1) = fraction(witness_json["first_p_one"].as_str().unwrap());
        let (n2, d2) = fraction(witness_json["second_p_one"].as_str().unwrap());
        assert_eq!(first_p * d1, n1 * 60_u128.pow(4), "axis {axis}");
        assert_eq!(second_p * d2, n2 * 60_u128.pow(4), "axis {axis}");
        assert_ne!(check.masses.0, check.masses.1);
    }
}

#[test]
fn graphs_outside_the_row_and_zero_mass_cells_are_refused_not_recovered() {
    // An edge from the tail variable to the head response is outside the row.
    let error = decide_chain_recovery(
        &build(&edges(0, false, &[(1, 2)])),
        &query(),
        &ExecutionContext::for_tests(1),
    )
    .unwrap_err();
    assert_eq!(error.detail, ChainRecoveryDetail::UnsupportedMechanism);
    // A zero complete-case cell is a positivity refusal, not a division by zero.
    let data = reference();
    let case = &data["cases"][0];
    let mut zeroed = law(case).both().to_owned();
    zeroed[0][1] = 0.0;
    let broken = ChainPatternLaw::new(
        zeroed,
        [0.0, 0.0],
        [0.0, 0.0],
        1.0 - zeroed.iter().flatten().sum::<f64>(),
    )
    .unwrap();
    let error = evaluate_chain_recovery(&plan(0, true), &broken).unwrap_err();
    assert_eq!(error.detail, ChainRecoveryDetail::Positivity);
}
