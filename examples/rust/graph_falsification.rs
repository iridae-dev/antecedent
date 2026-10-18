//! Graph falsification: is a DAG better than random relabellings of itself?
//!
//! Eight variables form four strongly dependent pairs. The graph that joins the pairs is
//! compared with 300 random relabellings of its nodes on the same data through the local
//! Markov implications it asserts; nothing is fitted. A graph that pairs the wrong
//! variables is not better than its relabellings, and a complete graph asserts nothing
//! that data could contradict.
//!
//! Run: `cargo run -p antecedent --example graph_falsification`
//! Source: `examples/rust/graph_falsification.rs`

use antecedent::gcm::{
    GraphFalsificationConfig, GraphFalsificationOutcome, GraphUninformativeReason, evaluate_graph,
};
use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

const TRUE_PAIRS: [(u32, u32); 4] = [(0, 1), (2, 3), (4, 5), (6, 7)];
const CROSSED_PAIRS: [(u32, u32); 4] = [(0, 2), (1, 3), (4, 6), (5, 7)];

fn graph(n: u32, edges: &[(u32, u32)]) -> Result<Dag, Box<dyn std::error::Error>> {
    let mut g = Dag::with_variables(n);
    for &(parent, child) in edges {
        g.insert_directed(DenseNodeId::from_raw(parent), DenseNodeId::from_raw(child))?;
    }
    Ok(g)
}

/// Four pairs `(x0,x1) (x2,x3) (x4,x5) (x6,x7)` with `x(2k+1) = 0.9 x(2k) + 0.45 noise`.
fn linear_pairs(rows: usize, seed: u64) -> Result<TabularData, Box<dyn std::error::Error>> {
    let mut rng = CausalRng::from_seed(seed);
    let mut columns: Vec<Vec<f64>> = (0..8).map(|_| Vec::with_capacity(rows)).collect();
    for _ in 0..rows {
        for pair in 0..4 {
            let root = standard_normal(&mut rng);
            columns[2 * pair].push(root);
            columns[2 * pair + 1].push(0.9 * root + 0.45 * standard_normal(&mut rng));
        }
    }
    let names: Vec<String> = (0..columns.len()).map(|i| format!("x{i}")).collect();
    Ok(TabularData::from_f64_columns(
        names.iter().map(String::as_str).zip(columns.iter().map(Vec::as_slice)),
    )?)
}

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let data = linear_pairs(2000, 5)?;
    let ctx = ExecutionContext::for_tests(1);
    let config = GraphFalsificationConfig {
        alpha: 0.01,
        n_permutations: 300,
        seed: Some(11),
        ..GraphFalsificationConfig::default()
    };

    // The graph that matches the data has far fewer violated implications than its
    // relabellings, so it is not rejected (which does not show it correct: Markov-equivalent
    // graphs cannot be told apart).
    let right = evaluate_graph(&graph(8, &TRUE_PAIRS)?, &data, &config, &ctx)?;
    assert_eq!(right.outcome, GraphFalsificationOutcome::NotRejected, "{right:?}");
    assert!(right.p_value.is_some_and(|p| p < 0.05));
    assert_eq!(right.permutations_completed, 300);
    assert_eq!((right.permutation_seed, right.execution_seed), (11, 1));

    // Pairing variables that are independent asserts independencies the data contradicts.
    let wrong = evaluate_graph(&graph(8, &CROSSED_PAIRS)?, &data, &config, &ctx)?;
    assert_eq!(wrong.outcome, GraphFalsificationOutcome::Rejected, "{wrong:?}");
    assert!(wrong.p_value.is_some_and(|p| p > 0.5));
    assert!(wrong.n_violations.is_some_and(|v| v >= 6));
    assert!(wrong.evidence.iter().any(|e| e.violated && e.p_value < 0.01));

    // The same seeds reproduce the verdict, the p-value and every permutation count.
    let again = evaluate_graph(&graph(8, &TRUE_PAIRS)?, &data, &config, &ctx)?;
    assert_eq!(again.p_value, right.p_value);
    assert_eq!(again.permutation_violations, right.permutation_violations);

    // A complete graph implies no independence, so no data can contradict it.
    let mut rng = CausalRng::from_seed(2);
    let columns: Vec<Vec<f64>> =
        (0..4).map(|_| (0..300).map(|_| standard_normal(&mut rng)).collect()).collect();
    let noise = TabularData::from_f64_columns(
        ["x0", "x1", "x2", "x3"].into_iter().zip(columns.iter().map(Vec::as_slice)),
    )?;
    let complete = graph(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])?;
    let nothing = evaluate_graph(&complete, &noise, &config, &ctx)?;
    assert_eq!(nothing.outcome, GraphFalsificationOutcome::Uninformative);
    assert_eq!(nothing.uninformative_reason, Some(GraphUninformativeReason::NoTestableImplication));
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}
