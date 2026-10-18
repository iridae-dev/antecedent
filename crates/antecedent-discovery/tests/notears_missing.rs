//! NOTEARS missing-value handling: refusal by default, declared complete case with accounting.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{MissingPolicy, TableView, TabularData};
use antecedent_discovery::{DiscoveryError, DiscoveryWorkspace, Notears};
use antecedent_graph::MarkedEdge;
use std::collections::BTreeSet;

/// Deterministic chain x0 -> x1 -> x2 with bounded pseudo-noise.
fn chain(n: usize) -> [Vec<f64>; 3] {
    let noise = |i: usize, k: f64| ((i as f64 * k) % 1.0) - 0.5;
    let x0: Vec<f64> = (0..n).map(|i| noise(i, 0.137)).collect();
    let x1: Vec<f64> = (0..n).map(|i| 0.8 * x0[i] + noise(i, 0.271)).collect();
    let x2: Vec<f64> = (0..n).map(|i| 0.8 * x1[i] + noise(i, 0.419)).collect();
    [x0, x1, x2]
}

fn table(columns: &[Vec<f64>; 3]) -> TabularData {
    TabularData::from_f64_columns(
        columns.iter().enumerate().map(|(i, c)| (format!("x{i}"), c.as_slice())),
    )
    .unwrap()
}

fn ids(data: &TabularData) -> Vec<VariableId> {
    data.schema().variables().iter().map(|v| v.id).collect()
}

fn edges(result: &antecedent_discovery::NotearsDiscoveryResult) -> BTreeSet<(u32, u32)> {
    result
        .discovery
        .evidence
        .graph
        .edges()
        .filter_map(MarkedEdge::parent_child)
        .map(|(a, b)| (a.raw(), b.raw()))
        .collect()
}

fn run(
    alg: &Notears,
    data: &TabularData,
) -> Result<antecedent_discovery::NotearsDiscoveryResult, DiscoveryError> {
    alg.run(data, &ids(data), &mut DiscoveryWorkspace::default(), &ExecutionContext::for_tests(1))
}

fn with_gaps(n: usize) -> ([Vec<f64>; 3], Vec<usize>) {
    let mut columns = chain(n);
    let dropped: Vec<usize> = (0..n).filter(|i| i % 9 == 4).collect();
    for &row in &dropped {
        columns[1][row] = f64::NAN;
    }
    (columns, dropped)
}

#[test]
fn a_missing_value_is_refused_naming_the_cell_and_the_policy() {
    let (columns, dropped) = with_gaps(600);
    let error = run(&Notears::new(), &table(&columns)).unwrap_err();
    let text = error.to_string();
    assert!(
        text.starts_with("reason=missing_policy_required: missing_data.policy_required"),
        "{text}"
    );
    assert!(text.contains(&format!("column `x1` at row {}", dropped[0])), "{text}");
    assert!(text.contains("complete_case"), "{text}");
}

#[test]
fn complete_case_equals_the_run_on_the_complete_rows_and_reports_the_rest() {
    let n = 600;
    let (columns, dropped) = with_gaps(n);
    let alg = Notears::new().with_lambda(0.05).with_threshold(0.2);
    let declared =
        run(&alg.clone().with_missing(MissingPolicy::CompleteCase), &table(&columns)).unwrap();

    let kept: Vec<usize> = (0..n).filter(|i| !dropped.contains(i)).collect();
    let reference_columns: [Vec<f64>; 3] =
        std::array::from_fn(|c| kept.iter().map(|&i| columns[c][i]).collect());
    let reference = run(&alg, &table(&reference_columns)).unwrap();

    assert_eq!(edges(&declared), edges(&reference));
    assert_eq!(declared.weights, reference.weights, "same rows must give the same fit");
    let accounting = &declared.row_accounting;
    assert_eq!((accounting.rows_total, accounting.rows_kept), (n, kept.len()));
    assert_eq!(
        accounting.dropped_rows.iter().map(|r| usize::try_from(*r).unwrap()).collect::<Vec<_>>(),
        dropped
    );
    let by_variable: Vec<_> =
        accounting.missing_by_variable.iter().map(|(v, c)| (v.to_string(), *c)).collect();
    assert_eq!(by_variable[1], ("x1".to_owned(), dropped.len()));
    assert_eq!(reference.row_accounting.rows_dropped(), 0);
}

#[test]
fn a_policy_this_route_cannot_execute_is_refused_even_on_complete_data() {
    let data = table(&chain(300));
    let error = run(&Notears::new().with_missing(MissingPolicy::BalancedPanel), &data).unwrap_err();
    assert!(error.to_string().contains("missing_data.policy_not_applicable"), "{error}");
}

#[test]
fn pairwise_available_recovers_the_chain_and_reports_pairs_and_repair() {
    let n = 900;
    let mut columns = chain(n);
    // Different rows are missing in different columns, so no row is complete in all three.
    for row in (0..n).filter(|i| i % 7 == 2) {
        columns[0][row] = f64::NAN;
    }
    for row in (0..n).filter(|i| i % 11 == 5) {
        columns[1][row] = f64::NAN;
    }
    for row in (0..n).filter(|i| i % 13 == 8) {
        columns[2][row] = f64::NAN;
    }
    let alg = Notears::new().with_lambda(0.05).with_threshold(0.2);
    let pairwise =
        run(&alg.clone().with_missing(MissingPolicy::PairwiseAvailable), &table(&columns)).unwrap();
    let complete = run(&alg, &table(&chain(n))).unwrap();
    assert_eq!(edges(&pairwise), edges(&complete));
    let accounting = &pairwise.row_accounting;
    assert_eq!(accounting.rows_dropped(), 0, "no row is deleted by a pairwise covariance");
    let pairs: Vec<_> =
        accounting.groups.iter().filter(|g| g.label.starts_with("pair:")).map(|g| g.kept).collect();
    assert_eq!(pairs.len(), 3);
    assert!(pairs.iter().all(|rows| *rows < n && *rows > n / 2));
    assert!(accounting.groups.iter().any(|g| g.label.starts_with("psd_repair:")));
    // The weights come from a covariance close to the complete-data one.
    for (a, b) in pairwise.weights.iter().zip(complete.weights.iter()) {
        assert!((a - b).abs() < 0.1, "{a} vs {b}");
    }
}
