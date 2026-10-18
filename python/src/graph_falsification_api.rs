//! Python bridge for graph permutation falsification (no fitted mechanisms).
//!
//! Python supplies the columns, the edges and the test settings; the implication tests, the
//! node-permutation comparison, every refusal and the self-describing report are Rust's. The
//! report crosses as JSON.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::gcm::{
    GraphFalsificationConfig, GraphFalsificationOutcome, GraphFalsificationReport,
    GraphUninformativeReason, evaluate_graph as facade_evaluate_graph,
};
use antecedent_core::VariableId;
use antecedent_data::TableView;
use pyo3::prelude::*;

use crate::{dag_from_named_edges, detach_catch, py_err, py_execution_context};

fn var_name(names: &[String], id: VariableId) -> String {
    names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("V{}", id.raw()))
}

fn graph_falsification_json(report: &GraphFalsificationReport, names: &[String]) -> String {
    let outcome = match report.outcome {
        GraphFalsificationOutcome::Rejected => "rejected",
        GraphFalsificationOutcome::NotRejected => "not_rejected",
        GraphFalsificationOutcome::Uninformative => "uninformative",
        GraphFalsificationOutcome::Incomplete => "incomplete",
    };
    let reason = report.uninformative_reason.map(|r| match r {
        GraphUninformativeReason::NoTestableImplication => "no_testable_implication",
        GraphUninformativeReason::EquivalenceClassShare => "equivalence_class_share",
    });
    let evidence: Vec<serde_json::Value> = report
        .evidence
        .iter()
        .map(|e| {
            serde_json::json!({
                "node": var_name(names, e.node),
                "other": var_name(names, e.other),
                "conditioning": e.conditioning.iter().map(|c| var_name(names, *c)).collect::<Vec<_>>(),
                "statistic": e.statistic,
                "p_value": e.p_value,
                "violated": e.violated,
            })
        })
        .collect();
    let receipt = report.search_receipt.as_ref().map(|r| {
        serde_json::json!({
            "stop": r.stop.code(),
            "operations_limit": r.operations_limit,
            "operations_consumed": r.operations_consumed,
            "summary": r.summary(),
        })
    });
    serde_json::json!({
        "outcome": outcome,
        "uninformative_reason": reason,
        "p_value": report.p_value,
        "n_implications": report.n_implications,
        "n_violations": report.n_violations,
        "evidence": evidence,
        "permutation_violations": report.permutation_violations.to_vec(),
        "permutations_requested": report.permutations_requested,
        "permutations_completed": report.permutations_completed,
        "exhaustive": report.exhaustive,
        "equivalent_permutations": report.equivalent_permutations,
        "equivalent_share": report.equivalent_share,
        "max_equivalent_share": report.max_equivalent_share,
        "ci_test": report.ci_test.as_ref(),
        "alpha": report.alpha,
        "significance": report.significance,
        "permutation_seed": report.permutation_seed,
        "execution_seed": report.execution_seed,
        "tests_run": report.tests_run,
        "cache_hits": report.cache_hits,
        "tests_skipped": report.tests_skipped,
        "max_ci_tests": report.max_ci_tests,
        "stop": report.stop.map(antecedent_core::SearchStop::code),
        "search_receipt": receipt,
        "limitations": report.limitations.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
    })
    .to_string()
}

/// Falsify a graph against data by graph-node permutation; returns the report as JSON.
///
/// No fitted mechanisms are involved. This is not the shuffled-outcome likelihood diagnostic.
#[pyfunction]
#[pyo3(
    name = "evaluate_graph",
    signature = (names, columns, edges, *, ci_test="parcorr", alpha=0.05, significance=0.05, n_permutations=200, min_permutations=20, max_equivalent_share=None, max_ci_tests=None, ci_replicates=199, permutation_seed=None, seed=0, threads=None, cancel=None)
)]
fn evaluate_graph_py(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    edges: Vec<(String, String)>,
    ci_test: &str,
    alpha: f64,
    significance: f64,
    n_permutations: usize,
    min_permutations: usize,
    max_equivalent_share: Option<f64>,
    max_ci_tests: Option<usize>,
    ci_replicates: u32,
    permutation_seed: Option<u64>,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let (data, _) = crate::tabular_from_py_columns(names.clone(), columns)?;
    detach_catch(py, move || {
        let g = dag_from_named_edges(data.schema(), &edges)?;
        let defaults = GraphFalsificationConfig::default();
        let config = GraphFalsificationConfig {
            ci_test: Arc::from(ci_test),
            alpha,
            significance,
            n_permutations,
            min_permutations,
            max_equivalent_share,
            max_ci_tests: max_ci_tests.unwrap_or(defaults.max_ci_tests),
            ci_replicates,
            seed: permutation_seed,
        };
        let mut ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        crate::apply_cancel(&mut ctx, cancel);
        let report = facade_evaluate_graph(&g, &data, &config, &ctx).map_err(py_err)?;
        Ok(graph_falsification_json(&report, &names))
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_graph_py, m)?)
}
