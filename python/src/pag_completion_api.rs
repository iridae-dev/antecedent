//! Python bridge for the circle-free completions of a PAG.
//!
//! The completions are Rust's: every yielded graph is an ancestral, maximal graph that keeps
//! the source PAG's unshielded colliders, and the audit says whether they are proved to be one
//! Markov equivalence class or only a prefix of it. A cap or a search bound that stopped the walk
//! is reported, never hidden.

use antecedent_graph::{CompletionSampler, Pag as GraphPag};
use pyo3::prelude::*;
use serde_json::json;

use crate::graphs::Pag;
use crate::py_err;

/// Up to `max_completions` circle-free completions of `pag` and the audit of the walk as JSON.
#[pyfunction]
#[pyo3(signature = (pag, max_completions=64))]
fn pag_completions(pag: PyRef<'_, Pag>, max_completions: usize) -> PyResult<(Vec<Pag>, String)> {
    if max_completions == 0 {
        return Err(crate::value_err("max_completions must be at least 1"));
    }
    let names = pag.names.clone();
    let source: GraphPag = pag.pag.clone();
    let mut sampler = CompletionSampler::new(source, max_completions).map_err(py_err)?;
    let completions: Vec<Pag> = sampler
        .by_ref()
        .map(|completion| Pag { pag: completion.graph, names: names.clone() })
        .collect();
    let report = sampler.validation_report();
    let audit = json!({
        "yielded": completions.len(),
        "max_completions": sampler.max_completions(),
        "circle_sites": sampler.n_circle_sites(),
        "represented_completions": report.represented_completions,
        "assignments_examined": report.assignments_examined,
        "rejected_constraints": report.rejected_constraints,
        "rejected_non_ancestral": report.rejected_non_ancestral,
        "rejected_nonmaximal": report.rejected_nonmaximal,
        "rejected_local_incompatible": report.rejected_local_incompatible,
        "ambiguous_global_class": report.ambiguous_global_class,
        "equivalence_audit_skipped": report.equivalence_audit_skipped,
        "search_stop": report.search_stop.map(|stop| stop.code()),
        "hit_cap": sampler.hit_cap(),
        "class_audit_incomplete": sampler.class_audit_incomplete(),
        "operations_consumed": report.operations_consumed,
    });
    Ok((completions, audit.to_string()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(pag_completions, module)?)
}
