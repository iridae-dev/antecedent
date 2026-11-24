//! The compiled graph falsification of a prepared study's own DAG.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use pyo3::prelude::*;

use super::PyPreparedAnalysis;
use crate::graph_falsification_api::{falsification_config, graph_falsification_json};
use crate::{detach_catch, py_err, py_execution_context};

#[pymethods]
impl PyPreparedAnalysis {
    /// Falsify the study's own DAG against the table it retained, by graph-node permutation,
    /// through the compiled `GraphFalsification` program; returns the report as JSON. The
    /// numbers are those of `evaluate_graph` on the same rows, edges, settings and seed.
    #[pyo3(signature = (*, ci_test="parcorr", alpha=0.05, significance=0.05, n_permutations=200, min_permutations=20, max_equivalent_share=None, max_ci_tests=None, ci_replicates=199, permutation_seed=None, seed=0, threads=None, cancel=None))]
    #[allow(clippy::too_many_arguments, reason = "one keyword per test setting")]
    fn falsify_graph(
        &self,
        py: Python<'_>,
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
        let config = falsification_config(
            ci_test,
            (alpha, significance),
            (n_permutations, min_permutations),
            max_equivalent_share,
            max_ci_tests,
            ci_replicates,
            permutation_seed,
        );
        // The study this handle's last execution ran on (the rebound branch after an
        // estimate on new rows), else the prepared one.
        let study = self.last_study.clone().unwrap_or_else(|| Arc::clone(&self.inner));
        let names = self.names.clone();
        detach_catch(py, move || {
            let mut ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
            crate::apply_cancel(&mut ctx, cancel);
            let compiled = study.falsify_graph(&config, &ctx).map_err(py_err)?;
            Ok(graph_falsification_json(compiled.report(), &names))
        })
    }
}
