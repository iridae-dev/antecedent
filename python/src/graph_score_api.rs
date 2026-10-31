//! Incremental graph scoring and rolling mechanism diagnostics as Python values.
//!
//! `GraphScoreCache` holds one dataset and a per-node local-score cache: a parent-set change
//! rescoring only the node it touches returns the score delta and the new total, equal to a
//! full recomputation. `RollingMechanismDiagnostics` keeps the linear-Gaussian mechanism fit,
//! residual summaries and CUSUM of the last `window` rows of one node.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent::CausalError;
use antecedent::state::{
    GraphScoreCacheKey, GraphScoreData, GraphScoreFamily, LocalScoreCache, ParentSetOp,
    RollingMechanismDiagnostics, StateError, full_graph_score,
};
use pyo3::prelude::*;

use crate::{py_err, value_err};

fn state_err(error: StateError) -> PyErr {
    py_err(CausalError::from(error))
}

fn family_of(name: &str) -> PyResult<GraphScoreFamily> {
    match name {
        "gaussian_bic" => Ok(GraphScoreFamily::GaussianBic),
        "laplace_bic" => Ok(GraphScoreFamily::LaplaceBic),
        other => Err(value_err(format!(
            "unknown graph score family {other:?}; use gaussian_bic or laplace_bic"
        ))),
    }
}

/// Per-node local scores of one dataset, with incremental parent-set changes.
#[pyclass(name = "GraphScoreCache")]
pub struct PyGraphScoreCache {
    data: GraphScoreData,
    family: GraphScoreFamily,
    cache: LocalScoreCache,
}

#[pymethods]
impl PyGraphScoreCache {
    /// One equal-length column per graph node; the family is `gaussian_bic` or `laplace_bic`.
    /// Both return -BIC/2, counting intercept, parent coefficients, and fitted variance or scale.
    #[new]
    #[pyo3(signature = (columns, *, family="gaussian_bic", byte_budget=None))]
    fn new(columns: Vec<Vec<f64>>, family: &str, byte_budget: Option<u64>) -> PyResult<Self> {
        let family = family_of(family)?;
        let n_rows = columns.first().map_or(0, Vec::len);
        if columns.iter().any(|column| column.len() != n_rows) {
            return Err(value_err("every column must have the same number of rows"));
        }
        if columns.iter().flatten().any(|value| !value.is_finite()) {
            return Err(value_err("graph score columns must be finite"));
        }
        let n_vars = columns.len();
        let flat: Vec<f64> = columns.into_iter().flatten().collect();
        let data = GraphScoreData::new(n_rows, n_vars, Arc::from(flat)).map_err(state_err)?;
        let key = GraphScoreCacheKey {
            data_version: 1,
            family,
            var_fingerprint: n_vars as u64,
            penalty_fingerprint: n_rows as u64,
        };
        let mut cache = LocalScoreCache::new(key);
        if let Some(bytes) = byte_budget {
            cache = cache.with_byte_budget(bytes);
        }
        Ok(Self { data, family, cache })
    }

    /// Rows of the dataset.
    #[getter]
    fn n_rows(&self) -> usize {
        self.data.n_rows
    }

    /// Graph nodes (columns) of the dataset.
    #[getter]
    fn n_vars(&self) -> usize {
        self.data.n_vars
    }

    /// Bytes the cache currently retains.
    #[getter]
    fn retained_bytes(&self) -> u64 {
        self.cache.retained_bytes()
    }

    /// Total score of the graph with the current parent sets.
    fn score(&mut self) -> PyResult<f64> {
        self.cache.score_graph(&self.data).map_err(state_err)
    }

    /// Replace `node`'s parent set: `(delta, new total)`. Only that node is rescored.
    fn set_parents(&mut self, node: u32, parents: Vec<u32>) -> PyResult<(f64, f64)> {
        self.cache
            .delta_score(&self.data, ParentSetOp::SetParents { node, parents: Arc::from(parents) })
            .map_err(state_err)
    }

    /// Current sorted parent set of `node` (empty when none was set).
    fn parents_of(&self, node: u32) -> Vec<u32> {
        self.cache.parents_of(node).to_vec()
    }

    /// Score of the graph with the given `{node: parents}` sets, recomputed from scratch
    /// without the cache. The incremental total equals this value.
    fn full_score(&self, parents: HashMap<u32, Vec<u32>>) -> PyResult<f64> {
        let parents = parents.into_iter().map(|(node, set)| (node, Arc::from(set))).collect();
        full_graph_score(&self.data, self.family, &parents).map_err(state_err)
    }

    /// Drop every cached score and parent set.
    fn clear(&mut self) {
        self.cache.clear();
    }

    fn __repr__(&self) -> String {
        format!("GraphScoreCache(n_rows={}, n_vars={})", self.data.n_rows, self.data.n_vars)
    }
}

/// Rolling linear-Gaussian mechanism fit and residual diagnostics over the last `window` rows.
#[pyclass(name = "RollingMechanismDiagnostics")]
pub struct PyRollingMechanismDiagnostics {
    inner: RollingMechanismDiagnostics,
}

#[pymethods]
impl PyRollingMechanismDiagnostics {
    #[new]
    fn new(ncols: usize, window: usize) -> PyResult<Self> {
        Ok(Self { inner: RollingMechanismDiagnostics::new(ncols, window).map_err(state_err)? })
    }

    /// Append one design row and response; the oldest row leaves a full window.
    fn append_row(&mut self, row: Vec<f64>, y: f64) -> PyResult<()> {
        self.inner.append_row(&row, y).map_err(state_err)
    }

    /// Append row-major design rows and their responses.
    fn append_batch(&mut self, rows: Vec<f64>, y: Vec<f64>) -> PyResult<()> {
        self.inner.append_batch(&rows, &y).map_err(state_err)
    }

    /// Recompute the coefficients, residual summaries and CUSUM from the retained window.
    fn refresh(&mut self) -> PyResult<()> {
        self.inner.refresh_summaries().map_err(state_err)
    }

    #[getter]
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    #[getter]
    fn window(&self) -> usize {
        self.inner.window
    }

    /// Rows currently retained.
    #[getter]
    fn n(&self) -> u64 {
        self.inner.n
    }

    /// Coefficients at the last refresh (empty before one).
    #[getter]
    fn beta(&self) -> Vec<f64> {
        self.inner.beta.clone()
    }

    #[getter]
    fn residual_sse(&self) -> f64 {
        self.inner.residual_sse
    }

    /// `SSE / (n - p)`, or `None` when the window holds no more rows than columns.
    #[getter]
    fn residual_var(&self) -> Option<f64> {
        self.inner.residual_var
    }

    #[getter]
    fn mean_abs_residual(&self) -> f64 {
        self.inner.mean_abs_residual
    }

    /// Maximum absolute CUSUM of the residuals, or `None` below four rows.
    #[getter]
    fn max_abs_cusum(&self) -> Option<f64> {
        self.inner.max_abs_cusum
    }

    /// Bytes currently retained.
    #[getter]
    fn bytes(&self) -> u64 {
        self.inner.bytes
    }

    /// Bytes a full window retains.
    #[getter]
    fn worst_case_bytes(&self) -> u64 {
        self.inner.worst_case_bytes()
    }

    fn __repr__(&self) -> String {
        format!(
            "RollingMechanismDiagnostics(ncols={}, window={}, n={})",
            self.inner.ncols(),
            self.inner.window,
            self.inner.n
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraphScoreCache>()?;
    m.add_class::<PyRollingMechanismDiagnostics>()?;
    Ok(())
}
