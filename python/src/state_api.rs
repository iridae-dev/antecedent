//! Incremental [`CausalState`] OO surface.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal::{
    CausalState, DataBatchRef, StateEvent, apply_state_event, new_causal_state,
};
use causal_core::{AverageEffectQuery, CacheBudget, CausalQuery, VariableId};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::{catch_ffi, columns_to_batch, py_err};

const DEFAULT_CACHE_BYTES: u64 = 1_048_576;

/// Incremental causal analysis state (versioned events, no auto-rerun).
#[pyclass(name = "CausalState")]
pub struct PyCausalState {
    inner: CausalState,
    next_batch: u64,
}

impl PyCausalState {
    fn append_ref(&mut self, nrows: u64, bytes: u64) -> PyResult<u64> {
        let id = Arc::from(format!("b{}", self.next_batch));
        self.next_batch = self.next_batch.wrapping_add(1);
        let ver = apply_state_event(
            &mut self.inner,
            StateEvent::AppendData(DataBatchRef { id, nrows, bytes }),
        )
        .map_err(py_err)?;
        Ok(ver.raw())
    }
}

#[pymethods]
impl PyCausalState {
    /// Create a fresh state with a cache budget of `cache_bytes`.
    #[new]
    #[pyo3(signature = (cache_bytes=DEFAULT_CACHE_BYTES))]
    fn new(cache_bytes: u64) -> Self {
        Self {
            inner: new_causal_state(CacheBudget::new(cache_bytes)),
            next_batch: 0,
        }
    }

    /// Monotonic state version (starts at 0; bumps on each applied event).
    #[getter]
    fn version(&self) -> u64 {
        self.inner.version.raw()
    }

    /// Number of registered queries whose cached results are currently stale.
    fn stale_query_count(&self) -> usize {
        self.inner.stale_queries().len()
    }

    /// Raw ids of registered queries that are currently stale.
    fn stale_queries(&self) -> Vec<u64> {
        self.inner
            .stale_queries()
            .into_iter()
            .map(|q| u64::from(q.raw()))
            .collect()
    }

    /// Append a tabular batch reference derived from named float64 columns.
    ///
    /// Returns the new state version. Column data is sized for the batch catalog
    /// but not retained (matches Rust `DataBatchRef` semantics).
    fn append_data(
        &mut self,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<u64> {
        let batch = columns_to_batch(&names, &columns)?;
        let nrows = u64::try_from(batch.num_rows()).unwrap_or(u64::MAX);
        let bytes = u64::try_from(batch.get_array_memory_size()).unwrap_or(u64::MAX);
        self.append_ref(nrows, bytes)
    }

    fn __repr__(&self) -> String {
        format!(
            "CausalState(version={}, stale_queries={}, batches={})",
            self.inner.version.raw(),
            self.inner.stale_queries().len(),
            self.inner.data_catalog.batches.len(),
        )
    }
}

/// Smoke / test helper: append synthetic batches and report `(version, stale_count)`.
#[pyfunction]
#[pyo3(signature = (n_appends=2, cache_bytes=DEFAULT_CACHE_BYTES))]
pub(crate) fn causal_state_append(n_appends: u64, cache_bytes: u64) -> PyResult<(u64, usize)> {
    catch_ffi(|| {
        let mut state = PyCausalState::new(cache_bytes);
        let q = state.inner.queries.register(CausalQuery::AverageEffect(
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)),
        ));
        let _ = state.inner.refresh_results(&[(q, 1, 16)]);
        for _ in 0..n_appends {
            state.append_ref(8, 64)?;
        }
        Ok((state.inner.version.raw(), state.inner.stale_queries().len()))
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCausalState>()?;
    m.add_function(wrap_pyfunction!(causal_state_append, m)?)?;
    Ok(())
}
