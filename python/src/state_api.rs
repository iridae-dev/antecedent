//! Incremental [`CausalState`] OO surface with retained column batches.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use causal::{
    CausalState, ConstraintId, DataBatchRef, GraphConstraintRecord, GraphEvidenceRecord,
    InterventionRecord, LgssmParams, LinearOlsSuffStats, ParticleFilterState, StateEvent,
    StreamingCovariance, apply_state_event, new_causal_state,
};
use causal_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    AverageEffectQuery, CacheBudget, CausalQuery, QueryId, VariableId,
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use crate::{catch_ffi, columns_to_batch, py_msg};

const DEFAULT_CACHE_BYTES: u64 = 1_048_576;

#[derive(Clone, Debug)]
struct RetainedBatch {
    names: Vec<String>,
    columns: Vec<Vec<f64>>,
}

/// Incremental causal analysis state (versioned events, no auto-rerun).
#[pyclass(name = "CausalState")]
pub struct PyCausalState {
    inner: CausalState,
    next_batch: u64,
    /// Python-owned column data keyed by batch id (Rust catalog stays metadata-only).
    retained: HashMap<String, RetainedBatch>,
}

impl PyCausalState {
    fn apply(&mut self, event: StateEvent) -> PyResult<u64> {
        let ver = apply_state_event(&mut self.inner, event).map_err(py_msg)?;
        Ok(ver.raw())
    }

    fn store_batch(
        &mut self,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<(Arc<str>, u64, u64)> {
        if names.len() != columns.len() {
            return Err(PyValueError::new_err("names/columns length mismatch"));
        }
        let batch = columns_to_batch(&names, &columns)?;
        let nrows = u64::try_from(batch.num_rows()).unwrap_or(u64::MAX);
        let bytes = u64::try_from(batch.get_array_memory_size()).unwrap_or(u64::MAX);
        let id_str = format!("b{}", self.next_batch);
        self.next_batch = self.next_batch.wrapping_add(1);
        let mut owned = Vec::with_capacity(columns.len());
        for col in &columns {
            owned.push(col.as_slice()?.to_vec());
        }
        self.retained.insert(
            id_str.clone(),
            RetainedBatch { names, columns: owned },
        );
        Ok((Arc::from(id_str), nrows, bytes))
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
            retained: HashMap::new(),
        }
    }

    /// Monotonic state version (starts at 0; bumps on each applied event).
    #[getter]
    fn version(&self) -> u64 {
        self.inner.version.raw()
    }

    /// Current data-catalog version.
    #[getter]
    fn data_version(&self) -> u64 {
        self.inner.data_catalog.version.raw()
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

    /// Batch ids currently retained in Python (catalog order).
    fn batch_ids(&self) -> Vec<String> {
        self.inner
            .data_catalog
            .batches
            .iter()
            .map(|b| b.id.to_string())
            .collect()
    }

    /// Append a tabular batch and retain its float64 columns.
    ///
    /// Returns the new state version.
    fn append_data(
        &mut self,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<u64> {
        let (id, nrows, bytes) = self.store_batch(names, columns)?;
        self.apply(StateEvent::AppendData(DataBatchRef { id, nrows, bytes }))
    }

    /// Replace all retained batches; optionally load one new batch.
    ///
    /// Returns the new state version.
    #[pyo3(signature = (names=None, columns=None))]
    fn replace_data(
        &mut self,
        names: Option<Vec<String>>,
        columns: Option<Vec<PyReadonlyArray1<'_, f64>>>,
    ) -> PyResult<u64> {
        self.retained.clear();
        self.inner.data_catalog.batches.clear();
        let new_ver = self.inner.data_catalog.version.next();
        let ver = self.apply(StateEvent::ReplaceData(new_ver))?;
        if let (Some(names), Some(columns)) = (names, columns) {
            let (id, nrows, bytes) = self.store_batch(names, columns)?;
            // Catalog was cleared by ReplaceData; append the replacement batch.
            return self.apply(StateEvent::AppendData(DataBatchRef { id, nrows, bytes }));
        }
        Ok(ver)
    }

    /// Fetch retained columns for `batch_id` as `(names, list[ndarray])`.
    fn get_batch<'py>(
        &self,
        py: Python<'py>,
        batch_id: &str,
    ) -> PyResult<(Vec<String>, Vec<Bound<'py, PyArray1<f64>>>)> {
        let batch = self
            .retained
            .get(batch_id)
            .ok_or_else(|| PyValueError::new_err(format!("unknown batch_id `{batch_id}`")))?;
        let cols: Vec<_> = batch
            .columns
            .iter()
            .map(|c| PyArray1::from_vec(py, c.clone()))
            .collect();
        Ok((batch.names.clone(), cols))
    }

    /// Row count for a retained batch.
    fn batch_nrows(&self, batch_id: &str) -> PyResult<usize> {
        let batch = self
            .retained
            .get(batch_id)
            .ok_or_else(|| PyValueError::new_err(format!("unknown batch_id `{batch_id}`")))?;
        Ok(batch.columns.first().map_or(0, Vec::len))
    }

    /// Add opaque graph evidence; returns new version.
    fn add_graph_evidence(&mut self, evidence_id: String, fingerprint: u64, bytes: u64) -> PyResult<u64> {
        self.apply(StateEvent::AddGraphEvidence(GraphEvidenceRecord {
            id: Arc::from(evidence_id),
            fingerprint,
            bytes,
        }))
    }

    /// List `(id, fingerprint, bytes)` graph-evidence records.
    fn graph_evidence(&self) -> Vec<(String, u64, u64)> {
        self.inner
            .graph_evidence
            .evidence
            .iter()
            .map(|e| (e.id.to_string(), e.fingerprint, e.bytes))
            .collect()
    }

    /// Add a graph constraint; returns new version.
    fn add_constraint(&mut self, constraint_id: String, fingerprint: u64) -> PyResult<u64> {
        self.apply(StateEvent::AddConstraint(GraphConstraintRecord {
            id: Arc::from(constraint_id),
            fingerprint,
        }))
    }

    /// Remove a graph constraint by id; returns new version.
    fn remove_constraint(&mut self, constraint_id: String) -> PyResult<u64> {
        self.apply(StateEvent::RemoveConstraint(ConstraintId(Arc::from(constraint_id))))
    }

    /// List active constraint `(id, fingerprint)` pairs.
    fn constraints(&self) -> Vec<(String, u64)> {
        self.inner
            .graph_evidence
            .constraints
            .iter()
            .map(|c| (c.id.to_string(), c.fingerprint))
            .collect()
    }

    /// Update / insert a named assumption (default provenance: stated / identification / assumed).
    fn update_assumption(&mut self, kind: &str) -> PyResult<u64> {
        let assumption = match kind {
            "causal_markov" => Assumption::CausalMarkov,
            "faithfulness" => Assumption::Faithfulness,
            "causal_sufficiency" => Assumption::CausalSufficiency,
            "consistency" => Assumption::Consistency,
            "positivity" => Assumption::Positivity,
            "no_interference" => Assumption::NoInterference,
            other => {
                return Err(PyValueError::new_err(format!("unknown assumption kind `{other}`")));
            }
        };
        self.apply(StateEvent::UpdateAssumption(AssumptionRecord {
            assumption,
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        }))
    }

    /// Register a binary average-effect query; returns `(version, query_id)`.
    fn register_average_effect(&mut self, treatment: u32, outcome: u32) -> PyResult<(u64, u64)> {
        let q = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(treatment),
            VariableId::from_raw(outcome),
        ));
        let before: Vec<_> = self.inner.queries.queries.keys().copied().collect();
        let ver = self.apply(StateEvent::RegisterQuery(q))?;
        let qid = self
            .inner
            .queries
            .queries
            .keys()
            .copied()
            .find(|k| !before.contains(k))
            .map(|k| u64::from(k.raw()))
            .ok_or_else(|| PyValueError::new_err("register_query did not assign an id"))?;
        Ok((ver, qid))
    }

    /// Record an opaque intervention; returns new version.
    fn record_intervention(&mut self, intervention_id: String, fingerprint: u64) -> PyResult<u64> {
        self.apply(StateEvent::RecordIntervention(InterventionRecord {
            id: Arc::from(intervention_id),
            fingerprint,
        }))
    }

    /// Mark queries fresh at fingerprints: list of `(query_id, fingerprint, bytes)`.
    fn refresh_results(&mut self, entries: Vec<(u64, u64, u64)>) -> PyResult<()> {
        let mapped: Vec<_> = entries
            .into_iter()
            .map(|(q, fp, bytes)| (QueryId::from_raw(q as u32), fp, bytes))
            .collect();
        self.inner.refresh_results(&mapped).map_err(py_msg)?;
        Ok(())
    }

    /// Ensure an OLS sufficient-stat slot exists with `ncols` predictors.
    fn ols_ensure(&mut self, key: String, ncols: usize) -> PyResult<()> {
        self.inner
            .suff_stats
            .ols
            .entry(Arc::from(key))
            .or_insert_with(|| LinearOlsSuffStats::new(ncols));
        Ok(())
    }

    /// Append one OLS design row / response under `key`.
    fn ols_append_row(&mut self, key: String, row: Vec<f64>, y: f64) -> PyResult<()> {
        let stats = self
            .inner
            .suff_stats
            .ols
            .get_mut(key.as_str())
            .ok_or_else(|| PyValueError::new_err(format!("unknown ols key `{key}`")))?;
        stats.append_row(&row, y).map_err(py_msg)?;
        Ok(())
    }

    /// Return OLS summary dict for `key`: `{n, ncols, xtx, xty, yty}`.
    fn ols_get<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyDict>> {
        let stats = self
            .inner
            .suff_stats
            .ols
            .get(key)
            .ok_or_else(|| PyValueError::new_err(format!("unknown ols key `{key}`")))?;
        let d = PyDict::new(py);
        d.set_item("n", stats.n)?;
        d.set_item("ncols", stats.ncols)?;
        d.set_item("xtx", stats.xtx.clone())?;
        d.set_item("xty", stats.xty.clone())?;
        d.set_item("yty", stats.yty)?;
        Ok(d)
    }

    /// Ensure a streaming-covariance slot of dimension `dim`.
    fn cov_ensure(&mut self, key: String, dim: usize) -> PyResult<()> {
        self.inner
            .suff_stats
            .cov
            .entry(Arc::from(key))
            .or_insert_with(|| StreamingCovariance::new(dim));
        Ok(())
    }

    /// Observe one row into streaming covariance `key`.
    fn cov_update(&mut self, key: String, row: Vec<f64>) -> PyResult<()> {
        let cov = self
            .inner
            .suff_stats
            .cov
            .get_mut(key.as_str())
            .ok_or_else(|| PyValueError::new_err(format!("unknown cov key `{key}`")))?;
        cov.append(&row).map_err(py_msg)?;
        Ok(())
    }

    /// Return streaming-cov summary `{n, dim, mean, m2}`.
    fn cov_get<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyDict>> {
        let cov = self
            .inner
            .suff_stats
            .cov
            .get(key)
            .ok_or_else(|| PyValueError::new_err(format!("unknown cov key `{key}`")))?;
        let d = PyDict::new(py);
        d.set_item("n", cov.n)?;
        d.set_item("dim", cov.dim)?;
        d.set_item("mean", cov.mean.clone())?;
        d.set_item("m2", cov.m2.clone())?;
        Ok(d)
    }

    /// Initialize a particle filter under `key`.
    #[pyo3(signature = (key, n_particles, *, a=0.9, process_std=0.3, obs_std=0.5, seed=1))]
    fn particle_filter_init(
        &mut self,
        key: String,
        n_particles: usize,
        a: f64,
        process_std: f64,
        obs_std: f64,
        seed: u64,
    ) -> PyResult<()> {
        let pf = ParticleFilterState::init(
            n_particles,
            LgssmParams { a, process_std, obs_std },
            self.inner.data_catalog.version.raw(),
            seed,
        )
        .map_err(py_msg)?;
        self.inner.suff_stats.particle_filters.insert(Arc::from(key), pf);
        Ok(())
    }

    /// Step particle filter `key` with observation `y`.
    fn particle_filter_step(&mut self, key: String, y: f64) -> PyResult<()> {
        let pf = self
            .inner
            .suff_stats
            .particle_filters
            .get_mut(key.as_str())
            .ok_or_else(|| PyValueError::new_err(format!("unknown particle filter `{key}`")))?;
        pf.step(y).map_err(py_msg)?;
        Ok(())
    }

    /// Particle-filter summary `{n_obs, n_particles, particles, log_weights}`.
    fn particle_filter_get<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyDict>> {
        let pf = self
            .inner
            .suff_stats
            .particle_filters
            .get(key)
            .ok_or_else(|| PyValueError::new_err(format!("unknown particle filter `{key}`")))?;
        let d = PyDict::new(py);
        d.set_item("n_obs", pf.n_obs)?;
        d.set_item("n_particles", pf.n_particles)?;
        d.set_item("particles", pf.particles.clone())?;
        d.set_item("log_weights", pf.log_weights.clone())?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "CausalState(version={}, stale_queries={}, batches={}, retained={})",
            self.inner.version.raw(),
            self.inner.stale_queries().len(),
            self.inner.data_catalog.batches.len(),
            self.retained.len(),
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
        for i in 0..n_appends {
            let id: Arc<str> = Arc::from(format!("synth{i}"));
            state.retained.insert(
                id.to_string(),
                RetainedBatch {
                    names: vec!["x".into()],
                    columns: vec![vec![0.0; 8]],
                },
            );
            apply_state_event(
                &mut state.inner,
                StateEvent::AppendData(DataBatchRef { id, nrows: 8, bytes: 64 }),
            )
            .map_err(py_msg)?;
        }
        Ok((state.inner.version.raw(), state.inner.stale_queries().len()))
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCausalState>()?;
    m.add_function(wrap_pyfunction!(causal_state_append, m)?)?;
    Ok(())
}
