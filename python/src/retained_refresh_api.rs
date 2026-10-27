//! Retained summary refresh, retained bootstrap replicates and their disk-backed spill.
//!
//! A prepared study's frozen estimator decides which summary a result rests on
//! (`PreparedStudy::retained_refresh_family`). Python states the rows and each later change as
//! JSON; the summary arithmetic, the verification against a fresh reconstruction, the retention
//! refusals and the admission-bounded spill are Rust's. The admission pool is the one of the
//! `Execution` the call runs under.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::PreparedStudy;
use antecedent::bootstrap_retention::{RetainedBootstrap, SpilledBootstrap};
use antecedent::retained_refresh::{
    RefreshMutation, RefreshObservation, RefreshResult, RetainedRefresh, SpilledRefresh,
    spill_registry_within_admission,
};
use antecedent::state::{ConjugateOlsPrior, RefreshOutcome, SpillRegistry};
use antecedent_state::retention::{RetainedOperation, RetentionMode, RetentionPolicy};
use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::contract_refusals_api::PyRetentionPolicy;
use crate::prepared_api::PyPreparedAnalysis;
use crate::{py_err, value_err};

fn ctx() -> antecedent_core::ExecutionContext {
    crate::py_execution_context(1, 1)
}

fn operation_of(name: &str) -> PyResult<RetainedOperation> {
    Ok(match name {
        "row_reconstruction" => RetainedOperation::RowReconstruction,
        "refit_from_retained_rows" => RetainedOperation::RefitFromRetainedRows,
        "exact_row_eviction" => RetainedOperation::ExactRowEviction,
        "in_place_column_insert" => RetainedOperation::InPlaceColumnInsert,
        "in_place_weight_or_mask_update" => RetainedOperation::InPlaceWeightOrMaskUpdate,
        "reuse_full_state" => RetainedOperation::ReuseFullState,
        "reuse_summary" => RetainedOperation::ReuseSummary,
        "read_disk_artifact" => RetainedOperation::ReadDiskArtifact,
        other => return Err(value_err(format!("unknown retained operation `{other}`"))),
    })
}

fn field<'a>(value: &'a Value, key: &str) -> PyResult<&'a Value> {
    value.get(key).ok_or_else(|| value_err(format!("missing `{key}`")))
}

fn text<'a>(value: &'a Value, key: &str) -> PyResult<&'a str> {
    field(value, key)?.as_str().ok_or_else(|| value_err(format!("`{key}` must be text")))
}

fn real(value: &Value, key: &str) -> PyResult<f64> {
    field(value, key)?.as_f64().ok_or_else(|| value_err(format!("`{key}` must be a number")))
}

fn word(value: &Value, key: &str) -> PyResult<u64> {
    field(value, key)?
        .as_u64()
        .ok_or_else(|| value_err(format!("`{key}` must be a non-negative integer")))
}

fn reals(value: &Value, key: &str) -> PyResult<Vec<f64>> {
    field(value, key)?
        .as_array()
        .ok_or_else(|| value_err(format!("`{key}` must be a list")))?
        .iter()
        .map(|item| item.as_f64().ok_or_else(|| value_err(format!("`{key}` holds a non-number"))))
        .collect()
}

fn observation_of(value: &Value) -> PyResult<RefreshObservation> {
    Ok(RefreshObservation {
        id: word(value, "id")?,
        x: reals(value, "x")?,
        y: real(value, "y")?,
        weight: real(value, "weight")?,
        masked: field(value, "masked")?
            .as_bool()
            .ok_or_else(|| value_err("`masked` must be true or false"))?,
    })
}

fn observations_of(value: &Value, key: &str) -> PyResult<Vec<RefreshObservation>> {
    field(value, key)?
        .as_array()
        .ok_or_else(|| value_err(format!("`{key}` must be a list")))?
        .iter()
        .map(observation_of)
        .collect()
}

fn mutation_of(value: &Value) -> PyResult<RefreshMutation> {
    Ok(match text(value, "kind")? {
        "append" => RefreshMutation::Append {
            source: Arc::from(text(value, "source")?),
            rows: observations_of(value, "rows")?,
        },
        "window" => RefreshMutation::Window {
            max_rows: usize::try_from(word(value, "max_rows")?)
                .map_err(|_| value_err("`max_rows` is too large"))?,
        },
        "evict" => RefreshMutation::Evict { id: word(value, "id")? },
        "set_weight" => {
            RefreshMutation::SetWeight { id: word(value, "id")?, weight: real(value, "weight")? }
        }
        "set_mask" => RefreshMutation::SetMask {
            id: word(value, "id")?,
            masked: field(value, "masked")?
                .as_bool()
                .ok_or_else(|| value_err("`masked` must be true or false"))?,
        },
        "replace" => RefreshMutation::Replace {
            source: Arc::from(text(value, "source")?),
            rows: observations_of(value, "rows")?,
        },
        "block" => RefreshMutation::in_block(
            text(value, "label")?,
            mutation_of(field(value, "mutation")?)?,
        ),
        other => return Err(value_err(format!("unknown refresh mutation `{other}`"))),
    })
}

fn result_json(result: &RefreshResult) -> Value {
    match result {
        RefreshResult::Ols(s) => json!({
            "family": "ols", "source": &*s.source, "n": s.n, "xtx": s.xtx, "xty": s.xty,
            "yty": s.yty, "beta": s.beta, "residual_variance": s.residual_variance,
            "source_digest": s.source_digest,
            "rows": s.ledger.rows, "live_rows": s.ledger.live_rows,
        }),
        RefreshResult::Weighted(s) => json!({
            "family": "weighted_ols", "source": &*s.source, "total_weight": s.total_weight,
            "effective_sample_size": s.effective_sample_size, "xtx": s.xtx, "xty": s.xty,
            "yty": s.yty, "beta": s.beta, "residual_variance": s.residual_variance,
            "beta_se": s.beta_se, "source_digest": s.source_digest,
        }),
        RefreshResult::Covariance(s) => json!({
            "family": "covariance", "source": &*s.source, "n": s.n, "mean": s.mean,
            "m2": s.m2, "sample_covariance": s.sample_covariance,
            "source_digest": s.source_digest,
            "rows": s.ledger.rows, "live_rows": s.ledger.live_rows,
        }),
        RefreshResult::Conjugate { summary, identity } => json!({
            "family": "conjugate_blocks", "mean": summary.mean, "precision": summary.precision,
            "scale": summary.scale, "shape": summary.shape, "rate": summary.rate,
            "n": summary.n, "identity": identity,
        }),
    }
}

const fn outcome_label(outcome: RefreshOutcome) -> &'static str {
    match outcome {
        RefreshOutcome::Updated => "updated",
        RefreshOutcome::Recomputed => "recomputed",
        _ => "recomputed",
    }
}

fn retention_label(mode: RetentionMode) -> &'static str {
    mode.as_str()
}

/// A directory-backed registry of spilled results with a byte budget.
#[pyclass(name = "SpillRegistryHandle", frozen, skip_from_py_object)]
pub(crate) struct SpillRegistryHandle {
    inner: Arc<SpillRegistry>,
}

#[pymethods]
impl SpillRegistryHandle {
    /// Directory the registry writes under.
    fn directory(&self) -> String {
        self.inner.directory().display().to_string()
    }

    /// Counters as JSON: entries, used, budget, spills, evictions, reloads.
    fn stats_json(&self) -> String {
        let s = self.inner.stats();
        json!({
            "entries": s.entries, "used_bytes": s.used_bytes, "budget_bytes": s.budget_bytes,
            "spills": s.spills, "evictions": s.evictions, "reloads": s.reloads,
        })
        .to_string()
    }

    /// Whether `key` is currently on disk.
    fn contains(&self, key: &str) -> bool {
        self.inner.contains(key)
    }

    /// Keys on disk, most recently used first.
    fn keys_by_recency(&self) -> Vec<String> {
        self.inner.keys_by_recency()
    }
}

/// A registry whose byte budget never exceeds the admission pool of the running `Execution`.
#[pyfunction]
fn spill_registry_open(directory: &str, requested_bytes: u64) -> PyResult<SpillRegistryHandle> {
    let inner =
        spill_registry_within_admission(&ctx(), directory, requested_bytes).map_err(py_err)?;
    Ok(SpillRegistryHandle { inner })
}

/// The summary a prepared study's estimator keeps, with its rows.
#[pyclass(name = "RetainedRefreshHandle", skip_from_py_object)]
pub(crate) struct RetainedRefreshHandle {
    study: Arc<PreparedStudy>,
    inner: RetainedRefresh,
}

#[pymethods]
impl RetainedRefreshHandle {
    /// Summary family (`ols`, `weighted_ols`, `covariance`, `conjugate_blocks`).
    fn family(&self) -> &'static str {
        self.inner.family().as_str()
    }

    /// Apply one change and compare the complete result with a fresh reconstruction from the
    /// retained rows; the summary is unchanged when either is refused. Returns `updated` or
    /// `recomputed`.
    fn apply(&mut self, mutation_json: &str) -> PyResult<&'static str> {
        let value: Value = serde_json::from_str(mutation_json)
            .map_err(|e| value_err(format!("invalid refresh mutation: {e}")))?;
        let mutation = mutation_of(&value)?;
        self.study
            .apply_retained_refresh(&mut self.inner, &mutation)
            .map(outcome_label)
            .map_err(py_err)
    }

    /// The complete cached result as JSON.
    fn result_json(&self) -> PyResult<String> {
        Ok(result_json(&self.inner.result().map_err(py_err)?).to_string())
    }

    /// The complete result rebuilt from the retained rows alone, as JSON.
    fn fresh_result_json(&self) -> PyResult<String> {
        Ok(result_json(&self.inner.fresh_result().map_err(py_err)?).to_string())
    }

    /// Compare the cached result with a fresh reconstruction; raises on the first field that
    /// disagrees or when the rows are not retained.
    fn check_against_fresh(&self) -> PyResult<()> {
        self.inner.check_against_fresh().map_err(py_err)
    }

    /// Stop retaining rows; the result stays.
    fn summary_only(&self) -> Self {
        Self { study: Arc::clone(&self.study), inner: self.inner.clone().into_summary_only() }
    }

    /// Apply a retention policy: a window of rows is kept as a trailing window, zero rows or
    /// sufficient statistics drop the rows.
    fn with_retention(&self, policy: PyRef<'_, PyRetentionPolicy>) -> PyResult<Self> {
        let policy: RetentionPolicy = policy.inner;
        let mut next = self.inner.clone();
        match policy.max_retained_rows() {
            Some(0) => next = next.into_summary_only(),
            Some(max_rows) => {
                let max_rows = usize::try_from(max_rows)
                    .map_err(|_| value_err("retained row cap is too large"))?;
                next.apply(&RefreshMutation::Window { max_rows }).map_err(py_err)?;
            }
            None => {}
        }
        Ok(Self { study: Arc::clone(&self.study), inner: next })
    }

    /// Accept `operation` under this summary's retention, or raise the refusal.
    fn require(&self, operation: &str) -> PyResult<()> {
        self.inner.require(operation_of(operation)?).map_err(py_err)
    }

    /// Bytes a spill or reload of this summary asks the admission pool for.
    fn frame_bytes(&self) -> u64 {
        self.inner.frame_bytes()
    }

    /// Write the summary under `registry` after admitting its frame.
    fn spill(
        &self,
        registry: PyRef<'_, SpillRegistryHandle>,
        key: &str,
    ) -> PyResult<SpilledRefreshHandle> {
        let spilled = self
            .study
            .spill_retained_refresh(&self.inner, &ctx(), &registry.inner, key)
            .map_err(py_err)?;
        Ok(SpilledRefreshHandle { study: Arc::clone(&self.study), inner: spilled })
    }
}

/// A refresh summary written to disk.
#[pyclass(name = "SpilledRefreshHandle", skip_from_py_object)]
pub(crate) struct SpilledRefreshHandle {
    study: Arc<PreparedStudy>,
    inner: SpilledRefresh,
}

#[pymethods]
impl SpilledRefreshHandle {
    fn family(&self) -> &'static str {
        self.inner.family().as_str()
    }

    fn mode(&self) -> &'static str {
        retention_label(self.inner.mode())
    }

    fn on_disk(&self) -> bool {
        self.inner.on_disk()
    }

    fn evicted(&self) -> Vec<String> {
        self.inner.evicted().to_vec()
    }

    fn key(&self) -> String {
        self.inner.handle().key().to_owned()
    }

    fn require(&self, operation: &str) -> PyResult<()> {
        self.inner.require(operation_of(operation)?).map_err(py_err)
    }

    /// Read the summary back after admitting its frame.
    fn reload(&self) -> PyResult<RetainedRefreshHandle> {
        let inner = self.study.reload_retained_refresh(&self.inner, &ctx()).map_err(py_err)?;
        Ok(RetainedRefreshHandle { study: Arc::clone(&self.study), inner })
    }
}

/// Start the summary a prepared study's estimator keeps over `rows_json` of `source`.
#[pyfunction]
#[pyo3(signature = (study, source, width, rows_json, prior_json=None))]
fn retained_refresh_start(
    study: PyRef<'_, PyPreparedAnalysis>,
    source: &str,
    width: usize,
    rows_json: &str,
    prior_json: Option<&str>,
) -> PyResult<RetainedRefreshHandle> {
    let rows_value: Value = serde_json::from_str(rows_json)
        .map_err(|e| value_err(format!("invalid refresh rows: {e}")))?;
    let rows = rows_value
        .as_array()
        .ok_or_else(|| value_err("rows must be a list"))?
        .iter()
        .map(observation_of)
        .collect::<PyResult<Vec<_>>>()?;
    let prior = prior_json
        .map(|text| -> PyResult<(ConjugateOlsPrior, String)> {
            let value: Value =
                serde_json::from_str(text).map_err(|e| value_err(format!("invalid prior: {e}")))?;
            let prior = ConjugateOlsPrior::new(
                reals(&value, "mean")?,
                reals(&value, "precision")?,
                real(&value, "shape")?,
                real(&value, "rate")?,
            )
            .map_err(py_err)?;
            Ok((prior, text_of(&value, "label")?))
        })
        .transpose()?;
    let inner = study
        .inner
        .start_retained_refresh(
            source,
            width,
            &rows,
            prior.as_ref().map(|(prior, label)| (prior, label.as_str())),
        )
        .map_err(py_err)?;
    Ok(RetainedRefreshHandle { study: Arc::clone(&study.inner), inner })
}

fn text_of(value: &Value, key: &str) -> PyResult<String> {
    text(value, key).map(str::to_owned)
}

/// The summary family a prepared study's estimator keeps.
#[pyfunction]
fn retained_refresh_family(study: PyRef<'_, PyPreparedAnalysis>) -> PyResult<&'static str> {
    study.inner.retained_refresh_family().map(|family| family.as_str()).map_err(py_err)
}

/// Replicate estimates of one bootstrap and the running summary they stand for.
#[pyclass(name = "RetainedBootstrapHandle", skip_from_py_object)]
pub(crate) struct RetainedBootstrapHandle {
    inner: RetainedBootstrap,
}

#[pymethods]
impl RetainedBootstrapHandle {
    #[new]
    #[pyo3(signature = (estimates, failed=0))]
    fn new(estimates: Vec<f64>, failed: u32) -> PyResult<Self> {
        Ok(Self { inner: RetainedBootstrap::from_replicates(estimates, failed).map_err(py_err)? })
    }

    fn retention(&self) -> &'static str {
        retention_label(self.inner.retention())
    }

    fn estimates(&self) -> Vec<f64> {
        self.inner.estimates().to_vec()
    }

    fn summary_only(&self) -> Self {
        Self { inner: self.inner.clone().into_summary_only() }
    }

    /// Add replicates in place under any retention.
    fn append(&mut self, estimates: Vec<f64>, failed: u32) -> PyResult<()> {
        self.inner.append(&estimates, failed).map_err(py_err)
    }

    /// Replicate counts and the standard error as JSON.
    fn accounting_json(&self) -> String {
        let a = self.inner.accounting();
        json!({"se": a.se, "replicates_ok": a.replicates_ok, "replicates_failed": a.replicates_failed})
            .to_string()
    }

    fn mean(&self) -> Option<f64> {
        self.inner.mean()
    }

    fn percentile_interval(&self, level: f64) -> PyResult<(f64, f64)> {
        self.inner.percentile_interval(level).map_err(py_err)
    }

    fn check_against_fresh(&self) -> PyResult<()> {
        self.inner.check_against_fresh().map_err(py_err)
    }

    fn frame_bytes(&self) -> u64 {
        self.inner.frame_bytes()
    }

    fn spill(
        &self,
        registry: PyRef<'_, SpillRegistryHandle>,
        key: &str,
    ) -> PyResult<SpilledBootstrapHandle> {
        let inner = self.inner.spill(&ctx(), &registry.inner, key).map_err(py_err)?;
        Ok(SpilledBootstrapHandle { inner })
    }
}

/// A bootstrap result written to disk.
#[pyclass(name = "SpilledBootstrapHandle", skip_from_py_object)]
pub(crate) struct SpilledBootstrapHandle {
    inner: SpilledBootstrap,
}

#[pymethods]
impl SpilledBootstrapHandle {
    fn mode(&self) -> &'static str {
        retention_label(self.inner.mode())
    }

    fn on_disk(&self) -> bool {
        self.inner.on_disk()
    }

    fn evicted(&self) -> Vec<String> {
        self.inner.evicted().to_vec()
    }

    fn require(&self, operation: &str) -> PyResult<()> {
        self.inner.require(operation_of(operation)?).map_err(py_err)
    }

    fn reload(&self) -> PyResult<RetainedBootstrapHandle> {
        Ok(RetainedBootstrapHandle { inner: self.inner.reload(&ctx()).map_err(py_err)? })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SpillRegistryHandle>()?;
    m.add_class::<RetainedRefreshHandle>()?;
    m.add_class::<SpilledRefreshHandle>()?;
    m.add_class::<RetainedBootstrapHandle>()?;
    m.add_class::<SpilledBootstrapHandle>()?;
    m.add_function(wrap_pyfunction!(spill_registry_open, m)?)?;
    m.add_function(wrap_pyfunction!(retained_refresh_start, m)?)?;
    m.add_function(wrap_pyfunction!(retained_refresh_family, m)?)?;
    Ok(())
}
