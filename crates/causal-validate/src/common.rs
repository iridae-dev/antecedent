//! Shared refuter types and data transforms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{AverageEffectQuery, ExecutionContext, VariableId};
use causal_data::{TabularData, TableView, ValidityBitmap};
use causal_estimate::{EffectEstimate, EstimationWorkspace, LinearAdjustmentAte};
use causal_identify::IdentifiedEstimand;

use crate::error::ValidationError;

/// Comparison of original vs refuted estimates.
#[derive(Clone, Debug)]
pub struct RefutationReport {
    /// Refuter id.
    pub refuter: Arc<str>,
    /// Original ATE.
    pub original_ate: f64,
    /// Refuted / transformed ATE (mean across replicates when applicable).
    pub refuted_ate: f64,
    /// Absolute difference `|refuted - original|` (or `|refuted|` for placebo).
    pub comparison: f64,
    /// Whether the check is informative for the estimator used.
    pub informative: bool,
    /// Whether the check passed the configured threshold.
    pub passed: bool,
    /// Failure condition description when `passed` is false.
    pub failure_condition: Option<Arc<str>>,
    /// Number of replicate estimates.
    pub replicates: u32,
}

/// Inputs shared by Phase 1 effect refuters.
#[derive(Clone, Debug)]
pub struct RefutationProblem<'a> {
    /// Tabular data.
    pub data: &'a TabularData,
    /// Identified estimand (backdoor adjustment).
    pub estimand: &'a IdentifiedEstimand,
    /// Average-effect query (levels / population).
    pub query: &'a AverageEffectQuery,
    /// Original point estimate.
    pub original: &'a EffectEstimate,
}

impl RefutationProblem<'_> {
    /// Treatment variable from the query.
    #[must_use]
    pub fn treatment(&self) -> VariableId {
        self.query.treatment
    }

    /// Outcome variable from the query.
    #[must_use]
    pub fn outcome(&self) -> VariableId {
        self.query.outcome
    }
}

/// Rebuild tabular data replacing one float column (preserves mask/weights/other columns).
pub(crate) fn with_replaced_float(
    data: &TabularData,
    id: VariableId,
    values: Arc<[f64]>,
) -> Result<TabularData, ValidationError> {
    data.with_replaced_float(id, values).map_err(|e| ValidationError::Data(e.to_string()))
}

/// Append an independent continuous covariate; returns new data and its id.
pub(crate) fn with_extra_float(
    data: &TabularData,
    name: &str,
    values: Arc<[f64]>,
) -> Result<(TabularData, VariableId), ValidationError> {
    data.with_appended_float(name, values).map_err(|e| ValidationError::Data(e.to_string()))
}

/// Fit linear adjustment once (no nested bootstrap pools).
pub(crate) fn fit_once(
    estimator: &LinearAdjustmentAte,
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, ValidationError> {
    let prep = estimator
        .prepare(data, estimand, query)
        .map_err(|e| ValidationError::Estimation(e.to_string()))?;
    estimator
        .fit(&prep, workspace, ctx, causal_core::AssumptionSet::new())
        .map_err(|e| ValidationError::Estimation(e.to_string()))
}

/// Copy a full-length float64 column (unmasked; caller handles missingness).
pub(crate) fn float64_full(data: &TabularData, id: VariableId) -> Result<Vec<f64>, ValidationError> {
    data.float64_values(id).map_err(|e| ValidationError::Data(e.to_string()))
}

/// Restrict analysis to a random `keep_fraction` of rows (Bernoulli per-row draw), intersected
/// with any existing analysis mask / column validity.
pub(crate) fn with_row_subset(
    data: &TabularData,
    keep_fraction: f64,
    ctx: &ExecutionContext,
    stream_id: u64,
) -> Result<TabularData, ValidationError> {
    let n = data.row_count();
    let mut rng = ctx.rng.stream(stream_id);
    let mut bytes = vec![0u8; n.div_ceil(8)];
    for i in 0..n {
        if rng.next_f64() < keep_fraction {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    let mask = ValidityBitmap::from_bytes(bytes, n).map_err(|e| ValidationError::Data(e.to_string()))?;
    data.with_analysis_mask(mask).map_err(|e| ValidationError::Data(e.to_string()))
}

/// Rebuild tabular data with `ids` columns resampled (with replacement) per `idx`; all other
/// columns and metadata are preserved. `idx.len()` must equal `data.row_count()`.
pub(crate) fn with_resampled_rows(
    data: &TabularData,
    resample_ids: &[VariableId],
    row_idx: &[usize],
) -> Result<TabularData, ValidationError> {
    let mut out = data.clone();
    for &id in resample_ids {
        let full = float64_full(&out, id)?;
        let resampled: Vec<f64> = row_idx.iter().map(|&i| full[i]).collect();
        out = with_replaced_float(&out, id, Arc::from(resampled))?;
    }
    Ok(out)
}

/// Sample standard deviation (`NaN` for fewer than 2 values).
pub(crate) fn sample_sd(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return f64::NAN;
    }
    #[allow(clippy::cast_precision_loss)]
    let n_f = n as f64;
    let mean = values.iter().sum::<f64>() / n_f;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n_f - 1.0);
    var.sqrt()
}

/// Standard-normal-ish draws via Box–Muller from [`ExecutionContext`] RNG.
pub(crate) fn fill_gaussian(out: &mut [f64], ctx: &ExecutionContext, stream_id: u64) {
    let mut rng = ctx.rng.stream(stream_id);
    let mut i = 0;
    while i < out.len() {
        let u1 = rng.next_f64().clamp(1e-12, 1.0);
        let u2 = rng.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = core::f64::consts::TAU * u2;
        out[i] = r * theta.cos();
        i += 1;
        if i < out.len() {
            out[i] = r * theta.sin();
            i += 1;
        }
    }
}
