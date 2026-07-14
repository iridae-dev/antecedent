//! Shared refuter types and data transforms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{AverageEffectQuery, ExecutionContext, VariableId};
use causal_data::{TableView, TabularData, ValidityBitmap};
use causal_estimate::{
    EffectEstimate, EstimationWorkspace, LinearAdjustmentAte, OverlapPolicy, OverlapReport,
};
use causal_identify::IdentifiedEstimand;
use causal_stats::{FaerBackend, GlmOptions, PropensityFit, PropensityWorkspace, fit_propensity};

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
    /// Scale-free comparison statistic. Replicate-based refuters store the two-sided
    /// p-value of the null value under the replicate distribution; sensitivity grids
    /// store the robustness value; overlap/e-value checks store their own statistic.
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

/// Inputs shared by effect refuters.
#[derive(Clone, Copy, Debug)]
pub struct RefutationProblem<'a> {
    /// Tabular data.
    pub data: &'a TabularData,
    /// Identified estimand (backdoor adjustment).
    pub estimand: &'a IdentifiedEstimand,
    /// Average-effect query (levels / population).
    pub query: &'a AverageEffectQuery,
    /// Original point estimate.
    pub original: &'a EffectEstimate,
    /// Estimator id used for the original fit (e.g. `linear.adjustment.ate`), when known.
    pub estimator: Option<&'a str>,
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
    data.with_replaced_float(id, values).map_err(ValidationError::from)
}

/// Append an independent continuous covariate; returns new data and its id.
pub(crate) fn with_extra_float(
    data: &TabularData,
    name: &str,
    values: Arc<[f64]>,
) -> Result<(TabularData, VariableId), ValidationError> {
    data.with_appended_float(name, values).map_err(ValidationError::from)
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
    let prep = estimator.prepare(data, estimand, query).map_err(ValidationError::from)?;
    estimator
        .fit(&prep, workspace, ctx, causal_core::AssumptionSet::new())
        .map_err(ValidationError::from)
}

/// Scores / treatment / optional outcome from a diagnostic propensity fit.
pub(crate) struct DiagnosticPropensityColumns {
    /// Fitted propensity scores.
    pub scores: Vec<f64>,
    /// Treatment column (complete cases).
    pub treatment: Vec<f64>,
    /// Outcome column when requested.
    pub outcome: Option<Vec<f64>>,
}

/// Diagnostic-only logistic propensity on treatment + adjustment covariates.
///
/// Used by overlap / Reisz validators when the original estimate has no propensity report.
pub(crate) fn fit_diagnostic_propensity(
    problem: &RefutationProblem<'_>,
    glm_options: &GlmOptions,
    include_outcome_in_mask: bool,
) -> Result<DiagnosticPropensityColumns, ValidationError> {
    let mut ids = vec![problem.treatment()];
    if include_outcome_in_mask {
        ids.push(problem.outcome());
    }
    ids.extend_from_slice(&problem.estimand.adjustment_set);
    let row_mask = problem.data.complete_case_mask(&ids).map_err(ValidationError::from)?;
    let treatment = problem
        .data
        .float64_masked(problem.treatment(), &row_mask)
        .map_err(ValidationError::from)?;
    let outcome = if include_outcome_in_mask {
        Some(
            problem
                .data
                .float64_masked(problem.outcome(), &row_mask)
                .map_err(ValidationError::from)?,
        )
    } else {
        None
    };
    let nrows = treatment.len();
    let ncols = 1 + problem.estimand.adjustment_set.len();
    let mut design = vec![0.0; nrows * ncols];
    for r in design.iter_mut().take(nrows) {
        *r = 1.0;
    }
    for (i, &z) in problem.estimand.adjustment_set.iter().enumerate() {
        let col = problem.data.float64_masked(z, &row_mask).map_err(ValidationError::from)?;
        let base = (1 + i) * nrows;
        design[base..base + nrows].copy_from_slice(&col);
    }
    let backend = FaerBackend;
    let mut ws = PropensityWorkspace::default();
    let fit: PropensityFit =
        fit_propensity(&design, nrows, ncols, &treatment, &backend, &mut ws, glm_options)
            .map_err(ValidationError::from)?;
    Ok(DiagnosticPropensityColumns { scores: fit.scores, treatment, outcome })
}

/// Build an [`OverlapReport`] from a diagnostic propensity fit.
pub(crate) fn diagnostic_overlap_report(
    problem: &RefutationProblem<'_>,
    glm_options: &GlmOptions,
    policy: OverlapPolicy,
) -> Result<OverlapReport, ValidationError> {
    let cols = fit_diagnostic_propensity(problem, glm_options, false)?;
    Ok(OverlapReport::from_propensities(&cols.scores, None, policy))
}

/// Linear adjustment with nested bootstrap disabled (refuters / sensitivity grids).
#[must_use]
pub(crate) fn linear_estimator_no_bootstrap() -> LinearAdjustmentAte {
    let mut estimator = LinearAdjustmentAte::new();
    estimator.bootstrap_replicates = 0;
    estimator
}

/// Which column a noise-replace refuter overwrites.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NoiseReplaceTarget {
    /// Replace the treatment column.
    Treatment,
    /// Replace the outcome column.
    Outcome,
}

/// Shared placebo / dummy-outcome loop: replace a column with Gaussian noise and refit.
///
/// Passes when the replicate ATE distribution is statistically consistent with zero
/// (two-sided normal test, `p >= alpha`), so the verdict is invariant to outcome units.
#[allow(clippy::too_many_arguments)]
pub(crate) fn noise_replace_refute(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
    estimator: &LinearAdjustmentAte,
    replicates: u32,
    alpha: f64,
    target: NoiseReplaceTarget,
    stream_base: u64,
    refuter_id: &'static str,
    failure_label: &'static str,
) -> Result<RefutationReport, ValidationError> {
    if replicates < 2 {
        return Err(ValidationError::NotApplicable {
            message: "noise-replace refuter requires replicates >= 2",
        });
    }
    let replace_id = match target {
        NoiseReplaceTarget::Treatment => problem.treatment(),
        NoiseReplaceTarget::Outcome => problem.outcome(),
    };
    let n = problem.data.row_count();
    let mut ates = Vec::with_capacity(replicates as usize);
    for r in 0..replicates {
        let mut noise = vec![0.0; n];
        fill_gaussian(&mut noise, ctx, stream_base.wrapping_add(u64::from(r)));
        let data = with_replaced_float(problem.data, replace_id, Arc::from(noise))?;
        let est = fit_once(estimator, &data, problem.estimand, problem.query, workspace, ctx)?;
        ates.push(est.ate);
    }
    let mean_ate = ates.iter().sum::<f64>() / f64::from(replicates);
    let p_value = replicate_p_value(&ates, 0.0);
    let passed = p_value >= alpha;
    Ok(RefutationReport {
        refuter: Arc::from(refuter_id),
        original_ate: problem.original.ate,
        refuted_ate: mean_ate,
        comparison: p_value,
        informative: true,
        passed,
        failure_condition: (!passed).then(|| {
            Arc::from(format!(
                "{failure_label} ATE distribution (mean {mean_ate}) is inconsistent with zero \
                 (p={p_value} < alpha={alpha})"
            ))
        }),
        replicates,
    })
}

/// Two-sided p-value of observing `hypothesized` under a normal fit to `samples`
/// (DoWhy-style refuter significance test). Degenerate spread compares means directly.
pub(crate) fn replicate_p_value(samples: &[f64], hypothesized: f64) -> f64 {
    if samples.len() < 2 {
        return 1.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let sd = sample_sd(samples);
    let scale = mean.abs().max(hypothesized.abs()).max(1.0);
    if !sd.is_finite() {
        return 1.0;
    }
    if sd <= 1e-12 * scale {
        return if (hypothesized - mean).abs() <= 1e-9 * scale { 1.0 } else { 0.0 };
    }
    let z = (hypothesized - mean) / sd;
    erfc_approx(z.abs() / std::f64::consts::SQRT_2)
}

// erfc via Abramowitz–Stegun 7.1.26 (max abs error ~1.5e-7, ample for refuter verdicts).
fn erfc_approx(x: f64) -> f64 {
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-ax * ax).exp();
    let signed = if x < 0.0 { -erf } else { erf };
    1.0 - signed
}

/// Copy a full-length float64 column (unmasked; caller handles missingness).
pub(crate) fn float64_full(
    data: &TabularData,
    id: VariableId,
) -> Result<Vec<f64>, ValidationError> {
    data.float64_values(id).map_err(ValidationError::from)
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
    let mask = ValidityBitmap::from_bytes(bytes, n).map_err(ValidationError::from)?;
    data.with_analysis_mask(mask).map_err(ValidationError::from)
}

/// Rebuild tabular data with `ids` columns resampled (with replacement) per `idx`; all other
/// columns and metadata are preserved. `idx.len()` must equal `data.row_count()` and every
/// index must point at a complete-case row for `resample_ids` (see [`complete_case_rows`]);
/// `keep` re-hides rows that were invalid in the source so the replicate keeps the original
/// effective sample size.
pub(crate) fn with_resampled_rows(
    data: &TabularData,
    resample_ids: &[VariableId],
    row_idx: &[usize],
    keep: &[bool],
) -> Result<TabularData, ValidationError> {
    let mut out = data.clone();
    for &id in resample_ids {
        let full = float64_full(&out, id)?;
        let resampled: Vec<f64> = row_idx.iter().map(|&i| full[i]).collect();
        out = with_replaced_float(&out, id, Arc::from(resampled))?;
    }
    if keep.iter().all(|&k| k) {
        return Ok(out);
    }
    let n = keep.len();
    let mut bytes = vec![0u8; n.div_ceil(8)];
    for (i, &k) in keep.iter().enumerate() {
        if k {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    let mask = ValidityBitmap::from_bytes(bytes, n).map_err(ValidationError::from)?;
    out.with_analysis_mask(mask).map_err(ValidationError::from)
}

/// Complete-case mask and the list of valid row indexes for `ids` (analysis mask included).
pub(crate) fn complete_case_rows(
    data: &TabularData,
    ids: &[VariableId],
) -> Result<(Vec<bool>, Vec<usize>), ValidationError> {
    let mask = data.complete_case_mask(ids).map_err(ValidationError::from)?;
    let valid: Vec<usize> = mask.iter().enumerate().filter_map(|(i, &k)| k.then_some(i)).collect();
    Ok((mask, valid))
}

/// Sample standard deviation of a column over the complete-case rows of `ids`.
pub(crate) fn masked_sample_sd(
    data: &TabularData,
    id: VariableId,
    mask: &[bool],
) -> Result<f64, ValidationError> {
    let vals = data.float64_masked(id, mask).map_err(ValidationError::from)?;
    Ok(sample_sd(&vals))
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
    let var = values
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / (n_f - 1.0);
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
