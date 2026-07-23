//! Shared refuter types and data transforms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, ExecutionContext, KernelPolicy, TemporalEffectQuery, VariableId,
};
use antecedent_data::TemporalIndexer;
use antecedent_data::{
    DiscoveryEstimationSplit, PanelData, PanelUnit, TableView, TabularData, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_estimate::{
    EffectEstimate, EstimationWorkspace, LinearAdjustmentAte, OverlapPolicy, OverlapReport,
    TemporalLinearAdjustment,
};
use antecedent_identify::IdentifiedEstimand;
use antecedent_kernels::erfc;
use antecedent_stats::{
    FaerBackend, GlmOptions, PropensityFit, PropensityWorkspace, fit_propensity_diagnostic,
};

use crate::error::ValidationError;

/// Context for lag-aware temporal refits (series or panel).
#[derive(Clone, Copy, Debug)]
pub struct TemporalRefitContext<'a> {
    /// Unfolded temporal indexer from identification.
    pub indexer: &'a TemporalIndexer,
    /// Temporal effect query (pulse / sustained).
    pub temporal_query: &'a TemporalEffectQuery,
    /// Optional discovery/estimation split.
    pub split: Option<&'a DiscoveryEstimationSplit>,
    /// Kernel policy for lag sample preparation.
    pub kernel_policy: &'a KernelPolicy,
    /// Series time index, or `None` when refitting a panel.
    pub time_index: Option<&'a TimeIndex>,
    /// Panel units when refitting stacked panel designs.
    pub panel: Option<&'a PanelData>,
}

impl TemporalRefitContext<'_> {
    /// True when this context targets a panel (stacked cluster design).
    #[must_use]
    pub fn is_panel(&self) -> bool {
        self.panel.is_some()
    }
}

/// Comparison of original vs refuted estimates.
#[derive(Clone, Debug)]
#[non_exhaustive]
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

impl RefutationReport {
    /// Construct a refutation report.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        refuter: impl Into<Arc<str>>,
        original_ate: f64,
        refuted_ate: f64,
        comparison: f64,
        informative: bool,
        passed: bool,
        failure_condition: Option<Arc<str>>,
        replicates: u32,
    ) -> Self {
        Self {
            refuter: refuter.into(),
            original_ate,
            refuted_ate,
            comparison,
            informative,
            passed,
            failure_condition,
            replicates,
        }
    }
}

/// Inputs shared by effect refuters.
#[derive(Clone, Copy, Debug)]
pub struct RefutationProblem<'a> {
    /// Tabular data (series storage wrap, or stacked panel rows for mutation).
    pub data: &'a TabularData,
    /// Identified estimand (backdoor adjustment).
    pub estimand: &'a IdentifiedEstimand,
    /// Average-effect query (levels / population).
    pub query: &'a AverageEffectQuery,
    /// Original point estimate.
    pub original: &'a EffectEstimate,
    /// Estimator id used for the original fit (e.g. `linear.adjustment.ate`), when known.
    pub estimator: Option<&'a str>,
    /// When set, refits use [`TemporalLinearAdjustment`] on the lag-aligned design.
    pub temporal: Option<TemporalRefitContext<'a>>,
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
        .fit(&prep, workspace, ctx, antecedent_core::AssumptionSet::new())
        .map_err(ValidationError::from)
}

/// Static or temporal effect refit (bootstrap disabled).
pub(crate) fn refit_effect(
    problem: &RefutationProblem<'_>,
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    extra_contemporaneous: &[VariableId],
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, ValidationError> {
    let Some(temporal) = problem.temporal else {
        let est = linear_estimator_no_bootstrap();
        return fit_once(&est, data, estimand, problem.query, workspace, ctx);
    };
    let mut estimator = TemporalLinearAdjustment::new();
    estimator.inner.bootstrap_replicates = 0;
    estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
    if let Some(panel) = temporal.panel {
        let rebuilt = panel_from_stacked(panel, data)?;
        let prep = if extra_contemporaneous.is_empty() {
            let (prep, _cluster_ids) = estimator
                .prepare_panel(
                    &rebuilt,
                    estimand,
                    temporal.temporal_query,
                    temporal.indexer,
                    temporal.split,
                    temporal.kernel_policy,
                )
                .map_err(ValidationError::from)?;
            prep
        } else {
            panel_prepare_with_extras(
                &estimator,
                &rebuilt,
                estimand,
                &temporal,
                extra_contemporaneous,
            )?
        };
        return estimator
            .fit(&prep, workspace, ctx, antecedent_core::AssumptionSet::new())
            .map_err(ValidationError::from);
    }
    let time_index = temporal.time_index.ok_or(ValidationError::NotApplicable {
        message: "temporal series refit requires time_index",
    })?;
    let series = TimeSeriesData::try_new(data.storage().clone(), time_index.clone())
        .map_err(ValidationError::from)?;
    let prep = estimator
        .prepare_with_extras(
            &series,
            estimand,
            temporal.temporal_query,
            temporal.indexer,
            temporal.split,
            temporal.kernel_policy,
            extra_contemporaneous,
        )
        .map_err(ValidationError::from)?;
    estimator
        .fit(&prep, workspace, ctx, antecedent_core::AssumptionSet::new())
        .map_err(ValidationError::from)
}

fn panel_prepare_with_extras(
    estimator: &TemporalLinearAdjustment,
    panel: &PanelData,
    estimand: &IdentifiedEstimand,
    temporal: &TemporalRefitContext<'_>,
    extra: &[VariableId],
) -> Result<antecedent_estimate::PreparedEstimationProblem, ValidationError> {
    // Stack per-unit prepare_with_extras designs (mirrors prepare_panel).
    let mut all_t = Vec::new();
    let mut all_y = Vec::new();
    let mut all_covs: Vec<(VariableId, Vec<f64>)> = Vec::new();
    let mut adj_keys: Vec<VariableId> = Vec::new();
    let mut active = 0.0;
    let mut control = 0.0;
    let mut treatment_delta = 0.0;
    let mut first = true;
    for unit in panel.units() {
        let prep = estimator
            .prepare_with_extras(
                &unit.series,
                estimand,
                temporal.temporal_query,
                temporal.indexer,
                temporal.split,
                temporal.kernel_policy,
                extra,
            )
            .map_err(ValidationError::from)?;
        if first {
            active = prep.active;
            control = prep.control;
            treatment_delta = prep.treatment_delta;
            adj_keys = prep.adjustment_set.to_vec();
            all_covs = adj_keys.iter().map(|&id| (id, Vec::new())).collect();
            first = false;
        }
        all_t.extend_from_slice(&prep.treatment);
        all_y.extend_from_slice(&prep.design.outcome);
        let nrows = prep.design.nrows;
        for (i, (_id, dest)) in all_covs.iter_mut().enumerate() {
            let base = (2 + i) * nrows;
            dest.extend_from_slice(&prep.design.matrix[base..base + nrows]);
        }
    }
    let cov_refs: Vec<(VariableId, &[f64])> =
        all_covs.iter().map(|(id, v)| (*id, v.as_slice())).collect();
    let selected: Vec<usize> = (0..all_t.len()).collect();
    let design =
        antecedent_stats::CompiledDesign::linear_adjustment(&all_t, &cov_refs, &all_y, &selected)
            .map_err(ValidationError::from)?;
    Ok(antecedent_estimate::PreparedEstimationProblem {
        design,
        method: Arc::from("temporal.linear.adjustment.panel"),
        adjustment_set: Arc::from(adj_keys),
        overlap: OverlapPolicy::ExplicitOverride,
        treatment_delta,
        target_population: antecedent_core::TargetPopulation::AllObserved,
        treatment: Arc::from(all_t),
        active,
        control,
    })
}

/// Rebuild a panel from stacked tabular mutations (same unit lengths as `original`).
pub(crate) fn panel_from_stacked(
    original: &PanelData,
    stacked: &TabularData,
) -> Result<PanelData, ValidationError> {
    let expected = original.total_rows();
    if stacked.row_count() != expected {
        return Err(ValidationError::data_msg(format!(
            "stacked panel refute rows {} != panel total_rows {expected}",
            stacked.row_count()
        )));
    }
    let mut offset = 0usize;
    let mut units = Vec::with_capacity(original.unit_count());
    for u in original.units() {
        let n = u.series.row_count();
        let slice = slice_tabular(stacked, offset, n)?;
        let series =
            TimeSeriesData::try_new(slice.storage().clone(), u.series.time_index().clone())
                .map_err(ValidationError::from)?;
        units.push(PanelUnit { unit_id: u.unit_id, series });
        offset += n;
    }
    PanelData::try_new(Arc::from(units)).map_err(ValidationError::from)
}

fn slice_tabular(
    data: &TabularData,
    start: usize,
    len: usize,
) -> Result<TabularData, ValidationError> {
    use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage};
    let storage = data.storage();
    let end = start + len;
    if end > data.row_count() {
        return Err(ValidationError::NotApplicable { message: "panel slice out of range" });
    }
    let mut cols = Vec::with_capacity(storage.columns().len());
    for col in storage.columns() {
        match col {
            OwnedColumn::Float64(c) => {
                let values: Arc<[f64]> = Arc::from(c.values[start..end].to_vec());
                let validity = ValidityBitmap::all_valid(len);
                cols.push(OwnedColumn::Float64(
                    Float64Column::new(c.id, values, validity).map_err(ValidationError::from)?,
                ));
            }
            _ => {
                return Err(ValidationError::NotApplicable {
                    message: "panel refute slice requires float64 columns",
                });
            }
        }
    }
    let mask = storage.analysis_mask().map(|m| {
        let mut bytes = vec![0u8; len.div_ceil(8)];
        for i in 0..len {
            if m.is_valid(start + i) {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }
        ValidityBitmap::from_bytes(bytes, len)
    });
    let mask = mask.transpose().map_err(ValidationError::from)?;
    let weights = storage.weights().map(|w| Arc::<[f64]>::from(w[start..end].to_vec()));
    let new_storage = OwnedColumnarStorage::try_new(storage.schema().clone(), cols, mask, weights)
        .map_err(ValidationError::from)?;
    Ok(TabularData::new(new_storage))
}

/// Stack panel units into one tabular table (row-major concat) for refute mutations.
pub fn stack_panel_tabular(panel: &PanelData) -> Result<TabularData, ValidationError> {
    use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage};
    let total = panel.total_rows();
    let schema = panel.schema().clone();
    let n_cols = schema.len();
    let mut col_vals: Vec<Vec<f64>> = (0..n_cols).map(|_| Vec::with_capacity(total)).collect();
    for u in panel.units() {
        for (j, dest) in col_vals.iter_mut().enumerate() {
            let id =
                VariableId::from_raw(u32::try_from(j).map_err(|_| {
                    ValidationError::data_msg("panel column index exceeds u32::MAX")
                })?);
            let vals = u.series.float64_values(id).map_err(ValidationError::from)?;
            dest.extend_from_slice(&vals);
        }
    }
    let mut cols = Vec::with_capacity(n_cols);
    for (j, values) in col_vals.into_iter().enumerate() {
        let id = VariableId::from_raw(
            u32::try_from(j)
                .map_err(|_| ValidationError::data_msg("panel column index exceeds u32::MAX"))?,
        );
        cols.push(OwnedColumn::Float64(
            Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(total))
                .map_err(ValidationError::from)?,
        ));
    }
    let storage =
        OwnedColumnarStorage::try_new(schema, cols, None, None).map_err(ValidationError::from)?;
    Ok(TabularData::new(storage))
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
    propensity: &mut PropensityWorkspace,
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
    let fit: PropensityFit = fit_propensity_diagnostic(
        &design,
        nrows,
        ncols,
        &treatment,
        &backend,
        propensity,
        glm_options,
    )
    .map_err(ValidationError::from)?;
    Ok(DiagnosticPropensityColumns { scores: fit.scores, treatment, outcome })
}

/// Build an [`OverlapReport`] from a diagnostic propensity fit.
pub(crate) fn diagnostic_overlap_report(
    problem: &RefutationProblem<'_>,
    glm_options: &GlmOptions,
    policy: OverlapPolicy,
) -> Result<OverlapReport, ValidationError> {
    let mut ws = PropensityWorkspace::default();
    diagnostic_overlap_report_with(problem, glm_options, policy, &mut ws)
}

/// Like [`diagnostic_overlap_report`], reusing a warmed propensity workspace.
pub(crate) fn diagnostic_overlap_report_with(
    problem: &RefutationProblem<'_>,
    glm_options: &GlmOptions,
    policy: OverlapPolicy,
    propensity: &mut PropensityWorkspace,
) -> Result<OverlapReport, ValidationError> {
    let cols = fit_diagnostic_propensity(problem, glm_options, false, propensity)?;
    // ATE IPW weights so ESS / extreme-weight fields are defined for the overlap refuter.
    let weights: Vec<f64> = cols
        .treatment
        .iter()
        .zip(cols.scores.iter())
        .map(|(&t, &p)| {
            let p = p.clamp(1e-9, 1.0 - 1e-9);
            if t > 0.5 { 1.0 / p } else { 1.0 / (1.0 - p) }
        })
        .collect();
    Ok(OverlapReport::from_propensities(&cols.scores, Some(&weights), policy))
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
    _estimator: &LinearAdjustmentAte,
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
        let est = refit_effect(problem, &data, problem.estimand, &[], workspace, ctx)?;
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
/// (pinned baseline-style refuter significance test). Degenerate spread compares means directly.
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
    erfc(z.abs() / std::f64::consts::SQRT_2)
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
    antecedent_stats::sample_std(values)
}

/// Standard-normal draws via Box–Muller from [`ExecutionContext`] RNG.
///
/// Emits both cos/sin components from each uniform pair (same stream use as
/// historical seeded tests).
pub(crate) fn fill_gaussian(out: &mut [f64], ctx: &ExecutionContext, stream_id: u64) {
    let mut rng = ctx.rng.stream(stream_id);
    antecedent_kernels::fill_standard_normal(&mut rng, out);
}
