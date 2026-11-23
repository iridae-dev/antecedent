//! Refuter table transforms, sample summaries and temporal design helpers.

use antecedent_data::TableView;

use super::{
    Arc, ColumnView, EstimationWorkspace, ExecutionContext, Float64Column, LinearAdjustmentAte,
    OwnedColumn, OwnedColumnarStorage, RefutationProblem, TabularData, ValidationError,
    ValidityBitmap, VariableId, refit_effect,
};

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

/// The same-effect refit of the *unperturbed* problem: the centre a perturbation refuter
/// compares its replicates against.
///
/// Refits always run the least-squares pipeline of [`refit_effect`], while `problem.original`
/// may be a different estimator's number (a Bayesian posterior mean shrunk toward its prior,
/// a conditional-effect scalar). Replicate spread is O(1/√n), the same order as that gap, so
/// centring on the published estimate would make a check of *stability under perturbation*
/// fail because of *estimator* differences. Centring on the full-sample refit compares like
/// with like.
pub(crate) fn full_sample_refit_ate(
    problem: &RefutationProblem<'_>,
    estimator: &LinearAdjustmentAte,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<f64, ValidationError> {
    let est =
        refit_effect(problem, problem.data, problem.estimand, &[], estimator, workspace, ctx)?;
    if est.ate.is_finite() {
        Ok(est.ate)
    } else {
        Err(ValidationError::estimation_msg(
            "non-finite full-sample refit effect: the refit did not produce a usable effect",
        ))
    }
}

/// Restrict a temporal series to one random contiguous window covering `keep_fraction` of
/// its rows, physically dropping the excluded prefix/suffix rows (never an interior mask).
///
/// [`with_row_subset`] cannot be reused for temporal designs: it hides rows behind an
/// analysis mask, but lag gathering ([`antecedent_data::LaggedSamplePlan::prepare`]) indexes
/// *raw row positions*, and `ensure_unmasked` rejects any non-fully-valid mask outright for
/// exactly that reason (a masked-out row would silently shift what "lag-1" reads once the
/// gather stepped over it). Physically dropping an interior hole instead of masking it would
/// dodge that error but not the underlying problem: rows on either side of the hole are no
/// longer adjacent in time, so a lag gather across the seam would quietly average together
/// samples separated by more than the intended step. A contiguous window sidesteps both
/// failure modes — every retained row's immediate predecessor in the window is still its
/// immediate predecessor in the original series, so lag-1/lag-2/etc. mean exactly what they
/// meant on the full data, and the shortened window is a genuine data-subset falsification
/// check (start-of-series and end-of-series windows are both reachable, so this is not
/// equivalent to a fixed train/test split).
///
/// Only float64 columns are supported (temporal designs are float64-typed end to end, same
/// restriction [`crate::panel_slice::PanelSliceTemplate`] applies to panel refits).
pub(crate) fn with_contiguous_row_window(
    data: &TabularData,
    keep_fraction: f64,
    ctx: &ExecutionContext,
    stream_id: u64,
) -> Result<TabularData, ValidationError> {
    let n = data.row_count();
    if n < 2 {
        return Err(ValidationError::NotApplicable {
            message: "contiguous row window requires at least 2 rows",
        });
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the floored product is at most the row count and `as` saturates negatives and NaN to 0; the result is clamped to 1..=n-1"
    )]
    let window_len = (((n as f64) * keep_fraction).floor() as usize).clamp(1, n - 1);
    let span = n - window_len;
    let mut rng = ctx.rng.stream(stream_id);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "next_f64 is in [0, 1), so the floored product is in [0, span]; it is non-negative and below the row count, then min-bounded to span"
    )]
    let start = (rng.next_f64() * (span as f64 + 1.0)).floor() as usize;
    let start = start.min(span);

    let storage = data.storage();
    let mut cols = Vec::with_capacity(storage.columns().len());
    for col in storage.columns() {
        cols.push(slice_float_column_window(col, start, window_len)?);
    }
    let mask =
        storage.analysis_mask().map(|m| slice_validity_window(m, start, window_len)).transpose()?;
    let weights =
        storage.weights().map(|w| Arc::<[f64]>::from(w[start..start + window_len].to_vec()));
    let new_storage = OwnedColumnarStorage::try_new(storage.schema().clone(), cols, mask, weights)
        .map_err(ValidationError::from)?;
    Ok(TabularData::new(new_storage))
}

fn slice_float_column_window(
    col: &OwnedColumn,
    start: usize,
    len: usize,
) -> Result<OwnedColumn, ValidationError> {
    match col {
        OwnedColumn::Float64(c) => {
            let values: Arc<[f64]> = Arc::from(c.values.as_slice()[start..start + len].to_vec());
            let validity = slice_validity_window(&c.validity, start, len)?;
            Ok(OwnedColumn::Float64(
                Float64Column::new(c.id, values, validity).map_err(ValidationError::from)?,
            ))
        }
        _ => Err(ValidationError::NotApplicable {
            message: "contiguous row window requires float64 columns",
        }),
    }
}

fn slice_validity_window(
    src: &ValidityBitmap,
    start: usize,
    len: usize,
) -> Result<ValidityBitmap, ValidationError> {
    let mut bytes = vec![0u8; len.div_ceil(8)];
    for i in 0..len {
        if src.is_valid(start + i) {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    ValidityBitmap::from_bytes(bytes, len).map_err(ValidationError::from)
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
    // One storage rebuild for all k resampled columns (weights/mask copied once), not one
    // per column: at 200 bootstrap replicates the per-column rebuild dominated the loop.
    // `resample_ids` are distinct by construction (treatment, outcome, then covariates), so
    // gathering every column from the source table matches the old sequential replacement.
    let mut replacements: Vec<(VariableId, Arc<[f64]>)> = Vec::with_capacity(resample_ids.len());
    for &id in resample_ids {
        // Gather straight from the borrowed float64 view when the column is natively float;
        // the coercing owned-copy fallback keeps non-float ids surfacing the same
        // `TypeMismatch` from the replace step as before.
        let resampled: Vec<f64> = if let Ok(ColumnView::Float64(c)) = data.column(id) {
            row_idx.iter().map(|&i| c.values[i]).collect()
        } else {
            let full = float64_full(data, id)?;
            row_idx.iter().map(|&i| full[i]).collect()
        };
        replacements.push((id, Arc::from(resampled)));
    }
    let out = data.with_replaced_floats(&replacements).map_err(ValidationError::from)?;
    if keep.iter().all(|&k| k) {
        return Ok(out);
    }
    out.with_analysis_mask(validity_from_flags(keep)?).map_err(ValidationError::from)
}

/// Validity bitmap with bit `i` set iff `keep[i]`.
pub(crate) fn validity_from_flags(keep: &[bool]) -> Result<ValidityBitmap, ValidationError> {
    let n = keep.len();
    let mut bytes = vec![0u8; n.div_ceil(8)];
    for (i, &k) in keep.iter().enumerate() {
        if k {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    ValidityBitmap::from_bytes(bytes, n).map_err(ValidationError::from)
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

/// Column-major `Xβ`: `out[r] = Σ_c matrix[c * n + r] * beta[c]`.
///
/// Each row accumulates in column order, the same operations as summing that
/// row's terms from the first column to the last.
pub(crate) fn column_major_dot(matrix: &[f64], n: usize, beta: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    for (c, &b) in beta.iter().enumerate() {
        let column = &matrix[c * n..(c + 1) * n];
        for (slot, &x) in out.iter_mut().zip(column) {
            *slot += x * b;
        }
    }
}

/// Standard-normal draws via Box–Muller from [`ExecutionContext`] RNG.
///
/// Emits both cos/sin components from each uniform pair (same stream use as
/// historical seeded tests).
pub(crate) fn fill_gaussian(out: &mut [f64], ctx: &ExecutionContext, stream_id: u64) {
    let mut rng = ctx.rng.stream(stream_id);
    antecedent_kernels::fill_standard_normal(&mut rng, out);
}

/// Compile the actual lag-aligned complete cases used by temporal diagnostics.
pub(crate) fn temporal_diagnostic_design(
    problem: &RefutationProblem<'_>,
) -> Result<antecedent_estimate::PreparedEstimationProblem, ValidationError> {
    let temporal = problem.temporal.ok_or(ValidationError::NotApplicable {
        message: "temporal diagnostic requires temporal context",
    })?;
    let estimator = antecedent_estimate::TemporalLinearAdjustment::new();
    let prep = if let Some(panel) = temporal.panel {
        estimator
            .prepare_panel(
                panel,
                problem.estimand,
                temporal.temporal_query,
                temporal.indexer,
                temporal.split,
                temporal.kernel_policy,
            )?
            .0
    } else {
        let time_index = temporal.time_index.ok_or(ValidationError::NotApplicable {
            message: "temporal e-value requires a series time index",
        })?;
        let series = antecedent_data::TimeSeriesData::try_new(
            problem.data.storage().clone(),
            time_index.clone(),
        )?;
        let inputs = antecedent_estimate::TemporalDesignInputs {
            data: &series,
            estimand: problem.estimand,
            query: temporal.temporal_query,
            indexer: temporal.indexer,
            split: temporal.split,
            policy: temporal.kernel_policy,
        };
        if temporal.gaps {
            estimator.prepare_with_gaps_extras(&inputs, &[])?.0
        } else {
            estimator.prepare(
                &series,
                problem.estimand,
                temporal.temporal_query,
                temporal.indexer,
                temporal.split,
                temporal.kernel_policy,
            )?
        }
    };
    Ok(prep)
}
