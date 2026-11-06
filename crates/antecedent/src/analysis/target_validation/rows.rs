//! Temporal row selection and series reconstruction for refits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_data::{TableView, ValidityBitmap};

use super::{
    Arc, CausalError, CausalRng, ExecutionContext, OwnedColumn, TabularData, TargetRefitError,
    TimeSeriesData, VariableId, data_failed, failed, unbiased_index, unsupported,
};

pub(super) fn mask_from(keep: &[bool]) -> Result<ValidityBitmap, TargetRefitError> {
    let mut bytes = vec![0_u8; keep.len().div_ceil(8)];
    for (i, &k) in keep.iter().enumerate() {
        if k {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    ValidityBitmap::from_bytes(bytes, keep.len()).map_err(data_failed)
}

/// `values` moved by `shift` periods (`out[t] = values[t + shift]`) and the rows that exist.
pub(super) fn shifted(values: &[f64], shift: isize) -> (Arc<[f64]>, Vec<bool>) {
    let n = values.len();
    let mut out = vec![0.0; n];
    let mut keep = vec![false; n];
    for t in 0..n {
        let Some(source) = t.checked_add_signed(shift).filter(|s| *s < n) else {
            continue;
        };
        out[t] = values[source];
        keep[t] = true;
    }
    (out.into(), keep)
}

/// Row weights that drop the design rows ending outside `[from, to)`: a shifted column has
/// no value there, and a zero weight removes the window without opening a gap in the series.
pub(super) fn window_weights(n: usize, from: usize, to: usize) -> Arc<[f64]> {
    (0..n).map(|t| if t >= from && t < to { 1.0 } else { 0.0 }).collect()
}

pub(super) fn times(a: &Arc<[f64]>, b: Option<Arc<[f64]>>) -> Arc<[f64]> {
    match b {
        Some(b) => a.iter().zip(b.iter()).map(|(x, y)| x * y).collect(),
        None => Arc::clone(a),
    }
}

/// Circular-block bootstrap of a series as row multiplicities: the design row ending at each
/// period keeps its own lag window, so no lag is read across a block join.
pub(super) fn block_weights(
    n: usize,
    span: usize,
    rng: &mut CausalRng,
) -> Result<Arc<[f64]>, TargetRefitError> {
    let length = antecedent_data::circular_block_length(span, n);
    let mut index = vec![0_usize; n];
    antecedent_data::fill_circular_block_indexes(n, length, rng, &mut index)
        .map_err(data_failed)?;
    let mut weights = vec![0.0; n];
    for i in index {
        weights[i] += 1.0;
    }
    Ok(weights.into())
}

/// Fraction of a window a contiguous subsample keeps. Windows this short vary about as much
/// as the full-window estimate would across samples (half-length windows vary too little and
/// reject a clean series on a quarter to a third of seeds; see the lead and control sweep in
/// `temporal_conditional_validation_truth`), so a null check against them is not over-eager.
const SUBSAMPLE_FRACTION: f64 = 1.0 / 3.0;

/// Start and width of a contiguous sub-window of `rows` periods for a route whose lag windows
/// may not cross a join: `SUBSAMPLE_FRACTION` of the periods, at least `span + 2` of them (or
/// all when fewer exist), starting uniformly among the positions that fit.
pub(super) fn subsample_bounds(rows: usize, span: usize, rng: &mut CausalRng) -> (usize, usize) {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the fraction lies in (0, 1), and the saturating cast is clamped into [1, rows]"
    )]
    let kept = (SUBSAMPLE_FRACTION * rows as f64).round() as usize;
    let width = kept.max(span + 2).min(rows);
    (unbiased_index(rng, rows - width + 1), width)
}

/// Eligible windows use observed consecutive source ticks and do not cross a
/// draw boundary, even if two blocks happen to have adjacent tick values.
pub(super) struct JoinFreeRows {
    values: Vec<bool>,
    _lease: antecedent_core::execution::AdmissionLease,
}
impl std::ops::Deref for JoinFreeRows {
    type Target = [bool];
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

pub(super) fn join_free_rows(
    map: &[usize],
    span: usize,
    block_length: Option<usize>,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<JoinFreeRows, TargetRefitError> {
    if ctx.cancellation.is_cancelled() {
        return Err(failed(&CausalError::Cancelled { stage: "temporal attribution" }));
    }
    if block_length == Some(0) {
        return Err(failed(&CausalError::Compile {
            message: "refit block length must be positive".into(),
        }));
    }
    let n = u64::try_from(map.len()).map_err(|_| {
        failed(&CausalError::Resource { message: "refit mask work overflow".into() })
    })?;
    budget.charge_work(n).map_err(|e| failed(&CausalError::Resource { message: e.to_string() }))?;
    let bytes = n.checked_mul(20).and_then(|v| v.checked_add(1024)).ok_or_else(|| {
        failed(&CausalError::Resource { message: "refit mask bytes overflow".into() })
    })?;
    if bytes > budget.limits().max_bytes {
        return Err(failed(&CausalError::Resource { message: "refit mask byte limit".into() }));
    }
    let lease = ctx.acquire_admission(0, bytes).map_err(|e| {
        failed(&match e {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "temporal attribution" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })
    })?;
    let mut values = Vec::new();
    values.try_reserve_exact(map.len()).map_err(|_| {
        failed(&CausalError::Resource { message: "refit mask allocation refused".into() })
    })?;
    let mut length = 0_usize;
    for (row, &tick) in map.iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(failed(&CausalError::Cancelled { stage: "temporal attribution" }));
        }
        let continues = row > 0
            && block_length.is_none_or(|block| row % block != 0)
            && map[row - 1].checked_add(1) == Some(tick);
        length = if continues { length.saturating_add(1) } else { 1 };
        values.push(length >= span.max(1));
    }
    Ok(JoinFreeRows { values, _lease: lease })
}

/// The analysis mask of a gathered series for a route that fits row-wise lag-aligned tables
/// (an attribution): the gathered original mask with every row whose lag window crosses a join
/// hidden, because such a route reads no row weights and would otherwise fit and score the
/// artificial lag pairs of the join.
pub(super) fn join_free_mask(
    storage: &antecedent_data::OwnedColumnarStorage,
    map: &[usize],
    span: usize,
    block_length: Option<usize>,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<ValidityBitmap, TargetRefitError> {
    let free = join_free_rows(map, span, block_length, budget, ctx)?;
    let n = u64::try_from(map.len()).map_err(|_| {
        failed(&CausalError::Resource { message: "refit mask work overflow".into() })
    })?;
    budget.charge_work(n).map_err(|e| failed(&CausalError::Resource { message: e.to_string() }))?;
    let mut keep = Vec::new();
    keep.try_reserve_exact(map.len()).map_err(|_| {
        failed(&CausalError::Resource { message: "refit mask allocation refused".into() })
    })?;
    for (&eligible, &source) in free.iter().zip(map) {
        if ctx.cancellation.is_cancelled() {
            return Err(failed(&CausalError::Cancelled { stage: "temporal attribution" }));
        }
        keep.push(eligible && storage.analysis_mask().is_none_or(|mask| mask.is_valid(source)));
    }
    mask_from(&keep)
}

pub(super) fn series_with(
    series: &TimeSeriesData,
    replacements: &[(VariableId, Arc<[f64]>)],
    keep: Option<&[bool]>,
    weights: Option<Arc<[f64]>>,
) -> Result<TimeSeriesData, TargetRefitError> {
    let mut out = series.clone();
    if !replacements.is_empty() {
        let table = TabularData::new(series.storage().clone())
            .with_replaced_floats(replacements)
            .map_err(data_failed)?;
        out = TimeSeriesData::try_new(table.storage().clone(), series.time_index().clone())
            .map_err(data_failed)?;
    }
    if let Some(keep) = keep {
        out = out.with_analysis_mask(mask_from(keep)?).map_err(data_failed)?;
    }
    if let Some(weights) = weights {
        let combined: Arc<[f64]> = match out.storage().weights() {
            Some(existing) => existing.iter().zip(weights.iter()).map(|(a, b)| a * b).collect(),
            None => weights,
        };
        out = out.with_weights(combined).map_err(data_failed)?;
    }
    Ok(out)
}

/// The periods `[start, end)` of a series as a complete, shorter series: every column is cut
/// to the same window, so the series stays lag-aligned and no analysis mask hides a row
/// (lagged temporal routes refuse an incomplete series). `None` when a column is not float64.
pub(super) fn truncated_series(
    series: &TimeSeriesData,
    start: usize,
    end: usize,
) -> Result<Option<TimeSeriesData>, TargetRefitError> {
    let storage = series.storage();
    let n = series.row_count();
    if !(start < end && end <= n) {
        return Err(unsupported("the window holds no period", "a longer series"));
    }
    let rows: Vec<usize> = (start..end).collect();
    let mut columns = Vec::with_capacity(storage.columns().len());
    for column in storage.columns() {
        let OwnedColumn::Float64(column) = column else {
            return Ok(None);
        };
        let values: Arc<[f64]> = Arc::from(&column.values[start..end]);
        let validity = column.validity.gather_rows(&rows).map_err(data_failed)?;
        columns.push(OwnedColumn::Float64(
            antecedent_data::Float64Column::new(column.id, values, validity)
                .map_err(data_failed)?,
        ));
    }
    let mask =
        storage.analysis_mask().map(|m| m.gather_rows(&rows)).transpose().map_err(data_failed)?;
    let weights = storage.weights().map(|w| Arc::<[f64]>::from(&w[start..end]));
    let storage = antecedent_data::OwnedColumnarStorage::try_new(
        storage.schema().clone(),
        columns,
        mask,
        weights,
    )
    .map_err(data_failed)?;
    let index = antecedent_data::TimeIndex {
        regularity: series.time_index().regularity.clone(),
        length: end - start,
    };
    TimeSeriesData::try_new(storage, index).map(Some).map_err(data_failed)
}
