//! Applying a resampling plan to time-series data.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Apply a resampling plan to produce a new float64 time series.
///
/// Index plans gather rows. [`ResamplingPlan::BayesianBootstrap`] keeps row
/// order and replaces storage weights with a fresh weight plan (multiplied by
/// any existing weights). Clustered plans require [`resample_timeseries_grouped`].
///
/// Serially dependent data must use a block plan: the IID row bootstrap and the
/// Bayesian bootstrap (IID Dirichlet weights) ignore autocorrelation and understate
/// uncertainty, so they are refused here. [`resample_timeseries_with_policy`] is the
/// explicit opt-in.
///
/// # Errors
///
/// Non-float columns, an IID plan on time-series data, or construction failures.
pub fn resample_timeseries(
    data: &TimeSeriesData,
    plan: ResamplingPlan,
    rng: &mut CausalRng,
    index_scratch: &mut Vec<GlobalId>,
) -> Result<TimeSeriesData, DataError> {
    if plan.needs_clusters() {
        return Err(DataError::InvalidArgument {
            message: "clustered plan requires resample_timeseries_grouped".into(),
        });
    }
    resample_timeseries_grouped(data, plan, None, rng, index_scratch)
}

/// Like [`resample_timeseries`], with optional cluster labels for clustered plans.
///
/// # Errors
///
/// Non-float columns, clustered plan without ids, or construction failures.
pub(crate) fn resample_timeseries_grouped(
    data: &TimeSeriesData,
    plan: ResamplingPlan,
    cluster_ids: Option<&[u32]>,
    rng: &mut CausalRng,
    index_scratch: &mut Vec<GlobalId>,
) -> Result<TimeSeriesData, DataError> {
    resample_timeseries_with_policy(
        data,
        plan,
        cluster_ids,
        rng,
        index_scratch,
        TemporalRandomPolicy::Refuse,
    )
}

/// [`resample_timeseries_grouped`] with an explicit [`TemporalRandomPolicy`].
///
/// [`TemporalRandomPolicy::Allow`] admits [`ResamplingPlan::IidBootstrap`] and
/// [`ResamplingPlan::BayesianBootstrap`], whose replicates carry no serial dependence.
///
/// # Errors
///
/// As [`resample_timeseries_grouped`]; an IID plan under [`TemporalRandomPolicy::Refuse`].
pub(crate) fn resample_timeseries_with_policy(
    data: &TimeSeriesData,
    plan: ResamplingPlan,
    cluster_ids: Option<&[u32]>,
    rng: &mut CausalRng,
    index_scratch: &mut Vec<GlobalId>,
    policy: TemporalRandomPolicy,
) -> Result<TimeSeriesData, DataError> {
    if matches!(plan, ResamplingPlan::IidBootstrap | ResamplingPlan::BayesianBootstrap) {
        crate::split::ensure_random_allowed_on_temporal(policy).map_err(|_| {
            DataError::InvalidArgument {
                message: "IID bootstrap refused on time-series data (ignores serial dependence); use a block plan or pass TemporalRandomPolicy::Allow".into(),
            }
        })?;
    }
    let n = data.row_count();
    if plan.is_weight_plan() {
        let mut weights = Vec::new();
        fill_resample_weights(plan, n, rng, &mut weights)?;
        return apply_weight_plan(data, &weights);
    }
    fill_resample_indexes_grouped(plan, n, cluster_ids, rng, index_scratch)?;
    apply_row_map(data, index_scratch)
}

pub(super) fn apply_weight_plan(data: &TimeSeriesData, weights: &[f64]) -> Result<TimeSeriesData, DataError> {
    let n = data.row_count();
    if weights.len() != n {
        return Err(DataError::LengthMismatch {
            expected: n,
            actual: weights.len(),
            context: "BayesianBootstrap weights",
        });
    }
    let schema = data.schema().clone();
    let mut cols = Vec::with_capacity(schema.len());
    for v in schema.variables() {
        // Weighting reweights rows; the columns themselves are shared unchanged, whatever
        // their type.
        cols.push(data.column(v.id)?.share());
    }
    let analysis_mask = data.storage().analysis_mask().cloned();
    let combined: Arc<[f64]> = match data.storage().weights() {
        Some(existing) => {
            Arc::from(existing.iter().zip(weights.iter()).map(|(a, b)| a * b).collect::<Vec<_>>())
        }
        None => Arc::from(weights.to_vec()),
    };
    let storage = OwnedColumnarStorage::try_new(schema, cols, analysis_mask, Some(combined))?;
    TimeSeriesData::try_new(storage, data.time_index().clone())
}

pub(super) fn apply_row_map(data: &TimeSeriesData, row_map: &[GlobalId]) -> Result<TimeSeriesData, DataError> {
    let n = row_map.len();
    let positions = row_map.iter().copied().map(row_pos).collect::<Result<Vec<_>, _>>()?;
    let schema = data.schema().clone();
    let mut cols = Vec::with_capacity(schema.len());
    for v in schema.variables() {
        // Each replicate slot inherits the source row's validity, whatever the column type.
        cols.push(data.column(v.id)?.gather_rows(&positions)?);
    }
    // Analysis mask and weights follow the same row map as the values.
    let analysis_mask =
        data.storage().analysis_mask().map(|m| m.gather_rows(&positions)).transpose()?;
    let weights = data
        .storage()
        .weights()
        .map(|w| Arc::from(positions.iter().map(|&r| w[r]).collect::<Vec<f64>>()));
    let storage = OwnedColumnarStorage::try_new(schema, cols, analysis_mask, weights)?;
    let mut time_index = data.time_index().clone();
    time_index.length = n;
    TimeSeriesData::try_new(storage, time_index)
}
