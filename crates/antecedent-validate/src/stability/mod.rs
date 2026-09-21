//! Discovery stability and validation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod env_holdout;
mod false_positive;
mod null_calibration;
mod orientation;
mod pcmci_grid;
mod regime;

use std::sync::Arc;

use antecedent_core::CausalRng;
use antecedent_data::{
    ColumnView, Float64Column, OwnedColumn, OwnedColumnarStorage, ResamplingPlan, TableView,
    TimeIndex, TimeSeriesData, resample_timeseries,
};

use crate::common::validity_from_flags;
use crate::error::ValidationError;

pub use env_holdout::{EnvironmentHoldout, EnvironmentHoldoutReport};
pub use false_positive::{FalsePositiveCheck, FalsePositiveCheckReport, NullTransform};
pub use null_calibration::{NullCalibrationReport, SyntheticNullCalibration};
pub use orientation::{OrientationStability, OrientationStabilityReport, UndirectedLinkStability};
pub use pcmci_grid::{
    AlphaThresholdSensitivity, BlockBootstrapStability, CiTestSensitivity,
    DiscoveryStabilityReport, LagWindowSensitivity, LinkStability,
};
pub use regime::{RegimeStability, RegimeStabilityReport};

/// A moving-block resample of a series whose block junctions are unusable to a lagged analysis.
pub(crate) struct GappedBootstrap {
    /// The resampled series: each block followed by `gap` invalid (missing) rows, except where
    /// the next block happens to start at the following original row.
    pub series: TimeSeriesData,
    /// Original row of every position of [`Self::series`]; a gap row repeats its predecessor's.
    pub source_rows: Vec<usize>,
}

/// Moving-block bootstrap of `data` that keeps every lagged window inside one block.
///
/// Concatenating resampled blocks and rebuilding lags on the result pairs, at every junction, an
/// outcome with regressors from an unrelated block (a fraction `lag / block` of each lag's
/// samples), which biases link frequencies toward independence. Following each junction with
/// `gap` invalid rows makes every lag window that would straddle it incomplete, and a lagged
/// analysis drops incomplete windows. A window spans `gap + 1` rows when `gap` is twice the
/// maximum lag (the depth PCMCI materializes), so no window can contain rows of two blocks
/// without also containing a gap row.
pub(crate) fn gapped_block_bootstrap(
    data: &TimeSeriesData,
    block: usize,
    gap: usize,
    rng: &mut CausalRng,
    scratch: &mut Vec<u32>,
) -> Result<GappedBootstrap, ValidationError> {
    let boot =
        resample_timeseries(data, ResamplingPlan::MovingBlock { length: block }, rng, scratch)
            .map_err(ValidationError::from)?;
    let n = scratch.len();
    // Position of each resampled row in the gapped series (`None` for a gap row).
    let mut layout: Vec<Option<usize>> = Vec::with_capacity(n + gap * (n / block.max(1) + 1));
    let mut source_rows = Vec::with_capacity(layout.capacity());
    for i in 0..n {
        if i > 0 && scratch[i] != scratch[i - 1] + 1 {
            for _ in 0..gap {
                layout.push(None);
                source_rows.push(scratch[i - 1] as usize);
            }
        }
        layout.push(Some(i));
        source_rows.push(scratch[i] as usize);
    }
    let len = layout.len();
    let mut columns = Vec::new();
    for variable in boot.schema().variables() {
        let ColumnView::Float64(src) = boot.column(variable.id).map_err(ValidationError::from)?
        else {
            return Err(ValidationError::NotApplicable {
                message: "block-bootstrap stability requires float64 columns",
            });
        };
        let values: Vec<f64> =
            layout.iter().map(|slot| slot.map_or(0.0, |i| src.values.as_slice()[i])).collect();
        let valid: Vec<bool> =
            layout.iter().map(|slot| slot.is_some_and(|i| src.validity.is_valid(i))).collect();
        columns.push(OwnedColumn::Float64(
            Float64Column::new(variable.id, Arc::from(values), validity_from_flags(&valid)?)
                .map_err(ValidationError::from)?,
        ));
    }
    let storage = boot.storage();
    let mask = storage
        .analysis_mask()
        .map(|m| {
            let flags: Vec<bool> =
                layout.iter().map(|slot| slot.is_some_and(|i| m.is_valid(i))).collect();
            validity_from_flags(&flags)
        })
        .transpose()?;
    let weights = storage.weights().map(|w| {
        Arc::<[f64]>::from(layout.iter().map(|slot| slot.map_or(0.0, |i| w[i])).collect::<Vec<_>>())
    });
    let new_storage = OwnedColumnarStorage::try_new(boot.schema().clone(), columns, mask, weights)
        .map_err(ValidationError::from)?;
    let mut time_index: TimeIndex = boot.time_index().clone();
    time_index.length = len;
    let series = TimeSeriesData::try_new(new_storage, time_index).map_err(ValidationError::from)?;
    Ok(GappedBootstrap { series, source_rows })
}

/// Lagged candidate links a PCMCI-family run scores: every ordered variable pair at each lag of
/// `min_lag..=max_lag`.
///
/// # Errors
///
/// `min_lag == 0` (contemporaneous links are oriented, not counted, per pair, so a per-link rate
/// over `n²` pairs would be wrong) or `max_lag < min_lag`.
pub(crate) fn lagged_link_family(
    n_vars: usize,
    min_lag: u32,
    max_lag: u32,
) -> Result<usize, ValidationError> {
    if min_lag == 0 {
        return Err(ValidationError::NotApplicable {
            message: "per-link false-positive calibration counts lagged links only; \
                      min_lag must be at least 1",
        });
    }
    if max_lag < min_lag {
        return Err(ValidationError::NotApplicable {
            message: "per-link false-positive calibration requires max_lag >= min_lag",
        });
    }
    Ok(n_vars * n_vars * (max_lag - min_lag + 1) as usize)
}

/// Standard error of a per-link false-positive rate `hits / (runs × family)` over `runs`
/// independent simulations.
///
/// The larger of the binomial value `sqrt(α(1−α) / (runs·family))` and the empirical run-to-run
/// spread `sd(hits_per_run) / (family·sqrt(runs))`. Links of one run share a sample, so the
/// hits of a run are more dispersed than binomial whenever the tests are positively dependent;
/// the empirical term makes the band honest about that instead of assuming independence.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn null_rate_se(alpha: f64, family: usize, hits_per_run: &[u64]) -> f64 {
    let runs = hits_per_run.len() as f64;
    let family = family as f64;
    let binomial = (alpha * (1.0 - alpha) / (runs * family)).sqrt();
    if hits_per_run.len() < 2 {
        return binomial;
    }
    let mean = hits_per_run.iter().sum::<u64>() as f64 / runs;
    let variance =
        hits_per_run.iter().map(|&h| (h as f64 - mean).powi(2)).sum::<f64>() / (runs - 1.0);
    binomial.max((variance / runs).sqrt() / family)
}

#[cfg(test)]
mod tests {
    use antecedent_core::ExecutionContext;
    use antecedent_data::{SamplingRegularity, TableView};

    use super::*;
    use crate::test_support::tabular;

    fn series_of_row_indices(n: usize) -> TimeSeriesData {
        let rows: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let data = tabular(&[rows.clone(), rows.iter().map(|v| -v).collect()]);
        TimeSeriesData::try_new(
            data.storage().clone(),
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap()
    }

    #[test]
    fn no_lag_window_straddles_a_block_junction() {
        // With gap = 2 * max_lag, any run of gap + 1 consecutive *valid* rows must be consecutive
        // in the original series: a lag window over them never mixes two blocks.
        let (n, block, max_lag) = (60_usize, 7_usize, 2_u32);
        let gap = 2 * max_lag as usize;
        let mut rng = ExecutionContext::for_tests(3).rng.stream(9);
        let mut scratch = Vec::new();
        let boot =
            gapped_block_bootstrap(&series_of_row_indices(n), block, gap, &mut rng, &mut scratch)
                .unwrap();
        let len = boot.series.row_count();
        assert!(len > n, "a 60-row series in blocks of 7 must have junctions");
        assert_eq!(boot.source_rows.len(), len);
        let ColumnView::Float64(col) =
            boot.series.column(antecedent_core::VariableId::from_raw(0)).unwrap()
        else {
            panic!("float64 column");
        };
        let valid: Vec<bool> = (0..len).map(|i| col.validity.is_valid(i)).collect();
        // Exactly the resampled rows are valid, each carrying its own original row index.
        assert_eq!(valid.iter().filter(|&&v| v).count(), n);
        for i in (0..len).filter(|&i| valid[i]) {
            assert_eq!(col.values.as_slice()[i], boot.source_rows[i] as f64);
        }
        for start in 0..=len - (gap + 1) {
            let window = start..=(start + gap);
            if window.clone().all(|i| valid[i]) {
                assert!(
                    window
                        .clone()
                        .skip(1)
                        .all(|i| boot.source_rows[i] == boot.source_rows[i - 1] + 1),
                    "window {window:?} straddles a junction"
                );
            }
        }
    }

    #[test]
    fn lagged_link_family_counts_ordered_pairs_over_the_lag_range() {
        // 3 variables, lags 1..=2: 3 * 3 * 2 = 18 candidate links.
        assert_eq!(lagged_link_family(3, 1, 2).unwrap(), 18);
        assert_eq!(lagged_link_family(2, 2, 4).unwrap(), 12);
        assert!(lagged_link_family(3, 0, 2).is_err());
        assert!(lagged_link_family(3, 3, 2).is_err());
    }

    #[test]
    fn null_rate_se_uses_the_number_of_link_trials_not_the_number_of_runs() {
        // alpha = 0.05, 40 runs, family 9 (3 variables, one lag): the binomial SE over
        // 360 trials is sqrt(0.05 * 0.95 / 360) = 0.011487; the run-level count of an
        // exactly-binomial run has the same dispersion, so the empirical term does not exceed it.
        let binomial = (0.05_f64 * 0.95 / 360.0).sqrt();
        let hits = vec![0_u64; 40];
        assert!((null_rate_se(0.05, 9, &hits) - binomial).abs() < 1e-15);
    }

    #[test]
    fn null_rate_se_widens_with_run_to_run_overdispersion() {
        // Runs alternate 0 and 9 hits of 9 links (all-or-nothing): sd(hits) = 4.5 * sqrt(40/39),
        // so the empirical SE is sd / (9 * sqrt(40)), far above the binomial value.
        let hits: Vec<u64> = (0..40).map(|i| if i % 2 == 0 { 0 } else { 9 }).collect();
        let mean = 4.5_f64;
        let sd = (hits.iter().map(|&h| (h as f64 - mean).powi(2)).sum::<f64>() / 39.0).sqrt();
        let expected = sd / (9.0 * 40.0_f64.sqrt());
        let se = null_rate_se(0.05, 9, &hits);
        assert!((se - expected).abs() < 1e-12, "{se} vs {expected}");
        assert!(se > (0.05_f64 * 0.95 / 360.0).sqrt());
    }
}
