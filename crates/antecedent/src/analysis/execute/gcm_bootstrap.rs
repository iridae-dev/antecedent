// SPDX-License-Identifier: MIT OR Apache-2.0
//! Frequentist whole-procedure row bootstrap of the fitted-mechanism (GCM) routes:
//! the static counterfactual, anomaly attribution and change attribution.
//!
//! Each replicate draws multinomial row counts under the declared sampling design
//! (independent subjects: a row is a subject, so a row draw is a subject draw),
//! refits every mechanism under those frequency weights on the original table, and
//! recomputes the published quantity. Mechanism families are the ones selected on the
//! original data, the same conditioning as the Bayesian-bootstrap cells. Counts come
//! from one semantic stream per replicate, so the result does not depend on the thread
//! count. A replicate whose refit or evaluation fails is kept as failed mass; it is
//! never redrawn or dropped from the accounting. The interval is the normal interval of
//! the bootstrap standard error; it is unsupported and its calibration is `not_measured`.

use super::{CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext};
use antecedent_estimate::BootstrapAccounting;

/// A replicate set publishes a standard error only when at least this share of the
/// requested replicates succeeded (and at least two did).
const MIN_SUCCESS_SHARE: f64 = 0.5;

/// Refuse a row bootstrap the declared sampling design does not allow.
///
/// With no declaration a static table is independent subjects, one row each.
pub(super) fn require_subject_resampling(
    design: Option<antecedent_core::DeclaredSamplingDesign>,
) -> Result<(), CausalError> {
    let design =
        design.unwrap_or(antecedent_core::DeclaredSamplingDesign::IndependentSubjectTrajectories);
    design.apply_resampling(antecedent_core::ResamplingRule::RowSubjectIid).map(|_| ()).map_err(
        |_| {
            crate::unsupported_reason!(
                "option_not_applicable",
                "the mechanism-refit bootstrap resamples independent subjects; a clustered \
                 or dependent-series design needs a whole-cluster or block resample. Next \
                 attempt: declare independent subjects, or omit bootstrap_replicates for the \
                 point value"
            )
        },
    )
}

/// Refuse a contiguous-block resample the declared sampling design does not allow.
///
/// With no declaration a lag-aligned unfolding of one series is a dependent series.
pub(super) fn require_series_resampling(
    design: Option<antecedent_core::DeclaredSamplingDesign>,
) -> Result<(), CausalError> {
    let design = design.unwrap_or(antecedent_core::DeclaredSamplingDesign::DependentTimeSeries);
    design.apply_resampling(antecedent_core::ResamplingRule::ContiguousBlock).map(|_| ()).map_err(
        |_| {
            crate::unsupported_reason!(
                "option_not_applicable",
                "the lag-aligned mechanism-refit bootstrap resamples contiguous blocks of one \
                 dependent series; an independent-subject or clustered design needs a \
                 subject or whole-cluster resample of separate series. Next attempt: declare \
                 a dependent series, or omit bootstrap_replicates for the point value"
            )
        },
    )
}

/// Live `f64` cells of a whole-table row bootstrap over `columns` float columns of `n` rows:
/// the columns read once up front, the resampled copy of every column of the replicate in
/// flight (the resampled table shares those buffers), and the replicate's row-index vector.
/// Pass the result as the `rows` of [`admit_replicates`].
pub(super) fn resampled_table_cells(n: usize, columns: usize) -> Option<usize> {
    columns.checked_mul(2)?.checked_add(1)?.checked_mul(n)
}

/// Circular-block row indices in draw order (`block == 1` draws independent rows).
pub(super) fn resample_rows(
    n: usize,
    block: usize,
    rng: &mut antecedent_core::CausalRng,
) -> Vec<usize> {
    let block = block.clamp(1, n.max(1));
    let blocks = n.div_ceil(block);
    let mut rows = Vec::with_capacity(blocks * block);
    for _ in 0..blocks {
        let start = antecedent_kernels::unbiased_index(rng, n);
        rows.extend((0..block).map(|offset| (start + offset) % n));
    }
    rows.truncate(n);
    rows
}

/// Nearest-order 95% endpoints of a sorted, nonempty sample.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the two fixed probabilities put the rounded rank inside the sample"
)]
pub(super) fn nearest_95_range(sorted: &[f64]) -> (f64, f64) {
    let quantile = |p: f64| {
        let index = (p * (sorted.len() - 1) as f64).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    };
    (quantile(0.025), quantile(0.975))
}

/// Lease `replicates * columns` replicate values plus one count vector of `rows` on
/// the execution pool before anything is allocated.
pub(super) fn admit_replicates(
    ctx: &ExecutionContext,
    replicates: u32,
    columns: usize,
    rows: usize,
) -> Result<antecedent_core::execution::AdmissionLease, CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::super::stage::STAGE_UNCERTAINTY });
    }
    let cells = usize::try_from(replicates)
        .ok()
        .and_then(|replicates| replicates.checked_mul(columns))
        .and_then(|cells| cells.checked_add(rows))
        .and_then(|cells| cells.checked_mul(size_of::<f64>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| CausalError::Resource {
            message: "mechanism-refit bootstrap size overflow".to_owned(),
        })?;
    ctx.acquire_admission(0, cells).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: super::super::stage::STAGE_UNCERTAINTY }
        }
        other => CausalError::Resource {
            message: format!(
                "the requested {replicates} bootstrap replicates are refused: {other}; lower \
                 the replicate count or raise the execution budget"
            ),
        },
    })
}

/// The semantic stream of replicate `replicate` of the route keyed by `route_key`.
pub(super) fn replicate_stream(
    ctx: &ExecutionContext,
    route_key: u64,
    replicate: u32,
) -> antecedent_core::CausalRng {
    ctx.rng.stream_for(
        antecedent_core::StreamDomain::Attribution,
        route_key.wrapping_add(u64::from(replicate).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
    )
}

/// Multinomial row counts: `n` draws with replacement over `n` rows.
pub(super) fn bootstrap_counts(rng: &mut antecedent_core::CausalRng, n: usize) -> Vec<f64> {
    let mut counts = vec![0.0_f64; n];
    if n == 0 {
        return counts;
    }
    for _ in 0..n {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a uniform in [0, 1) times n floors to a row index below n"
        )]
        let row = ((rng.next_f64() * n as f64) as usize).min(n - 1);
        counts[row] += 1.0;
    }
    counts
}

/// Replicate accounting of one bootstrapped quantity: its bootstrap standard error
/// (sample standard deviation of the successful replicates) and the counts.
pub(super) fn accounting(values: &[f64], failed: u32, requested: u32) -> BootstrapAccounting {
    let finite: Vec<f64> = values.iter().copied().filter(|value| value.is_finite()).collect();
    let ok = u32::try_from(finite.len()).unwrap_or(u32::MAX);
    let failed =
        failed.saturating_add(u32::try_from(values.len() - finite.len()).unwrap_or(u32::MAX));
    let enough = ok >= 2 && f64::from(ok) >= MIN_SUCCESS_SHARE * f64::from(requested);
    let se = enough.then(|| {
        let n = finite.len() as f64;
        let mean = finite.iter().sum::<f64>() / n;
        (finite.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    });
    BootstrapAccounting {
        se,
        replicates_ok: Some(ok),
        replicates_failed: Some(failed),
        cancelled: false,
        early_stopped: false,
    }
}

/// Disclosure of the Frequentist mechanism-refit bootstrap interval.
pub(super) fn bootstrap_diagnostic(target: &str, accounting: &BootstrapAccounting) -> Diagnostic {
    bootstrap_diagnostic_over(
        target,
        "whole-procedure multinomial row bootstrap over independent subjects",
        accounting,
    )
}

pub(super) fn bootstrap_diagnostic_over(
    target: &str,
    resample: &str,
    accounting: &BootstrapAccounting,
) -> Diagnostic {
    let ok = accounting.replicates_ok.unwrap_or(0);
    let failed = accounting.replicates_failed.unwrap_or(0);
    Diagnostic::new(
        "gcm.frequentist_bootstrap",
        DiagnosticKind::Scientific,
        if accounting.se.is_some() {
            DiagnosticSeverity::Info
        } else {
            DiagnosticSeverity::Warning
        },
        format!(
            "{target}: {resample}; each replicate refits every mechanism under the replicate's row counts \
             (conditional on the mechanism families selected on the original data, as the \
             Bayesian-bootstrap cell is) and recomputes the quantity on the original units. \
             replicates_ok={ok}, replicates_failed={failed} (failed replicates are retained \
             as failed mass). The interval is estimate ± z·SE from the bootstrap standard \
             error{}. It is unsupported and its calibration is not_measured: no coverage \
             record names this construction.",
            if accounting.se.is_some() {
                ""
            } else {
                "; too few replicates succeeded, so no standard error is published"
            }
        ),
    )
    .with_fields([
        ("replicates_ok", ok.to_string()),
        ("replicates_failed", failed.to_string()),
        ("calibration", "not_measured".to_owned()),
    ])
}

#[cfg(test)]
mod tests {
    use super::{nearest_95_range, resample_rows, resampled_table_cells};

    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the fixed probabilities put the rounded rank inside each nonempty sample"
    )]
    fn nearest_range_preserves_legacy_ranks_and_value_bits() {
        for mut values in [
            vec![2.0, 1.0],
            vec![0.0, -0.0],
            (0..21).map(f64::from).collect(),
            vec![f64::NAN, 0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY],
        ] {
            values.sort_by(f64::total_cmp);
            let before: Vec<_> = values.iter().map(|value| value.to_bits()).collect();
            let legacy = |p: f64| {
                let index = (p * (values.len() - 1) as f64).round() as usize;
                values[index.min(values.len() - 1)].to_bits()
            };
            let (lower, upper) = nearest_95_range(&values);
            assert_eq!((lower.to_bits(), upper.to_bits()), (legacy(0.025), legacy(0.975)));
            assert_eq!(values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(), before);
        }
    }

    #[test]
    fn circular_resampling_preserves_rows_and_stream_position() {
        for (n, block) in [(0, 0), (7, 1), (7, 3), (7, 20)] {
            let mut actual_rng = antecedent_core::CausalRng::from_seed(91);
            let mut reference_rng = antecedent_core::CausalRng::from_seed(91);
            let actual = resample_rows(n, block, &mut actual_rng);
            let width = block.clamp(1, n.max(1));
            let starts: Vec<_> = (0..n.div_ceil(width))
                .map(|_| antecedent_kernels::unbiased_index(&mut reference_rng, n))
                .collect();
            let expected: Vec<_> = starts
                .iter()
                .flat_map(|start| (0..width).map(move |offset| (start + offset) % n))
                .take(n)
                .collect();
            assert_eq!(actual, expected);
            assert_eq!(actual_rng.next_u64(), reference_rng.next_u64());
        }
    }

    #[test]
    fn a_table_bootstrap_leases_the_source_columns_and_the_replicate_copy() {
        // 100 rows, 4 columns: 400 source cells, 400 resampled cells, 100 row indices.
        assert_eq!(resampled_table_cells(100, 4), Some(900));
        // Strictly more than the source table plus the index vector alone (the old lease).
        assert!(resampled_table_cells(100, 4).unwrap() > 100 * (4 + 1));
        assert_eq!(resampled_table_cells(0, 4), Some(0));
        assert_eq!(resampled_table_cells(usize::MAX, 2), None);
        assert_eq!(resampled_table_cells(2, usize::MAX), None);
    }
}
