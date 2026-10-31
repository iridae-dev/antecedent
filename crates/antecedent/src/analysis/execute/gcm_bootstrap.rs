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
//! the bootstrap standard error; it is unlicensed and its calibration is unmeasured.

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
             error{}. It is unlicensed and its calibration is unmeasured: no coverage \
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
        ("calibration", "unmeasured".to_owned()),
    ])
}
