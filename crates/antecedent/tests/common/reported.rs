//! The interval a study result reports by default, read the way a consumer
//! reads it, plus the same construction at the calibration gate's level.
//!
//! The facade publishes every interval at 0.95 unless the caller asks for
//! another level ([`REPORTED_LEVEL`]; `IntervalBinding::level`,
//! `ContinuousResponseOptions::confidence_level`, the posterior `q025`/`q975`
//! summaries). The `v19` gate measures at 0.90 ([`GATE_LEVEL`]). The `v110`
//! coverage tests score both levels from the same replicates:
//!
//! * a normal interval `est ± z·se` is re-formed at 0.90 from the reported SE
//!   (the 0.95 bounds are checked to be `est ± z_0.975·se` first, so the
//!   0.90 interval is the facade's own construction);
//! * a posterior-quantile interval is re-formed from the effect draws with the
//!   summaries' rounding rule (checked against `q025` / `q975` at 0.95);
//! * a response interval whose quantile rule is internal to the estimator is
//!   re-run at `confidence_level = 0.90` on the same data and seed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code)]

use antecedent::StudyResult;
use antecedent_core::ResponseUncertainty;

use super::calibration::{Z90, normal_interval, quantile_interval};

/// Level every facade interval is published at by default.
///
/// Re-exported from the harness, not restated: `common::calibration` owns the
/// levels and their critical values, and every suite that scores "the reported
/// interval" must mean the same one.
pub use super::calibration::{REPORTED_LEVEL, Z95};

/// Level of the calibration gate.
pub const GATE_LEVEL: f64 = 0.9;

/// Replicate count of a test whose tallies are strongly dependent, floored at
/// `floor` unless the caller set `ANTECEDENT_CALIBRATION_NSIM` explicitly.
///
/// The coordinates of a pointwise band are nearly the same event: one
/// replicate's fit moves every coordinate together, so a band test applies the
/// `level ± 3·MCSE` acceptance band to ten strongly dependent tallies at once
/// and a high or low block of 400 seeds trips several of them together. Such a
/// test measures at [`super::calibration::PRECISION_N_SIM`] or more instead,
/// where the harness also enforces the one-sided precision floor. That is a
/// tighter gate than the default count, never a widened tolerance.
#[must_use]
pub fn n_sim_at_least(floor: u32) -> u32 {
    if std::env::var_os("ANTECEDENT_CALIBRATION_NSIM").is_some() {
        super::calibration::n_sim()
    } else {
        super::calibration::n_sim().max(floor)
    }
}

/// Two-sided normal critical value for the two supported levels.
#[must_use]
pub fn z_for(level: f64) -> f64 {
    if (level - REPORTED_LEVEL).abs() < 1e-12 {
        Z95
    } else if (level - GATE_LEVEL).abs() < 1e-12 {
        Z90
    } else {
        panic!("no critical value tabulated for level {level}")
    }
}

/// `est ± z(level)·se`, or `None` when either is not finite / `se <= 0`.
///
/// The level-taking face of [`super::calibration::normal_interval`], which is
/// the one body that forms a normal interval in these suites.
#[must_use]
pub fn normal_at(est: f64, se: f64, level: f64) -> Option<(f64, f64)> {
    normal_interval(est, Some(se), z_for(level))
}

/// SE behind a scalar Frequentist result's reported interval: the bootstrap
/// SE when bootstrap replicates succeeded, otherwise the analytic SE. This is
/// the rule `execute_helpers` uses to choose the interval method.
#[must_use]
pub fn scalar_reported_se(result: &StudyResult) -> (f64, &'static str) {
    match result.estimate.se_bootstrap.filter(|s| s.is_finite() && *s > 0.0) {
        Some(se) if result.estimate.bootstrap_replicates_ok.unwrap_or(0) > 0 => {
            (se, "bootstrap_se")
        }
        _ => (result.estimate.se_analytic, "analytic_se"),
    }
}

/// Reported (0.95) and gate-level (0.90) normal intervals of a scalar result.
#[must_use]
pub fn scalar_normal_pair(result: &StudyResult) -> [Option<(f64, f64)>; 2] {
    let (se, _) = scalar_reported_se(result);
    [
        normal_at(result.estimate.ate, se, REPORTED_LEVEL),
        normal_at(result.estimate.ate, se, GATE_LEVEL),
    ]
}

/// Reported posterior interval of column `col` (`q025`, `q975`) and the same
/// equal-tailed rule at the gate level from the draws.
///
/// # Panics
///
/// When the draw-level 0.95 rule does not reproduce the published summaries,
/// i.e. the 0.90 interval would not be the facade's construction.
#[must_use]
pub fn posterior_pair(result: &StudyResult, col: usize) -> [Option<(f64, f64)>; 2] {
    let Some(posterior) = result.posterior.as_ref() else {
        return [None, None];
    };
    let reported = (posterior.summaries.q025[col], posterior.summaries.q975[col]);
    let Ok(draws) = posterior.draws.column(col) else {
        return [Some(reported), None];
    };
    let rebuilt = quantile_interval(draws, REPORTED_LEVEL);
    if let Some((lo, hi)) = rebuilt {
        assert!(
            (lo - reported.0).abs() <= 1e-12 * lo.abs().max(1.0)
                && (hi - reported.1).abs() <= 1e-12 * hi.abs().max(1.0),
            "posterior summaries [{}, {}] are not the equal-tailed draw quantiles [{lo}, {hi}]",
            reported.0,
            reported.1
        );
    }
    [Some(reported), quantile_interval(draws, GATE_LEVEL)]
}

/// Scalar response interval `(lower, upper)` with its level and SE.
#[must_use]
pub fn response_scalar(result: &StudyResult) -> Option<(f64, f64, f64, f64)> {
    match result.response.as_ref()?.uncertainty {
        ResponseUncertainty::Scalar { standard_error, level, lower, upper } => {
            Some((lower, upper, level, standard_error))
        }
        _ => None,
    }
}

/// Pointwise band `(lower, upper, level)` of a response surface.
#[must_use]
pub fn response_band(result: &StudyResult) -> Option<(Vec<f64>, Vec<f64>, f64)> {
    match &result.response.as_ref()?.uncertainty {
        ResponseUncertainty::PointwiseBand { level, lower, upper } => {
            Some((lower.to_vec(), upper.to_vec(), *level))
        }
        _ => None,
    }
}

/// Reported and gate-level intervals of a Frequentist scalar response whose
/// interval is `value ± z·se` (checked).
///
/// # Panics
///
/// When the reported interval is not at [`REPORTED_LEVEL`] or not symmetric
/// `± z_0.975·se` around its midpoint.
#[must_use]
pub fn response_normal_pair(result: &StudyResult) -> [Option<(f64, f64)>; 2] {
    let Some((lower, upper, level, se)) = response_scalar(result) else {
        return [None, None];
    };
    assert!((level - REPORTED_LEVEL).abs() < 1e-12, "response interval level {level}");
    let mid = 0.5 * (lower + upper);
    assert!(
        ((upper - lower) - 2.0 * Z95 * se).abs() <= 1e-9 * (1.0 + se),
        "response interval [{lower}, {upper}] is not ± z·se with se={se}"
    );
    [Some((lower, upper)), normal_at(mid, se, GATE_LEVEL)]
}

/// Record `[reported, gate]` intervals into the matching pair of tallies.
pub fn record_pair(
    tallies: &mut [super::calibration::CoverageTally; 2],
    intervals: [Option<(f64, f64)>; 2],
    truth: f64,
) {
    for (tally, interval) in tallies.iter_mut().zip(intervals) {
        tally.record(interval, truth);
    }
}

/// Skip a replicate in both tallies.
pub fn skip_pair(tallies: &mut [super::calibration::CoverageTally; 2]) {
    for tally in tallies.iter_mut() {
        tally.skip();
    }
}

/// Gate every tally, printing every line before failing on any of them.
/// `measured[i] = Some([m0, m1, m2])` asserts tally `i` as a named boundary cell
/// against its measured coverage at each sample-size grid point; `None` asserts it
/// at nominal.
///
/// # Panics
///
/// When any tally fails its gate.
pub fn gate(
    tallies: &[super::calibration::CoverageTally],
    measured: &[Option<[f64; super::calibration::GRID_POINTS]>],
) {
    assert_eq!(tallies.len(), measured.len(), "one gate entry per tally");
    let failures: Vec<String> = tallies
        .iter()
        .zip(measured)
        .filter_map(|(tally, measured)| {
            std::panic::catch_unwind(|| match measured {
                Some(m) => tally.assert_boundary_at(m.map(Some)),
                None => tally.assert(),
            })
            .err()
            .map(|e| {
                e.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_else(|| "coverage failure".into())
            })
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

/// Gate every tally per sample-size grid point.
///
/// `measured[i] = [m0, m1, m2]`: `Some(m)` at a point names that tally a
/// boundary held to `m`; `None` gates it at nominal. A record is a boundary
/// over its whole range when any point is.
///
/// # Panics
///
/// When any tally fails its gate.
pub fn gate_at(
    tallies: &[super::calibration::CoverageTally],
    measured: &[[Option<f64>; super::calibration::GRID_POINTS]],
) {
    assert_eq!(tallies.len(), measured.len(), "one gate entry per tally");
    let failures: Vec<String> = tallies
        .iter()
        .zip(measured)
        .filter_map(|(tally, measured)| {
            std::panic::catch_unwind(|| tally.assert_boundary_at(*measured)).err().map(|e| {
                e.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_else(|| "coverage failure".into())
            })
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("; "));
}
