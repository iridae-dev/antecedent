//! CI calibration helpers (null / alternative recovery rates).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use causal_core::ExecutionContext;

use crate::ci::types::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependence, SignificanceMethod,
};
use crate::error::StatsError;

/// Summary of a calibration sweep at a fixed significance level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationReport {
    /// Monte Carlo trials under the null.
    pub null_trials: u32,
    /// Trials with `p < alpha` under the null (Type I).
    pub null_rejections: u32,
    /// Monte Carlo trials under the alternative.
    pub alt_trials: u32,
    /// Trials with `p < alpha` under the alternative (power).
    pub alt_rejections: u32,
    /// Nominal alpha.
    pub alpha: f64,
}

impl CalibrationReport {
    /// Empirical Type I error rate.
    #[must_use]
    pub fn type_i_rate(self) -> f64 {
        if self.null_trials == 0 {
            return 0.0;
        }
        f64::from(self.null_rejections) / f64::from(self.null_trials)
    }

    /// Empirical power.
    #[must_use]
    pub fn power(self) -> f64 {
        if self.alt_trials == 0 {
            return 0.0;
        }
        f64::from(self.alt_rejections) / f64::from(self.alt_trials)
    }
}

/// Run ParCorr-style calibration: independent Gaussian null + linear alternative.
///
/// # Errors
///
/// Propagates CI failures.
pub fn calibrate_parcorr_like(
    ci: &dyn ConditionalIndependence,
    n: usize,
    trials: u32,
    alpha: f64,
    seed: u64,
) -> Result<CalibrationReport, StatsError> {
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::for_tests(seed);
    let mut null_rej = 0u32;
    let mut alt_rej = 0u32;
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];

    for t in 0..trials {
        let mut rng = ctx.rng.stream(0xCA11_u64.wrapping_add(u64::from(t)));
        let x: Vec<f64> = (0..n)
            .map(|_| {
                let u = (rng.next_u64() as f64) / (u64::MAX as f64);
                let v = (rng.next_u64() as f64) / (u64::MAX as f64);
                (-2.0 * u.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
            })
            .collect();
        let y_null: Vec<f64> = (0..n)
            .map(|_| {
                let u = (rng.next_u64() as f64) / (u64::MAX as f64);
                let v = (rng.next_u64() as f64) / (u64::MAX as f64);
                (-2.0 * u.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
            })
            .collect();
        let cols_null: [&[f64]; 2] = [&x, &y_null];
        let req = CiBatchRequest {
            columns: &cols_null,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
        };
        let out = ci.test_batch(&req, &mut ws, &ctx)?;
        if out.results[0].p_value < alpha {
            null_rej += 1;
        }

        let y_alt: Vec<f64> = x
            .iter()
            .map(|&xi| {
                let u = (rng.next_u64() as f64) / (u64::MAX as f64);
                let v = (rng.next_u64() as f64) / (u64::MAX as f64);
                let e = (-2.0 * u.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos();
                0.7 * xi + 0.3 * e
            })
            .collect();
        let cols_alt: [&[f64]; 2] = [&x, &y_alt];
        let req_alt = CiBatchRequest {
            columns: &cols_alt,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
        };
        let out_alt = ci.test_batch(&req_alt, &mut ws, &ctx)?;
        if out_alt.results[0].p_value < alpha {
            alt_rej += 1;
        }
    }

    Ok(CalibrationReport {
        null_trials: trials,
        null_rejections: null_rej,
        alt_trials: trials,
        alt_rejections: alt_rej,
        alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci::PartialCorrelation;

    #[test]
    fn parcorr_calibration_type_i_near_alpha_and_power() {
        let report = calibrate_parcorr_like(&PartialCorrelation::new(), 200, 80, 0.05, 7).unwrap();
        // Loose bounds: Type I not wildly inflated; power clearly above alpha.
        assert!(
            report.type_i_rate() < 0.20,
            "type I too high: {}",
            report.type_i_rate()
        );
        assert!(
            report.power() > 0.50,
            "power too low: {}",
            report.power()
        );
    }
}
