//! CI calibration helpers (null / alternative recovery rates).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use causal_core::ExecutionContext;

use crate::ci::types::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependence, ConditionalIndependenceTest,
    ConfidenceMethod, SignificanceMethod,
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
            confidence: ConfidenceMethod::default(),
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
            confidence: ConfidenceMethod::default(),
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

/// Discrete G² calibration: independent categorical null + dependent alternative.
///
/// # Errors
///
/// Propagates CI failures.
pub fn calibrate_gsquared(
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
    let levels = 3i32;

    for t in 0..trials {
        let mut rng = ctx.rng.stream(0x65_u64.wrapping_add(u64::from(t)));
        let x: Vec<f64> = (0..n).map(|_| (rng.next_u64() % levels as u64) as f64).collect();
        let y_null: Vec<f64> = (0..n).map(|_| (rng.next_u64() % levels as u64) as f64).collect();
        let cols_null: [&[f64]; 2] = [&x, &y_null];
        let req = CiBatchRequest {
            columns: &cols_null,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let out = ci.test_batch(&req, &mut ws, &ctx)?;
        if out.results[0].p_value < alpha {
            null_rej += 1;
        }

        let y_alt: Vec<f64> =
            x.iter()
                .map(|&xi| {
                    if rng.next_u64() % 5 == 0 {
                        (rng.next_u64() % levels as u64) as f64
                    } else {
                        xi
                    }
                })
                .collect();
        let cols_alt: [&[f64]; 2] = [&x, &y_alt];
        let req_alt = CiBatchRequest {
            columns: &cols_alt,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
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

/// Within ~2 SE of α under a Bernoulli(α) null with `trials` Monte Carlo draws.
#[must_use]
pub fn type_i_within_two_se(rate: f64, alpha: f64, trials: u32) -> bool {
    let n = f64::from(trials);
    let se = (alpha * (1.0 - alpha) / n).sqrt();
    (rate - alpha).abs() <= 2.0 * se + 1e-12
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci::{
        GSquared, Gpdc, KnnCmi, MixedKnnCmi, MultivariatePartialCorrelation, PartialCorrelation,
        RegressionCi, RobustPartialCorrelation, SymbolicCmi, WeightedPartialCorrelation,
    };

    #[test]
    fn parcorr_calibration_type_i_near_alpha_and_power() {
        let trials = 200u32;
        let alpha = 0.05;
        let report =
            calibrate_parcorr_like(&PartialCorrelation::new(), 250, trials, alpha, 7).unwrap();
        assert!(
            type_i_within_two_se(report.type_i_rate(), alpha, trials)
                || report.type_i_rate() < 0.12,
            "type I off nominal: {} (2SE band around {})",
            report.type_i_rate(),
            alpha
        );
        assert!(report.power() > 0.50, "power too low: {}", report.power());
    }

    #[test]
    fn gsquared_calibration_type_i_and_power() {
        let report = calibrate_gsquared(&GSquared::new(), 300, 120, 0.05, 11).unwrap();
        assert!(report.type_i_rate() < 0.15, "G² type I too high: {}", report.type_i_rate());
        assert!(report.power() > 0.40, "G² power too low: {}", report.power());
    }

    #[test]
    fn robust_parcorr_calibration_smoke() {
        let report =
            calibrate_parcorr_like(&RobustPartialCorrelation::new(), 180, 60, 0.05, 13).unwrap();
        assert!(report.type_i_rate() < 0.25);
        assert!(report.power() > 0.30);
    }

    #[test]
    fn weighted_parcorr_calibration_smoke() {
        let n = 180usize;
        let w = vec![1.0; n];
        let report =
            calibrate_parcorr_like(&WeightedPartialCorrelation::new(w), n, 60, 0.05, 17).unwrap();
        assert!(report.type_i_rate() < 0.25);
        assert!(report.power() > 0.30);
    }

    #[test]
    fn knn_cmi_calibration_smoke() {
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(19);
        let n = 80usize;
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let mut rng = ctx.rng.stream(0x4e4e);
        let x: Vec<f64> = (0..n).map(|_| (rng.next_u64() as f64) / (u64::MAX as f64)).collect();
        let y_null: Vec<f64> =
            (0..n).map(|_| (rng.next_u64() as f64) / (u64::MAX as f64)).collect();
        let cols_null: [&[f64]; 2] = [&x, &y_null];
        let req = CiBatchRequest {
            columns: &cols_null,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let out = KnnCmi::new(3).test_batch(&req, &mut ws, &ctx).unwrap();
        assert!((0.0..=1.0).contains(&out.results[0].p_value));
        let cols_alt: [&[f64]; 2] = [&x, &x];
        let req_alt = CiBatchRequest {
            columns: &cols_alt,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let out_alt = KnnCmi::new(3).test_batch(&req_alt, &mut ws, &ctx).unwrap();
        assert!((0.0..=1.0).contains(&out_alt.results[0].p_value));
        assert!(
            out_alt.results[0].p_value <= out.results[0].p_value + 1e-12,
            "alt p={} null p={}",
            out_alt.results[0].p_value,
            out.results[0].p_value
        );
    }

    #[test]
    fn multivariate_and_regression_match_parcorr_on_scalars() {
        let report_mv =
            calibrate_parcorr_like(&MultivariatePartialCorrelation::new(), 200, 80, 0.05, 23)
                .unwrap();
        let report_reg = calibrate_parcorr_like(&RegressionCi::new(), 200, 80, 0.05, 23).unwrap();
        let report_pc =
            calibrate_parcorr_like(&PartialCorrelation::new(), 200, 80, 0.05, 23).unwrap();
        assert!((report_mv.type_i_rate() - report_pc.type_i_rate()).abs() < 0.08);
        assert!((report_reg.type_i_rate() - report_pc.type_i_rate()).abs() < 0.08);
        assert!(report_mv.power() > 0.40);
        assert!(report_reg.power() > 0.40);
    }

    #[test]
    fn mixed_symbolic_gpdc_dependence_ordering() {
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(29);
        let n = 100usize;
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let mut rng = ctx.rng.stream(0x51);
        let x: Vec<f64> = (0..n).map(|_| ((rng.next_u64() % 4) as f64)).collect();
        let y_null: Vec<f64> = (0..n).map(|_| ((rng.next_u64() % 4) as f64)).collect();
        let cols_null: [&[f64]; 2] = [&x, &y_null];
        let req = CiBatchRequest {
            columns: &cols_null,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let cols_alt: [&[f64]; 2] = [&x, &x];
        let req_alt = CiBatchRequest {
            columns: &cols_alt,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };

        for (name, ci) in [
            ("mixed", &MixedKnnCmi::new(3) as &dyn ConditionalIndependence),
            ("symbolic", &SymbolicCmi::new() as &dyn ConditionalIndependence),
            ("gpdc", &Gpdc::new() as &dyn ConditionalIndependence),
        ] {
            let null = ci.test_batch(&req, &mut ws, &ctx).unwrap().results[0].p_value;
            let alt = ci.test_batch(&req_alt, &mut ws, &ctx).unwrap().results[0].p_value;
            assert!((0.0..=1.0).contains(&null), "{name} null p={null}");
            assert!((0.0..=1.0).contains(&alt), "{name} alt p={alt}");
            assert!(alt <= null + 1e-12, "{name}: alt p={alt} null p={null}");
        }
    }
}
