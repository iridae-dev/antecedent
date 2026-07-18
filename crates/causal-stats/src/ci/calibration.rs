//! CI calibration helpers (null / alternative recovery rates).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use causal_core::ExecutionContext;
use causal_kernels::standard_normal;

use crate::ci::types::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependence, ConfidenceMethod,
    SignificanceMethod,
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
#[allow(clippy::many_single_char_names)]
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
        let x: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
        let y_null: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
        let cols_null: [&[f64]; 2] = [&x, &y_null];
        let req = CiBatchRequest {
            columns: &cols_null,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let out = ci.test_batch_adhoc(&req, &mut ws, &ctx)?;
        if out.results[0].p_value < alpha {
            null_rej += 1;
        }

        let y_alt: Vec<f64> = x
            .iter()
            .map(|&xi| {
                let e = standard_normal(&mut rng);
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
        let out_alt = ci.test_batch_adhoc(&req_alt, &mut ws, &ctx)?;
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
        let x: Vec<f64> =
            (0..n).map(|_| (rng.next_u64() % u64::try_from(levels).unwrap_or(1)) as f64).collect();
        let y_null: Vec<f64> =
            (0..n).map(|_| (rng.next_u64() % u64::try_from(levels).unwrap_or(1)) as f64).collect();
        let cols_null: [&[f64]; 2] = [&x, &y_null];
        let req = CiBatchRequest {
            columns: &cols_null,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let out = ci.test_batch_adhoc(&req, &mut ws, &ctx)?;
        if out.results[0].p_value < alpha {
            null_rej += 1;
        }

        let y_alt: Vec<f64> = x
            .iter()
            .map(|&xi| {
                if rng.next_u64() % 5 == 0 {
                    (rng.next_u64() % u64::try_from(levels).unwrap_or(1)) as f64
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
        let out_alt = ci.test_batch_adhoc(&req_alt, &mut ws, &ctx)?;
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

/// Within ~3 SE of α (slightly looser band for nonparametric / discrete CI tests).
#[must_use]
pub fn type_i_within_three_se(rate: f64, alpha: f64, trials: u32) -> bool {
    let n = f64::from(trials);
    let se = (alpha * (1.0 - alpha) / n).sqrt();
    (rate - alpha).abs() <= 3.0 * se + 1e-12
}

/// Pearson χ² goodness-of-fit of `p_values` vs U[0,1] over `n_bins` equal bins.
///
/// Returns `(chi2, df)`. Permutation p-values live on a discrete lattice
/// `(1+k)/(1+R)`; with enough trials and bins ≪ lattice size the continuous
/// Uniform approximation is adequate for a calibration gate.
#[must_use]
pub fn uniform_bin_chi2(p_values: &[f64], n_bins: usize) -> (f64, usize) {
    let n = p_values.len();
    if n == 0 || n_bins == 0 {
        return (0.0, 0);
    }
    let mut counts = vec![0u32; n_bins];
    for &p in p_values {
        let p = p.clamp(0.0, 1.0 - f64::EPSILON);
        let b = ((p * n_bins as f64).floor() as usize).min(n_bins - 1);
        counts[b] += 1;
    }
    let expected = n as f64 / n_bins as f64;
    let mut chi2 = 0.0;
    for c in counts {
        let d = f64::from(c) - expected;
        chi2 += d * d / expected;
    }
    (chi2, n_bins.saturating_sub(1))
}

/// Critical value for χ²_{df} at ~0.001 (conservative gate; 9 df ≈ 27.9).
#[must_use]
pub fn chi2_crit_approx(df: usize) -> f64 {
    // Rough Wilson–Hilferty / tabulated anchors for small df used by gates.
    match df {
        0 => 0.0,
        1 => 10.83,
        2 => 13.82,
        3 => 16.27,
        4 => 18.47,
        5 => 20.52,
        6 => 22.46,
        7 => 24.32,
        8 => 26.12,
        9 => 27.88,
        10 => 29.59,
        _ => {
            let k = df as f64;
            // Mean + ~3.3 SD of χ²_df ≈ 0.001 upper tail for moderate df.
            k + 3.3 * (2.0 * k).sqrt()
        }
    }
}

/// Collect null p-values under independent Gaussian noise for a ParCorr-like CI.
///
/// # Errors
///
/// Propagates CI failures.
pub fn collect_null_pvalues_parcorr_like(
    ci: &dyn ConditionalIndependence,
    n: usize,
    trials: u32,
    seed: u64,
    significance: SignificanceMethod,
) -> Result<Vec<f64>, StatsError> {
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::for_tests(seed);
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
    let mut out = Vec::with_capacity(trials as usize);
    for t in 0..trials {
        let mut rng = ctx.rng.stream(0xCA11_u64.wrapping_add(u64::from(t)));
        let x: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
        let y: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
        let cols: [&[f64]; 2] = [&x, &y];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance,
            confidence: ConfidenceMethod::None,
        };
        let res = ci.test_batch_adhoc(&req, &mut ws, &ctx)?;
        out.push(res.results[0].p_value);
    }
    Ok(out)
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
        let trials = 800u32;
        let alpha = 0.05;
        let report =
            calibrate_parcorr_like(&PartialCorrelation::new(), 250, trials, alpha, 7).unwrap();
        assert!(
            type_i_within_two_se(report.type_i_rate(), alpha, trials),
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
        // Loose every-PR smoke. Tighter Type I lives in
        // `robust_parcorr_calibration_gate` (scripts/gate_calibration.sh).
        let report =
            calibrate_parcorr_like(&RobustPartialCorrelation::new(), 180, 60, 0.05, 13).unwrap();
        assert!(report.type_i_rate() < 0.25);
        assert!(report.power() > 0.30);
    }

    #[test]
    fn weighted_parcorr_calibration_smoke() {
        // Loose every-PR smoke. Tighter Type I lives in
        // `weighted_parcorr_calibration_gate` (scripts/gate_calibration.sh).
        let n = 180usize;
        let w = vec![1.0; n];
        let report =
            calibrate_parcorr_like(&WeightedPartialCorrelation::new(w), n, 60, 0.05, 17).unwrap();
        assert!(report.type_i_rate() < 0.25);
        assert!(report.power() > 0.30);
    }

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn robust_parcorr_calibration_gate() {
        let trials = 400u32;
        let alpha = 0.05;
        let report =
            calibrate_parcorr_like(&RobustPartialCorrelation::new(), 220, trials, alpha, 31)
                .unwrap();
        assert!(
            type_i_within_two_se(report.type_i_rate(), alpha, trials)
                || (report.type_i_rate() - alpha).abs() < 0.04,
            "robust ParCorr type I off nominal: {}",
            report.type_i_rate()
        );
        assert!(report.power() > 0.40, "power={}", report.power());
    }

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn weighted_parcorr_calibration_gate() {
        let trials = 400u32;
        let alpha = 0.05;
        let n = 220usize;
        let w = vec![1.0; n];
        let report =
            calibrate_parcorr_like(&WeightedPartialCorrelation::new(w), n, trials, alpha, 37)
                .unwrap();
        assert!(
            type_i_within_two_se(report.type_i_rate(), alpha, trials)
                || (report.type_i_rate() - alpha).abs() < 0.04,
            "weighted ParCorr type I off nominal: {}",
            report.type_i_rate()
        );
        assert!(report.power() > 0.40, "power={}", report.power());
    }

    /// G² Type I near α (scheduled). Every-PR smoke uses a looser ceiling.
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn gsquared_calibration_gate() {
        let trials = 400u32;
        let alpha = 0.05;
        let report = calibrate_gsquared(&GSquared::new(), 400, trials, alpha, 41).unwrap();
        assert!(
            type_i_within_three_se(report.type_i_rate(), alpha, trials)
                || (report.type_i_rate() - alpha).abs() < 0.035,
            "G² type I off nominal: {}",
            report.type_i_rate()
        );
        assert!(report.power() > 0.40, "G² power={}", report.power());
    }

    /// kNN-CMI actual Type I rate under independent noise (not just alt≺null ordering).
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn knn_cmi_calibration_gate() {
        let trials = 200u32;
        let alpha = 0.05;
        let n = 120usize;
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(43);
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let mut null_rej = 0u32;
        let mut alt_rej = 0u32;
        let ci = KnnCmi::new(3);
        for t in 0..trials {
            let mut rng = ctx.rng.stream(0x4e4e_u64.wrapping_add(u64::from(t)));
            let x: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
            let y_null: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
            let cols_null: [&[f64]; 2] = [&x, &y_null];
            let req = CiBatchRequest {
                columns: &cols_null,
                queries: &queries,
                z_flat: &[],
                significance: SignificanceMethod::BlockShuffle {
                    replicates: 49,
                    block_size: 1,
                },
                confidence: ConfidenceMethod::None,
            };
            let out = ci.test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
            if out.results[0].p_value < alpha {
                null_rej += 1;
            }
            let y_alt: Vec<f64> = x
                .iter()
                .map(|&xi| 0.85 * xi + 0.4 * standard_normal(&mut rng))
                .collect();
            let cols_alt: [&[f64]; 2] = [&x, &y_alt];
            let req_alt = CiBatchRequest {
                columns: &cols_alt,
                queries: &queries,
                z_flat: &[],
                significance: SignificanceMethod::BlockShuffle {
                    replicates: 49,
                    block_size: 1,
                },
                confidence: ConfidenceMethod::None,
            };
            let out_alt = ci.test_batch_adhoc(&req_alt, &mut ws, &ctx).unwrap();
            if out_alt.results[0].p_value < alpha {
                alt_rej += 1;
            }
        }
        let type_i = f64::from(null_rej) / f64::from(trials);
        let power = f64::from(alt_rej) / f64::from(trials);
        // Discrete permutation p-values + finite kNN bias → allow ~3 SE or |Δ|<0.06.
        assert!(
            type_i_within_three_se(type_i, alpha, trials) || (type_i - alpha).abs() < 0.06,
            "kNN-CMI type I off nominal: {type_i}"
        );
        assert!(power > 0.35, "kNN-CMI power too low: {power}");
    }

    /// ParCorr block-shuffle (block_size=1 ⇒ row permutation) p-values ≈ U[0,1] under null.
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn parcorr_perm_pvalue_uniformity_gate() {
        let trials = 400u32;
        let n_bins = 10usize;
        let pvals = collect_null_pvalues_parcorr_like(
            &PartialCorrelation::new(),
            200,
            trials,
            47,
            SignificanceMethod::BlockShuffle { replicates: 99, block_size: 1 },
        )
        .unwrap();
        let (chi2, df) = uniform_bin_chi2(&pvals, n_bins);
        let crit = chi2_crit_approx(df);
        assert!(
            chi2 <= crit,
            "ParCorr-perm p-values not uniform: χ²={chi2:.2} df={df} crit={crit:.2}"
        );
        let alpha = 0.05;
        let rej = pvals.iter().filter(|&&p| p < alpha).count() as u32;
        let rate = f64::from(rej) / f64::from(trials);
        assert!(
            type_i_within_three_se(rate, alpha, trials) || (rate - alpha).abs() < 0.04,
            "ParCorr-perm type I={rate}"
        );
    }

    /// kNN-CMI permutation p-values ≈ U[0,1] under independent noise.
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn knn_perm_pvalue_uniformity_gate() {
        let trials = 200u32;
        let n = 100usize;
        let n_bins = 8usize;
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(53);
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let ci = KnnCmi::new(3);
        let mut pvals = Vec::with_capacity(trials as usize);
        for t in 0..trials {
            let mut rng = ctx.rng.stream(0x6e4e_u64.wrapping_add(u64::from(t)));
            let x: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
            let y: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
            let cols: [&[f64]; 2] = [&x, &y];
            let req = CiBatchRequest {
                columns: &cols,
                queries: &queries,
                z_flat: &[],
                significance: SignificanceMethod::BlockShuffle {
                    replicates: 49,
                    block_size: 1,
                },
                confidence: ConfidenceMethod::None,
            };
            let out = ci.test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
            pvals.push(out.results[0].p_value);
        }
        let (chi2, df) = uniform_bin_chi2(&pvals, n_bins);
        let crit = chi2_crit_approx(df);
        // kNN MI proxy is slightly conservative/discrete; allow 1.5× the 0.001 critical.
        assert!(
            chi2 <= crit * 1.5,
            "kNN-perm p-values not uniform: χ²={chi2:.2} df={df} crit*1.5={:.2}",
            crit * 1.5
        );
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
        let out = KnnCmi::new(3).test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!((0.0..=1.0).contains(&out.results[0].p_value));
        let cols_alt: [&[f64]; 2] = [&x, &x];
        let req_alt = CiBatchRequest {
            columns: &cols_alt,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let out_alt = KnnCmi::new(3).test_batch_adhoc(&req_alt, &mut ws, &ctx).unwrap();
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
            let null = ci.test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0].p_value;
            let alt = ci.test_batch_adhoc(&req_alt, &mut ws, &ctx).unwrap().results[0].p_value;
            assert!((0.0..=1.0).contains(&null), "{name} null p={null}");
            assert!((0.0..=1.0).contains(&alt), "{name} alt p={alt}");
            assert!(alt <= null + 1e-12, "{name}: alt p={alt} null p={null}");
        }
    }
}
