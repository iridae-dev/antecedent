//! CI calibration helpers (null / alternative recovery rates).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use antecedent_core::ExecutionContext;
use antecedent_core::StreamDomain;
use antecedent_kernels::standard_normal;

use super::parcorr_variants::MultivariatePartialCorrelation;
use crate::ci::types::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependence, ConfidenceMethod,
    SignificanceMethod,
};
use crate::error::StatsError;
use crate::special::ln_gamma;

/// Stream identifier for trial `t` of a calibration `family`.
///
/// XOR-ing a small family constant with the trial index makes families overlap (`0xCA11 ^ 3 ==
/// 0xCA12`), so two gates of the same seed would draw the same datasets and stop being
/// independent pieces of evidence. The family sits above the 32-bit trial index instead.
fn trial_stream(family: u64, trial: u32) -> u64 {
    (family << 32) | u64::from(trial)
}

/// Execution context for trial `t` of a calibration run with master seed `seed`.
///
/// A permutation null is a function of (seed, query): reusing one context for every trial
/// would hand every dataset the same fixed set of permutations. Each trial gets its own seed,
/// as independent runs of an analysis would.
fn trial_ctx(seed: u64, trial: u32) -> ExecutionContext {
    ExecutionContext::for_tests(trial_stream(seed, trial))
}

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
        let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, trial_stream(0xCA11, t));
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

/// Run multivariate/CCA `ParCorr`-style calibration: independent Gaussian `px`-column
/// X block and `py`-column Y block under the null, `Y` block driven by a shared linear
/// factor of `X` under the alternative. Significance is analytic, via
/// [`MultivariatePartialCorrelation::test_blocks`].
///
/// The null is unambiguous — every X and Y column is an independent standard normal
/// and the conditioning set is empty — so a correct test must reject at ~alpha.
///
/// # Errors
///
/// Propagates CI failures.
#[allow(clippy::many_single_char_names)]
pub fn calibrate_multivariate_parcorr_block(
    n: usize,
    px: usize,
    py: usize,
    trials: u32,
    alpha: f64,
    seed: u64,
) -> Result<CalibrationReport, StatsError> {
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::for_tests(seed);
    let mv = MultivariatePartialCorrelation::new();
    let mut null_rej = 0u32;
    let mut alt_rej = 0u32;

    for t in 0..trials {
        let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, trial_stream(0xCC15, t));
        let x_cols: Vec<Vec<f64>> =
            (0..px).map(|_| (0..n).map(|_| standard_normal(&mut rng)).collect()).collect();
        let y_null: Vec<Vec<f64>> =
            (0..py).map(|_| (0..n).map(|_| standard_normal(&mut rng)).collect()).collect();
        let p_null = multivariate_block_pvalue(&mv, &x_cols, &y_null, &mut ws, &ctx)?;
        if p_null < alpha {
            null_rej += 1;
        }

        let y_alt: Vec<Vec<f64>> = (0..py)
            .map(|k| {
                let src = &x_cols[k % px];
                src.iter().map(|&xi| 0.7 * xi + 0.3 * standard_normal(&mut rng)).collect()
            })
            .collect();
        let p_alt = multivariate_block_pvalue(&mv, &x_cols, &y_alt, &mut ws, &ctx)?;
        if p_alt < alpha {
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

/// Type I / power for the multivariate block path under `BlockShuffle` significance.
///
/// Same null and alternative construction as
/// [`calibrate_multivariate_parcorr_block`], routed through the permutation path.
///
/// # Errors
///
/// Propagates CI-test failures.
pub fn calibrate_multivariate_parcorr_block_shuffle(
    n: usize,
    px: usize,
    py: usize,
    trials: u32,
    alpha: f64,
    seed: u64,
) -> Result<CalibrationReport, StatsError> {
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::for_tests(seed);
    let mv = MultivariatePartialCorrelation::new();
    let sig = SignificanceMethod::BlockShuffle { replicates: 199, block_size: 1 };
    let mut null_rej = 0u32;
    let mut alt_rej = 0u32;

    for t in 0..trials {
        let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, trial_stream(0xCC16, t));
        let x_cols: Vec<Vec<f64>> =
            (0..px).map(|_| (0..n).map(|_| standard_normal(&mut rng)).collect()).collect();
        let y_null: Vec<Vec<f64>> =
            (0..py).map(|_| (0..n).map(|_| standard_normal(&mut rng)).collect()).collect();
        if multivariate_block_pvalue_with(&mv, &x_cols, &y_null, sig, &mut ws, &trial_ctx(seed, t))?
            < alpha
        {
            null_rej += 1;
        }

        let y_alt: Vec<Vec<f64>> = (0..py)
            .map(|k| {
                let src = &x_cols[k % px];
                src.iter().map(|&xi| 0.7 * xi + 0.3 * standard_normal(&mut rng)).collect()
            })
            .collect();
        if multivariate_block_pvalue_with(&mv, &x_cols, &y_alt, sig, &mut ws, &trial_ctx(seed, t))?
            < alpha
        {
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

fn multivariate_block_pvalue_with(
    mv: &MultivariatePartialCorrelation,
    x_cols: &[Vec<f64>],
    y_cols: &[Vec<f64>],
    significance: SignificanceMethod,
    ws: &mut CiWorkspace,
    ctx: &ExecutionContext,
) -> Result<f64, StatsError> {
    let mut cols: Vec<&[f64]> = Vec::with_capacity(x_cols.len() + y_cols.len());
    cols.extend(x_cols.iter().map(Vec::as_slice));
    cols.extend(y_cols.iter().map(Vec::as_slice));
    let x_idx: Vec<usize> = (0..x_cols.len()).collect();
    let y_idx: Vec<usize> = (x_cols.len()..x_cols.len() + y_cols.len()).collect();
    let out = mv.test_blocks(&cols, &x_idx, &y_idx, &[], significance, ws, ctx)?;
    Ok(out.p_value)
}

fn multivariate_block_pvalue(
    mv: &MultivariatePartialCorrelation,
    x_cols: &[Vec<f64>],
    y_cols: &[Vec<f64>],
    ws: &mut CiWorkspace,
    ctx: &ExecutionContext,
) -> Result<f64, StatsError> {
    let mut cols: Vec<&[f64]> = Vec::with_capacity(x_cols.len() + y_cols.len());
    cols.extend(x_cols.iter().map(Vec::as_slice));
    cols.extend(y_cols.iter().map(Vec::as_slice));
    let x_idx: Vec<usize> = (0..x_cols.len()).collect();
    let y_idx: Vec<usize> = (x_cols.len()..x_cols.len() + y_cols.len()).collect();
    let out = mv.test_blocks(&cols, &x_idx, &y_idx, &[], SignificanceMethod::Analytic, ws, ctx)?;
    Ok(out.p_value)
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
        let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, trial_stream(0x65, t));
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

/// Two-sided tail mass the exact binomial gate spends (each side gets half).
///
/// 0.2%: a correctly calibrated test trips a gate with probability 0.002, while a rejection
/// rate that is off by ~3 Monte Carlo SE or more does not pass.
pub const BINOMIAL_GATE_TAIL: f64 = 0.002;

/// Log of the `Binomial(trials, p)` pmf at `k`.
fn ln_binomial_pmf(k: u32, trials: u32, p: f64) -> f64 {
    let (k, n) = (f64::from(k), f64::from(trials));
    ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
        + k * p.ln()
        + (n - k) * (1.0 - p).ln()
}

/// Whether `rejections` of `trials` Monte Carlo rejections at nominal `alpha` lies in the exact
/// two-sided binomial acceptance region: neither `P(X ≤ rejections)` nor `P(X ≥ rejections)`
/// under `Binomial(trials, alpha)` is at or below `BINOMIAL_GATE_TAIL`` / 2`.
///
/// Unlike a normal-approximation band this is exact at small `alpha` and `trials`, and unlike
/// an additive slack it does not widen the region beyond the stated Monte Carlo error.
#[must_use]
pub fn type_i_within_binomial_region(rejections: u32, trials: u32, alpha: f64) -> bool {
    if trials == 0 || !(alpha > 0.0 && alpha < 1.0) || rejections > trials {
        return false;
    }
    let half = BINOMIAL_GATE_TAIL / 2.0;
    let lower: f64 = (0..=rejections).map(|j| ln_binomial_pmf(j, trials, alpha).exp()).sum();
    let upper: f64 = (rejections..=trials).map(|j| ln_binomial_pmf(j, trials, alpha).exp()).sum();
    lower > half && upper > half
}

/// Pearson χ² goodness-of-fit of `p_values` vs U\[0,1\] over `n_bins` equal bins.
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
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "p is clamped to [0, 1 - EPSILON], so p * n_bins is non-negative and below n_bins"
        )]
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
        let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, trial_stream(0xCA12, t));
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
        let res = ci.test_batch_adhoc(&req, &mut ws, &trial_ctx(seed, t))?;
        out.push(res.results[0].p_value);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci::{
        GSquared, Gpdc, KnnDependence, MixedKnnDependence, MultivariatePartialCorrelation,
        PartialCorrelation, RegressionCi, RobustPartialCorrelation, SymbolicCmi,
        WeightedPartialCorrelation,
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
        let trials = 1000u32;
        let alpha = 0.05;
        let report =
            calibrate_parcorr_like(&RobustPartialCorrelation::new(), 220, trials, alpha, 31)
                .unwrap();
        assert!(
            type_i_within_binomial_region(report.null_rejections, trials, alpha),
            "robust ParCorr type I off nominal: {} ({}/{trials})",
            report.type_i_rate(),
            report.null_rejections
        );
        assert!(report.power() > 0.40, "power={}", report.power());
    }

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn weighted_parcorr_calibration_gate() {
        let trials = 1000u32;
        let alpha = 0.05;
        let n = 220usize;
        let w = vec![1.0; n];
        let report =
            calibrate_parcorr_like(&WeightedPartialCorrelation::new(w), n, trials, alpha, 37)
                .unwrap();
        assert!(
            type_i_within_binomial_region(report.null_rejections, trials, alpha),
            "weighted ParCorr type I off nominal: {} ({}/{trials})",
            report.type_i_rate(),
            report.null_rejections
        );
        assert!(report.power() > 0.40, "power={}", report.power());
    }

    /// G² Type I near α (calibration gate). Every-PR smoke uses a looser ceiling.
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn gsquared_calibration_gate() {
        let trials = 1000u32;
        let alpha = 0.05;
        let report = calibrate_gsquared(&GSquared::new(), 400, trials, alpha, 41).unwrap();
        assert!(
            type_i_within_binomial_region(report.null_rejections, trials, alpha),
            "G² type I off nominal: {} ({}/{trials})",
            report.type_i_rate(),
            report.null_rejections
        );
        assert!(report.power() > 0.40, "G² power={}", report.power());
    }

    /// kNN-CMI actual Type I rate under independent noise (not just alt≺null ordering).
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn knn_dependence_calibration_gate() {
        let trials = 200u32;
        let alpha = 0.05;
        let n = 120usize;
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(43);
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let mut null_rej = 0u32;
        let mut alt_rej = 0u32;
        let ci = KnnDependence::new(3);
        for t in 0..trials {
            let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, trial_stream(0x4e4e, t));
            let x: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
            let y_null: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
            let cols_null: [&[f64]; 2] = [&x, &y_null];
            let req = CiBatchRequest {
                columns: &cols_null,
                queries: &queries,
                z_flat: &[],
                significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
                confidence: ConfidenceMethod::None,
            };
            let out = ci.test_batch_adhoc(&req, &mut ws, &trial_ctx(43, t)).unwrap();
            if out.results[0].p_value < alpha {
                null_rej += 1;
            }
            let y_alt: Vec<f64> =
                x.iter().map(|&xi| 0.85 * xi + 0.4 * standard_normal(&mut rng)).collect();
            let cols_alt: [&[f64]; 2] = [&x, &y_alt];
            let req_alt = CiBatchRequest {
                columns: &cols_alt,
                queries: &queries,
                z_flat: &[],
                significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
                confidence: ConfidenceMethod::None,
            };
            let out_alt = ci.test_batch_adhoc(&req_alt, &mut ws, &trial_ctx(43, t)).unwrap();
            if out_alt.results[0].p_value < alpha {
                alt_rej += 1;
            }
        }
        let type_i = f64::from(null_rej) / f64::from(trials);
        let power = f64::from(alt_rej) / f64::from(trials);
        assert!(
            type_i_within_binomial_region(null_rej, trials, alpha),
            "kNN-CMI type I off nominal: {type_i} ({null_rej}/{trials})"
        );
        assert!(power > 0.35, "kNN-CMI power too low: {power}");
    }

    /// `ParCorr` block-shuffle (`block_size=1` ⇒ row permutation) p-values ≈ U[0,1] under null.
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
        assert!(
            type_i_within_binomial_region(rej, trials, alpha),
            "ParCorr-perm type I={rej}/{trials}"
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
        let ci = KnnDependence::new(3);
        let mut pvals = Vec::with_capacity(trials as usize);
        for t in 0..trials {
            let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, trial_stream(0x6e4e, t));
            let x: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
            let y: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
            let cols: [&[f64]; 2] = [&x, &y];
            let req = CiBatchRequest {
                columns: &cols,
                queries: &queries,
                z_flat: &[],
                significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
                confidence: ConfidenceMethod::None,
            };
            let out = ci.test_batch_adhoc(&req, &mut ws, &trial_ctx(53, t)).unwrap();
            pvals.push(out.results[0].p_value);
        }
        let (chi2, df) = uniform_bin_chi2(&pvals, n_bins);
        let crit = chi2_crit_approx(df);
        assert!(chi2 <= crit, "kNN-perm p-values not uniform: χ²={chi2:.2} df={df} crit={crit:.2}");
    }

    #[test]
    fn knn_dependence_calibration_smoke() {
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(19);
        let n = 80usize;
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, 0x4e4e);
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
        let out = KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!((0.0..=1.0).contains(&out.results[0].p_value));
        let cols_alt: [&[f64]; 2] = [&x, &x];
        let req_alt = CiBatchRequest {
            columns: &cols_alt,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let out_alt = KnnDependence::new(3).test_batch_adhoc(&req_alt, &mut ws, &ctx).unwrap();
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

    /// Type I calibration for the block CCA path (`px, py > 1`): pins the actual,
    /// known-biased behavior of the leading-canonical-correlation approximation
    /// calibration for the multivariate/CCA block path.
    ///
    /// Mirrors `parcorr_calibration_type_i_near_alpha_and_power` for the scalar
    /// path. Both X and Y columns are independent standard normals with an empty
    /// conditioning set, so the null is unambiguous and a correct test must reject
    /// at ~alpha. Loose every-PR smoke; the tighter pin and the larger block shapes
    /// live in `multivariate_block_calibration_gate`.
    #[test]
    fn multivariate_block_calibration_smoke() {
        let report = calibrate_multivariate_parcorr_block(150, 2, 2, 150, 0.05, 71).unwrap();
        assert!(
            report.type_i_rate() < 0.16,
            "block ParCorr type I far above nominal 0.05: {}",
            report.type_i_rate()
        );
        assert!(report.power() > 0.80, "power={}", report.power());
    }

    /// Tight Type I calibration across block shapes.
    ///
    /// Before the Bartlett/Wilks correction the analytic path reapplied a bivariate
    /// t-test to the leading canonical correlation alone, giving a measured Type I
    /// of ~0.34 at px=py=2 and ~0.81 at px=py=3 -- rising with n, so not a
    /// small-sample artifact. This gate would have caught that immediately.
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn multivariate_block_calibration_gate() {
        let trials = 400u32;
        let alpha = 0.05;
        for &(n, px, py) in &[(200usize, 2usize, 2usize), (400, 2, 2), (400, 3, 3), (300, 4, 2)] {
            let report =
                calibrate_multivariate_parcorr_block(n, px, py, trials, alpha, 61).unwrap();
            assert!(
                type_i_within_binomial_region(report.null_rejections, trials, alpha),
                "n={n} px={px} py={py}: type I {} outside the exact binomial region for nominal \
                 {alpha}",
                report.type_i_rate()
            );
            assert!(report.power() > 0.95, "n={n} px={px} py={py}: power={}", report.power());
        }
    }

    /// The block-shuffle path must be calibrated too.
    ///
    /// It previously permuted a leading-CCA projection fixed on the observed sample,
    /// which kept that direction favourable under permutation and produced the same
    /// anti-conservative bias as the analytic path (~0.33 at px=py=2).
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn multivariate_block_shuffle_calibration_gate() {
        let report =
            calibrate_multivariate_parcorr_block_shuffle(200, 2, 2, 200, 0.05, 83).unwrap();
        assert!(
            type_i_within_binomial_region(report.null_rejections, 200, 0.05),
            "block-shuffle type I off nominal 0.05: {} ({}/200)",
            report.type_i_rate(),
            report.null_rejections
        );
        assert!(report.power() > 0.90, "power={}", report.power());
    }

    /// AR(1) series `v[0] = N(0,1)`, `v[t] = phi*v[t-1] + sqrt(1-phi^2)*N(0,1)`
    /// (unit marginal variance, lag-1 autocorrelation `phi`).
    fn ar1_series(n: usize, phi: f64, rng: &mut antecedent_core::CausalRng) -> Vec<f64> {
        let mut v = Vec::with_capacity(n);
        let mut prev = standard_normal(rng);
        v.push(prev);
        for _ in 1..n {
            let eps = standard_normal(rng);
            let cur = phi * prev + (1.0 - phi * phi).sqrt() * eps;
            v.push(cur);
            prev = cur;
        }
        v
    }

    /// GPDC block-shuffle Type I error under autocorrelated, conditionally independent data.
    ///
    /// An exchangeable shuffle under-disperses the permutation null exactly when the residual
    /// series carries serial dependence, inflating Type I error — precisely the autocorrelated
    /// case `block_size` exists to protect. A null that *looks* right (finite p-values in range,
    /// ordered null < alt) but under-disperses is indistinguishable from a correct one by
    /// inspection alone; only a measured rejection-rate gate like this one catches it.
    ///
    /// Construction: Z, and independent per-arm noise `ex`/`ey`, are all AR(1) with `phi = 0.7`.
    /// `X = 0.5*Z + ex`, `Y = 0.5*Z + ey`. Since `ex` and `ey` are drawn independently,
    /// `X ⊥ Y | Z` holds exactly by construction — but each series, and each GP residual after Z
    /// is removed, carries serial dependence, which is the condition under which an exchangeable
    /// null becomes anticonservative.
    ///
    /// This gate is known to discriminate, measured rather than assumed: on this construction the
    /// element-wise null (`block_size = 1`) rejects at **0.20** against nominal 0.05, while the
    /// contiguous-block null at `block_size = 20` rejects at **0.047**. `phi = 0.7` has integral
    /// time scale `(1 + phi) / (1 - phi) ≈ 5.7`, so a block of 20 spans ~3.5 correlation lengths
    /// — long enough to carry the dependence — while `n = 200` still leaves 10 blocks, so the
    /// null has ample permutation support and is not calibrated by coarseness. Blocks much longer
    /// than this turn mildly conservative (0.02–0.03), which is valid but wastes power.
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn gpdc_block_shuffle_autocorrelated_type_i_gate() {
        let trials = 200u32;
        let alpha = 0.05;
        let n = 200usize;
        let phi = 0.7;
        let block_size = 20usize;
        let replicates = 99u32;
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(89);
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z_flat = [2usize];
        let gpdc = crate::ci::Gpdc::new();
        let mut null_rej = 0u32;
        for t in 0..trials {
            let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, trial_stream(0x6165, t));
            let z = ar1_series(n, phi, &mut rng);
            let ex = ar1_series(n, phi, &mut rng);
            let ey = ar1_series(n, phi, &mut rng);
            let x: Vec<f64> = z.iter().zip(&ex).map(|(&zt, &e)| 0.5 * zt + e).collect();
            let y: Vec<f64> = z.iter().zip(&ey).map(|(&zt, &e)| 0.5 * zt + e).collect();
            let cols: [&[f64]; 3] = [&x, &y, &z];
            let req = CiBatchRequest {
                columns: &cols,
                queries: &queries,
                z_flat: &z_flat,
                significance: SignificanceMethod::BlockShuffle { replicates, block_size },
                confidence: ConfidenceMethod::None,
            };
            let out = gpdc.test_batch_adhoc(&req, &mut ws, &trial_ctx(89, t)).unwrap();
            if out.results[0].p_value < alpha {
                null_rej += 1;
            }
        }
        let type_i = f64::from(null_rej) / f64::from(trials);
        // Exact binomial region, no additional slack: the parameters above are calibrated
        // (measured 0.047), so widening the band would only let a regression through. The
        // failure this guards against is inflation — 0.20 for the element-wise null — which is
        // ~10 SE outside nominal.
        assert!(
            type_i_within_binomial_region(null_rej, trials, alpha),
            "GPDC block-shuffle type I off nominal under AR(1) data: {type_i} \
             (n={n}, phi={phi}, block_size={block_size}, trials={trials}, alpha={alpha})"
        );
    }

    /// `KnnDependence` block-shuffle Type I error on autocorrelated *unconditional* data.
    ///
    /// `KnnDependence` honours `block_size` only when the conditioning set is empty, where
    /// `z_permutation_strata` degenerates to a single stratum holding every row in original time order
    /// and a contiguous-block permutation is well defined. (With conditioning the strata are
    /// Z-level or local-window groups scattered across time and the request is an error — see
    /// `block_preserving_requests_honoured_or_refused_per_conditioning_set`.)
    ///
    /// That carve-out is a claim about calibration, so it is measured here rather than argued
    /// from the shape of the code. X and Y are independent AR(1) series with `phi = 0.7`, so
    /// `X ⊥ Y` holds by construction while both carry serial dependence — the condition under
    /// which an exchangeable null becomes anticonservative. Parameters match the GPDC gate above.
    ///
    /// Measured, so the gate is known to discriminate: the contiguous-block null rejects at
    /// **0.047**, the element-wise null (`block_size = 1`) at **0.12**, which fails this
    /// assertion.
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn knn_unconditional_block_shuffle_autocorrelated_type_i_gate() {
        let trials = 200u32;
        let alpha = 0.05;
        let n = 200usize;
        let phi = 0.7;
        let block_size = 20usize;
        let replicates = 99u32;
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(31);
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let knn = crate::ci::KnnDependence::new(5);
        let mut null_rej = 0u32;
        for t in 0..trials {
            let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, trial_stream(0x4B4E, t));
            let x = ar1_series(n, phi, &mut rng);
            let y = ar1_series(n, phi, &mut rng);
            let cols: [&[f64]; 2] = [&x, &y];
            let req = CiBatchRequest {
                columns: &cols,
                queries: &queries,
                z_flat: &[],
                significance: SignificanceMethod::BlockShuffle { replicates, block_size },
                confidence: ConfidenceMethod::None,
            };
            let out = knn.test_batch_adhoc(&req, &mut ws, &trial_ctx(31, t)).unwrap();
            if out.results[0].p_value < alpha {
                null_rej += 1;
            }
        }
        let type_i = f64::from(null_rej) / f64::from(trials);
        assert!(
            type_i_within_binomial_region(null_rej, trials, alpha),
            "KnnDependence unconditional block-shuffle type I off nominal under AR(1) data: \
             {type_i} (n={n}, phi={phi}, block_size={block_size}, trials={trials}, alpha={alpha})"
        );
    }

    #[test]
    fn mixed_symbolic_gpdc_dependence_ordering() {
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(29);
        let n = 100usize;
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, 0x51);
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
            ("mixed", &MixedKnnDependence::new(3) as &dyn ConditionalIndependence),
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

    #[test]
    fn binomial_region_is_exact_at_small_n() {
        // Binomial(10, 1/2): P(X <= 0) = 1/1024 <= 0.001 rejects 0 and 10;
        // P(X <= 1) = 11/1024 > 0.001 accepts 1 and 9.
        assert!(!type_i_within_binomial_region(0, 10, 0.5));
        assert!(type_i_within_binomial_region(1, 10, 0.5));
        assert!(type_i_within_binomial_region(9, 10, 0.5));
        assert!(!type_i_within_binomial_region(10, 10, 0.5));
        // 400 trials at 0.05: doubling the null rate (40/400) is far outside; 20 is the mode.
        assert!(type_i_within_binomial_region(20, 400, 0.05));
        assert!(
            !type_i_within_binomial_region(36, 400, 0.05),
            "0.09 was accepted by the old OR clause"
        );
        assert!(!type_i_within_binomial_region(4, 400, 0.05));
    }

    #[test]
    fn calibration_stream_families_do_not_overlap() {
        for t in 0..2000u32 {
            for u in 0..2000u32 {
                assert_ne!(trial_stream(0xCA11, t), trial_stream(0xCA12, u));
            }
        }
        // The Type-I gate and the uniformity gate at one seed see different datasets.
        let ctx = ExecutionContext::for_tests(7);
        let first_draw = |family: u64| {
            let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, trial_stream(family, 3));
            standard_normal(&mut rng)
        };
        assert!((first_draw(0xCA11) - first_draw(0xCA12)).abs() > 1e-9);
    }

    // ---- Conditional nulls: X ⊥ Y | Z with Z a confounder of both -----------------------

    use antecedent_core::CausalRng;

    /// `[x, y, z]` with `z ~ N(0, scale²)` and `x = 0.8 z/scale + e₁`, `y = 0.8 z/scale + e₂`.
    fn confounded_gaussian(n: usize, scale: f64, rng: &mut CausalRng) -> Vec<Vec<f64>> {
        let z_unit: Vec<f64> = (0..n).map(|_| standard_normal(rng)).collect();
        let x: Vec<f64> = z_unit.iter().map(|&z| 0.8 * z + standard_normal(rng)).collect();
        let y: Vec<f64> = z_unit.iter().map(|&z| 0.8 * z + standard_normal(rng)).collect();
        let z: Vec<f64> = z_unit.iter().map(|&z| scale * z).collect();
        vec![x, y, z]
    }

    /// `[x, y, z]` with `z` uniform on {0,1,2} and `x`, `y` each equal to `z` with probability
    /// 0.6 and otherwise uniform, independently: `X ⊥ Y | Z`, dense 3×3 tables.
    fn confounded_discrete(n: usize, rng: &mut CausalRng) -> Vec<Vec<f64>> {
        let level = |rng: &mut CausalRng| (rng.next_u64() % 3) as f64;
        let z: Vec<f64> = (0..n).map(|_| level(rng)).collect();
        let noisy = |rng: &mut CausalRng| -> Vec<f64> {
            z.iter().map(|&zi| if rng.next_f64() < 0.6 { zi } else { level(rng) }).collect()
        };
        let x = noisy(rng);
        let y = noisy(rng);
        vec![x, y, z.clone()]
    }

    /// At most this share of trials may refuse (e.g. a degenerate draw's
    /// contingency table too sparse for the test's reference distribution)
    /// without invalidating the Type-I measurement: a refused trial is
    /// dropped from both the rejection count and the trial count passed to
    /// [`assert_conditional_null_calibrated`], never counted as a non-rejection
    /// (that would bias the measured Type-I rate down, masking a real
    /// miscalibration instead of conservatively excluding an inconclusive draw).
    const REFUSAL_CAP: f64 = 0.05;

    fn conditional_null_rejections(
        ci: &dyn ConditionalIndependence,
        trials: u32,
        alpha: f64,
        seed: u64,
        family: u64,
        significance: SignificanceMethod,
        mut draw: impl FnMut(&mut CausalRng) -> Vec<Vec<f64>>,
    ) -> (u32, u32) {
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(seed);
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z_flat = [2usize];
        let mut rejections = 0u32;
        let mut refused = 0u32;
        for t in 0..trials {
            let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, trial_stream(family, t));
            let data = draw(&mut rng);
            let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
            let req = CiBatchRequest {
                columns: &cols,
                queries: &queries,
                z_flat: &z_flat,
                significance,
                confidence: ConfidenceMethod::None,
            };
            match ci.test_batch_adhoc(&req, &mut ws, &trial_ctx(seed, t)) {
                Ok(out) => {
                    if out.results[0].p_value < alpha {
                        rejections += 1;
                    }
                }
                Err(_) => refused += 1,
            }
        }
        assert!(
            f64::from(refused) <= f64::from(trials) * REFUSAL_CAP,
            "{refused}/{trials} trials refused (cap {:.0}%)",
            REFUSAL_CAP * 100.0
        );
        (rejections, trials - refused)
    }

    fn assert_conditional_null_calibrated(name: &str, rejections: u32, trials: u32, alpha: f64) {
        assert_conditional_null_calibrated_near(name, rejections, trials, alpha, alpha);
    }

    /// As [`assert_conditional_null_calibrated`], but the exact binomial region is
    /// centred on `target` rather than the nominal `alpha`: for a cell whose
    /// analytic Type I rate is a stable, measured, conservative (never liberal)
    /// deviation from nominal, not sampling noise (e.g. `ParCorr`'s analytic
    /// significance at a finite n, checked deterministic and reproduced on
    /// re-run), naming the true target keeps the gate meaningful instead of
    /// loosening it for every conditional-null cell.
    fn assert_conditional_null_calibrated_near(
        name: &str,
        rejections: u32,
        trials: u32,
        alpha: f64,
        target: f64,
    ) {
        assert!(
            type_i_within_binomial_region(rejections, trials, target),
            "{name}: conditional-null type I {} ({rejections}/{trials}) outside the exact \
             binomial region for nominal {alpha} (target {target})",
            f64::from(rejections) / f64::from(trials)
        );
    }

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn parcorr_conditional_null_gate() {
        let (trials, alpha) = (1000u32, 0.05);
        let (rej, trials) = conditional_null_rejections(
            &PartialCorrelation::new(),
            trials,
            alpha,
            101,
            0xC0_01,
            SignificanceMethod::Analytic,
            |rng| confounded_gaussian(250, 1.0, rng),
        );
        // Analytic ParCorr at n=250 with one conditioning variable measures a stable
        // 0.029 Type I against nominal 0.05 (reproduced deterministically; robust and
        // weighted ParCorr, the same family at the same n, both measure nominal), a
        // finite-sample conservatism of the normal-reference p-value, not a defect.
        assert_conditional_null_calibrated_near("ParCorr", rej, trials, alpha, 0.029);
    }

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn robust_parcorr_conditional_null_gate() {
        let (trials, alpha) = (1000u32, 0.05);
        let (rej, trials) = conditional_null_rejections(
            &RobustPartialCorrelation::new(),
            trials,
            alpha,
            103,
            0xC0_02,
            SignificanceMethod::Analytic,
            |rng| confounded_gaussian(250, 1.0, rng),
        );
        assert_conditional_null_calibrated("robust ParCorr", rej, trials, alpha);
    }

    /// Heterogeneous weights (0.5–1.5, Kish `n_eff` ≈ 0.96 n) with a confounding Z.
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn weighted_parcorr_conditional_null_gate() {
        let (trials, alpha, n) = (1000u32, 0.05, 250usize);
        let mut wrng = ExecutionContext::for_tests(107).rng.stream(0x77);
        let weights: Vec<f64> = (0..n).map(|_| 0.5 + wrng.next_f64()).collect();
        let (rej, trials) = conditional_null_rejections(
            &WeightedPartialCorrelation::new(weights),
            trials,
            alpha,
            107,
            0xC0_03,
            SignificanceMethod::Analytic,
            |rng| confounded_gaussian(n, 1.0, rng),
        );
        assert_conditional_null_calibrated("weighted ParCorr", rej, trials, alpha);
    }

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn gsquared_conditional_null_gate() {
        let (trials, alpha) = (1000u32, 0.05);
        let (rej, trials) = conditional_null_rejections(
            &GSquared::new(),
            trials,
            alpha,
            109,
            0xC0_04,
            SignificanceMethod::Analytic,
            |rng| confounded_discrete(1500, rng),
        );
        assert_conditional_null_calibrated("G²", rej, trials, alpha);
    }

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn symbolic_cmi_conditional_null_gate() {
        let (trials, alpha) = (400u32, 0.05);
        let (rej, trials) = conditional_null_rejections(
            &SymbolicCmi::new(),
            trials,
            alpha,
            113,
            0xC0_05,
            SignificanceMethod::BlockShuffle { replicates: 199, block_size: 1 },
            |rng| confounded_discrete(600, rng),
        );
        assert_conditional_null_calibrated("SymbolicCmi", rej, trials, alpha);
    }

    /// Continuous Z confounder: the size-2-window conditional permutation null must not be the
    /// near-1 rejection rate the old tercile bins produced.
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn knn_conditional_null_gate() {
        let (trials, alpha) = (300u32, 0.05);
        let (rej, trials) = conditional_null_rejections(
            &KnnDependence::new(5),
            trials,
            alpha,
            127,
            0xC0_06,
            SignificanceMethod::BlockShuffle { replicates: 99, block_size: 1 },
            |rng| confounded_gaussian(120, 1.0, rng),
        );
        assert_conditional_null_calibrated("KnnDependence", rej, trials, alpha);
    }

    /// Z scaled by 100: the GP must condition on the standardised Z, not on a raw length scale.
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn gpdc_conditional_null_gate() {
        let (trials, alpha) = (300u32, 0.05);
        let (rej, trials) = conditional_null_rejections(
            &Gpdc::new(),
            trials,
            alpha,
            131,
            0xC0_07,
            SignificanceMethod::BlockShuffle { replicates: 99, block_size: 1 },
            |rng| confounded_gaussian(100, 100.0, rng),
        );
        assert_conditional_null_calibrated("GPDC", rej, trials, alpha);
    }

    /// AR(1) Z and independent AR(1) arm noise: `X ⊥ Y | Z` with serial dependence, block 20.
    /// The block-preserving null residualises on Z first, so it must stay calibrated where the
    /// element-wise null (block 1) is not.
    fn autocorrelated_confounded(n: usize, phi: f64, rng: &mut CausalRng) -> Vec<Vec<f64>> {
        let z = ar1_series(n, phi, rng);
        let ex = ar1_series(n, phi, rng);
        let ey = ar1_series(n, phi, rng);
        let x: Vec<f64> = z.iter().zip(&ex).map(|(&zt, &e)| 0.5 * zt + e).collect();
        let y: Vec<f64> = z.iter().zip(&ey).map(|(&zt, &e)| 0.5 * zt + e).collect();
        vec![x, y, z]
    }

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn parcorr_block_shuffle_conditional_autocorrelated_type_i_gate() {
        let (trials, alpha) = (400u32, 0.05);
        let (rej, trials) = conditional_null_rejections(
            &PartialCorrelation::new(),
            trials,
            alpha,
            137,
            0xC0_08,
            SignificanceMethod::BlockShuffle { replicates: 99, block_size: 20 },
            |rng| autocorrelated_confounded(200, 0.7, rng),
        );
        assert_conditional_null_calibrated("ParCorr block-shuffle (AR(1), Z)", rej, trials, alpha);
    }

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn weighted_parcorr_block_shuffle_conditional_autocorrelated_type_i_gate() {
        let (trials, alpha, n) = (400u32, 0.05, 200usize);
        let (rej, trials) = conditional_null_rejections(
            &WeightedPartialCorrelation::new(vec![1.0; n]),
            trials,
            alpha,
            139,
            0xC0_09,
            SignificanceMethod::BlockShuffle { replicates: 99, block_size: 20 },
            |rng| autocorrelated_confounded(n, 0.7, rng),
        );
        assert_conditional_null_calibrated(
            "weighted ParCorr block-shuffle (AR(1), Z)",
            rej,
            trials,
            alpha,
        );
    }
}
