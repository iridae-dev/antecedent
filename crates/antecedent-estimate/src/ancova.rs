//! Independent-unit randomized ANCOVA point and HC0 sandwich variance.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, hc2_coefficient_bell_mccaffrey,
    student_t_ppf,
};

use crate::randomized_scores::MIN_EXPECTED_ARM_ROWS_FOR_INTERVAL;

/// Fitted treatment coefficient, HC0 variance, and covariate coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct AncovaFit {
    /// Coefficient on realized assignment.
    pub effect: f64,
    /// Independent-row HC0 sandwich variance for the treatment coefficient.
    pub hc0_variance: f64,
    /// Coefficients in the supplied covariate order.
    pub covariate_coefficients: Vec<f64>,
    /// Number assigned treatment.
    pub treated: usize,
    /// Number assigned control.
    pub control: usize,
}

/// Pointwise interval calibrated for independently assigned Bernoulli rows.
///
/// The interval needs 30 observed rows in each arm and at least
/// [`MIN_EXPECTED_ARM_ROWS_FOR_INTERVAL`](crate::randomized_scores::MIN_EXPECTED_ARM_ROWS_FOR_INTERVAL)
/// expected rows in each arm (`n·p` treated, `n·(1−p)` control): HC0 omits the leverage correction,
/// whose size is set by how few rows an arm holds, not by the total row count. The coefficient and
/// HC0 variance remain available below this measured support boundary; an interval does not,
/// but [`small_arm_interval_95`] publishes the HC2 Student interval built for those arms.
#[must_use]
pub fn calibrated_bernoulli_interval_95(fit: &AncovaFit, probability: f64) -> Option<[f64; 2]> {
    use antecedent_stats::NORMAL_Q975 as NORMAL_95;
    let rows = (fit.treated + fit.control) as f64;
    if fit.covariate_coefficients.is_empty()
        || fit.covariate_coefficients.len() > 2
        || fit.treated < 30
        || fit.control < 30
        || !probability.is_finite()
        || probability + 1e-12 < 0.2
        || probability > 0.8 + 1e-12
        || rows * probability + 1e-9 < MIN_EXPECTED_ARM_ROWS_FOR_INTERVAL
        || rows * (1.0 - probability) + 1e-9 < MIN_EXPECTED_ARM_ROWS_FOR_INTERVAL
        || !fit.hc0_variance.is_finite()
        || fit.hc0_variance <= 0.0
    {
        return None;
    }
    let radius = NORMAL_95 * fit.hc0_variance.sqrt();
    Some([fit.effect - radius, fit.effect + radius])
}

/// Column-major design of an intercept, the realized assignment and the covariates.
fn assignment_design(assignment: &[bool], covariates: &[&[f64]]) -> Vec<f64> {
    let n = assignment.len();
    let mut design = Vec::with_capacity(n * (covariates.len() + 2));
    design.extend(std::iter::repeat_n(1.0, n));
    design.extend(assignment.iter().map(|&value| f64::from(value)));
    for covariate in covariates {
        design.extend_from_slice(covariate);
    }
    design
}

const RANK_DEFICIENT: &str = "ANCOVA design is rank deficient or covariates are collinear";

/// Fit OLS with intercept, assignment, and pre-assignment covariates.
pub fn fit_ancova(
    outcome: &[f64],
    assignment: &[bool],
    covariates: &[&[f64]],
) -> Result<AncovaFit, &'static str> {
    let n = outcome.len();
    let k = covariates.len();
    let q = k + 2;
    if n <= q || assignment.len() != n || k == 0 || covariates.iter().any(|x| x.len() != n) {
        return Err(
            "ANCOVA requires aligned rows, at least one covariate, and residual degrees of freedom",
        );
    }
    if outcome.iter().chain(covariates.iter().flat_map(|x| x.iter())).any(|v| !v.is_finite()) {
        return Err("outcomes and covariates must be finite");
    }
    let treated = assignment.iter().filter(|&&value| value).count();
    let control = n - treated;
    if treated == 0 || control == 0 {
        return Err("ANCOVA requires observed treated and control units");
    }
    let design = assignment_design(assignment, covariates);
    // QR least squares for the coefficients and residuals (the normal equations square the
    // condition number); the HC0 bread is the shared scale-free `(X'X)^-1`.
    let fit = FaerBackend
        .least_squares(&design, n, q, outcome, &mut LeastSquaresWorkspace::default())
        .map_err(|_| RANK_DEFICIENT)?;
    let inverse = crate::util::xtx_inverse(&design, n, q).ok_or(RANK_DEFICIENT)?;
    let beta = fit.coefficients;
    // HC0 sandwich for the treatment coefficient: `Σ_i (e_i · [(X'X)^-1 x_i]_T)²`.
    let mut variance = 0.0;
    for (i, residual) in fit.residuals.iter().enumerate() {
        let leverage_row: f64 = (0..q).map(|j| inverse[q + j] * design[j * n + i]).sum();
        variance += (residual * leverage_row).powi(2);
    }
    let variance = variance.max(0.0);
    if !beta.iter().all(|v| v.is_finite()) || !variance.is_finite() {
        return Err("ANCOVA coefficients or variance are not finite");
    }
    Ok(AncovaFit {
        effect: beta[1],
        hc0_variance: variance,
        covariate_coefficients: beta[2..].to_vec(),
        treated,
        control,
    })
}

/// Fewest observed rows in each arm of a [`small_arm_interval_95`].
pub const MIN_UNITS_PER_ARM_FOR_SMALL_ARM_INTERVAL: usize = 2;

/// Fewest rows in all of a trial that publishes a [`small_arm_interval_95`]: a very small trial
/// stays point-only whatever its arms, and the construction is measured from 150 rows upward.
/// The function itself is exact for any sample with two rows per arm; the route applies this
/// floor.
pub const MIN_ROWS_FOR_SMALL_ARM_INTERVAL: usize = 100;

/// Covariates up to which the small-arm interval's coverage is measured, the envelope of the
/// Bernoulli interval. [`small_arm_interval_95`] publishes inside it;
/// [`small_arm_interval_any_covariates_95`] computes the same HC2 Student interval for any
/// covariate count, outside the measured envelope.
pub const MAX_COVARIATES_FOR_SMALL_ARM_INTERVAL: usize = 2;

/// Treatment coefficient with an HC2 Student interval for arms too small for a normal one.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct SmallArmInterval {
    /// Coefficient on realized assignment (the difference in means when no covariate is given).
    pub effect: f64,
    /// HC2 sandwich variance of the coefficient.
    pub variance: f64,
    /// Bell–McCaffrey degrees of freedom of the Student reference.
    pub degrees_of_freedom: f64,
    /// Pointwise 95% interval `effect ± t · √variance`.
    pub interval_95: [f64; 2],
}

/// Pointwise 95% interval of the treatment coefficient that holds in arms of any size above one.
///
/// The standard error is the HC2 sandwich, which corrects the leverage the normal interval's
/// HC0 omits, and the reference is Student-t with the Bell–McCaffrey degrees of freedom
/// (Bell–McCaffrey 2002; Imbens–Kolesár 2016, *Review of Economics and Statistics*
/// 98(4)). With no covariates the regression is the difference in means, the HC2 variance is
/// the per-arm sample-variance sum of Neyman, and the interval is Welch's with equal-variance
/// degrees of freedom. The interval conditions on the realized arm sizes, so it is also the
/// interval of a complete randomization with those sizes.
///
/// Returns `None` for misaligned or non-finite inputs, fewer than
/// [`MIN_UNITS_PER_ARM_FOR_SMALL_ARM_INTERVAL`] rows in an arm, more than
/// [`MAX_COVARIATES_FOR_SMALL_ARM_INTERVAL`] covariates (use
/// [`small_arm_interval_any_covariates_95`]), a rank-deficient design, a row of leverage 1, or
/// a variance that is not positive (constant outcomes within each arm).
#[must_use]
pub fn small_arm_interval_95(
    outcome: &[f64],
    assignment: &[bool],
    covariates: &[&[f64]],
) -> Option<SmallArmInterval> {
    if covariates.len() > MAX_COVARIATES_FOR_SMALL_ARM_INTERVAL {
        return None;
    }
    small_arm_interval_any_covariates_95(outcome, assignment, covariates)
}

/// [`small_arm_interval_95`] for any number of covariates. The HC2 leverage correction and the
/// Bell–McCaffrey degrees of freedom are computed for the fitted design, so the interval is
/// the same construction; its coverage is measured only up to
/// [`MAX_COVARIATES_FOR_SMALL_ARM_INTERVAL`] covariates, so beyond that it is not calibrated.
#[must_use]
pub fn small_arm_interval_any_covariates_95(
    outcome: &[f64],
    assignment: &[bool],
    covariates: &[&[f64]],
) -> Option<SmallArmInterval> {
    let n = outcome.len();
    let q = covariates.len() + 2;
    let treated = assignment.iter().filter(|&&value| value).count();
    if assignment.len() != n
        || covariates.iter().any(|x| x.len() != n)
        || n <= q
        || treated < MIN_UNITS_PER_ARM_FOR_SMALL_ARM_INTERVAL
        || n - treated < MIN_UNITS_PER_ARM_FOR_SMALL_ARM_INTERVAL
        || outcome.iter().chain(covariates.iter().flat_map(|x| x.iter())).any(|v| !v.is_finite())
    {
        return None;
    }
    let design = assignment_design(assignment, covariates);
    let fit = FaerBackend
        .least_squares(&design, n, q, outcome, &mut LeastSquaresWorkspace::default())
        .ok()?;
    // Rounding leaves residuals near 1e-16 of the outcome scale when every arm is constant; that
    // is zero variance, not a resolvable interval.
    let spread: f64 = fit.residuals.iter().map(|e| e * e).sum();
    let scale: f64 = outcome.iter().map(|y| y * y).sum();
    if spread.partial_cmp(&(1e-24 * scale)).is_none_or(std::cmp::Ordering::is_le) {
        return None;
    }
    let hc2 = hc2_coefficient_bell_mccaffrey(&design, n, q, &fit.residuals, 1).ok()?;
    let effect = fit.coefficients[1];
    let radius = student_t_ppf(0.975, hc2.degrees_of_freedom) * hc2.variance.sqrt();
    (effect.is_finite() && hc2.variance > 0.0 && radius.is_finite()).then_some(SmallArmInterval {
        effect,
        variance: hc2.variance,
        degrees_of_freedom: hc2.degrees_of_freedom,
        interval_95: [effect - radius, effect + radius],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_linear_effect_and_refuses_collinearity() {
        let x1 = [0., 1., 2., 3., 4., 5., 6., 7.];
        let x2 = [1., 0., 1., 0., 1., 0., 1., 0.];
        let assignment = [false, true, false, true, true, false, true, false];
        let y = (0..8)
            .map(|i| 3. + 2. * f64::from(assignment[i]) + 4. * x1[i] - x2[i])
            .collect::<Vec<_>>();
        let fit = fit_ancova(&y, &assignment, &[&x1, &x2]).unwrap();
        assert!((fit.effect - 2.).abs() < 1e-10);
        assert!(fit.hc0_variance < 1e-20);
        assert!(fit_ancova(&y, &assignment, &[&x1, &x1]).is_err());
    }

    #[test]
    fn collinearity_is_judged_scale_free() {
        // Covariates in units of 1e-9 put the Gram pivots far below the former absolute 1e-12
        // cutoff; the design is still perfectly conditioned, so the fit is the same one.
        let x1 = [0., 1., 2., 3., 4., 5., 6., 7.].map(|v| v * 1e-9);
        let x2 = [1., 0., 1., 0., 1., 0., 1., 0.].map(|v| v * 1e-9);
        let assignment = [false, true, false, true, true, false, true, false];
        let y = (0..8)
            .map(|i| 3. + 2. * f64::from(assignment[i]) + 4e9 * x1[i] - 1e9 * x2[i])
            .collect::<Vec<_>>();
        let fit = fit_ancova(&y, &assignment, &[&x1, &x2]).unwrap();
        assert!((fit.effect - 2.).abs() < 1e-8, "{}", fit.effect);
        assert!((fit.covariate_coefficients[0] - 4e9).abs() < 1.0);
        // A covariate that is an exact multiple of another is refused at the same scale.
        let doubled = x1.map(|v| 2.0 * v);
        let e = fit_ancova(&y, &assignment, &[&x1, &doubled]).unwrap_err();
        assert!(e.contains("rank deficient"), "{e}");
    }

    /// Cells `(rows, covariates, probability)` from the 400-row boundary of the earlier blanket rule
    /// down to the arm-support boundary: 150 rows at 0.5, a 2:1 allocation, and 200 rows at 0.2
    /// (exactly 40 expected treated rows), plus 300 and 600 rows at the probability boundaries.
    const ANCOVA_COVERAGE_CELLS: [(usize, usize, f64); 20] = [
        (400, 1, 0.2),
        (400, 1, 0.5),
        (400, 1, 0.8),
        (400, 2, 0.2),
        (400, 2, 0.5),
        (400, 2, 0.8),
        (150, 1, 0.5),
        (150, 2, 0.5),
        (150, 2, 1.0 / 3.0),
        (200, 2, 0.2),
        (200, 2, 0.8),
        (300, 1, 0.2),
        (300, 1, 0.5),
        (300, 2, 0.2),
        (300, 2, 0.5),
        (300, 2, 0.8),
        (300, 2, 1.0 / 3.0),
        (600, 1, 0.2),
        (600, 2, 0.5),
        (600, 2, 0.8),
    ];

    #[test]
    fn bernoulli_ancova_hc0_covers_known_finite_population_effect() {
        const REPLICATES: usize = 2_000;
        for (n, covariate_count, probability) in ANCOVA_COVERAGE_CELLS {
            let x1 = (0..n).map(|i| (0.17 * i as f64).sin()).collect::<Vec<_>>();
            let x2 = (0..n).map(|i| (0.11 * i as f64).cos()).collect::<Vec<_>>();
            let effect = (0..n).map(|i| 1.4 + 0.2 * x1[i]).sum::<f64>() / n as f64;
            let (mut covered, mut published) = (0, 0);
            for rep in 0..REPLICATES {
                let assignment = (0..n)
                    .map(|i| crate::splitmix::keyed_unit(rep as u64, i as u64) < probability)
                    .collect::<Vec<_>>();
                let outcome = (0..n)
                    .map(|i| {
                        2.0 + 1.2 * x1[i] - 0.7 * x2[i]
                            + 0.5 * (0.29 * i as f64).sin()
                            + f64::from(assignment[i]) * (1.4 + 0.2 * x1[i])
                    })
                    .collect::<Vec<_>>();
                let covariates: Vec<&[f64]> =
                    if covariate_count == 1 { vec![&x1] } else { vec![&x1, &x2] };
                let fit = fit_ancova(&outcome, &assignment, &covariates).unwrap();
                assert!(fit.hc0_variance > 0.0);
                // An allocation that realizes fewer than 30 rows in an arm publishes no interval;
                // coverage is scored over the allocations that do.
                let Some([lower, upper]) = calibrated_bernoulli_interval_95(&fit, probability)
                else {
                    continue;
                };
                published += 1;
                covered += usize::from(lower <= effect && effect <= upper);
            }
            assert!(published * 10 >= REPLICATES * 9, "n={n} p={probability}: {published}");
            let rate = covered as f64 / published as f64;
            eprintln!(
                "Bernoulli ANCOVA HC0 n={n} covariates={covariate_count} p={probability:.3} known-truth coverage: {covered}/{published} = {rate:.4}"
            );
            assert!((0.93..=0.985).contains(&rate));
        }
    }

    /// Measurement harness for the covariate count beyond the licence: the HC0 normal interval's
    /// known-truth coverage with three to six covariates under the cells of the licensed record.
    /// Run with `--ignored --nocapture`; the licence widens only for counts whose every measured
    /// cell falls inside the licensed coverage band.
    #[test]
    #[ignore = "measurement harness: prints coverage for covariate counts outside the licence"]
    fn bernoulli_ancova_hc0_coverage_beyond_two_covariates_is_measured_not_licensed() {
        const REPLICATES: usize = 2_000;
        for (n, covariate_count, probability) in
            [(400, 3, 0.5), (400, 4, 0.5), (600, 5, 0.5), (600, 6, 0.5), (400, 3, 0.2)]
        {
            let xs: Vec<Vec<f64>> = (0..covariate_count)
                .map(|j| (0..n).map(|i| (0.17 * (j + 1) as f64 * i as f64 + j as f64).sin()).collect())
                .collect();
            let refs: Vec<&[f64]> = xs.iter().map(Vec::as_slice).collect();
            let effect = (0..n).map(|i| 1.4 + 0.2 * xs[0][i]).sum::<f64>() / n as f64;
            let (mut covered, mut scored) = (0, 0);
            for rep in 0..REPLICATES {
                let assignment = (0..n)
                    .map(|i| crate::splitmix::keyed_unit(rep as u64, i as u64) < probability)
                    .collect::<Vec<_>>();
                let outcome = (0..n)
                    .map(|i| {
                        2.0 + 1.2 * xs[0][i] - 0.7 * xs[1][i]
                            + 0.5 * (0.29 * i as f64).sin()
                            + f64::from(assignment[i]) * (1.4 + 0.2 * xs[0][i])
                    })
                    .collect::<Vec<_>>();
                let Ok(fit) = fit_ancova(&outcome, &assignment, &refs) else { continue };
                let radius = 1.959_963_984_540_054 * fit.hc0_variance.sqrt();
                scored += 1;
                covered += usize::from((fit.effect - effect).abs() <= radius);
            }
            eprintln!(
                "Bernoulli ANCOVA HC0 n={n} covariates={covariate_count} p={probability:.3} known-truth coverage: {covered}/{scored} = {:.4}",
                covered as f64 / scored.max(1) as f64
            );
        }
    }

    #[test]
    fn calibrated_interval_refuses_sparse_low_probability_and_degenerate_variance() {
        let fit = AncovaFit {
            effect: 1.0,
            hc0_variance: 0.1,
            covariate_coefficients: vec![2.0],
            treated: 29,
            control: 371,
        };
        assert!(calibrated_bernoulli_interval_95(&fit, 0.5).is_none());
        let fit = AncovaFit { treated: 200, control: 200, ..fit };
        assert!(calibrated_bernoulli_interval_95(&fit, 0.1).is_none());
        assert!(calibrated_bernoulli_interval_95(&fit, 0.5).is_some());
        // 300 balanced rows are supported; a design expecting 39 treated rows is not, however many
        // the allocation happened to realize, and 40 expected rows are the boundary.
        let balanced = AncovaFit { treated: 150, control: 150, ..fit.clone() };
        assert!(calibrated_bernoulli_interval_95(&balanced, 0.5).is_some());
        let sparse_design = AncovaFit { treated: 45, control: 150, ..fit.clone() };
        assert!(calibrated_bernoulli_interval_95(&sparse_design, 0.2).is_none());
        let boundary = AncovaFit { treated: 45, control: 155, ..fit.clone() };
        assert!(calibrated_bernoulli_interval_95(&boundary, 0.2).is_some());
        let fit = AncovaFit { hc0_variance: 0.0, ..fit };
        assert!(calibrated_bernoulli_interval_95(&fit, 0.5).is_none());
        let fit = AncovaFit { hc0_variance: 0.1, covariate_coefficients: vec![0.0; 3], ..fit };
        assert!(calibrated_bernoulli_interval_95(&fit, 0.5).is_none());
    }

    /// Three treated and five control rows: s₁² = 7, s₀² = 5.5, effect 3 − 2 = 1.
    const TREATED: [f64; 3] = [1.0, 2.0, 6.0];
    const CONTROL: [f64; 5] = [0.0, 1.0, 1.0, 2.0, 6.0];

    fn two_arm_rows() -> (Vec<f64>, Vec<bool>) {
        let outcome = TREATED.iter().chain(CONTROL.iter()).copied().collect();
        let assignment = (0..8).map(|i| i < 3).collect();
        (outcome, assignment)
    }

    #[test]
    fn small_arm_difference_in_means_is_welch_with_the_per_arm_variance_sum() {
        let (outcome, assignment) = two_arm_rows();
        let fit = small_arm_interval_95(&outcome, &assignment, &[]).unwrap();
        assert!((fit.effect - 1.0).abs() < 1e-12);
        // HC2 of the two-group regression is s₁²/n₁ + s₀²/n₀.
        assert!((fit.variance - (7.0 / 3.0 + 5.5 / 5.0)).abs() < 1e-12, "{}", fit.variance);
        // Hand computation: (1/3 + 1/5)² / (1/(9·2) + 1/(25·4)) = 512/118.
        assert!((fit.degrees_of_freedom - 512.0 / 118.0).abs() < 1e-10);
        // Student-t critical value between t(5) = 2.5706 and t(4) = 2.7764 at df ≈ 4.34.
        let t = (fit.interval_95[1] - fit.effect) / fit.variance.sqrt();
        assert!(t > 2.5706 && t < 2.7764, "{t}");
        assert!(
            (fit.effect - fit.interval_95[0] - (fit.interval_95[1] - fit.effect)).abs() < 1e-12
        );
    }

    #[test]
    fn small_arm_interval_with_a_covariate_shares_the_ancova_coefficient() {
        let x = [0.3, 1.1, -0.4, 0.9, -1.2, 0.5, 0.0, 1.7];
        let noise = [0.2, -0.1, 0.3, -0.2, 0.1, 0.0, -0.3, 0.1];
        let (_, assignment) = two_arm_rows();
        let y: Vec<f64> =
            (0..8).map(|i| 2.0 + 1.5 * f64::from(assignment[i]) + 0.8 * x[i] + noise[i]).collect();
        let fit = small_arm_interval_95(&y, &assignment, &[&x]).unwrap();
        let full = fit_ancova(&y, &assignment, &[&x]).unwrap();
        assert!((fit.effect - full.effect).abs() < 1e-10);
        // HC2 divides each squared residual by 1 − h_i < 1, so it never falls below HC0.
        assert!(fit.variance >= full.hc0_variance);
        // The quadratic form has rank at most n − p = 5.
        assert!(fit.degrees_of_freedom > 0.0 && fit.degrees_of_freedom <= 5.0 + 1e-9);
    }

    #[test]
    fn small_arm_interval_is_refused_below_the_minimum_arm_and_for_unsupported_inputs() {
        let (outcome, assignment) = two_arm_rows();
        let one_treated: Vec<bool> = (0..8).map(|i| i == 0).collect();
        assert!(small_arm_interval_95(&outcome, &one_treated, &[]).is_none());
        let one_control: Vec<bool> = (0..8).map(|i| i != 0).collect();
        assert!(small_arm_interval_95(&outcome, &one_control, &[]).is_none());
        // Two rows per arm is the minimum supported.
        let arms = [true, true, false, false];
        assert!(small_arm_interval_95(&[1.0, 3.0, 0.0, 2.0], &arms, &[]).is_some());
        // Constant outcomes within each arm have no variance to build an interval from.
        assert!(small_arm_interval_95(&[3.0, 3.0, 1.0, 1.0], &arms, &[]).is_none());
        // More than two covariates, misaligned and non-finite inputs are refused.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5];
        assert!(small_arm_interval_95(&outcome, &assignment, &[&x, &x, &x]).is_none());
        assert!(small_arm_interval_95(&outcome[..7], &assignment, &[]).is_none());
        let mut bad = outcome.clone();
        bad[0] = f64::NAN;
        assert!(small_arm_interval_95(&bad, &assignment, &[]).is_none());
    }
}
