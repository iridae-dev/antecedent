//! Independent-unit randomized ANCOVA point and HC0 sandwich variance.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

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
/// The coefficient and HC0 variance remain available below this measured
/// support boundary; an interval does not.
#[must_use]
pub fn calibrated_bernoulli_interval_95(fit: &AncovaFit, probability: f64) -> Option<[f64; 2]> {
    use antecedent_stats::NORMAL_Q975 as NORMAL_95;
    if fit.covariate_coefficients.is_empty()
        || fit.covariate_coefficients.len() > 2
        || fit.treated + fit.control < 400
        || fit.treated < 30
        || fit.control < 30
        || !probability.is_finite()
        || probability + 1e-12 < 0.2
        || probability > 0.8 + 1e-12
        || !fit.hc0_variance.is_finite()
        || fit.hc0_variance <= 0.0
    {
        return None;
    }
    let radius = NORMAL_95 * fit.hc0_variance.sqrt();
    Some([fit.effect - radius, fit.effect + radius])
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
    let mut design = Vec::with_capacity(n * q);
    design.extend(std::iter::repeat_n(1.0, n));
    design.extend(assignment.iter().map(|&value| f64::from(value)));
    for covariate in covariates {
        design.extend_from_slice(covariate);
    }
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

    #[test]
    fn bernoulli_ancova_hc0_covers_known_finite_population_effect() {
        const REPLICATES: usize = 2_000;
        const N: usize = 400;
        let x1 = (0..N).map(|i| (0.17 * i as f64).sin()).collect::<Vec<_>>();
        let x2 = (0..N).map(|i| (0.11 * i as f64).cos()).collect::<Vec<_>>();
        let effect = (0..N).map(|i| 1.4 + 0.2 * x1[i]).sum::<f64>() / N as f64;
        for (covariate_count, probability) in
            [(1, 0.2), (1, 0.5), (1, 0.8), (2, 0.2), (2, 0.5), (2, 0.8)]
        {
            let mut covered = 0;
            for rep in 0..REPLICATES {
                let assignment = (0..N)
                    .map(|i| crate::splitmix::keyed_unit(rep as u64, i as u64) < probability)
                    .collect::<Vec<_>>();
                let outcome = (0..N)
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
                let [lower, upper] = calibrated_bernoulli_interval_95(&fit, probability).unwrap();
                covered += usize::from(lower <= effect && effect <= upper);
            }
            let rate = covered as f64 / REPLICATES as f64;
            eprintln!(
                "Bernoulli ANCOVA HC0 covariates={covariate_count} p={probability} known-truth coverage: {covered}/{REPLICATES} = {rate:.4}"
            );
            assert!((0.93..=0.985).contains(&rate));
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
        let fit = AncovaFit { hc0_variance: 0.0, ..fit };
        assert!(calibrated_bernoulli_interval_95(&fit, 0.5).is_none());
        let fit = AncovaFit { hc0_variance: 0.1, covariate_coefficients: vec![0.0; 3], ..fit };
        assert!(calibrated_bernoulli_interval_95(&fit, 0.5).is_none());
    }
}
