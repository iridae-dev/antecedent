//! Independent oracles for summary-data synthesis: hand-calculated pooling,
//! DerSimonian-Laird on a worked example, REML against a likelihood grid,
//! hierarchical borrowing against numerical integration, and the dependence
//! envelope against a brute-force correlation grid.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::synthesis::{
    BetweenStudyMethod, MeanPrior, SynthesisError, TauPrior, check_covariance, equicorrelated_gls,
    fixed_effect_gls, fixed_effect_inverse_variance, hierarchical_normal_borrowing,
    pooled_dependence_envelope, random_effects_dersimonian_laird, random_effects_reml,
};

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: expected {expected}, got {actual} (tolerance {tolerance})"
    );
}

/// Gaussian elimination with partial pivoting: solves `a x = b` (row-major
/// `k x k`). Deliberately unrelated to the library's Cholesky path.
fn solve_dense(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let mut m: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            let mut row = a[i * k..(i + 1) * k].to_vec();
            row.push(b[i]);
            row
        })
        .collect();
    for col in 0..k {
        let pivot =
            (col..k).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs())).expect("pivot");
        m.swap(col, pivot);
        for row in (col + 1)..k {
            let factor = m[row][col] / m[col][col];
            for c in col..=k {
                let value = m[col][c];
                m[row][c] -= factor * value;
            }
        }
    }
    let mut x = vec![0.0; k];
    for i in (0..k).rev() {
        let mut sum = m[i][k];
        for (j, &xj) in x.iter().enumerate().take(k).skip(i + 1) {
            sum -= m[i][j] * xj;
        }
        x[i] = sum / m[i][i];
    }
    x
}

/// Brute-force GLS for an equicorrelated covariance: `(estimate, variance)`.
fn brute_force_gls(y: &[f64], s: &[f64], rho: f64) -> (f64, f64) {
    let k = y.len();
    let mut cov = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            cov[i * k + j] = if i == j { s[i] * s[i] } else { rho * s[i] * s[j] };
        }
    }
    let ones = vec![1.0; k];
    let x = solve_dense(&cov, &ones, k);
    let denominator: f64 = x.iter().sum();
    let estimate: f64 = x.iter().zip(y).map(|(w, v)| w * v).sum::<f64>() / denominator;
    (estimate, 1.0 / denominator)
}

// -- fixed effect --------------------------------------------------------------------------

/// y = (1, 3), standard errors (1, 2), so S = [[1, 2 rho], [2 rho, 4]].
///
/// rho = 0: weights (4/5, 1/5), estimate 4/5 + 3/5 = 7/5, variance 1/(1 + 1/4) = 4/5.
/// rho = 1/4: S = [[1, 1/2], [1/2, 4]], det 15/4, S^-1 1 = (7/2, 1/2)/(15/4), weights
/// (7/8, 1/8), estimate 7/8 + 3/8 = 5/4, variance (15/4)/4 = 15/16.
/// rho = 1/2: S = [[1, 1], [1, 4]], det 3, 1'S^-1 = (3, 0)/3, weights (1, 0), estimate 1,
/// variance 1.
#[test]
fn generalized_least_squares_matches_the_two_study_closed_form() {
    let y = [1.0, 3.0];
    let cases = [
        (0.0, 1.4, 0.8, [0.8, 0.2]),
        (0.25, 1.25, 0.9375, [0.875, 0.125]),
        (0.5, 1.0, 1.0, [1.0, 0.0]),
    ];
    for (rho, estimate, variance, weights) in cases {
        let cov = [1.0, 2.0 * rho, 2.0 * rho, 4.0];
        let fit = fixed_effect_gls(&y, &cov).expect("positive definite");
        close(fit.estimate, estimate, 1e-12, "estimate");
        close(fit.variance, variance, 1e-12, "variance");
        close(fit.weights[0], weights[0], 1e-12, "weight 0");
        close(fit.weights[1], weights[1], 1e-12, "weight 1");
        let (e, v) = equicorrelated_gls(&y, &[1.0, 2.0], rho).expect("closed form");
        close(e, estimate, 1e-12, "equicorrelated estimate");
        close(v, variance, 1e-12, "equicorrelated variance");
    }
}

/// Q = sum w (y - mu)^2 = 1 * 0.4^2 + (1/4) * 1.6^2 = 0.16 + 0.64 = 0.8, df = 1.
#[test]
fn inverse_variance_pooling_and_q_match_hand_values() {
    let fit = fixed_effect_inverse_variance(&[1.0, 3.0], &[1.0, 4.0]).expect("pool");
    close(fit.estimate, 1.4, 1e-12, "estimate");
    close(fit.variance, 0.8, 1e-12, "variance");
    close(fit.heterogeneity_q, 0.8, 1e-12, "Q");
    assert_eq!(fit.degrees_of_freedom, 1);
    let diagonal = fixed_effect_gls(&[1.0, 3.0], &[1.0, 0.0, 0.0, 4.0]).expect("diagonal GLS");
    close(diagonal.estimate, fit.estimate, 1e-12, "GLS diagonal estimate");
    close(diagonal.heterogeneity_q, 0.8, 1e-12, "GLS diagonal Q");
}

#[test]
fn pooling_is_invariant_to_study_order() {
    let y = [0.3, -1.2, 2.5, 0.9];
    let s = [0.5, 1.1, 0.8, 2.0];
    let cov = |order: &[usize]| {
        let k = order.len();
        let mut c = vec![0.0; k * k];
        for (a, &i) in order.iter().enumerate() {
            for (b, &j) in order.iter().enumerate() {
                c[a * k + b] = if i == j { s[i] * s[i] } else { 0.3 * s[i] * s[j] };
            }
        }
        c
    };
    let forward = [0, 1, 2, 3];
    let shuffled = [2, 0, 3, 1];
    let y2: Vec<f64> = shuffled.iter().map(|&i| y[i]).collect();
    let a = fixed_effect_gls(&y, &cov(&forward)).expect("fit");
    let b = fixed_effect_gls(&y2, &cov(&shuffled)).expect("fit");
    close(a.estimate, b.estimate, 1e-12, "estimate");
    close(a.variance, b.variance, 1e-12, "variance");
    close(a.heterogeneity_q, b.heterogeneity_q, 1e-10, "Q");
}

#[test]
fn covariance_must_be_positive_semidefinite_and_definite() {
    let y = [1.0, 2.0];
    // eigenvalues 3 and -1
    assert!(matches!(
        fixed_effect_gls(&y, &[1.0, 2.0, 2.0, 1.0]),
        Err(SynthesisError::NotPositiveSemidefinite { .. })
    ));
    // perfectly correlated: semidefinite, singular
    assert!(matches!(
        fixed_effect_gls(&y, &[1.0, 1.0, 1.0, 1.0]),
        Err(SynthesisError::Singular { .. })
    ));
    assert!(matches!(
        fixed_effect_gls(&y, &[1.0, 0.5, 0.4, 1.0]),
        Err(SynthesisError::NotSymmetric { .. })
    ));
    assert!(matches!(
        fixed_effect_gls(&y, &[1.0, f64::NAN, f64::NAN, 1.0]),
        Err(SynthesisError::NonFinite(_))
    ));
    assert!(matches!(fixed_effect_gls(&y, &[1.0, 0.0, 0.0]), Err(SynthesisError::Shape(_))));
    // the check alone reports the structure
    let check = check_covariance(&[1.0, 0.0, 0.0, 4.0], 2).expect("diagonal");
    assert!(!check.has_cross_covariance);
    close(check.min_eigenvalue, 1.0, 1e-12, "min eigenvalue");
    close(check.max_eigenvalue, 4.0, 1e-12, "max eigenvalue");
    let check = check_covariance(&[1.0, 1.0, 1.0, 1.0], 2).expect("semidefinite is accepted");
    assert!(check.has_cross_covariance);
    assert!(matches!(
        fixed_effect_inverse_variance(&[1.0, 2.0], &[1.0, 0.0]),
        Err(SynthesisError::NonPositiveVariance(_))
    ));
}

// -- random effects ------------------------------------------------------------------------

/// y = (0, 2, 6), v = (1, 2, 4); w = (1, 1/2, 1/4), W = 7/4.
/// `mu_FE` = (0 + 1 + 3/2)/(7/4) = 10/7; Q = sum w y^2 - (sum w y)^2 / W
///   = (0 + 2 + 9) - (5/2)^2/(7/4) = 11 - 25/7 = 52/7; df = 2;
/// C = W - sum w^2 / W = 7/4 - (21/16)/(7/4) = 7/4 - 3/4 = 1;
/// tau^2 = (52/7 - 2)/1 = 38/7 = 5.428571...
/// RE weights 1/(v + 38/7) = 7/45, 7/52, 7/66, so
/// `mu_RE` = 3330/1457 = 2.285518188..., variance 25740/10199 = 2.523776841...
/// I^2 = (Q - df)/Q = 19/26.
#[test]
fn dersimonian_laird_matches_the_hand_derivation() {
    let fit = random_effects_dersimonian_laird(&[0.0, 2.0, 6.0], &[1.0, 2.0, 4.0]).expect("fit");
    assert_eq!(fit.method, BetweenStudyMethod::DerSimonianLaird);
    close(fit.q, 52.0 / 7.0, 1e-12, "Q");
    assert_eq!(fit.degrees_of_freedom, 2);
    close(fit.tau_squared, 38.0 / 7.0, 1e-12, "tau^2");
    close(fit.estimate, 3330.0 / 1457.0, 1e-12, "estimate");
    close(fit.variance, 25740.0 / 10199.0, 1e-12, "variance");
    close(fit.i_squared, 19.0 / 26.0, 1e-12, "I^2");
    close(fit.weights.iter().sum::<f64>(), 1.0, 1e-12, "weights sum");
}

/// Homogeneous studies: Q below its degrees of freedom truncates tau^2 at zero and the
/// random-effects result equals the fixed-effect result.
#[test]
fn homogeneous_studies_truncate_tau_squared_at_zero() {
    let y = [1.0, 1.1, 0.9];
    let v = [1.0, 1.0, 1.0];
    let re = random_effects_dersimonian_laird(&y, &v).expect("fit");
    let fe = fixed_effect_inverse_variance(&y, &v).expect("fit");
    close(re.tau_squared, 0.0, 0.0, "tau^2");
    close(re.estimate, fe.estimate, 1e-12, "estimate");
    close(re.variance, fe.variance, 1e-12, "variance");
    let reml = random_effects_reml(&y, &v).expect("fit");
    close(reml.tau_squared, 0.0, 0.0, "REML tau^2");
}

/// For k = 2 the restricted likelihood depends on u = v1 + v2 + 2 tau^2 only through
/// log u + d^2/u, minimised at u = d^2, so tau^2 = (d^2 - (v1 + v2)) / 2; the method of
/// moments gives the same number: y = (0, 3), v = (1, 1): (9 - 2)/2 = 7/2.
#[test]
fn two_study_reml_and_moments_agree_with_the_closed_form() {
    let y = [0.0, 3.0];
    let v = [1.0, 1.0];
    let reml = random_effects_reml(&y, &v).expect("fit");
    let dl = random_effects_dersimonian_laird(&y, &v).expect("fit");
    close(reml.tau_squared, 3.5, 1e-10, "REML tau^2");
    close(dl.tau_squared, 3.5, 1e-12, "DL tau^2");
    // equal variances: the pooled mean is the midpoint for any tau^2
    close(reml.estimate, 1.5, 1e-12, "estimate");
    // variance = 1 / (2 / (1 + 3.5)) = 9/4
    close(reml.variance, 2.25, 1e-9, "variance");
}

/// k = 3 REML against a brute-force maximisation of the restricted log-likelihood
/// `-1/2 [sum log(v + t) + log W + sum w (y - mu)^2]` on a fine grid (step 1e-4).
#[test]
fn reml_matches_a_likelihood_grid() {
    let y = [0.0, 2.0, 6.0];
    let v = [1.0, 2.0, 4.0];
    let log_likelihood = |t: f64| {
        let w: Vec<f64> = v.iter().map(|a| 1.0 / (a + t)).collect();
        let total: f64 = w.iter().sum();
        let mu: f64 = w.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>() / total;
        -0.5 * (v.iter().map(|a| (a + t).ln()).sum::<f64>()
            + total.ln()
            + w.iter().zip(&y).map(|(a, b)| a * (b - mu) * (b - mu)).sum::<f64>())
    };
    let mut best = (0.0, f64::NEG_INFINITY);
    for step in 0..=300_000 {
        let t = f64::from(step) * 1e-4;
        let value = log_likelihood(t);
        if value > best.1 {
            best = (t, value);
        }
    }
    let fit = random_effects_reml(&y, &v).expect("fit");
    close(fit.tau_squared, best.0, 2e-4, "REML tau^2 vs grid");
    // independent value from a separate bisection of the score: 6.1442331122877
    close(fit.tau_squared, 6.144_233_112_287_695, 1e-9, "REML tau^2");
    close(fit.estimate, 2.316_509_187_449_771, 1e-9, "REML estimate");
    close(fit.variance, 2.767_495_902_004_193, 1e-9, "REML variance");
}

#[test]
fn random_effects_need_two_studies() {
    assert!(matches!(
        random_effects_dersimonian_laird(&[1.0], &[1.0]),
        Err(SynthesisError::TooFewStudies { needed: 2, got: 1 })
    ));
    assert!(matches!(
        random_effects_reml(&[1.0], &[1.0]),
        Err(SynthesisError::TooFewStudies { needed: 2, got: 1 })
    ));
}

// -- hierarchical borrowing ----------------------------------------------------------------

/// One support point tau = 1, flat mean prior, y = (0, 2, 6), v = (1, 2, 4):
/// w = 1/(v + 1) = (1/2, 1/3, 1/5), W = 31/30, `mu_hat` = (0 + 2/3 + 6/5)/(31/30) = 56/31,
/// variance 30/31. Study i: B = 1/(1 + `v_i`); mean B y + (1 - B) `mu_hat`,
/// variance B v + (1 - B)^2 30/31 (0.9032..., 0.7419...; 1.8710, 1.0968; 2.6452, 1.4194).
#[test]
fn hierarchical_borrowing_at_a_single_tau_matches_the_closed_form() {
    let prior = TauPrior { support: vec![1.0], probabilities: vec![1.0] };
    let fit =
        hierarchical_normal_borrowing(&[0.0, 2.0, 6.0], &[1.0, 2.0, 4.0], MeanPrior::Flat, &prior)
            .expect("fit");
    close(fit.mean, 56.0 / 31.0, 1e-12, "mean");
    close(fit.variance, 30.0 / 31.0, 1e-12, "variance");
    let expected_means =
        [0.903_225_806_451_612_9, 1.870_967_741_935_483_8, 2.645_161_290_322_580_6];
    let expected_variances =
        [0.741_935_483_870_967_7, 1.096_774_193_548_387, 1.419_354_838_709_677_5];
    for i in 0..3 {
        close(fit.study_means[i], expected_means[i], 1e-12, "study mean");
        close(fit.study_variances[i], expected_variances[i], 1e-12, "study variance");
    }
    close(fit.tau_posterior[0], 1.0, 0.0, "posterior mass");
}

/// A three-point tau prior (0, 1, 3 with probabilities 0.2, 0.5, 0.3). The posterior over
/// tau is proportional to prior x marginal likelihood, the marginal likelihood being the
/// integral of prod `N(y_i`; mu, `v_i` + tau^2) over mu (flat prior) or against N(1, 4); the
/// expected numbers come from Simpson integration with 40001 nodes over `mu_hat` +- 40 sd,
/// independent of the closed-form `log` marginal in the library.
#[test]
fn hierarchical_mixture_matches_numerical_integration() {
    let y = [0.0, 2.0, 6.0];
    let v = [1.0, 2.0, 4.0];
    let prior = TauPrior { support: vec![0.0, 1.0, 3.0], probabilities: vec![0.2, 0.5, 0.3] };

    let flat = hierarchical_normal_borrowing(&y, &v, MeanPrior::Flat, &prior).expect("fit");
    let flat_post = [0.086_654_132_365_457_08, 0.452_320_524_159_984_25, 0.461_025_343_474_558_65];
    for (a, b) in flat.tau_posterior.iter().zip(flat_post) {
        close(*a, b, 1e-9, "flat tau posterior");
    }
    close(flat.mean, 2.048_310_559_640_661, 1e-9, "flat mean");
    close(flat.variance, 2.326_010_099_248_764, 1e-9, "flat variance");
    close(flat.study_means[0], 0.643_081_568_029_725_4, 1e-9, "flat study 0 mean");

    let normal = hierarchical_normal_borrowing(
        &y,
        &v,
        MeanPrior::Normal { mean: 1.0, variance: 4.0 },
        &prior,
    )
    .expect("fit");
    let normal_post = [0.105_702_436_657_962_72, 0.505_806_923_033_306_8, 0.388_490_640_308_730_43];
    for (a, b) in normal.tau_posterior.iter().zip(normal_post) {
        close(*a, b, 1e-9, "normal tau posterior");
    }
    close(normal.mean, 1.649_812_318_907_936, 1e-9, "normal mean");
    close(normal.variance, 1.207_401_505_896_837_3, 1e-9, "normal variance");
    close(normal.study_means[0], 0.629_489_188_035_437_9, 1e-9, "normal study 0 mean");
}

#[test]
fn hierarchical_prior_declarations_are_validated() {
    let y = [0.0, 2.0];
    let v = [1.0, 1.0];
    let bad = [
        TauPrior { support: vec![], probabilities: vec![] },
        TauPrior { support: vec![1.0, 2.0], probabilities: vec![0.5] },
        TauPrior { support: vec![-1.0], probabilities: vec![1.0] },
        TauPrior { support: vec![1.0, 2.0], probabilities: vec![0.5, 0.6] },
        TauPrior { support: vec![1.0, 2.0], probabilities: vec![1.0, 0.0] },
    ];
    for prior in bad {
        assert!(matches!(
            hierarchical_normal_borrowing(&y, &v, MeanPrior::Flat, &prior),
            Err(SynthesisError::InvalidPrior(_))
        ));
    }
    let ok = TauPrior { support: vec![1.0], probabilities: vec![1.0] };
    assert!(matches!(
        hierarchical_normal_borrowing(&y, &v, MeanPrior::Normal { mean: 0.0, variance: 0.0 }, &ok),
        Err(SynthesisError::InvalidPrior(_))
    ));
}

// -- dependence envelope -------------------------------------------------------------------

/// s = (1, 2), y = (1, 3): V(rho) = 4 (1 - rho^2) / (5 - 4 rho), dV/drho has the sign of
/// 16 rho^2 - 40 rho + 16 = 8 (2 rho - 1)(rho - 2): the variance rises to 1 at rho = 1/2
/// (an interior maximum equal to the smaller variance s1^2) and falls after it.
/// mu(rho) = (1 + 3 ... ) is monotone: 7/5 at 0, 1 at 1/2, -1/7 at 9/10.
/// Over [0, 9/10]: estimate in [-1/7, 7/5]; variance in [19/35, 1] with the maximum at 1/2.
#[test]
fn the_envelope_finds_an_interior_variance_extreme() {
    let envelope =
        pooled_dependence_envelope(&[1.0, 3.0], &[1.0, 2.0], 0.0, 0.9).expect("envelope");
    assert_eq!(envelope.label(), "assumption_range");
    close(envelope.estimate_min, -1.0 / 7.0, 1e-12, "estimate min");
    close(envelope.rho_at_estimate_min, 0.9, 0.0, "estimate min at");
    close(envelope.estimate_max, 1.4, 1e-12, "estimate max");
    close(envelope.rho_at_estimate_max, 0.0, 0.0, "estimate max at");
    close(envelope.variance_max, 1.0, 1e-12, "variance max");
    close(envelope.rho_at_variance_max, 0.5, 1e-12, "variance max at");
    close(envelope.variance_min, 19.0 / 35.0, 1e-12, "variance min");
    close(envelope.rho_at_variance_min, 0.9, 0.0, "variance min at");
}

/// Compare the exact envelope with a 20001-point brute-force grid of full-matrix GLS.
/// The exact extreme is at least as extreme as any grid value and within the grid's
/// second-order resolution of the best grid value.
#[test]
fn the_envelope_agrees_with_brute_force_grids() {
    let cases: [(&[f64], &[f64], f64, f64); 4] = [
        (&[1.0, 3.0], &[1.0, 2.0], 0.2, 0.8),
        (&[0.5, -0.4, 1.7], &[0.6, 1.3, 2.2], -0.4, 0.7),
        (&[2.0, 1.0, 4.0, 0.0], &[1.0, 0.5, 1.5, 3.0], -0.3, 0.85),
        (&[2.0, 1.0, 4.0, 0.0], &[1.0, 0.5, 1.5, 3.0], 0.05, 0.95),
    ];
    for (y, s, lo, hi) in cases {
        let envelope = pooled_dependence_envelope(y, s, lo, hi).expect("envelope");
        let mut grid_est = (f64::INFINITY, f64::NEG_INFINITY);
        let mut grid_var = (f64::INFINITY, f64::NEG_INFINITY);
        let n = 20_000;
        for step in 0..=n {
            let rho = lo + (hi - lo) * f64::from(step) / f64::from(n);
            let (e, v) = brute_force_gls(y, s, rho);
            grid_est = (grid_est.0.min(e), grid_est.1.max(e));
            grid_var = (grid_var.0.min(v), grid_var.1.max(v));
        }
        let slack = 1e-6;
        assert!(envelope.estimate_min <= grid_est.0 + 1e-12);
        assert!(envelope.estimate_max >= grid_est.1 - 1e-12);
        assert!(envelope.variance_min <= grid_var.0 + 1e-12);
        assert!(envelope.variance_max >= grid_var.1 - 1e-12);
        close(envelope.estimate_min, grid_est.0, slack, "estimate min vs grid");
        close(envelope.estimate_max, grid_est.1, slack, "estimate max vs grid");
        close(envelope.variance_min, grid_var.0, slack, "variance min vs grid");
        close(envelope.variance_max, grid_var.1, slack, "variance max vs grid");
        // the reported argmax reproduces the reported extreme through the full matrix
        let (e, v) = brute_force_gls(y, s, envelope.rho_at_variance_max);
        close(v, envelope.variance_max, 1e-10, "variance at its argmax");
        let _ = e;
    }
}

#[test]
fn the_envelope_is_invariant_to_study_order() {
    let a = pooled_dependence_envelope(&[1.0, 3.0, -2.0], &[1.0, 2.0, 0.7], -0.2, 0.8)
        .expect("envelope");
    let b = pooled_dependence_envelope(&[-2.0, 1.0, 3.0], &[0.7, 1.0, 2.0], -0.2, 0.8)
        .expect("envelope");
    close(a.estimate_min, b.estimate_min, 1e-12, "estimate min");
    close(a.estimate_max, b.estimate_max, 1e-12, "estimate max");
    close(a.variance_min, b.variance_min, 1e-12, "variance min");
    close(a.variance_max, b.variance_max, 1e-12, "variance max");
}

#[test]
fn the_envelope_refuses_degenerate_and_non_definite_ranges() {
    let y = [1.0, 3.0, 2.0];
    let s = [1.0, 2.0, 1.5];
    // single correlation is a covariance declaration, not a range
    assert!(matches!(
        pooled_dependence_envelope(&y, &s, 0.3, 0.3),
        Err(SynthesisError::InvalidCorrelationRange(_))
    ));
    // rho = 1 is singular
    assert!(matches!(
        pooled_dependence_envelope(&y, &s, 0.0, 1.0),
        Err(SynthesisError::InvalidCorrelationRange(_))
    ));
    // k = 3: the equicorrelation matrix is positive definite only above -1/2
    assert!(matches!(
        pooled_dependence_envelope(&y, &s, -0.5, 0.2),
        Err(SynthesisError::InvalidCorrelationRange(_))
    ));
    assert!(matches!(
        pooled_dependence_envelope(&y, &s, f64::NAN, 0.2),
        Err(SynthesisError::InvalidCorrelationRange(_))
    ));
    assert!(matches!(
        pooled_dependence_envelope(&[1.0], &[1.0], 0.0, 0.5),
        Err(SynthesisError::TooFewStudies { .. })
    ));
    assert!(matches!(
        pooled_dependence_envelope(&y, &[1.0, 0.0, 1.0], 0.0, 0.5),
        Err(SynthesisError::NonPositiveVariance(_))
    ));
    assert!(matches!(
        equicorrelated_gls(&y, &s, 1.0),
        Err(SynthesisError::InvalidCorrelationRange(_))
    ));
}
