//! Independent exact checks for the two fixed-covariate (CUPED) randomized rows.
//!
//! * `cuped_ht_effect` (Bernoulli, known propensity): the pooled slope, the adjusted
//!   Horvitz-Thompson contrast and its independent-score standard error are computed here from
//!   their definitions on a design where the covariate is balanced across arms, so the slope and
//!   the effect are analytic: with `y = 2 a + 3 x` and `x` orthogonal to `a`, the slope is 3 and
//!   the adjusted outcome is `2 a + 3 mean(x)`, so the contrast is exactly 2.
//! * `complete_cluster_itt` on covariate-adjusted cluster totals (complete cluster
//!   randomization): every complete allocation of 4 of 8 unequal clusters is enumerated. For any
//!   fixed coefficient the estimator is exactly unbiased for the unadjusted per-row ITT, its
//!   exact variance is Neyman's `S1^2/m + S0^2/m - S_tau^2/K` over the adjusted totals, and the
//!   mean of the reported variance bound exceeds that variance by exactly `S_tau^2/K`.
//!
//! The coverage simulations in the library modules are calibration evidence and are a separate
//! claim.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::randomized_neyman::complete_cluster_itt;
use antecedent_estimate::randomized_scores::cuped_ht_effect;

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol * (1.0 + want.abs()), "{what}: got {got}, want {want}");
}

#[test]
fn cuped_ht_contrast_slope_and_standard_error_match_the_closed_form() {
    let assignment = [true, false, true, false, true, false, true, false];
    let covariate = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]; // same distribution in both arms
    let outcome: Vec<f64> =
        assignment.iter().zip(&covariate).map(|(&a, &x)| 2.0 * f64::from(a) + 3.0 * x).collect();
    let got = cuped_ht_effect(&outcome, &covariate, &assignment, &[0.5]).unwrap();

    // Slope: pooled Cov(y, x) / Var(x) = 3, since a is uncorrelated with x.
    close(got.beta, 3.0, 1e-12, "pooled slope");
    // Adjusted outcome 2a + 3 mean(x) = 2a + 7.5: treated 9.5, control 7.5. Scores are
    // 9.5 / 0.5 = 19 and -7.5 / 0.5 = -15 and the contrast is (4 * 19 - 4 * 15) / 8 = 2.
    close(got.effect, 2.0, 1e-12, "adjusted contrast equals the planted effect");
    // Variance of the mean score: sum (score - 2)^2 / (n (n - 1)) = 8 * 17^2 / 56 = 289 / 7.
    close(got.standard_error, (289.0_f64 / 7.0).sqrt(), 1e-12, "score standard error");

    // The same quantities from the definitions with unbalanced, non-constant propensities.
    let outcome2 = [1.0, 4.0, 2.5, 3.0, 7.0, 2.0, 5.5, 0.5];
    let covariate2 = [0.5, 2.0, 1.0, 3.5, 4.0, 1.5, 2.5, 0.0];
    let assignment2 = [true, false, true, true, false, false, true, false];
    let p = [0.5, 0.4, 0.6, 0.5, 0.3, 0.5, 0.45, 0.55];
    let n = 8.0;
    let (my, mx) = (outcome2.iter().sum::<f64>() / n, covariate2.iter().sum::<f64>() / n);
    let beta = outcome2.iter().zip(&covariate2).map(|(y, x)| (y - my) * (x - mx)).sum::<f64>()
        / covariate2.iter().map(|x| (x - mx).powi(2)).sum::<f64>();
    let scores: Vec<f64> = (0..8)
        .map(|i| {
            let adjusted = outcome2[i] - beta * (covariate2[i] - mx);
            if assignment2[i] { adjusted / p[i] } else { -adjusted / (1.0 - p[i]) }
        })
        .collect();
    let effect = scores.iter().sum::<f64>() / n;
    let variance = scores.iter().map(|s| (s - effect).powi(2)).sum::<f64>() / (n * (n - 1.0));
    let got = cuped_ht_effect(&outcome2, &covariate2, &assignment2, &p).unwrap();
    close(got.beta, beta, 1e-12, "slope");
    close(got.effect, effect, 1e-12, "contrast");
    close(got.standard_error, variance.sqrt(), 1e-12, "standard error");
}

fn variance(values: &[f64]) -> f64 {
    let m = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

#[test]
fn covariate_adjusted_cluster_totals_have_the_exact_neyman_mean_and_variance() {
    let sizes = [2_usize, 3, 2, 4, 3, 2, 3, 3];
    let covariate = [1.5, 2.0, 0.5, 3.5, 2.5, 1.0, 3.0, 2.0]; // fixed pre-assignment totals
    let control = [3.0, 5.5, 1.0, 9.0, 6.0, 2.5, 7.5, 5.0];
    let effect = [1.0, 2.5, 0.5, 4.0, 2.0, 0.0, 3.5, 1.5]; // heterogeneous unit-total effects
    let treated: Vec<f64> = control.iter().zip(&effect).map(|(c, e)| c + e).collect();
    let k = sizes.len();
    let rows: usize = sizes.iter().sum();
    let scale = k as f64 / rows as f64;
    let truth = effect.iter().sum::<f64>() / rows as f64;

    for theta in [0.0, 0.7, -1.3, 2.4] {
        let adj1: Vec<f64> = treated.iter().zip(&covariate).map(|(y, x)| y - theta * x).collect();
        let adj0: Vec<f64> = control.iter().zip(&covariate).map(|(y, x)| y - theta * x).collect();
        let (mut count, mut mean_effect, mut mean_square, mut mean_bound) =
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for mask in 0_u32..(1 << k) {
            if mask.count_ones() != 4 {
                continue;
            }
            let pick = |totals: &[f64], want: bool| -> Vec<f64> {
                (0..k).filter(|&c| ((mask >> c) & 1 == 1) == want).map(|c| totals[c]).collect()
            };
            let fit = complete_cluster_itt(&pick(&adj1, true), &pick(&adj0, false), rows).unwrap();
            count += 1.0;
            mean_effect += fit.effect;
            mean_square += (fit.effect - truth).powi(2);
            mean_bound += fit.variance_upper_bound;
        }
        assert!((count - 70.0).abs() < 1e-12);
        close(
            mean_effect / count,
            truth,
            1e-12,
            &format!("unbiased for the unadjusted ITT (theta {theta})"),
        );
        let s_tau = variance(&effect);
        let exact =
            scale * scale * (variance(&adj1) / 4.0 + variance(&adj0) / 4.0 - s_tau / k as f64);
        close(mean_square / count, exact, 1e-10, &format!("exact variance (theta {theta})"));
        let expected_bound = scale * scale * (variance(&adj1) / 4.0 + variance(&adj0) / 4.0);
        close(
            mean_bound / count,
            expected_bound,
            1e-10,
            &format!("mean variance bound (theta {theta})"),
        );
        close(
            mean_bound / count - mean_square / count,
            scale * scale * s_tau / k as f64,
            1e-10,
            &format!("conservativeness gap equals S_tau^2 / K (theta {theta})"),
        );
    }
}
