//! Randomized-trial arm posteriors against independent references.
//!
//! References: `scipy.integrate.quad` of the Beta (Student-t) densities against the other arm's
//! CDF for the two-arm superpopulation laws; brute-force enumeration of the two Beta-Binomial
//! missing-count laws for the exact finite-population law; and 2e7-draw Monte Carlo from the
//! hierarchical model for the 2x2 factorial contrasts and the Gaussian finite-population law
//! (tolerances sit above the Monte Carlo error).

use antecedent_estimate::multi_arm::estimate_multi_arm;
use antecedent_estimate::randomized_posterior::{
    ArmBetaPrior, ArmContrastRequest, ArmNormalPrior, ArmOutcomeModel, ArmPosteriorSpec,
    RandomizedEstimand, posterior_arm_contrasts, posterior_factorial_2x2, posterior_two_arm,
};
use antecedent_stats::adaptive_inference::{BetaPosterior, probability_superior};

#[path = "common/checks.rs"]
mod checks;

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!((actual - expected).abs() <= tolerance, "{what}: {actual} vs {expected}");
}

const UNIFORM: ArmOutcomeModel = ArmOutcomeModel::Bernoulli(ArmBetaPrior { alpha: 1.0, beta: 1.0 });

fn spec(model: ArmOutcomeModel, estimand: RandomizedEstimand) -> ArmPosteriorSpec {
    ArmPosteriorSpec { model, estimand, credible_level: 0.9 }
}

/// Treated: successes 3, failures 1 (4 units); control: successes 2, failures 4 (6 units).
fn binary_trial() -> (Vec<f64>, Vec<bool>) {
    let rows = [
        (true, 1.0),
        (false, 1.0),
        (false, 0.0),
        (true, 1.0),
        (false, 0.0),
        (true, 0.0),
        (false, 1.0),
        (true, 1.0),
        (false, 0.0),
        (false, 0.0),
    ];
    (rows.iter().map(|r| r.1).collect(), rows.iter().map(|r| r.0).collect())
}

#[test]
fn superpopulation_two_arm_matches_quadrature_of_the_beta_difference() {
    let (y, treated) = binary_trial();
    let posterior = posterior_two_arm(
        &y,
        &treated,
        &[0.5; 10],
        &spec(UNIFORM, RandomizedEstimand::Superpopulation),
    )
    .unwrap();
    let c = &posterior.contrasts[0];
    // Beta(4, 2) minus Beta(3, 5): mean 4/6 - 3/8, variance 8/252 + 15/576.
    close(c.mean, 0.291_666_666_667, 1e-3, "mean");
    close(c.standard_deviation, 0.240_390_720_313, 1e-3, "sd");
    // scipy quad / brentq of P(theta1 - theta0 <= d).
    close(c.interval[0], -0.126_115_479_034, 3e-3, "5% quantile");
    close(c.interval[1], 0.665_059_267_836, 3e-3, "95% quantile");
    close(c.probability_positive, 29.0 / 33.0, 3e-3, "P(delta > 0)");
    let exact = probability_superior(
        BetaPosterior::new(4.0, 2.0).unwrap(),
        BetaPosterior::new(3.0, 5.0).unwrap(),
        &|| false,
    )
    .unwrap();
    close(c.probability_positive, exact, 3e-3, "against the incomplete-beta quadrature");
    assert_eq!(posterior.arms[1].observed_units, 4);
    close(posterior.arms[1].mean, 4.0 / 6.0, 1e-12, "treated arm mean");
}

#[test]
fn finite_population_two_arm_is_the_exact_beta_binomial_convolution() {
    let (y, treated) = binary_trial();
    let posterior = posterior_two_arm(
        &y,
        &treated,
        &[0.5; 10],
        &spec(UNIFORM, RandomizedEstimand::FinitePopulation),
    )
    .unwrap();
    let c = &posterior.contrasts[0];
    // Enumeration of BetaBinomial(6; 4, 2) and BetaBinomial(4; 3, 5) missing counts.
    close(c.mean, 0.35, 1e-12, "mean");
    close(c.standard_deviation, 0.188_034_951_158, 1e-9, "sd");
    close(c.probability_positive, 0.934_474_616_293, 1e-9, "P(delta > 0)");
    close(c.interval[0], 0.0, 1e-12, "lower endpoint");
    close(c.interval[1], 0.6, 1e-12, "upper endpoint");
    // The finite-population law is tighter than the superpopulation one (0.188 < 0.240).
    let superpopulation = posterior_two_arm(
        &y,
        &treated,
        &[0.5; 10],
        &spec(UNIFORM, RandomizedEstimand::Superpopulation),
    )
    .unwrap();
    assert!(c.standard_deviation < superpopulation.contrasts[0].standard_deviation);
    assert_eq!(c.estimand, RandomizedEstimand::FinitePopulation);
}

#[test]
fn factorial_main_effects_and_interaction_match_monte_carlo() {
    // Cells 00, 01, 10, 11 with 10 units each and 3, 5, 6, 9 successes.
    let successes = [3_usize, 5, 6, 9];
    let (mut y, mut a, mut b) = (vec![], vec![], vec![]);
    for (cell, &s) in successes.iter().enumerate() {
        for r in 0..10 {
            y.push(if r < s { 1.0 } else { 0.0 });
            a.push(cell >= 2);
            b.push(cell % 2 == 1);
        }
    }
    let posterior = posterior_factorial_2x2(
        &y,
        &a,
        &b,
        &[0.5; 40],
        &[0.5; 40],
        &spec(UNIFORM, RandomizedEstimand::Superpopulation),
    )
    .unwrap();
    // (exact mean, exact sd, Monte Carlo P(>0), Monte Carlo 5% and 95% quantiles).
    let expected = [
        (0.291_666_666_667, 0.128_165_057_84, 0.986_184, 0.075_764, 0.497_717),
        (0.208_333_333_333, 0.128_165_057_84, 0.945_542, -0.005_511, 0.416_340),
        (0.083_333_333_333, 0.256_330_115_68, 0.625_345, -0.337_071, 0.506_756),
    ];
    for (c, (mean, sd, positive, lo, hi)) in posterior.contrasts.iter().zip(expected) {
        close(c.mean, mean, 1e-3, "mean");
        close(c.standard_deviation, sd, 1e-3, "sd");
        close(c.probability_positive, positive, 4e-3, "P(> 0)");
        close(c.interval[0], lo, 4e-3, "5% quantile");
        close(c.interval[1], hi, 4e-3, "95% quantile");
    }
    // The interaction weights are [1, -1, -1, 1] and the main effects are half-differences.
    assert_eq!(posterior.contrasts[2].weights, vec![1.0, -1.0, -1.0, 1.0]);
}

#[test]
fn gaussian_arms_superpopulation_and_finite_population() {
    let treated_values = [2.1, 1.7, 2.9, 2.4];
    let control_values = [1.0, 0.4, 1.3, 0.9, 0.2];
    let mut y = treated_values.to_vec();
    y.extend(control_values);
    let treated: Vec<bool> = (0..9).map(|i| i < 4).collect();
    let model = ArmOutcomeModel::Gaussian(ArmNormalPrior {
        mean: 0.0,
        pseudo_units: 0.5,
        variance_shape: 2.0,
        variance_scale: 1.0,
    });
    let sp = posterior_two_arm(
        &y,
        &treated,
        &[0.5; 9],
        &spec(model, RandomizedEstimand::Superpopulation),
    )
    .unwrap();
    let c = &sp.contrasts[0];
    // Difference of two Student-t marginals (scipy quad).
    close(c.mean, 1.331_313_131_313, 1e-3, "SP mean");
    close(c.standard_deviation, 0.517_256_026_302, 2e-3, "SP sd");
    close(c.interval[0], 0.491_327_247_833, 4e-3, "SP 5%");
    close(c.interval[1], 2.171_299_030_231, 4e-3, "SP 95%");
    close(c.probability_positive, 0.992_294_762_214, 3e-3, "SP P(> 0)");

    let fp = posterior_two_arm(
        &y,
        &treated,
        &[0.5; 9],
        &spec(model, RandomizedEstimand::FinitePopulation),
    )
    .unwrap();
    let c = &fp.contrasts[0];
    // Hierarchical Monte Carlo of the missing sums; the mean is exact.
    close(c.mean, 1.405_274_971_942, 2e-3, "FP mean");
    close(c.standard_deviation, 0.384_068_652_6, 4e-3, "FP sd");
    close(c.interval[0], 0.782_124_6, 6e-3, "FP 5%");
    close(c.interval[1], 2.028_485_7, 6e-3, "FP 95%");
    close(c.probability_positive, 0.999_03, 2e-3, "FP P(> 0)");
}

#[test]
fn posterior_is_a_function_of_the_observed_counts_not_of_the_known_probabilities() {
    let (y, treated) = binary_trial();
    let at = |p: f64| {
        posterior_two_arm(
            &y,
            &treated,
            &[p; 10],
            &spec(UNIFORM, RandomizedEstimand::Superpopulation),
        )
        .unwrap()
    };
    let (even, skewed) = (at(0.5), at(0.3));
    assert_eq!(even.contrasts, skewed.contrasts);
    assert_ne!(even.evidence.arm_probabilities, skewed.evidence.arm_probabilities);
    close(skewed.evidence.arm_probabilities[1], 0.3, 1e-12, "retained treated probability");
    assert_eq!(skewed.evidence.units, 10);
}

#[test]
fn large_balanced_trial_posterior_mean_agrees_with_the_horvitz_thompson_target() {
    let n = 2000;
    let treated: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    // 1000 units per arm; treated success rate 0.6, control 0.3.
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let k = i / 2;
            let rate = if i % 2 == 0 { 6 } else { 3 };
            if k % 10 < rate { 1.0 } else { 0.0 }
        })
        .collect();
    let assignment: Vec<usize> = treated.iter().map(|t| usize::from(*t)).collect();
    let probabilities = vec![vec![0.5, 0.5]; n];
    let ht = estimate_multi_arm(&y, &assignment, &probabilities).unwrap();
    let ht_effect = ht[1].value - ht[0].value;
    let posterior = posterior_two_arm(
        &y,
        &treated,
        &vec![0.5; n],
        &spec(UNIFORM, RandomizedEstimand::Superpopulation),
    )
    .unwrap();
    close(ht_effect, 0.3, 1e-12, "HT contrast");
    close(posterior.contrasts[0].mean, ht_effect, 2e-3, "posterior mean against HT");
    // Posterior sd near the binomial standard error sqrt(.24/1000 + .21/1000).
    close(posterior.contrasts[0].standard_deviation, (0.45_f64 / 1000.0).sqrt(), 1e-3, "sd");
}

#[test]
fn refusals_name_the_missing_premise_and_the_next_attempt() {
    let (y, treated) = binary_trial();
    let mut varying = vec![0.5; 10];
    varying[3] = 0.4;
    let error = posterior_two_arm(
        &y,
        &treated,
        &varying,
        &spec(UNIFORM, RandomizedEstimand::Superpopulation),
    )
    .unwrap_err();
    let (code, message) = checks::coded_refusal(error);
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("randomized_posterior.varying_propensity"), "{message}");
    assert!(message.contains("stratum") && message.contains("Horvitz-Thompson"), "{message}");

    let mut real = y.clone();
    real[0] = 0.7;
    let error = posterior_two_arm(
        &real,
        &treated,
        &[0.5; 10],
        &spec(UNIFORM, RandomizedEstimand::Superpopulation),
    )
    .unwrap_err();
    assert!(checks::coded_refusal(error).1.contains("randomized_posterior.non_binary_outcome"));

    let all_treated = vec![true; 10];
    let error = posterior_two_arm(
        &y,
        &all_treated,
        &[0.5; 10],
        &spec(UNIFORM, RandomizedEstimand::Superpopulation),
    )
    .unwrap_err();
    let (code, _) = checks::coded_refusal(error);
    assert_eq!(code, "arm_not_populated");

    let bad_prior = ArmOutcomeModel::Bernoulli(ArmBetaPrior { alpha: 0.0, beta: 1.0 });
    let error = posterior_two_arm(
        &y,
        &treated,
        &[0.5; 10],
        &spec(bad_prior, RandomizedEstimand::Superpopulation),
    )
    .unwrap_err();
    assert!(checks::coded_refusal(error).1.contains("randomized_posterior.invalid_prior"));

    let assignment: Vec<usize> = treated.iter().map(|t| usize::from(*t)).collect();
    let rows = vec![vec![0.5, 0.5]; 10];
    for weights in [vec![-1.0, 0.7], vec![-1.0, 2.0], vec![1.0]] {
        let error = posterior_arm_contrasts(&ArmContrastRequest {
            outcome: &y,
            assignment: &assignment,
            probabilities: &rows,
            model: UNIFORM,
            contrasts: &[weights],
            estimand: RandomizedEstimand::Superpopulation,
            credible_level: 0.9,
        })
        .unwrap_err();
        assert!(checks::coded_refusal(error).1.contains("randomized_posterior.invalid_contrast"));
    }
    let error = posterior_two_arm(
        &y,
        &treated,
        &[0.5; 10],
        &ArmPosteriorSpec {
            model: UNIFORM,
            estimand: RandomizedEstimand::Superpopulation,
            credible_level: 1.0,
        },
    )
    .unwrap_err();
    assert!(checks::coded_refusal(error).1.contains("randomized_posterior.invalid_level"));
}

#[test]
fn three_arm_pairwise_contrasts_follow_each_pair() {
    // Arm 0: 1 of 3, arm 1: 2 of 3, arm 2: 3 of 3 successes; uniform priors give Beta(2, 3),
    // Beta(3, 2) and Beta(4, 1) with means 0.4, 0.6 and 0.8.
    let y = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    let assignment = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    let rows = vec![vec![1.0 / 3.0; 3]; 9];
    let posterior = posterior_arm_contrasts(&ArmContrastRequest {
        outcome: &y,
        assignment: &assignment,
        probabilities: &rows,
        model: UNIFORM,
        contrasts: &[vec![-1.0, 1.0, 0.0], vec![-1.0, 0.0, 1.0], vec![0.0, -1.0, 1.0]],
        estimand: RandomizedEstimand::Superpopulation,
        credible_level: 0.9,
    })
    .unwrap();
    close(posterior.contrasts[0].mean, 0.2, 2e-3, "1 - 0");
    close(posterior.contrasts[1].mean, 0.8 - 0.4, 2e-3, "2 - 0");
    close(posterior.contrasts[2].mean, 0.8 - 0.6, 2e-3, "2 - 1");
    let exact = probability_superior(
        BetaPosterior::new(4.0, 1.0).unwrap(),
        BetaPosterior::new(2.0, 3.0).unwrap(),
        &|| false,
    )
    .unwrap();
    close(posterior.contrasts[1].probability_positive, exact, 3e-3, "P(2 > 0)");
}
